"""Step 1: PDFs in docs/ -> data/pages.json and data/chunks.json.

Usage:
    python ingest.py                 # PDFs that have a text layer
    python ingest.py --ocr           # also OCR pages with no text (scanned PDFs)

Extraction is the slow step, so its result is cached per file in
data/pages_cache/. Chunking is fast and is re-done on every run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import settings
from chunking import build_chunks


def normalise_line(line: str) -> str:
    """Lower-case and replace digits, so 'Page 3 of 10' and 'Page 4 of 10' match."""
    return re.sub(r"\d+", "#", line.strip().lower())


def strip_repeated_lines(page_texts: list[str]) -> list[str]:
    """Remove running headers/footers: short lines that appear on most pages."""
    n = len(page_texts)
    if n < 4:
        return page_texts
    counts: Counter = Counter()
    for text in page_texts:
        counts.update({normalise_line(ln) for ln in text.splitlines() if ln.strip()})
    limit = max(3, int(0.6 * n))
    repeated = {line for line, c in counts.items() if c >= limit and len(line) < 120}
    return [
        "\n".join(ln for ln in text.splitlines() if normalise_line(ln) not in repeated)
        for text in page_texts
    ]


def ocr_pages(pdf_path: Path, indices: list[int]) -> dict[int, str]:
    """OCR the given 0-based page indices. Needs: pip install -r requirements-ocr.txt"""
    import numpy as np
    import pypdfium2 as pdfium
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    pdf = pdfium.PdfDocument(str(pdf_path))
    out: dict[int, str] = {}
    try:
        for n, i in enumerate(indices, start=1):
            print(f"    OCR page {i + 1} ({n}/{len(indices)})", flush=True)
            image = pdf[i].render(scale=2).to_pil().convert("RGB")
            result, _ = engine(np.array(image))
            out[i] = " ".join(item[1] for item in (result or []))
    finally:
        pdf.close()
    return out


def extract_pdf(pdf_path: Path, use_ocr: bool = False, ocr_fn=ocr_pages) -> tuple[list[dict], dict]:
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        raw = [page.extract_text() or "" for page in pdf.pages]
    texts = strip_repeated_lines(raw)
    weak = [i for i, t in enumerate(raw) if len(t.split()) < settings.MIN_WORDS]
    ocr_done = 0
    if use_ocr and weak:
        for i, text in ocr_fn(pdf_path, weak).items():
            texts[i] = text
            ocr_done += 1
    pages = [{"doc": pdf_path.name, "page": i, "text": t} for i, t in enumerate(texts, start=1)]
    stats = {
        "pages": len(pages),
        "weak_pages": len(weak),
        "ocr_pages": ocr_done,
        "empty_pages": sum(1 for p in pages if len(p["text"].split()) < settings.MIN_WORDS),
    }
    return pages, stats


def extract_cached(pdf_path: Path, use_ocr: bool, cache_dir: Path) -> tuple[list[dict], dict]:
    digest = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:16]
    cache = cache_dir / f"{digest}_{int(use_ocr)}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        for p in data["pages"]:
            p["doc"] = pdf_path.name  # same bytes may be renamed
        return data["pages"], data["stats"]
    pages, stats = extract_pdf(pdf_path, use_ocr)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"pages": pages, "stats": stats}, ensure_ascii=False), encoding="utf-8")
    return pages, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", default=str(settings.DOCS_DIR))
    ap.add_argument("--data", default=str(settings.DATA_DIR))
    ap.add_argument("--ocr", action="store_true", help="OCR pages that have no text layer")
    ap.add_argument("--chunk-words", type=int, default=settings.CHUNK_WORDS)
    ap.add_argument("--overlap", type=int, default=settings.OVERLAP)
    args = ap.parse_args(argv)

    docs_dir, data_dir = Path(args.docs), Path(args.data)
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {docs_dir}. Put your public PDFs there first.")
        return 1

    all_pages: list[dict] = []
    print(f"{'file':52} {'pages':>5} {'empty':>5} {'ocr':>4} {'chunks':>6}")
    for pdf_path in pdfs:
        try:
            pages, stats = extract_cached(pdf_path, args.ocr, data_dir / "pages_cache")
        except Exception as err:  # locked, encrypted or corrupt PDF: report and continue
            print(f"{pdf_path.name:52} SKIPPED ({type(err).__name__})")
            continue
        all_pages.extend(pages)
        n_chunks = len(build_chunks(pages, args.chunk_words, args.overlap, settings.MIN_WORDS))
        flag = ""
        if stats["pages"] and stats["empty_pages"] > stats["pages"] / 2:
            flag = "   <-- mostly empty: scanned PDF? run again with --ocr"
        print(f"{pdf_path.name:52} {stats['pages']:5d} {stats['empty_pages']:5d} "
              f"{stats['ocr_pages']:4d} {n_chunks:6d}{flag}")

    chunks = build_chunks(all_pages, args.chunk_words, args.overlap, settings.MIN_WORDS)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "pages.json").write_text(json.dumps(all_pages, ensure_ascii=False), encoding="utf-8")
    (data_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nTotal: {len(all_pages)} pages -> {len(chunks)} chunks in {data_dir}")
    if len(chunks) < 100:
        print("WARNING: fewer than 100 chunks. With so few chunks, hit@k looks better than it really is; "
              "add more text-based documents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
