"""Measure retrieval and refusal on your own labeled questions.

Usage:
    python evaluate.py                       # sweep chunk sizes x methods, write results.md + failures.csv
    python evaluate.py --apply-best          # also save the best chunks.json + data/config.json for the app
    python evaluate.py --sweep 180 --model none    # quick TF-IDF-only run

Only rows with verified=yes in eval_questions.csv are used, so a label you have not
checked against the PDF can never silently inflate your numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

import settings
from answer import top_score
from chunking import build_chunks
from retrieval import retriever_from_chunks

K_VALUES = (1, 3, 5)


def load_questions(path) -> tuple[list[dict], int]:
    """Return (usable questions, number of skipped rows)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    questions, skipped = [], 0
    for r in rows:
        text = (r.get("question") or "").strip()
        if (r.get("verified") or "").strip().lower() != "yes" or not text or text.startswith("<"):
            skipped += 1
            continue
        answerable = (r.get("answerable") or "").strip().lower() == "yes"
        expected: set[tuple[str, int]] = set()
        if answerable:
            doc = (r.get("expected_doc") or "").strip()
            for part in re.split(r"[|;]", r.get("expected_page") or ""):
                if part.strip():
                    expected.add((doc, int(part)))
            if not doc or not expected:
                raise ValueError(f"question {r.get('id')}: answerable rows need expected_doc and expected_page")
        questions.append({"id": r.get("id", ""), "question": text, "answerable": answerable,
                          "expected": expected, "notes": r.get("notes", "")})
    return questions, skipped


def run_method(questions, retriever, method: str, k: int = max(K_VALUES)) -> list[dict]:
    rows = []
    for q in questions:
        hits = retriever.search(q["question"], k=k, method=method)
        found = [(h.chunk["doc"], h.chunk["page"]) for h in hits]
        rank = None
        if q["answerable"]:
            rank = next((i for i, f in enumerate(found, start=1) if f in q["expected"]), None)
        rows.append({**q, "hits": hits, "rank": rank, "top_score": top_score(hits)})
    return rows


def best_tau(rows) -> tuple[float, float, float]:
    """Pick the refusal threshold; return (tau, answered_rate_on_answerable, refused_rate_on_unanswerable).

    A question is handled well if: answerable, correct page retrieved, score >= tau;
    or unanswerable and score < tau. Ties prefer the higher tau (refusing is safer than a wrong answer).
    """
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    scores = sorted({r["top_score"] for r in rows})
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [(scores[-1] + 1e-6) if scores else 1.0]
    best = None
    for tau in candidates:
        good_a = sum(1 for r in answerable if r["rank"] is not None and r["top_score"] >= tau)
        good_u = sum(1 for r in unanswerable if r["top_score"] < tau)
        key = ((good_a + good_u) / max(len(rows), 1), tau)
        if best is None or key >= best[0]:
            best = (key, good_a, good_u)
    (_, tau), good_a, good_u = best
    return (tau,
            good_a / len(answerable) if answerable else 0.0,
            good_u / len(unanswerable) if unanswerable else 0.0)


def summarise(rows) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    n = max(len(answerable), 1)
    out = {f"hit@{k}": sum(1 for r in answerable if r["rank"] and r["rank"] <= k) / n for k in K_VALUES}
    out["mrr"] = sum(1 / r["rank"] for r in answerable if r["rank"]) / n
    out["tau"], out["answered"], out["refused"] = best_tau(rows)
    return out


def format_table(results: list[dict]) -> str:
    head = ("| chunk words | method | hit@1 | hit@3 | hit@5 | MRR | answered (answerable) | "
            "refused (unanswerable) | tau |\n|---|---|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| {r['chunk_words']} | {r['method']} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['hit@5']:.2f} | "
        f"{r['mrr']:.2f} | {r['answered']:.2f} | {r['refused']:.2f} | {r['tau']:.3f} |\n" for r in results)
    return head + body


def write_failures(path: Path, rows, tau: float) -> int:
    count = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "kind", "expected", "rank", "top_score", "top3"])
        for r in rows:
            if r["answerable"] and (r["rank"] is None or r["rank"] > 1):
                kind = "missed" if r["rank"] is None else f"found at rank {r['rank']}"
            elif not r["answerable"] and r["top_score"] >= tau:
                kind = "wrongly answered (should refuse)"
            else:
                continue
            top3 = "; ".join(f"{h.chunk['doc']} p.{h.chunk['page']} ({h.score:.2f})" for h in r["hits"][:3])
            expected = "; ".join(f"{d} p.{p}" for d, p in sorted(r["expected"]))
            w.writerow([r["id"], r["question"], kind, expected, r["rank"], f"{r['top_score']:.3f}", top3])
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default=str(settings.EVAL_PATH))
    ap.add_argument("--pages", default=str(settings.PAGES_PATH))
    ap.add_argument("--data", default=str(settings.DATA_DIR))
    ap.add_argument("--model", default=settings.EMBED_MODEL, help="embedding model name, or 'none'")
    ap.add_argument("--sweep", default="120,180,250", help="comma-separated chunk sizes in words")
    ap.add_argument("--overlap", type=int, default=settings.OVERLAP)
    ap.add_argument("--out", default=str(settings.ROOT))
    ap.add_argument("--apply-best", action="store_true", help="write chunks.json and config.json for the best setting")
    args = ap.parse_args(argv)

    questions, skipped = load_questions(args.eval)
    n_ans = sum(q["answerable"] for q in questions)
    n_unans = len(questions) - n_ans
    print(f"Using {len(questions)} verified questions ({n_ans} answerable, {n_unans} unanswerable); "
          f"skipped {skipped} unverified or placeholder rows.")
    if n_ans == 0:
        print("No verified answerable questions. Fill eval_questions.csv and set verified=yes first.")
        return 1
    if n_ans < 20 or n_unans < 5:
        print("WARNING: aim for at least 20 answerable and 5 unanswerable verified questions; "
              "smaller sets give unreliable numbers.")

    pages = json.loads(Path(args.pages).read_text(encoding="utf-8"))
    results, kept = [], {}
    for size in [int(s) for s in args.sweep.split(",") if s.strip()]:
        overlap = min(args.overlap, size // 4)   # a small chunk size must not be swallowed by the overlap
        chunks = build_chunks(pages, size, overlap, settings.MIN_WORDS)
        if not chunks:
            continue
        retriever = retriever_from_chunks(chunks, args.model, args.data)
        for method in retriever.available_methods():
            rows = run_method(questions, retriever, method)
            summary = {"chunk_words": size, "overlap": overlap, "method": method,
                       "n_chunks": len(chunks), **summarise(rows)}
            results.append(summary)
            kept[(size, method)] = (rows, chunks)
    if not results:
        print("No chunks produced; run ingest.py first.")
        return 1

    best = max(results, key=lambda r: (r["hit@5"], r["mrr"], r["hit@1"]))   # ties keep the first (hybrid, small)
    table = format_table(results)
    print("\n" + table)
    print(f"Best: chunk_words={best['chunk_words']} method={best['method']} "
          f"hit@5={best['hit@5']:.2f} mrr={best['mrr']:.2f} tau={best['tau']:.3f}")

    out_dir = Path(args.out)
    rows, chunks = kept[(best["chunk_words"], best["method"])]
    n_fail = write_failures(out_dir / "failures.csv", rows, best["tau"])
    (out_dir / "results.md").write_text(
        f"# Evaluation results ({date.today().isoformat()})\n\n"
        f"Questions: {len(questions)} verified ({n_ans} answerable, {n_unans} unanswerable). "
        f"Corpus: {len(set(c['doc'] for c in chunks))} documents, {best['n_chunks']} chunks at the best setting.\n"
        f"Embedding model: {args.model}.\n\n{table}\n"
        "Notes: hit@k = the labeled (document, page) is among the top k chunks. "
        "tau (refusal threshold) is tuned on these same questions, so the answered/refused columns are optimistic. "
        f"{n_fail} questions are listed in failures.csv for failure analysis.\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'results.md'} and {out_dir / 'failures.csv'} ({n_fail} rows)")

    if args.apply_best:
        data_dir = Path(args.data)
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
        settings.save_config({
            "method": best["method"], "chunk_words": best["chunk_words"], "overlap": best["overlap"],
            "embedder": args.model, "tau": round(best["tau"], 4), "n_questions": len(questions),
            "tuned_on": Path(args.eval).name,
        }, data_dir / "config.json")
        print(f"Saved {data_dir / 'chunks.json'} and {data_dir / 'config.json'} for the app.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
