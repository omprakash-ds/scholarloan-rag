import textwrap

import pytest

import ingest

reportlab = pytest.importorskip("reportlab")
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

BODIES = [
    "Applicants must submit income certificate caste certificate and marksheets before the last date every year. ",
    "The scholarship is paid directly into the bank account linked with Aadhaar of the selected student only. ",
    "Students who fail to get promoted to the next class lose the scholarship for that academic year entirely. ",
    "Grievances regarding the provisional merit list may be raised on the portal within one month of publication. ",
    "The gap between passing the qualifying examination and admission should not be more than two years. ",
]


def make_pdf(path, bodies, header=True, font=11):
    c = canvas.Canvas(str(path))
    total = len(bodies)
    for n, body in enumerate(bodies, start=1):
        c.setFont("Helvetica", font)
        if header:
            c.drawString(72, 800, "National Scholarship Guidelines Header")
            c.drawString(72, 40, f"Page {n} of {total}")
        y = 700
        for line in textwrap.wrap(body * 2, 70) if body else []:
            c.drawString(72, y, line)
            y -= 18
        c.showPage()
    c.save()


def test_strip_repeated_lines_removes_headers_and_footers():
    bodies = ["alpha eligibility rules", "beta payment rules", "gamma renewal rules", "delta grievance rules", "epsilon limits"]
    pages = [f"Running Header\n{body}\nPage {i} of 5" for i, body in enumerate(bodies, start=1)]
    cleaned = ingest.strip_repeated_lines(pages)
    assert all("Running Header" not in p and "Page" not in p for p in cleaned)
    assert [p.strip() for p in cleaned] == bodies


def test_short_documents_are_left_alone():
    pages = ["Header\nbody a", "Header\nbody b"]
    assert ingest.strip_repeated_lines(pages) == pages


def test_extract_pdf_removes_header_and_keeps_body(tmp_path):
    pdf = tmp_path / "guide.pdf"
    make_pdf(pdf, BODIES)
    pages, stats = ingest.extract_pdf(pdf)
    assert stats["pages"] == 5 and stats["empty_pages"] == 0 and stats["weak_pages"] == 0
    assert all("Header" not in p["text"] for p in pages)
    assert "Aadhaar" in pages[1]["text"] and pages[1]["page"] == 2 and pages[1]["doc"] == "guide.pdf"


def test_blank_page_triggers_injected_ocr(tmp_path):
    pdf = tmp_path / "scan.pdf"
    make_pdf(pdf, [BODIES[0], "", BODIES[2], BODIES[3], BODIES[4]])
    seen = {}

    def fake_ocr(path, indices):
        seen["indices"] = indices
        return {i: "text recovered by ocr" for i in indices}

    pages, stats = ingest.extract_pdf(pdf, use_ocr=True, ocr_fn=fake_ocr)
    assert seen["indices"] == [1] and stats["ocr_pages"] == 1
    assert pages[1]["text"] == "text recovered by ocr"
    _, no_ocr = ingest.extract_pdf(pdf, use_ocr=False, ocr_fn=fake_ocr)
    assert no_ocr["empty_pages"] == 1 and no_ocr["ocr_pages"] == 0


def test_extract_cached_reuses_result(tmp_path, monkeypatch):
    pdf = tmp_path / "guide.pdf"
    make_pdf(pdf, BODIES)
    cache = tmp_path / "cache"
    first, _ = ingest.extract_cached(pdf, False, cache)
    monkeypatch.setattr(ingest, "extract_pdf", lambda *a, **k: (_ for _ in ()).throw(AssertionError("not cached")))
    second, _ = ingest.extract_cached(pdf, False, cache)
    assert first == second


def test_main_reports_and_writes_files(tmp_path, capsys):
    docs, data = tmp_path / "docs", tmp_path / "data"
    docs.mkdir()
    make_pdf(docs / "a.pdf", BODIES)
    (docs / "broken.pdf").write_bytes(b"this is not a pdf")
    assert ingest.main(["--docs", str(docs), "--data", str(data), "--chunk-words", "40", "--overlap", "5"]) == 0
    out = capsys.readouterr().out
    assert "a.pdf" in out and "broken.pdf" in out and "SKIPPED" in out
    assert (data / "pages.json").exists() and (data / "chunks.json").exists()


def test_main_without_pdfs_returns_error(tmp_path):
    (tmp_path / "docs").mkdir()
    assert ingest.main(["--docs", str(tmp_path / "docs"), "--data", str(tmp_path / "data")]) == 1


def test_real_ocr_on_an_image_only_pdf(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    pdfium = pytest.importorskip("pypdfium2")
    text_pdf = tmp_path / "text.pdf"
    make_pdf(text_pdf, ["Scholarship payment requires an Aadhaar linked bank account for every student."], header=False, font=22)
    doc = pdfium.PdfDocument(str(text_pdf))
    image = doc[0].render(scale=2).to_pil().convert("RGB")
    doc.close()
    scanned = tmp_path / "scanned.pdf"
    c = canvas.Canvas(str(scanned), pagesize=image.size)
    c.drawImage(ImageReader(image), 0, 0, width=image.size[0], height=image.size[1])
    c.save()
    pages, stats = ingest.extract_pdf(scanned, use_ocr=True)
    assert stats["ocr_pages"] == 1
    assert "aadhaar" in pages[0]["text"].lower() and "scholarship" in pages[0]["text"].lower()
