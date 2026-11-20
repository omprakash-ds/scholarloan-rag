import csv
import json

import pytest

import evaluate
from conftest import PAGES

HEADER = ["id", "question", "answerable", "expected_doc", "expected_page", "verified", "notes"]


def write_csv(path, rows, bom=False):
    with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


GOOD_ROWS = [
    [1, "Is interest subsidy available for both undergraduate and postgraduate courses?", "yes", "loan.pdf", "2", "yes", ""],
    [2, "Is an Aadhaar linked bank account mandatory for scholarship payment?", "yes", "scholarship.pdf", "2", "yes", ""],
    [3, "Which fees does the education loan cover, such as hostel and tuition?", "yes", "loan.pdf", "1", "yes", ""],
    [4, "How is selection done for the girl students scholarship?", "yes", "scholarship.pdf", "1", "yes", ""],
    [5, "What is the weather in Mumbai today?", "no", "", "", "yes", ""],
    [6, "Which cricket team won the world cup?", "no", "", "", "yes", ""],
]


def test_load_questions_skips_unverified_and_placeholders(tmp_path):
    path = tmp_path / "q.csv"
    write_csv(path, [
        [1, "A real question?", "yes", "loan.pdf", "1|2", "yes", ""],
        [2, "Not checked yet?", "yes", "loan.pdf", "1", "no", ""],
        [3, "<write a question>", "yes", "loan.pdf", "1", "yes", ""],
        [4, "Unanswerable one?", "no", "", "", "yes", ""],
    ], bom=True)
    questions, skipped = evaluate.load_questions(path)
    assert skipped == 2 and [q["id"] for q in questions] == ["1", "4"]
    assert questions[0]["expected"] == {("loan.pdf", 1), ("loan.pdf", 2)}
    assert questions[1]["expected"] == set()


def test_answerable_row_without_page_is_an_error(tmp_path):
    path = tmp_path / "q.csv"
    write_csv(path, [[1, "Question?", "yes", "loan.pdf", "", "yes", ""]])
    with pytest.raises(ValueError):
        evaluate.load_questions(path)


def test_run_and_summarise_on_tfidf(tmp_path, tfidf_retriever):
    path = tmp_path / "q.csv"
    write_csv(path, GOOD_ROWS)
    questions, _ = evaluate.load_questions(path)
    rows = evaluate.run_method(questions, tfidf_retriever, "tfidf")
    summary = evaluate.summarise(rows)
    assert summary["hit@5"] == 1.0 and summary["hit@1"] >= 0.75 and 0 < summary["mrr"] <= 1
    assert summary["refused"] >= 0.5     # off-topic questions score low


def test_best_tau_separates_the_two_groups():
    rows = [
        {"answerable": True, "rank": 1, "top_score": 0.8},
        {"answerable": True, "rank": 1, "top_score": 0.7},
        {"answerable": False, "rank": None, "top_score": 0.2},
        {"answerable": False, "rank": None, "top_score": 0.3},
    ]
    tau, answered, refused = evaluate.best_tau(rows)
    assert 0.3 < tau <= 0.7 and answered == 1.0 and refused == 1.0


def test_missed_retrieval_does_not_count_as_good_answer():
    rows = [{"answerable": True, "rank": None, "top_score": 0.9},
            {"answerable": False, "rank": None, "top_score": 0.1}]
    tau, answered, refused = evaluate.best_tau(rows)
    assert answered == 0.0 and refused == 1.0


def test_main_end_to_end_writes_reports_and_config(tmp_path):
    (tmp_path / "pages.json").write_text(json.dumps(PAGES), encoding="utf-8")
    write_csv(tmp_path / "q.csv", GOOD_ROWS)
    code = evaluate.main(["--eval", str(tmp_path / "q.csv"), "--pages", str(tmp_path / "pages.json"),
                          "--data", str(tmp_path), "--out", str(tmp_path), "--model", "none",
                          "--sweep", "30,60", "--apply-best"])
    assert code == 0
    assert "| chunk words | method |" in (tmp_path / "results.md").read_text(encoding="utf-8")
    assert (tmp_path / "failures.csv").exists() and (tmp_path / "chunks.json").exists()
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["method"] == "tfidf" and cfg["embedder"] == "none" and cfg["chunk_words"] in (30, 60)


def test_main_refuses_when_nothing_is_verified(tmp_path):
    (tmp_path / "pages.json").write_text(json.dumps(PAGES), encoding="utf-8")
    write_csv(tmp_path / "q.csv", [[1, "Q?", "yes", "loan.pdf", "1", "no", ""]])
    assert evaluate.main(["--eval", str(tmp_path / "q.csv"), "--pages", str(tmp_path / "pages.json"),
                          "--data", str(tmp_path), "--out", str(tmp_path), "--model", "none"]) == 1
