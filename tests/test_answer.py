from answer import (NOT_FOUND, SYSTEM_PROMPT, answer_question, best_sentences, build_prompt,
                    parse_citations)
from conftest import FakeLLM
from llm import LLMError

Q = "Is interest subsidy available for both undergraduate and postgraduate courses?"


def test_refuses_when_best_score_is_below_tau(tfidf_retriever):
    ans = answer_question("zzzz qqqq", tfidf_retriever, method="tfidf", tau=0.1)
    assert not ans.answered and ans.mode == "refused" and ans.text == NOT_FOUND
    assert ans.citations == [] and ans.closest


def test_extractive_answer_has_citations(tfidf_retriever):
    ans = answer_question(Q, tfidf_retriever, method="tfidf", tau=0.0)
    assert ans.answered and ans.mode == "extractive"
    assert ans.citations[0].doc == "loan.pdf" and ans.citations[0].page == 2
    assert "[1]" in ans.text and "only once" in ans.text


def test_llm_answer_keeps_only_cited_passages(tfidf_retriever):
    llm = FakeLLM(reply="Subsidy is available only once, for UG or PG. [1]")
    ans = answer_question(Q, tfidf_retriever, method="tfidf", llm=llm)
    assert ans.mode == "llm" and [c.n for c in ans.citations] == [1]
    assert (ans.citations[0].doc, ans.citations[0].page) == ("loan.pdf", 2)
    assert "Passages:" in llm.prompts[0]


def test_llm_not_found_becomes_refusal(tfidf_retriever):
    ans = answer_question(Q, tfidf_retriever, method="tfidf", llm=FakeLLM(reply="NOT_FOUND"))
    assert not ans.answered and ans.mode == "refused"


def test_llm_answer_without_valid_citation_falls_back(tfidf_retriever):
    ans = answer_question(Q, tfidf_retriever, method="tfidf", llm=FakeLLM(reply="Yes, definitely."))
    assert ans.mode == "extractive" and "no valid citation" in ans.note


def test_invalid_citation_numbers_are_ignored_and_reported(tfidf_retriever):
    ans = answer_question(Q, tfidf_retriever, method="tfidf", llm=FakeLLM(reply="Only once [1] and also [9]."), k=3)
    assert ans.mode == "llm" and [c.n for c in ans.citations] == [1]
    assert "9" in ans.note


def test_llm_failure_falls_back_to_extractive(tfidf_retriever):
    ans = answer_question(Q, tfidf_retriever, method="tfidf", llm=FakeLLM(error=LLMError("HTTP 429")))
    assert ans.answered and ans.mode == "extractive" and "unavailable" in ans.note


def test_source_metadata_reaches_the_citation(tfidf_retriever):
    sources = {"loan.pdf": {"title": "Model Loan Scheme", "publisher": "IBA", "url": "https://example.org/x", "as_of": "2022"}}
    ans = answer_question(Q, tfidf_retriever, method="tfidf", sources=sources)
    c = ans.citations[0]
    assert (c.title, c.publisher, c.url, c.as_of) == ("Model Loan Scheme", "IBA", "https://example.org/x", "2022")


def test_prompt_numbers_passages_and_system_prompt_marks_them_untrusted(tfidf_retriever):
    hits = tfidf_retriever.search(Q, k=2, method="tfidf")
    prompt = build_prompt(Q, hits)
    assert "[1] (" in prompt and "[2] (" in prompt and Q in prompt
    assert "untrusted" in SYSTEM_PROMPT and "NOT_FOUND" in SYSTEM_PROMPT


def test_parse_citations():
    assert parse_citations("a [2] b [1] c [2] d [7]", 3) == ([2, 1], [7])
    assert parse_citations("no citations", 3) == ([], [])


def test_best_sentences_prefers_overlap_and_keeps_order():
    text = "Fees are paid yearly. Aadhaar is mandatory for payment. The portal opens in July."
    assert best_sentences(text, "Is Aadhaar mandatory?", n=1) == ["Aadhaar is mandatory for payment."]
    assert best_sentences(text, "portal Aadhaar", n=2) == ["Aadhaar is mandatory for payment.", "The portal opens in July."]


def test_rupee_abbreviation_is_not_split():
    text = "The amount is Rs. 50,000 per year. Students must apply online."
    assert best_sentences(text, "amount per year", n=1) == ["The amount is Rs. 50,000 per year."]
