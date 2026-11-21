# 🎓 ScholarLoan RAG: Cited Indian Education Aid Q&A

> Ask about Indian education loans and scholarships. Get an answer with the **exact document, page and date**, or an honest "not found".

[![Live App](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://omprakash-ds-scholarloan-rag-app-tczgd0.streamlit.app/)
[![CI Tests](https://github.com/omprakash-ds/scholarloan-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/omprakash-ds/scholarloan-rag/actions/workflows/tests.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-54%20Passed-2EA44F?logo=pytest&logoColor=white)](#-run-locally--tests)
[![Hackathon](https://img.shields.io/badge/Hackathon-Build%20Your%20First%20AI%20App%202026-6C47FF)](https://devpost.com/)

**👉 Live demo: https://omprakash-ds-scholarloan-rag-app-tczgd0.streamlit.app/**

| **Cited answer with page and date** | **Refusal on an unsupported question** |
| :---: | :---: |
| ![Answer with page citations](assets/demo_answer.png) | ![Refusal on an unsupported question](assets/demo_refusal.png) |

---

## 📌 At a glance

| | |
|---|---|
| **Corpus** | 9 public PDFs, 237 searchable passages |
| **Retrieval** | TF-IDF + MiniLM embeddings, hybrid rank fusion |
| **Best setting** | 120-word chunks, hybrid |
| **hit@1 / hit@3 / hit@5** | 0.79 / 0.92 / 0.96 |
| **MRR** | 0.86 |
| **Refusal threshold** | tau = 0.437 (tuned on my labeled questions) |
| **Tests** | 54 automated tests, run in CI |
| **Needs an API key?** | No. Works in extractive mode without one |

---

## 🎯 Problem

A student in India comparing the **IBA Model Education Loan Scheme**, **PM-Vidyalaxmi**, **CSIS** and **AICTE scholarships** has to read long government and banking PDFs, several of them scanned, just to verify one rule such as margin, collateral or an income ceiling. A general chatbot can blend rules from similar schemes, gives no page-level audit trail, and may answer questions the documents never cover.

**ScholarLoan RAG** answers only from **9 public PDF documents**, shows `(document, page, as-of date)` for every source, checks that LLM citations point to passages that were actually retrieved, and refuses when retrieval confidence is below a threshold. It **reduces** unsupported answers. It does not eliminate them (see Results and Limitations).

---

## 🏗️ How it works

```text
PDFs --ingest.py--> pages (pdfplumber + OCR fallback) --chunking.py--> chunks (doc, page, 120 words, 30 overlap)
chunks --retrieval.py--> document name prepended --> TF-IDF (1,2)-grams + MiniLM dense --> hybrid ranking (RRF, k=60)
question --> top-k passages --> confidence (0.7*dense + 0.3*tfidf) < 0.437?
                 yes --> "not found" + closest low-confidence passages
                 no  --> answer.py: LLM answer with checked [n] citations, or cited extracted sentences (no API key)
evaluate.py: sweep of chunk size and method (hit@k, MRR, tau tuning, failures.csv)
app.py: Streamlit UI with latency and confidence shown per answer
```

### Design choices and why

| Choice | Reason |
|---|---|
| Per-page chunks | Every citation maps to an exact page |
| Hybrid retrieval | TF-IDF catches exact terms ("Aadhaar", "moratorium"); dense embeddings catch paraphrase; rank fusion combines them |
| Document name added to each chunk before indexing | Inner pages rarely mention the scheme name. This raised hit@1 from 0.33 to 0.79 on my test set |
| Brute-force NumPy search | Exact and fast at 237 chunks. A vector database adds nothing at this size |
| PDF text treated as untrusted data in the prompt | Instructions hidden inside a document are not followed |
| Citation check | An invalid `[n]` from the model is rejected and the app falls back to extracted sentences |
| Optional LLM | Any OpenAI-compatible endpoint via environment variables (see `.env.example`) |

---

## 📊 Results (my own labeled test set)

31 verified questions: **24 answerable** across all 9 documents and **7 unanswerable** (2 are near-misses: a scheme and a lender that are not in the corpus).

| chunk words | method | hit@1 | hit@3 | hit@5 | MRR | answered (answerable) | refused (unanswerable) | tau |
|---|---|---|---|---|---|---|---|---|
| **120** | **hybrid** | **0.79** | **0.92** | **0.96** | **0.86** | **0.96** | **0.71** | **0.437** |
| 120 | dense | 0.62 | 0.88 | 0.92 | 0.75 | 0.92 | 0.43 | 0.503 |
| 120 | tfidf | 0.46 | 0.88 | 0.96 | 0.66 | 0.92 | 0.86 | 0.584 |
| 180 | hybrid | 0.75 | 0.92 | 0.96 | 0.83 | 0.88 | 0.86 | 0.464 |
| 250 | hybrid | 0.58 | 0.83 | 0.88 | 0.71 | 0.79 | 1.00 | 0.447 |

Full sweep: [`results.md`](results.md).

**How to read it**
- hit@k means the labeled `(document, page)` is among the top k chunks.
- `tau` was tuned on these same 31 questions, so the answered/refused columns are **optimistic**. With only 7 unanswerable questions, one question moves the refusal rate by 14 points.
- **2 of 7 unanswerable questions were wrongly answered** (ids 16 and 17).
- **Held-out check:** 10 new questions written after tuning, including 2 answerable ones that do not name the scheme. Result: **X/6 answerable answered correctly, Y/4 out-of-scope refused.** *(run `python held_out_check.py` and fill in X and Y)*

---

## 🔍 Failure analysis

7 of 31 questions are listed in [`failures.csv`](failures.csv). Three examples:

1. **Missed (id 25):** "Can the Dr. Ambedkar overseas interest subsidy continue if the moratorium period is extended?"
   The labeled answer is on page 8. The top result was page 7 of the same document (score 0.648) and page 8 was not in the top 5.
   *Cause:* WRITE WHAT YOU FOUND AFTER READING PAGES 7 AND 8.
   *Planned fix:* chunks that cross a page break while keeping both page numbers.
2. **Wrongly answered (id 17):** "What interest rate does a specific bank charge on education loans?"
   Should be refused, but IBA p.7 scored 0.501, above tau.
   *Cause:* WRITE WHAT TEXT ON IBA P.7 MATCHED. A single similarity threshold cannot separate "same topic, different question" from "answerable".
   *Planned fix:* check that the entity named in the question (the bank) appears in the passage, or add an LLM answerability check.
3. **Right document, wrong page (id 11):** "Can I claim CSIS interest subsidy for both an undergraduate and a postgraduate course?"
   Labeled p.1, found at rank 4 behind p.4 and p.2 of the same PDF.
   *Cause:* CHECK WHETHER P.2 OR P.4 ALSO STATES THE RULE. If yes, page-level scoring undercounts.

---

## ⚠️ Limitations

- Scheme rules change. Each source shows its document date. **Confirm on the official source before applying.** This is not financial or legal advice.
- Small corpus: 9 documents, 237 passages. Schemes outside them (for example UGC Ishan Uday, or private lenders like HDFC Credila) are refused by design.
- `central_sector_scholarship_guidelines.pdf` applies **only to residents of Puducherry**.
- Most labeled questions name the scheme, while real students often do not. The held-out set is a first check, not proof.
- Retrieval is scored at page level, not sentence level, and the refusal threshold is tuned on the test questions.
- ADD: which PDFs have no date found, and that the IBA copy is hosted by a third party.

---

## 🧪 Run locally & tests

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows. macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt    # add requirements-ocr.txt only for scanned PDFs
python ingest.py                   # add --ocr for scanned PDFs
python evaluate.py --apply-best
streamlit run app.py
pip install -r requirements-dev.txt && python -m pytest    # 54 tests
```

Optional LLM answers: copy `.env.example` to `.env` and fill in `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. Never commit `.env`.
Tests use a fake embedder and a fake LLM, so they run offline.

---

## 🧑‍💻 My contribution and AI tools

REPLACE THIS WITH YOUR OWN TRUE STATEMENT. Suggested structure, keep only what is true:

- I chose the 9 documents and wrote and verified 31 labeled questions against the PDFs.
- I compared chunk sizes and retrieval methods and chose 120-word hybrid.
- I found the document-name fix (hit@1 0.33 to 0.79).
- I wrote the held-out check.
- AI assistants (Claude, ADD OTHERS) helped with the code skeleton, tests and UI. I reviewed and changed WHAT and can explain every file.

---

## 📚 Sources and credits

Documents (titles, publishers, URLs and dates in [`docs/sources.csv`](docs/sources.csv)): AICTE Pragati, Saksham and Swanath scholarship guidelines; PM-USP CSIS guidelines (2022); PM-USP CSSS FAQ; IBA Model Education Loan Scheme; PM-Vidyalaxmi guidelines; Dr. Ambedkar overseas interest subsidy scheme; Puducherry DHTE scholarship guidelines.

Libraries: pdfplumber, scikit-learn, sentence-transformers, Streamlit, NumPy, requests (optional: RapidOCR, pypdfium2).
Model: `all-MiniLM-L6-v2` (see its model card for the license).
