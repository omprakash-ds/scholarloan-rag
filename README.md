# 🎓 ScholarLoan RAG — Cited Indian Education Aid Q&A

[![Live App](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://omprakash-ds-scholarloan-rag-app-tczgd0.streamlit.app/)
[![CI Tests](https://github.com/omprakash-ds/scholarloan-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/omprakash-ds/scholarloan-rag/actions/workflows/tests.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-54%20Passed-2EA44F?logo=pytest&logoColor=white)](#verification--local-setup)

**Live Demo:** https://omprakash-ds-scholarloan-rag-app-tczgd0.streamlit.app/ | **Demo Video:** `[PASTE DEMO VIDEO URL]` | **Hackathon:** Build Your First AI App 2026 (Devpost)

| **Cited Answer with Page & Date Audit Trail** | **Safe Refusal on Out-of-Scope Query (`< 0.437`)** |
| :---: | :---: |
| ![Answer with page citations](assets/demo_answer.png) | ![Refusal on an unsupported question](assets/demo_refusal.png) |

---

## 🎯 Business Problem

A student in India comparing the **IBA Model Education Loan Scheme**, **PM-Vidyalaxmi**, **CSIS**, and **AICTE scholarships** has to dig through 15–30 page government and banking PDFs—several of them scanned—just to verify a single rule like margin requirements, collateral limits, or income ceilings. Standard LLMs fail in this regulated domain in three ways: they **blend rules across similar schemes**, provide **no verifiable page-level audit trail**, and **answer out-of-scope questions** (like private bank fees or foreign scholarships) instead of abstaining.

**ScholarLoan RAG** answers strictly from **9 official public PDF guidelines** (**237 searchable passages**), attaches exact **`(document, page, as_of)`** citations to every claim, verifies LLM citations programmatically, and refuses when retrieval confidence drops below a calibrated threshold (`tau = 0.437`).

---

## 🏗️ System Architecture & Engineering Tradeoffs

```text
PDFs --ingest.py--> pages (pdfplumber + OCR fallback) --chunking.py--> chunks (doc, page, 120w, 30w overlap)
chunks --retrieval.py--> Doc-Context Injection --> TF-IDF (1,2)-grams + MiniLM Dense --> Hybrid RRF (k=60)
question --> top-k passages --> confidence (0.7*dense + 0.3*tfidf) < 0.437? --Yes--> Safe Refusal + Low-Confidence Passages
                                                                            --No---> answer.py (LLM with [n] citation check, or extractive fallback)
evaluate.py: grid sweep (hit@k, MRR, tau tuning, failures.csv)   |   app.py: Streamlit UI + latency/confidence telemetry
