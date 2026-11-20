# Build guide: ScholarLoan RAG (hackathon deadline: Nov 27, 2026)

The code in this folder runs. Your job is to make it yours: run it on your documents, label your questions,
read the failures, change something, and be able to explain every file without notes.
Plan to finish by **Nov 26**; the Devpost page shows Nov 28, 3:30 am IST, but never leave submission to the last night.

## How to use this clone without becoming dependent on it
1. Read each file below before running it. Close it, then explain it aloud in 60 seconds.
2. Type `chunking.py`, `retrieval.py` and `evaluate.py` yourself in a new file next to the original, then diff. This takes about 2 hours and is where the learning happens.
3. Make 2 changes of your own and measure them (examples: a different chunk size, a different sentence picker, a reranking step). Put the before/after numbers in the README.
4. Write the README's "My contribution" section last, and only list what is true.

## Setup (Windows PowerShell)
```
cd D:\scholarloan-rag                 # your project folder. Run `git init` ONLY here, never in D:\
python -m venv .venv                  # venv inside the project, not in D:\
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # tests only
python -m pytest                      # 54 tests should pass (the OCR test is skipped without requirements-ocr.txt)
```
Put your PDFs in `docs\` and list each one in `docs\sources.csv`. Never push personal PDFs (challans, forms) to GitHub.

## 7-day plan
| Day | Output you can show |
|---|---|
| Nov 20 | Tests pass. `python ingest.py` runs on your PDFs. IBA PDF handled (see OCR below). Goal: 8+ text documents, 100+ chunks. `sources.csv` filled. |
| Nov 21 | `eval_questions.csv`: seeds verified against the PDFs, plus new questions. At least 20 answerable and 5 unanswerable, all `verified=yes`. |
| Nov 22 | `python evaluate.py --apply-best` run. `failures.csv` read. One improvement made and re-measured. |
| Nov 23 | App works locally. Optional LLM key set. 10 manual questions tried, including 3 tricky ones. Deployed to Hugging Face Spaces. |
| Nov 24 | README results and failure analysis written from real numbers. Code pushed to GitHub. |
| Nov 25 | 2-3 minute demo video, screenshots, Devpost text drafted. |
| Nov 26 | Buffer. Submit. |

## The scanned IBA PDF (OCR)
If `ingest.py` shows most pages empty for a file, the PDF is a scan. First try to find a text version from the bank or IBA.
Otherwise: `pip install -r requirements-ocr.txt`, then `python ingest.py --ocr`. OCR can misread digits, so check amounts
and limits against the PDF before using them in labeled questions. Results are cached in `data\pages_cache\`.

## Writing the labeled questions (the part that wins)
- Write the question as a student would ask it, not copying the PDF's words.
- Open the PDF, find the answer, and write the page number. If the answer spans pages, write `2|3`.
- Only set `verified=yes` after you have checked the page yourself. `evaluate.py` ignores everything else.
- Unanswerable questions: make sure NO loaded document answers them. Include a few close-but-not-covered ones (for example a question about a scheme you did not load), not only silly off-topic ones.
- The 12 seeded questions came from text fragments, so their pages may be wrong. Treat them as drafts.

## What each file does and what you must be able to explain
| File | Job | Decision to explain | Interview question |
|---|---|---|---|
| `settings.py` | Paths and defaults | Why settings live in one place | What changes if the chunk size changes? |
| `chunking.py` | Word windows per page with overlap | Per-page chunks keep exact citations; overlap avoids cutting answers in half | Why not chunk across pages? |
| `ingest.py` | PDF to pages, header/footer removal, OCR fallback, cache | Running headers pollute every chunk; OCR is slow so it is cached | How do you know a PDF is scanned? |
| `retrieval.py` | TF-IDF, dense embeddings, hybrid fusion | Lexical search catches exact terms ("Aadhaar"); dense catches paraphrase; fusion combines ranks | Why reciprocal rank fusion and not adding raw scores? |
| `answer.py` | Refusal, extractive answer, LLM answer, citation check | Refuse when the best score is below tau; reject LLM answers whose citations were not retrieved | What happens if the model invents `[7]`? |
| `llm.py` | OpenAI-compatible client with retries | Retry only on 429/5xx; never put the key in an error | Why temperature 0? |
| `evaluate.py` | hit@k, MRR, tau tuning, failures | Metrics on labeled questions; only verified rows count | Why is the refusal number optimistic? |
| `app.py` | Streamlit UI | Show sources, dates and a verify warning | What does the user see when the app refuses? |
| `tests/` | 54 tests | Fake embedder and fake LLM make tests fast and offline | How do you test an LLM feature without the LLM? |

## Optional: connect a language model
Without a key the app shows cited extracted sentences, which is acceptable for the hackathon. For written answers, pick a provider
with an OpenAI-compatible API (for example Groq, Google Gemini's compatibility endpoint, OpenRouter, or a local Ollama).
Copy `.env.example` to `.env`, fill `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` from the provider's current docs, and check its
free-tier limits and terms yourself. Never commit `.env`. On Hugging Face Spaces, put the three values in the Space's secrets.

## Deploy on Hugging Face Spaces
1. Create a new Space, SDK Streamlit (check the Spaces docs for the current README front-matter fields, usually `sdk: streamlit` and `app_file: app.py`).
2. Push: `app.py`, `settings.py`, `chunking.py`, `retrieval.py`, `answer.py`, `llm.py`, `sources.py`, `requirements.txt`, `docs/sources.csv`, `data/chunks.json`, `data/config.json`. The PDFs themselves are not needed at runtime.
3. Keep `data/emb_*.npy` out of the push; the Space recomputes embeddings on startup.
4. Open the live URL in a private window and ask 5 questions before putting it in Devpost.

## Demo video script (2-3 minutes)
1. 15 s: the problem, one sentence, one example confusing document.
2. 45 s: ask a question the documents answer; show the answer, the page citation and the date.
3. 30 s: ask a question they do not answer; show the refusal.
4. 45 s: show `results.md` (your table) and one row of `failures.csv` you fixed.
5. 15 s: limits (documents change, not advice) and what you would do next.

## Devpost submission checklist
- [ ] Title and short description (problem, solution, who benefits). Do not write "zero hallucination".
- [ ] Working app link, opened and tested from a private window.
- [ ] AI implementation explained: embeddings, TF-IDF, hybrid ranking, optional LLM, citation check.
- [ ] GitHub repo link. No `.env`, API keys, personal files. Search the history: `git log -p | findstr /i "api_key"`.
- [ ] Demo video or screenshots.
- [ ] README with real numbers, failure analysis, limitations, sources and credits.
- [ ] "Your contribution": what you built, what you learned, which AI tools you used, truthfully.

## If something breaks
| Symptom | Fix |
|---|---|
| `SKIPPED (PdfminerException)` | The PDF is password-protected or corrupt; download it again or skip it |
| Most pages empty | Scanned PDF: use `--ocr` |
| SSL error downloading a government PDF | Download in the browser instead of disabling certificate checks |
| First app start is slow | The embedding model downloads once; later starts are fast |
| `No verified answerable questions` | Set `verified=yes` only on rows you checked |
| hit@5 is 1.00 everywhere | Corpus too small: add documents (target 100+ chunks) |
| Streamlit says no index | Run `ingest.py`, then `evaluate.py --apply-best` |

## Honest limits to state, not hide
Refusal threshold tuned on the same questions; page-level (not sentence-level) retrieval; small corpus; documents may be outdated;
extractive mode quotes passages instead of reasoning across them. Judges respect measured limits more than big claims.
