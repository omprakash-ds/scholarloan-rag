# ScholarLoan RAG: Cited Answers for Indian Student Aid

A document question-answering app for Indian students trying to understand education loans and scholarships.
Every answer shows the document and page it came from. If the loaded documents do not support an answer,
the app says so instead of guessing. Retrieval quality is measured on a hand-labeled question set.

**Live demo:** <link> | **Demo video:** <link> | **Hackathon:** Build Your First AI App 2026 (Devpost)

## Problem
<2-3 sentences in your own words: who the student is, which documents confuse them, what goes wrong today.>

## What it does
- Answers questions from public loan and scholarship PDFs, with document title, page and "as of" date for each source.
- Refuses when the best match is below a confidence threshold tuned on labeled questions.
- Works without any API key (shows the most relevant sentences, cited). With a key it writes a short answer
  and the app checks that every citation points to a passage that was actually retrieved.
- Treats PDF text as untrusted data in the prompt, so instructions hidden inside a document are not followed.

## How it works
```
PDFs --ingest.py--> pages (text, OCR fallback) --chunking.py--> chunks (doc, page, text)
chunks --retrieval.py--> TF-IDF + dense embeddings --> hybrid ranking (reciprocal rank fusion)
question --> top-k passages --> score below tau? --> "not found"
                          \--> answer.py --> extracted sentences, or LLM answer with checked [n] citations
evaluate.py: hit@k, MRR, refusal tuning, failures.csv      app.py: Streamlit UI
```
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (local, free). Search is brute-force numpy: exact and fast at this corpus size.
- Language model (optional): any OpenAI-compatible endpoint, configured by environment variables. See `.env.example`.

## Results (my own labeled test set)
<Paste the table from results.md here after running `python evaluate.py --apply-best`. Then state the numbers in one sentence.>

How to read it: hit@k means the labeled (document, page) appears in the top k chunks. The refusal threshold is tuned on the
same questions, so the answered/refused columns are optimistic. The test set has <N> answerable and <M> unanswerable questions.

## Failure analysis
<Three real failures from failures.csv: the question, what was retrieved, why it failed, and what you changed or would change.>

## Limitations (please read)
- Scheme rules change. The app shows each document's date, but you must confirm on the official source before applying.
- Not financial or legal advice.
- Small corpus (<N> documents). Questions about schemes outside these documents are refused by design.
- One scholarship document applies only to residents of Puducherry. <Note any other scope limits.>
- Retrieval is evaluated at page level, not at sentence level.

## Run locally
```
python -m venv .venv
.venv\Scripts\activate            # Windows. On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt   # add requirements-ocr.txt only for scanned PDFs
python ingest.py                  # add --ocr for scanned PDFs
python evaluate.py --apply-best   # needs verified rows in eval_questions.csv
streamlit run app.py
python -m pytest                  # needs requirements-dev.txt
```

## My contribution and AI tools used
<Be specific. Example structure: I chose the documents and wrote N labeled questions by reading each PDF. I decided the
chunking approach, the hybrid retrieval and the refusal threshold after comparing the results above. AI assistants
(<names>) helped draft boilerplate, tests and the UI; I reviewed and changed <what> and can explain every file.>

## Sources and credits
Documents (see `docs/sources.csv` for titles, publishers, URLs and dates): <list them>.
Libraries: pdfplumber, scikit-learn, sentence-transformers, Streamlit, NumPy, requests (optional: RapidOCR, pypdfium2).
Model: all-MiniLM-L6-v2 (see its model card for the license).
