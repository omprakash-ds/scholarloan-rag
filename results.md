# Evaluation results (2026-11-20)

Questions: 31 verified (24 answerable, 7 unanswerable). Corpus: 9 documents, 237 chunks at the best setting.
Embedding model: sentence-transformers/all-MiniLM-L6-v2.

| chunk words | method | hit@1 | hit@3 | hit@5 | MRR | answered (answerable) | refused (unanswerable) | tau |
|---|---|---|---|---|---|---|---|---|
| 120 | hybrid | 0.79 | 0.92 | 0.96 | 0.86 | 0.96 | 0.71 | 0.437 |
| 120 | dense | 0.62 | 0.88 | 0.92 | 0.75 | 0.92 | 0.43 | 0.503 |
| 120 | tfidf | 0.46 | 0.88 | 0.96 | 0.66 | 0.92 | 0.86 | 0.584 |
| 180 | hybrid | 0.75 | 0.92 | 0.96 | 0.83 | 0.88 | 0.86 | 0.464 |
| 180 | dense | 0.58 | 0.83 | 0.92 | 0.73 | 0.92 | 0.43 | 0.485 |
| 180 | tfidf | 0.46 | 0.88 | 0.96 | 0.67 | 0.96 | 0.57 | 0.512 |
| 250 | hybrid | 0.58 | 0.83 | 0.88 | 0.71 | 0.79 | 1.00 | 0.447 |
| 250 | dense | 0.58 | 0.88 | 0.96 | 0.73 | 0.88 | 0.71 | 0.571 |
| 250 | tfidf | 0.38 | 0.88 | 0.92 | 0.61 | 0.88 | 0.71 | 0.557 |

Notes: hit@k = the labeled (document, page) is among the top k chunks. tau (refusal threshold) is tuned on these same questions, so the answered/refused columns are optimistic. 7 questions are listed in failures.csv for failure analysis.
