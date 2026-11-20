import settings
from answer import answer_question
from retrieval import build_retriever
from sources import load_sources
cfg = settings.load_config()
retriever = build_retriever(settings.CHUNKS_PATH, cfg.get("embedder", settings.EMBED_MODEL), settings.DATA_DIR)
sources = load_sources()
held_out = [
    # 6 answerable (first 2 intentionally omit the scheme name!)
    ("yes", "Which scholarship is available for girl students in technical degree courses?"),
    ("yes", "Is there an interest subsidy for students going abroad for higher studies?"),
    ("yes", "What is the family income ceiling for the AICTE Saksham scholarship?"),
    ("yes", "Who is eligible for the AICTE Swanath scholarship?"),
    ("yes", "What is the maximum loan amount eligible for credit guarantee under PM-Vidyalaxmi?"),
    ("yes", "Is collateral required for an IBA education loan up to Rs. 7.5 lakh?"),
    # 4 out-of-scope (should refuse)
    ("no", "How do I apply for a Fulbright scholarship to the USA?"),
    ("no", "What is SBI's current floating interest rate for home loans?"),
    ("no", "What is the cut-off rank for IIT Delhi Computer Science?"),
    ("no", "Does the Bihar Student Credit Card scheme cover hostel fees?"),
]
for expected, q in held_out:
    res = answer_question(q, retriever, method=cfg["method"], tau=cfg["tau"], k=settings.TOP_K, llm=None, sources=sources)
    top = res.citations[0] if res.citations else (res.closest[0] if res.closest else None)
    top_info = f"{top.doc} p.{top.page} ({top.score:.3f})" if top else "none"
    status = "ANSWERED" if res.answered else "REFUSED"
    print(f"[expected={expected:3s} | got={status:8s}] {q}\n   -> top1: {top_info}")
