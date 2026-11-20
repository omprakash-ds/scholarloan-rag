import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGES = [
    {"doc": "loan.pdf", "page": 1, "text": (
        "The education loan covers tuition fees, hostel charges, examination fees and library fees. "
        "The repayment holiday is the course period plus one year. Loans up to a fixed limit need no "
        "collateral security. A co-borrower is required for every student borrower.")},
    {"doc": "loan.pdf", "page": 2, "text": (
        "Interest subsidy under the scheme is available only once, either for the undergraduate or the "
        "postgraduate course. Eligible students belong to economically weaker sections with family income "
        "below the stated limit.")},
    {"doc": "scholarship.pdf", "page": 1, "text": (
        "The scholarship amount is paid every year for the duration of the course. Girl students admitted "
        "to approved technical institutions are eligible. Selection is based on merit of the qualifying "
        "examination marks.")},
    {"doc": "scholarship.pdf", "page": 2, "text": (
        "Applicants must apply online through the national portal once every year. Grievances about the "
        "merit list must be raised within one month. An Aadhaar linked bank account is mandatory for "
        "payment of the scholarship.")},
]


class HashEmbedder:
    """Deterministic bag-of-words embedder so tests need no model download."""
    dim = 128

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in text.lower().split():
                out[row, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.prompts = reply, error, []

    def complete(self, system, user, max_tokens=400):
        self.prompts.append(user)
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def pages():
    return [dict(p) for p in PAGES]


@pytest.fixture
def chunks(pages):
    from chunking import build_chunks
    return build_chunks(pages, size=60, overlap=10, min_words=5)


@pytest.fixture
def tfidf_retriever(chunks):
    from retrieval import Retriever
    return Retriever(chunks)


@pytest.fixture
def dense_retriever(chunks):
    from retrieval import Retriever
    emb = HashEmbedder()
    return Retriever(chunks, emb)
