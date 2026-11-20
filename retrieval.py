"""Retrieval: TF-IDF, dense embeddings, and hybrid (reciprocal rank fusion).
Brute-force numpy search is used on purpose: with a few hundred to a few thousand
chunks it is exact and fast. A vector index (FAISS, Chroma) only pays off at
much larger scale.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import settings
METHODS = ("hybrid", "dense", "tfidf")   # order matters: ties prefer the earlier method
RRF_K = 60
@dataclass
class Hit:
    chunk: dict
    score: float   # cosine similarity on the retriever's scorer (dense if loaded, else TF-IDF)
    rank: int      # 1-based
class SentenceTransformerEmbedder:
    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.model = SentenceTransformer(name)
    def encode(self, texts) -> np.ndarray:
        vectors = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        return np.asarray(vectors, dtype=np.float32)
def _index_text(chunk: dict) -> str:
    """Prepend cleaned document name for indexing so inner pages keep scheme context."""
    doc = chunk.get("doc", "").removesuffix(".pdf").replace("_", " ").strip()
    return f"{doc}: {chunk['text']}" if doc else chunk["text"]
class Retriever:
    def __init__(self, chunks: list[dict], embedder=None, doc_vectors=None):
        if not chunks:
            raise ValueError("no chunks to search")
        self.chunks = chunks
        self.embedder = embedder
        texts = [_index_text(c) for c in chunks]
        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._tfidf_matrix = self._tfidf.fit_transform(texts)
        self._dense = None
        if embedder is not None:
            vectors = doc_vectors if doc_vectors is not None else embedder.encode(texts)
            self._dense = np.asarray(vectors, dtype=np.float32)
    @property
    def has_dense(self) -> bool:
        return self._dense is not None
    @property
    def scorer(self) -> str:
        return "dense" if self.has_dense else "tfidf"
    def available_methods(self) -> tuple[str, ...]:
        return METHODS if self.has_dense else ("tfidf",)
    def _tfidf_scores(self, query: str) -> np.ndarray:
        q = self._tfidf.transform([query])
        return (self._tfidf_matrix @ q.T).toarray().ravel()
    def _dense_scores(self, query: str) -> np.ndarray:
        q = np.asarray(self.embedder.encode([query]), dtype=np.float32)[0]
        return self._dense @ q
    @staticmethod
    def _ranking(scores: np.ndarray) -> np.ndarray:
        return np.argsort(-scores, kind="stable")
    def search(self, query: str, k: int = settings.TOP_K, method: str = "hybrid") -> list[Hit]:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
        if method != "tfidf" and not self.has_dense:
            raise ValueError("dense model not loaded; use method='tfidf'")
        tfidf = self._tfidf_scores(query)
        dense = self._dense_scores(query) if self.has_dense else None
        if method == "tfidf":
            order = self._ranking(tfidf)
        elif method == "dense":
            order = self._ranking(dense)
        else:  # reciprocal rank fusion of the two rankings
            fused = np.zeros(len(self.chunks))
            for scores in (tfidf, dense):
                for rank, idx in enumerate(self._ranking(scores), start=1):
                    fused[idx] += 1.0 / (RRF_K + rank)
            order = self._ranking(fused)
        shown = (0.7 * dense + 0.3 * tfidf) if (method == "hybrid" and self.has_dense) else (dense if self.has_dense else tfidf)
        return [Hit(self.chunks[i], float(shown[i]), r) for r, i in enumerate(order[:k], start=1)]
def retriever_from_chunks(chunks, model_name=None, cache_dir=None, embedder=None) -> Retriever:
    """Build a retriever; dense vectors are cached on disk, keyed by model + chunk text."""
    model_name = model_name or settings.EMBED_MODEL
    if model_name.lower() == "none" and embedder is None:
        return Retriever(chunks)
    embedder = embedder or SentenceTransformerEmbedder(model_name)
    texts = [_index_text(c) for c in chunks]
    key = hashlib.sha1((model_name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    cache = Path(cache_dir or settings.DATA_DIR) / f"emb_{key}.npy"
    if cache.exists():
        vectors = np.load(cache)
    else:
        vectors = embedder.encode(texts)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
    return Retriever(chunks, embedder, vectors)
def build_retriever(chunks_path=None, model_name=None, cache_dir=None) -> Retriever:
    chunks = json.loads(Path(chunks_path or settings.CHUNKS_PATH).read_text(encoding="utf-8"))
    return retriever_from_chunks(chunks, model_name, cache_dir)
