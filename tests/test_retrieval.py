import numpy as np
import pytest

from conftest import HashEmbedder
from retrieval import Retriever, retriever_from_chunks


def top_page(hits):
    return hits[0].chunk["doc"], hits[0].chunk["page"]


def test_tfidf_finds_the_right_page(tfidf_retriever):
    hits = tfidf_retriever.search("Is an Aadhaar linked bank account mandatory?", k=3, method="tfidf")
    assert top_page(hits) == ("scholarship.pdf", 2)
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score >= hits[1].score


@pytest.mark.parametrize("method", ["dense", "hybrid", "tfidf"])
def test_all_methods_work_with_dense_vectors(dense_retriever, method):
    hits = dense_retriever.search("Can I get interest subsidy for both undergraduate and postgraduate?", k=3, method=method)
    assert top_page(hits) == ("loan.pdf", 2)
    assert dense_retriever.scorer == "dense"


def test_tfidf_only_retriever_rejects_dense_methods(tfidf_retriever):
    assert tfidf_retriever.available_methods() == ("tfidf",)
    with pytest.raises(ValueError):
        tfidf_retriever.search("anything", method="dense")


def test_unknown_method_and_empty_corpus():
    with pytest.raises(ValueError):
        Retriever([])
    with pytest.raises(ValueError):
        Retriever([{"doc": "a", "page": 1, "text": "hello world"}]).search("q", method="nope")


def test_query_with_no_overlap_scores_zero(tfidf_retriever):
    hits = tfidf_retriever.search("zzzz qqqq", k=2, method="tfidf")
    assert hits[0].score == 0.0


def test_embedding_cache_is_written_and_reused(chunks, tmp_path):
    first = HashEmbedder()
    r1 = retriever_from_chunks(chunks, "fake-model", tmp_path, embedder=first)
    assert first.calls == 1
    assert len(list(tmp_path.glob("emb_*.npy"))) == 1
    second = HashEmbedder()
    r2 = retriever_from_chunks(chunks, "fake-model", tmp_path, embedder=second)
    assert second.calls == 0                      # vectors came from disk
    assert np.allclose(r1._dense, r2._dense)


def test_changed_chunks_do_not_reuse_stale_cache(chunks, tmp_path):
    retriever_from_chunks(chunks, "fake-model", tmp_path, embedder=HashEmbedder())
    changed = [dict(c) for c in chunks]
    changed[0]["text"] += " extra words"
    emb = HashEmbedder()
    retriever_from_chunks(changed, "fake-model", tmp_path, embedder=emb)
    assert emb.calls == 1
    assert len(list(tmp_path.glob("emb_*.npy"))) == 2


def test_model_none_gives_tfidf_only(chunks):
    assert not retriever_from_chunks(chunks, "none").has_dense
