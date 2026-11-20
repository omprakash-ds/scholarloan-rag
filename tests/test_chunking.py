import pytest

from chunking import build_chunks, split_words


def test_windows_overlap_and_cover_text():
    text = " ".join(f"w{i}" for i in range(100))
    pieces = list(split_words(text, size=40, overlap=10, min_words=5))
    assert pieces[0].split()[:2] == ["w0", "w1"]
    assert pieces[1].split()[0] == "w30"          # step = size - overlap
    assert pieces[-1].split()[-1] == "w99"         # nothing is lost at the end
    assert len(pieces) == 3                        # no overlap-only trailing piece


def test_short_text_below_minimum_is_dropped():
    assert list(split_words("only five words in here", size=40, overlap=5, min_words=20)) == []


def test_short_text_above_minimum_is_kept():
    text = " ".join(["word"] * 25)
    assert list(split_words(text, size=180, overlap=30, min_words=20)) == [text]


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1), (10, 11)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        list(split_words("a b c", size, overlap, 1))


def test_build_chunks_keeps_doc_page_and_sequential_ids(pages):
    chunks = build_chunks(pages, size=20, overlap=5, min_words=5)
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert {(c["doc"], c["page"]) for c in chunks} == {("loan.pdf", 1), ("loan.pdf", 2),
                                                       ("scholarship.pdf", 1), ("scholarship.pdf", 2)}
    assert all(len(c["text"].split()) <= 20 for c in chunks)
