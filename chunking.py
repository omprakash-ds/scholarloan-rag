"""Pure chunking functions (no file I/O) so they are easy to test and to sweep."""
from __future__ import annotations


def split_words(text: str, size: int, overlap: int, min_words: int):
    """Yield word windows of `size` words that overlap by `overlap` words."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = text.split()
    step = size - overlap
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) >= min_words:
            yield " ".join(piece)
        if start + size >= len(words):
            break


def build_chunks(pages: list[dict], size: int, overlap: int, min_words: int) -> list[dict]:
    """Chunk page by page, so every chunk keeps an exact (doc, page) for citations."""
    chunks: list[dict] = []
    for page in pages:
        for piece in split_words(page["text"], size, overlap, min_words):
            chunks.append({
                "chunk_id": len(chunks),
                "doc": page["doc"],
                "page": page["page"],
                "text": piece,
            })
    return chunks
