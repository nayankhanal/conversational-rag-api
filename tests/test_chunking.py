import pytest

from app.schemas.ingestion import ChunkingStrategy
from app.services.chunking import FixedSizeChunker, RecursiveChunker, build_chunker

TEXT = ("Alpha beta gamma. " * 20 + "\n\n") * 5


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_chunks_respect_size_and_cover_text(strategy: ChunkingStrategy) -> None:
    chunks = build_chunker(strategy, 200, 30).split(TEXT)
    assert len(chunks) > 1
    assert all(0 < len(c) <= 200 for c in chunks)


def test_fixed_overlap() -> None:
    chunks = FixedSizeChunker(10, 3).split("abcdefghijklmnopqrstuvwxyz")
    assert chunks[0] == "abcdefghij"
    assert chunks[1].startswith("hij")


def test_recursive_prefers_paragraph_boundaries() -> None:
    chunks = RecursiveChunker(30, 0).split("one one one.\n\ntwo two two.\n\nthree three three.")
    assert chunks == ["one one one. two two two.", "three three three."]


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(10, 10)
