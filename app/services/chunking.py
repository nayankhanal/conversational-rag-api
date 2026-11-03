from typing import Protocol

from app.schemas.ingestion import ChunkingStrategy


class Chunker(Protocol):
    def split(self, text: str) -> list[str]: ...


def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")


class FixedSizeChunker:
    """Sliding character window; simple, predictable, may cut mid-sentence."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        _validate(chunk_size, overlap)
        self._size = chunk_size
        self._step = chunk_size - overlap

    def split(self, text: str) -> list[str]:
        text = text.strip()
        chunks = [text[i : i + self._size].strip() for i in range(0, len(text), self._step)]
        # The last window can be pure overlap of the previous one; drop it.
        if (
            len(chunks) > 1
            and len(text) - (len(chunks) - 1) * self._step <= self._size - self._step
        ):
            chunks.pop()
        return [c for c in chunks if c]


class RecursiveChunker:
    """Splits on paragraph, line, sentence, then word boundaries, then packs pieces
    greedily up to ``chunk_size`` with a trailing overlap between chunks."""

    _SEPARATORS = ("\n\n", "\n", ". ", " ")

    def __init__(self, chunk_size: int, overlap: int) -> None:
        _validate(chunk_size, overlap)
        self._size = chunk_size
        self._overlap = overlap

    def split(self, text: str) -> list[str]:
        pieces = self._atomize(text.strip(), self._SEPARATORS)
        return self._pack(pieces)

    def _atomize(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self._size:
            return [text] if text else []
        if not separators:
            return [text[i : i + self._size] for i in range(0, len(text), self._size)]
        head, rest = separators[0], separators[1:]
        pieces: list[str] = []
        for part in text.split(head):
            part = part.strip()
            if part:
                pieces.extend(self._atomize(part, rest))
        return pieces

    def _pack(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        window: list[str] = []
        length = 0
        for piece in pieces:
            if window and length + 1 + len(piece) > self._size:
                chunks.append(" ".join(window))
                while window and (length > self._overlap or length + 1 + len(piece) > self._size):
                    length -= len(window.pop(0)) + (1 if window else 0)
            length += len(piece) + (1 if window else 0)
            window.append(piece)
        if window:
            chunks.append(" ".join(window))
        return chunks


def build_chunker(strategy: ChunkingStrategy, chunk_size: int, overlap: int) -> Chunker:
    match strategy:
        case ChunkingStrategy.FIXED:
            return FixedSizeChunker(chunk_size, overlap)
        case ChunkingStrategy.RECURSIVE:
            return RecursiveChunker(chunk_size, overlap)
