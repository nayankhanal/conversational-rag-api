from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.schemas.chat import BookingDraft, ChatMessage


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


@dataclass(frozen=True, slots=True)
class TurnAnalysis:
    """Structured LLM reading of a user turn."""

    wants_booking: bool
    standalone_question: str
    booking: BookingDraft


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    async def upsert(
        self,
        document_id: str,
        filename: str,
        chunks: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    async def search(self, vector: Sequence[float], top_k: int) -> list[RetrievedChunk]: ...


class ChatMemory(Protocol):
    async def get_history(self, session_id: str) -> list[ChatMessage]: ...

    async def append(self, session_id: str, *messages: ChatMessage) -> None: ...

    async def get_booking_draft(self, session_id: str) -> BookingDraft: ...

    async def set_booking_draft(self, session_id: str, draft: BookingDraft | None) -> None: ...

    async def clear(self, session_id: str) -> None: ...


class ChatLLM(Protocol):
    async def analyze_turn(
        self, history: Sequence[ChatMessage], message: str, draft: BookingDraft
    ) -> TurnAnalysis: ...

    async def answer(
        self, history: Sequence[ChatMessage], question: str, context: Sequence[RetrievedChunk]
    ) -> str: ...
