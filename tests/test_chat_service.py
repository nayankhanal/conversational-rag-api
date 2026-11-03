from collections.abc import Sequence
from datetime import date, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.session import create_engine, create_session_factory, init_models
from app.repositories.bookings import BookingRepository
from app.schemas.chat import BookingDraft, ChatMessage, Intent
from app.services.chat import ChatService
from app.services.protocols import RetrievedChunk, TurnAnalysis


class FakeMemory:
    def __init__(self) -> None:
        self.history: dict[str, list[ChatMessage]] = {}
        self.drafts: dict[str, BookingDraft] = {}

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        return list(self.history.get(session_id, []))

    async def append(self, session_id: str, *messages: ChatMessage) -> None:
        self.history.setdefault(session_id, []).extend(messages)

    async def get_booking_draft(self, session_id: str) -> BookingDraft:
        return self.drafts.get(session_id, BookingDraft())

    async def set_booking_draft(self, session_id: str, draft: BookingDraft | None) -> None:
        if draft is None:
            self.drafts.pop(session_id, None)
        else:
            self.drafts[session_id] = draft

    async def clear(self, session_id: str) -> None:
        self.history.pop(session_id, None)
        self.drafts.pop(session_id, None)


class FakeEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class FakeStore:
    async def upsert(self, *args: object) -> None: ...

    async def search(self, vector: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        return [RetrievedChunk("d1", "doc.txt", 0, 0.9, "Paris is the capital of France.")]


class ScriptedLLM:
    def __init__(self, analyses: list[TurnAnalysis]) -> None:
        self._analyses = analyses

    async def analyze_turn(
        self, history: Sequence[ChatMessage], message: str, draft: BookingDraft
    ) -> TurnAnalysis:
        return self._analyses.pop(0)

    async def answer(
        self, history: Sequence[ChatMessage], question: str, context: Sequence[RetrievedChunk]
    ) -> str:
        return f"answer from {len(context)} chunk(s)"


@pytest.fixture
async def factory() -> async_sessionmaker[AsyncSession]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await init_models(engine)
    return create_session_factory(engine)


def _analysis(booking: BookingDraft, wants: bool = True) -> TurnAnalysis:
    return TurnAnalysis(wants_booking=wants, standalone_question="q", booking=booking)


async def test_question_returns_sources(factory: async_sessionmaker[AsyncSession]) -> None:
    async with factory() as db:
        service = ChatService(
            llm=ScriptedLLM([_analysis(BookingDraft(), wants=False)]),
            embedder=FakeEmbedder(),
            vector_store=FakeStore(),
            memory=FakeMemory(),
            bookings=BookingRepository(db),
            top_k=3,
        )
        result = await service.handle(None, "What is the capital of France?")
    assert result.intent is Intent.QUESTION
    assert result.sources[0].filename == "doc.txt"


async def test_multi_turn_booking_collects_slots_then_persists(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    memory = FakeMemory()
    llm = ScriptedLLM(
        [
            _analysis(BookingDraft(name="Ada")),
            _analysis(BookingDraft(email="ada@example.com", date=tomorrow)),
            _analysis(BookingDraft(time="14:30")),
        ]
    )
    async with factory() as db:
        service = ChatService(
            llm=llm,
            embedder=FakeEmbedder(),
            vector_store=FakeStore(),
            memory=memory,
            bookings=BookingRepository(db),
            top_k=3,
        )
        first = await service.handle("s1", "Book an interview, I'm Ada")
        assert first.booking is None and "email" in first.answer
        second = await service.handle("s1", "ada@example.com, tomorrow")
        assert second.booking is None and "time" in second.answer
        third = await service.handle("s1", "2:30pm")
    assert third.booking is not None
    assert third.booking.name == "Ada"
    assert "s1" not in memory.drafts
    assert len(memory.history["s1"]) == 6


async def test_invalid_email_is_reasked(factory: async_sessionmaker[AsyncSession]) -> None:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    full = BookingDraft(name="Ada", email="not-an-email", date=tomorrow, time="10:00")
    memory = FakeMemory()
    async with factory() as db:
        service = ChatService(
            llm=ScriptedLLM([_analysis(full)]),
            embedder=FakeEmbedder(),
            vector_store=FakeStore(),
            memory=memory,
            bookings=BookingRepository(db),
            top_k=3,
        )
        result = await service.handle("s2", "book it")
    assert result.booking is None
    assert memory.drafts["s2"].email is None
