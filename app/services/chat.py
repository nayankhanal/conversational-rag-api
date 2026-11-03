import uuid
from dataclasses import asdict
from datetime import date

from pydantic import ValidationError

from app.repositories.bookings import BookingRepository
from app.schemas.chat import (
    BookingCreate,
    BookingDraft,
    BookingRead,
    ChatMessage,
    ChatResponse,
    Intent,
    SourceChunk,
)
from app.services.protocols import ChatLLM, ChatMemory, Embedder, VectorStore

_FIELD_LABELS = {
    "name": "full name",
    "email": "email address",
    "date": "preferred date",
    "time": "preferred time",
}


class ChatService:
    """Custom RAG pipeline: analyze turn -> (book | retrieve -> generate) -> persist memory."""

    def __init__(
        self,
        *,
        llm: ChatLLM,
        embedder: Embedder,
        vector_store: VectorStore,
        memory: ChatMemory,
        bookings: BookingRepository,
        top_k: int,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._vector_store = vector_store
        self._memory = memory
        self._bookings = bookings
        self._top_k = top_k

    async def handle(self, session_id: str | None, message: str) -> ChatResponse:
        session_id = session_id or str(uuid.uuid4())
        history = await self._memory.get_history(session_id)
        draft = await self._memory.get_booking_draft(session_id)

        analysis = await self._llm.analyze_turn(history, message, draft)
        if analysis.wants_booking:
            response = await self._handle_booking(session_id, draft.merge(analysis.booking))
        else:
            response = await self._handle_question(
                session_id, history, analysis.standalone_question
            )

        await self._memory.append(
            session_id,
            ChatMessage(role="user", content=message),
            ChatMessage(role="assistant", content=response.answer),
        )
        return response

    async def _handle_question(
        self, session_id: str, history: list[ChatMessage], question: str
    ) -> ChatResponse:
        [vector] = await self._embedder.embed([question])
        chunks = await self._vector_store.search(vector, self._top_k)
        answer = await self._llm.answer(history, question, chunks)
        return ChatResponse(
            session_id=session_id,
            answer=answer,
            intent=Intent.QUESTION,
            sources=[SourceChunk(**asdict(c)) for c in chunks],
        )

    async def _handle_booking(self, session_id: str, draft: BookingDraft) -> ChatResponse:
        missing = draft.missing()
        if missing:
            await self._memory.set_booking_draft(session_id, draft)
            labels = ", ".join(_FIELD_LABELS[f] for f in missing)
            return ChatResponse(
                session_id=session_id,
                answer=f"Happy to book your interview. I still need your: {labels}.",
                intent=Intent.BOOKING,
            )

        try:
            booking = BookingCreate.model_validate(draft.model_dump())
            if booking.date < date.today():
                raise ValueError("date in the past")
        except (ValidationError, ValueError) as exc:
            bad = _invalid_fields(exc)
            await self._memory.set_booking_draft(
                session_id, draft.model_copy(update={f: None for f in bad})
            )
            labels = ", ".join(_FIELD_LABELS[f] for f in bad)
            return ChatResponse(
                session_id=session_id,
                answer=f"Some details look invalid. Could you provide a valid {labels}?",
                intent=Intent.BOOKING,
            )

        record = await self._bookings.add(session_id, booking)
        await self._memory.set_booking_draft(session_id, None)
        return ChatResponse(
            session_id=session_id,
            answer=(
                f"Your interview is booked, {booking.name}: {booking.date.isoformat()} at "
                f"{booking.time.strftime('%H:%M')}. A confirmation will go to {booking.email}."
            ),
            intent=Intent.BOOKING,
            booking=BookingRead.model_validate(record),
        )


def _invalid_fields(exc: Exception) -> list[str]:
    if isinstance(exc, ValidationError):
        fields = {str(err["loc"][0]) for err in exc.errors() if err["loc"]}
        return [f for f in _FIELD_LABELS if f in fields] or ["date"]
    return ["date"]
