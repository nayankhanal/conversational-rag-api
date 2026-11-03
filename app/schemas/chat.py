from datetime import date, datetime, time
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class Intent(StrEnum):
    QUESTION = "question"
    BOOKING = "booking"


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None, max_length=64, description="Omit to start a new conversation."
    )
    message: str = Field(min_length=1, max_length=4000)


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


class BookingCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    date: date
    time: time


class BookingRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    name: str
    email: str
    interview_date: date
    interview_time: time
    created_at: datetime


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: Intent
    sources: list[SourceChunk] = []
    booking: BookingRead | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class BookingDraft(BaseModel):
    """Booking details gathered so far across turns; fields stay None until supplied."""

    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None

    def merge(self, other: "BookingDraft") -> "BookingDraft":
        updates = {k: v for k, v in other.model_dump().items() if v}
        return self.model_copy(update=updates)

    def missing(self) -> list[str]:
        return [k for k, v in self.model_dump().items() if not v]
