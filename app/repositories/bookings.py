from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import BookingRecord
from app.schemas.chat import BookingCreate


class BookingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, session_id: str, booking: BookingCreate) -> BookingRecord:
        record = BookingRecord(
            session_id=session_id,
            name=booking.name,
            email=str(booking.email),
            interview_date=booking.date,
            interview_time=booking.time,
        )
        self._session.add(record)
        await self._session.commit()
        return record
