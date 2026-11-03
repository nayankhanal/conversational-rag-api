from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentRecord


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, record: DocumentRecord) -> DocumentRecord:
        self._session.add(record)
        await self._session.commit()
        return record

    async def get(self, document_id: str) -> DocumentRecord | None:
        return await self._session.get(DocumentRecord, document_id)

    async def list(self) -> Sequence[DocumentRecord]:
        result = await self._session.scalars(
            select(DocumentRecord).order_by(DocumentRecord.created_at.desc())
        )
        return result.all()
