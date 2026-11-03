from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.container import Container
from app.repositories.bookings import BookingRepository
from app.repositories.documents import DocumentRepository
from app.services.chat import ChatService
from app.services.ingestion import IngestionService


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


async def get_db_session(container: ContainerDep) -> AsyncIterator[AsyncSession]:
    async with container.session_factory() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(get_db_session)]


def get_document_repository(session: DbSession) -> DocumentRepository:
    return DocumentRepository(session)


DocumentRepoDep = Annotated[DocumentRepository, Depends(get_document_repository)]


def get_ingestion_service(container: ContainerDep, documents: DocumentRepoDep) -> IngestionService:
    return IngestionService(container.embedder, container.vector_store, documents)


def get_chat_service(container: ContainerDep, session: DbSession) -> ChatService:
    return ChatService(
        llm=container.llm,
        embedder=container.embedder,
        vector_store=container.vector_store,
        memory=container.memory,
        bookings=BookingRepository(session),
        top_k=container.settings.retrieval_top_k,
    )


IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
