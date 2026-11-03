from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status

from app.api.deps import ContainerDep, DocumentRepoDep, IngestionServiceDep
from app.core.exceptions import FileTooLargeError, InvalidChunkingError, NotFoundError
from app.schemas.ingestion import ChunkingStrategy, DocumentRead

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    container: ContainerDep,
    service: IngestionServiceDep,
    file: Annotated[UploadFile, File(description=".pdf or .txt file")],
    strategy: Annotated[ChunkingStrategy, Form()] = ChunkingStrategy.RECURSIVE,
    chunk_size: Annotated[int, Form(ge=100, le=8000)] = 1000,
    chunk_overlap: Annotated[int, Form(ge=0, le=2000)] = 150,
) -> DocumentRead:
    if chunk_overlap >= chunk_size:
        raise InvalidChunkingError("chunk_overlap must be smaller than chunk_size.")
    limit = container.settings.max_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise FileTooLargeError(f"File exceeds {limit} bytes.")

    record = await service.ingest(
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        data=data,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return DocumentRead.model_validate(record)


@router.get("", response_model=list[DocumentRead])
async def list_documents(documents: DocumentRepoDep) -> list[DocumentRead]:
    return [DocumentRead.model_validate(r) for r in await documents.list()]


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(document_id: str, documents: DocumentRepoDep) -> DocumentRead:
    record = await documents.get(document_id)
    if record is None:
        raise NotFoundError("Document not found.")
    return DocumentRead.model_validate(record)
