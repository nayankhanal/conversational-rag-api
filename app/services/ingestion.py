import uuid

from app.core.exceptions import EmptyDocumentError
from app.db.models import DocumentRecord
from app.repositories.documents import DocumentRepository
from app.schemas.ingestion import ChunkingStrategy
from app.services.chunking import build_chunker
from app.services.protocols import Embedder, VectorStore
from app.services.text_extraction import extract_text


class IngestionService:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        documents: DocumentRepository,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._documents = documents

    async def ingest(
        self,
        *,
        filename: str,
        content_type: str,
        data: bytes,
        strategy: ChunkingStrategy,
        chunk_size: int,
        chunk_overlap: int,
    ) -> DocumentRecord:
        text = await extract_text(filename, data)
        chunks = build_chunker(strategy, chunk_size, chunk_overlap).split(text)
        if not chunks:
            raise EmptyDocumentError("Document produced no chunks.")

        document_id = str(uuid.uuid4())
        vectors = await self._embedder.embed(chunks)
        await self._vector_store.upsert(document_id, filename, chunks, vectors)

        record = DocumentRecord(
            id=document_id,
            filename=filename,
            content_type=content_type,
            chunking_strategy=strategy.value,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            char_count=len(text),
            chunk_count=len(chunks),
        )
        return await self._documents.add(record)
