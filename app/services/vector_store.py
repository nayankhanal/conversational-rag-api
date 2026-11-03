import uuid
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.services.protocols import RetrievedChunk

_UPSERT_BATCH = 64


class QdrantVectorStore:
    def __init__(self, client: AsyncQdrantClient, collection: str, dimensions: int) -> None:
        self._client = client
        self._collection = collection
        self._dimensions = dimensions

    async def ensure_collection(self) -> None:
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                self._collection,
                vectors_config=VectorParams(size=self._dimensions, distance=Distance.COSINE),
            )

    async def upsert(
        self,
        document_id: str,
        filename: str,
        chunks: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.UUID(document_id), str(index))),
                vector=list(vector),
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": index,
                    "text": text,
                },
            )
            for index, (text, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        for start in range(0, len(points), _UPSERT_BATCH):
            await self._client.upsert(self._collection, points[start : start + _UPSERT_BATCH])

    async def search(self, vector: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        result = await self._client.query_points(
            self._collection, query=list(vector), limit=top_k, with_payload=True
        )
        return [
            RetrievedChunk(
                document_id=str(point.payload["document_id"]),
                filename=str(point.payload["filename"]),
                chunk_index=int(point.payload["chunk_index"]),
                score=point.score,
                text=str(point.payload["text"]),
            )
            for point in result.points
            if point.payload is not None
        ]
