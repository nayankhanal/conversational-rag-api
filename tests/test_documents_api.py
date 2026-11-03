from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_container, get_ingestion_service
from app.db.models import DocumentRecord
from app.main import create_app
from app.schemas.ingestion import ChunkingStrategy


class FakeIngestion:
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
        from datetime import UTC, datetime

        return DocumentRecord(
            id="1",
            filename=filename,
            content_type=content_type,
            chunking_strategy=strategy.value,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            char_count=len(data),
            chunk_count=1,
            created_at=datetime.now(UTC),
        )


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_container] = lambda: SimpleNamespace(
        settings=SimpleNamespace(max_upload_bytes=100)
    )
    app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestion()
    yield TestClient(app)


def test_upload_with_strategy(client: TestClient) -> None:
    r = client.post(
        "/api/v1/documents",
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"strategy": "fixed", "chunk_size": "200", "chunk_overlap": "20"},
    )
    assert r.status_code == 201
    assert r.json()["chunking_strategy"] == "fixed"


def test_overlap_must_be_smaller(client: TestClient) -> None:
    r = client.post(
        "/api/v1/documents",
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"chunk_size": "200", "chunk_overlap": "200"},
    )
    assert r.status_code == 422


def test_too_large(client: TestClient) -> None:
    r = client.post("/api/v1/documents", files={"file": ("a.txt", b"x" * 101, "text/plain")})
    assert r.status_code == 413
