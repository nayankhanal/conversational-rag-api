from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_type: str
    chunking_strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    char_count: int
    chunk_count: int
    created_at: datetime
