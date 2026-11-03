from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Any OpenAI-compatible provider; defaults to Google Gemini's free tier.
    llm_api_key: SecretStr
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    chat_model: str = "gemini-3.5-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 3072  # must match the model's output size
    send_embedding_dimensions: bool = False  # pass `dimensions` to the provider (OpenAI does)

    # Embedded on-disk Qdrant by default; set QDRANT_URL to use a server or Qdrant Cloud.
    qdrant_path: str = "./qdrant_data"
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"

    redis_url: str = "redis://localhost:6379/0"
    chat_history_ttl_seconds: int = 60 * 60 * 24
    chat_history_max_messages: int = 20

    database_url: str = "sqlite+aiosqlite:///./app.db"

    retrieval_top_k: int = 5
    max_upload_bytes: int = 10 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()  # populated from env
