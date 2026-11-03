from dataclasses import dataclass

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.session import create_engine, create_session_factory, init_models
from app.services.embeddings import OpenAIEmbedder
from app.services.llm import OpenAIChatLLM
from app.services.memory import RedisChatMemory
from app.services.vector_store import QdrantVectorStore


@dataclass(slots=True)
class Container:
    """Process-wide singletons, created at startup and closed at shutdown."""

    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis: Redis
    qdrant: AsyncQdrantClient
    embedder: OpenAIEmbedder
    llm: OpenAIChatLLM
    vector_store: QdrantVectorStore
    memory: RedisChatMemory

    @classmethod
    async def create(cls, settings: Settings) -> "Container":
        engine = create_engine(settings.database_url)
        await init_models(engine)

        openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(), base_url=settings.openai_base_url
        )
        qdrant = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        )
        vector_store = QdrantVectorStore(
            qdrant, settings.qdrant_collection, settings.embedding_dimensions
        )
        await vector_store.ensure_collection()

        redis: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
        return cls(
            settings=settings,
            engine=engine,
            session_factory=create_session_factory(engine),
            redis=redis,
            qdrant=qdrant,
            embedder=OpenAIEmbedder(
                openai_client, settings.embedding_model, settings.embedding_dimensions
            ),
            llm=OpenAIChatLLM(openai_client, settings.chat_model),
            vector_store=vector_store,
            memory=RedisChatMemory(
                redis, settings.chat_history_ttl_seconds, settings.chat_history_max_messages
            ),
        )

    async def aclose(self) -> None:
        await self.redis.aclose()
        await self.qdrant.close()
        await self.engine.dispose()
