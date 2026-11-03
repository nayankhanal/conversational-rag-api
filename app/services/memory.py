from redis.asyncio import Redis

from app.schemas.chat import BookingDraft, ChatMessage


class RedisChatMemory:
    """Per-session chat history (capped list) and in-progress booking draft."""

    def __init__(self, redis: Redis, ttl_seconds: int, max_messages: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._max = max_messages

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"chat:{session_id}:messages"

    @staticmethod
    def _draft_key(session_id: str) -> str:
        return f"chat:{session_id}:booking_draft"

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        raw = await self._redis.lrange(self._history_key(session_id), 0, -1)
        return [ChatMessage.model_validate_json(item) for item in raw]

    async def append(self, session_id: str, *messages: ChatMessage) -> None:
        key = self._history_key(session_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *(m.model_dump_json() for m in messages))
            pipe.ltrim(key, -self._max, -1)
            pipe.expire(key, self._ttl)
            await pipe.execute()

    async def get_booking_draft(self, session_id: str) -> BookingDraft:
        raw = await self._redis.get(self._draft_key(session_id))
        return BookingDraft.model_validate_json(raw) if raw else BookingDraft()

    async def set_booking_draft(self, session_id: str, draft: BookingDraft | None) -> None:
        key = self._draft_key(session_id)
        if draft is None:
            await self._redis.delete(key)
        else:
            await self._redis.set(key, draft.model_dump_json(), ex=self._ttl)

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._history_key(session_id), self._draft_key(session_id))
