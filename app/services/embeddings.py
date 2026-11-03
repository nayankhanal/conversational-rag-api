from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI

_BATCH_SIZE = 96


class OpenAIEmbedder:
    def __init__(self, client: AsyncOpenAI, model: str, dimensions: int | None) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = list(texts[start : start + _BATCH_SIZE])
            extra: dict[str, Any] = (
                {} if self._dimensions is None else {"dimensions": self._dimensions}
            )
            response = await self._client.embeddings.create(model=self._model, input=batch, **extra)
            vectors.extend(item.embedding for item in response.data)
        return vectors
