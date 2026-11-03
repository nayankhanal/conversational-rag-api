from collections.abc import Sequence

from openai import AsyncOpenAI

_BATCH_SIZE = 96


class OpenAIEmbedder:
    def __init__(self, client: AsyncOpenAI, model: str, dimensions: int) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = list(texts[start : start + _BATCH_SIZE])
            response = await self._client.embeddings.create(
                model=self._model, input=batch, dimensions=self._dimensions
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors
