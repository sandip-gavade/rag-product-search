from openai import OpenAI

from .base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    # 1536-dimensional output — must match catalog.models.EMBEDDING_DIMENSIONS.
    model = "text-embedding-3-small"
    dimensions = 1536

    def __init__(self, client: OpenAI | None = None):
        # OpenAI() reads OPENAI_API_KEY from the environment automatically;
        # a client can be injected instead for testing without an API key.
        self._client = client or OpenAI()

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        # The API guarantees response.data is ordered to match the input list.
        return [item.embedding for item in response.data]
