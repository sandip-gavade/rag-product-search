from django.conf import settings
from openai import OpenAI

from .base import EmbeddingProvider


class LMStudioEmbeddingProvider(EmbeddingProvider):
    """Local embeddings via LM Studio's OpenAI-compatible local server.

    Uses the `openai` client (not a LangChain wrapper) pointed at LM
    Studio's base_url — the embeddings endpoint is plain OpenAI-compatible
    REST, so no LangChain integration is needed here the way the LLM
    providers need langchain_openai/langchain_ollama for structured output.

    dimensions=768 matches the default LMSTUDIO_EMBEDDING_MODEL
    (nomic-embed-text-v1.5). Swapping to a different local embedding model
    with a different output width needs this updated to match, alongside
    a migration on catalog.models.EMBEDDING_DIMENSIONS (the pgvector
    column width) — see that constant's comment.
    """

    dimensions = 768

    def __init__(self, client: OpenAI | None = None):
        self._client = client or OpenAI(
            base_url=settings.LMSTUDIO_BASE_URL,
            api_key="lm-studio",  # unchecked by LM Studio, but the client requires a non-empty string
        )
        self.model = settings.LMSTUDIO_EMBEDDING_MODEL

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
