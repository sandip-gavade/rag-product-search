from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface every embedding backend implements.

    Kept swappable (rather than calling OpenAI directly from the task) so
    the provider can later change to a local `sentence-transformers` model
    without touching ingestion/tasks.py — only a new provider class plus
    an EMBEDDING_PROVIDER env var value.
    """

    #: Vector width this provider returns. Must match
    #: catalog.models.EMBEDDING_DIMENSIONS (the pgvector column width) —
    #: switching providers with a different width requires a migration.
    dimensions: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""
