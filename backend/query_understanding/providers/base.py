from abc import ABC, abstractmethod
from typing import Iterator

from query_understanding.schema import ParsedQuery


class LLMProvider(ABC):
    """Interface every LLM backend implements — mirrors ingestion's
    EmbeddingProvider pattern so the active LLM can be swapped via
    LLM_PROVIDER without touching chain.py or rag/synth.py.
    """

    @abstractmethod
    def parse_query(self, prompt: str) -> ParsedQuery:
        """Return structured filters extracted from `prompt`, via the
        model's native tool-calling / structured-output support."""

    @abstractmethod
    def synthesize_answer(self, prompt: str) -> str:
        """Return a freeform text completion for `prompt` (blocking)."""

    @abstractmethod
    def stream_answer(self, prompt: str) -> Iterator[str]:
        """Yield a freeform text completion for `prompt` incrementally,
        chunk by chunk, for Phase 5's streamed RAG answer endpoint."""
