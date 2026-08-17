from abc import ABC, abstractmethod

from query_understanding.schema import ParsedQuery


class LLMProvider(ABC):
    """Interface every LLM backend implements — mirrors ingestion's
    EmbeddingProvider pattern so the active LLM can be swapped via
    LLM_PROVIDER without touching chain.py or rag/synth.py (Phase 5).
    """

    @abstractmethod
    def parse_query(self, prompt: str) -> ParsedQuery:
        """Return structured filters extracted from `prompt`, via the
        model's native tool-calling / structured-output support."""

    @abstractmethod
    def synthesize_answer(self, prompt: str) -> str:
        """Return a freeform text completion for `prompt`."""
