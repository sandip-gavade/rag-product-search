from django.conf import settings
from langchain_ollama import ChatOllama

from query_understanding.schema import ParsedQuery

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Local Qwen3 via Ollama/LM Studio — zero-cost for iteration, but
    weaker at structured extraction than Claude: smaller open models are
    more prone to malformed tool-call JSON on edge cases (ambiguous
    quantities, mixed units, sarcasm). chain.py's fallback-to-plain-query
    path is the safety net for that, and matters more with this provider
    than with AnthropicProvider.
    """

    def __init__(self, chat_model=None):
        self._chat_model = chat_model or ChatOllama(
            model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL,
        )
        self._structured_model = self._chat_model.with_structured_output(ParsedQuery)

    def parse_query(self, prompt: str) -> ParsedQuery:
        return self._structured_model.invoke(prompt)

    def synthesize_answer(self, prompt: str) -> str:
        response = self._chat_model.invoke(prompt)
        return response.content
