from django.conf import settings
from langchain_openai import ChatOpenAI

from query_understanding.schema import ParsedQuery

from .base import LLMProvider


class LMStudioProvider(LLMProvider):
    """Local Qwen3 via LM Studio's OpenAI-compatible local server.

    LM Studio speaks the OpenAI Chat Completions wire format (not
    Ollama's native API), so this uses langchain_openai.ChatOpenAI
    pointed at LM Studio's local base_url instead of langchain_ollama —
    same model family as OllamaProvider, different serving layer. The
    same structured-output-reliability caveat from OllamaProvider applies
    here: local models are more prone to malformed tool-call JSON than
    Claude, which is why chain.py's fallback path exists.
    """

    def __init__(self, chat_model=None):
        self._chat_model = chat_model or ChatOpenAI(
            model=settings.LMSTUDIO_MODEL,
            base_url=settings.LMSTUDIO_BASE_URL,
            # LM Studio's local server doesn't check the key, but the
            # OpenAI client requires a non-empty string to be set.
            api_key="lm-studio",
        )
        # method="json_schema" is required here, not just a preference:
        # with_structured_output()'s default method forces a specific tool
        # by passing an object-shaped tool_choice, and LM Studio's server
        # (llama.cpp's OpenAI-compatible layer) only accepts the string
        # values none/auto/required — an object tool_choice 400s. Confirmed
        # against a live LM Studio server during development.
        self._structured_model = self._chat_model.with_structured_output(
            ParsedQuery, method="json_schema",
        )

    def parse_query(self, prompt: str) -> ParsedQuery:
        return self._structured_model.invoke(prompt)

    def synthesize_answer(self, prompt: str) -> str:
        response = self._chat_model.invoke(prompt)
        return response.content

    def stream_answer(self, prompt: str):
        for chunk in self._chat_model.stream(prompt):
            if chunk.content:
                yield chunk.content
