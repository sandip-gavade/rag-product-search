from langchain_anthropic import ChatAnthropic

from query_understanding.schema import ParsedQuery

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    # claude-opus-5 — current flagship model; not downgraded for cost here
    # since that's a deploy-time decision, not a code default. Swap models
    # via LLM_PROVIDER / a different provider class, not by editing this
    # string per environment.
    #
    # No temperature/top_p is set: Claude Opus 5 rejects sampling
    # parameters outright (400), unlike older models where they were just
    # optional.
    model = "claude-opus-5"

    def __init__(self, chat_model=None):
        self._chat_model = chat_model or ChatAnthropic(model=self.model)
        self._structured_model = self._chat_model.with_structured_output(ParsedQuery)

    def parse_query(self, prompt: str) -> ParsedQuery:
        return self._structured_model.invoke(prompt)

    def synthesize_answer(self, prompt: str) -> str:
        response = self._chat_model.invoke(prompt)
        return response.content
