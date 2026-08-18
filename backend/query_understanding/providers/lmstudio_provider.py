from django.conf import settings
from langchain_openai import ChatOpenAI
from openai import OpenAI

from query_understanding.schema import ParsedQuery

from .base import LLMProvider


class LMStudioProvider(LLMProvider):
    """Local Qwen3 via LM Studio's OpenAI-compatible local server.

    parse_query() uses the raw `openai` client, not LangChain's
    with_structured_output() — discovered live against a real
    qwen/qwen3.5-9b instance that LangChain's structured-output parser
    hard-fails here. Qwen3's hybrid reasoning mode routes the model's
    structured JSON answer into a `reasoning_content` field instead of the
    standard `content` field when response_format is json_schema; LangChain
    only ever looks at `content`, sees it empty, and raises rather than
    falling back. Handling both fields ourselves is the fix — and this
    provider also sets a generous max_tokens, since a low default
    truncates the model mid-thought (finish_reason: "length") before it
    reaches an answer at all, which looked like the same symptom initially
    but is a separate failure mode from the field-routing one.

    synthesize_answer()/stream_answer() stay on LangChain — freeform chat
    doesn't hit either issue (reasoning naturally precedes the final
    answer in `content`, doesn't replace it).
    """

    def __init__(self, chat_model=None, client=None):
        self._chat_model = chat_model or ChatOpenAI(
            model=settings.LMSTUDIO_MODEL,
            base_url=settings.LMSTUDIO_BASE_URL,
            api_key="lm-studio",  # unchecked by LM Studio, but the client requires a non-empty string
            max_tokens=8192,
        )
        self._client = client or OpenAI(base_url=settings.LMSTUDIO_BASE_URL, api_key="lm-studio")
        self.model = settings.LMSTUDIO_MODEL

    def parse_query(self, prompt: str) -> ParsedQuery:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "parsed_query", "schema": ParsedQuery.model_json_schema()},
            },
        )
        message = response.choices[0].message
        raw_json = message.content or getattr(message, "reasoning_content", None) or ""
        return ParsedQuery.model_validate_json(raw_json)

    def synthesize_answer(self, prompt: str) -> str:
        response = self._chat_model.invoke(prompt)
        return response.content

    def stream_answer(self, prompt: str):
        for chunk in self._chat_model.stream(prompt):
            if chunk.content:
                yield chunk.content
