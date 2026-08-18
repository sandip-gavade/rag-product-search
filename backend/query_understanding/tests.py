from unittest.mock import patch

from django.test import TestCase, override_settings

from .chain import understand_query
from .schema import ParsedQuery


class FakeLLMProvider:
    def __init__(self, parsed=None, raise_error=False):
        self._parsed = parsed
        self._raise_error = raise_error
        self.last_prompt = None

    def parse_query(self, prompt):
        self.last_prompt = prompt
        if self._raise_error:
            raise RuntimeError("provider unavailable")
        return self._parsed

    def synthesize_answer(self, prompt):
        return "n/a"


class UnderstandQueryTests(TestCase):
    def test_extracts_price_and_category_from_example_query(self):
        # The exact example from the project plan's Phase 4 done-when check.
        provider = FakeLLMProvider(parsed=ParsedQuery(
            price_max=3000, category="Footwear", semantic_query="waterproof hiking boots",
        ))
        parsed = understand_query("waterproof hiking boots under ₹3000", llm_provider=provider)

        self.assertEqual(parsed.price_max, 3000)
        self.assertEqual(parsed.category, "Footwear")
        self.assertEqual(parsed.semantic_query, "waterproof hiking boots")

    def test_category_matched_case_insensitively(self):
        provider = FakeLLMProvider(parsed=ParsedQuery(category="footwear", semantic_query="boots"))
        parsed = understand_query("boots", llm_provider=provider)
        self.assertEqual(parsed.category, "Footwear")

    def test_unknown_category_is_dropped_not_filtered_on(self):
        provider = FakeLLMProvider(parsed=ParsedQuery(category="Not A Real Category", semantic_query="boots"))
        parsed = understand_query("boots", llm_provider=provider)
        self.assertIsNone(parsed.category)

    def test_empty_semantic_query_falls_back_to_original_text(self):
        provider = FakeLLMProvider(parsed=ParsedQuery(semantic_query="   "))
        parsed = understand_query("waterproof hiking boots", llm_provider=provider)
        self.assertEqual(parsed.semantic_query, "waterproof hiking boots")

    def test_provider_failure_falls_back_to_raw_query_as_semantic_query(self):
        provider = FakeLLMProvider(raise_error=True)
        parsed = understand_query("some ambiguous or malformed query!!", llm_provider=provider)

        self.assertEqual(parsed.semantic_query, "some ambiguous or malformed query!!")
        self.assertIsNone(parsed.category)
        self.assertIsNone(parsed.price_min)
        self.assertIsNone(parsed.price_max)

    def test_prompt_includes_valid_category_list(self):
        provider = FakeLLMProvider(parsed=ParsedQuery(semantic_query="boots"))
        understand_query("boots", llm_provider=provider)
        self.assertIn("Footwear", provider.last_prompt)
        self.assertIn("Electronics", provider.last_prompt)


class GetLLMProviderTests(TestCase):
    @override_settings(LLM_PROVIDER="anthropic")
    def test_defaults_to_anthropic_provider(self):
        from .providers import get_llm_provider
        from .providers.anthropic_provider import AnthropicProvider

        with patch("query_understanding.providers.anthropic_provider.ChatAnthropic") as MockChat:
            MockChat.return_value.with_structured_output.return_value = object()
            provider = get_llm_provider()
        self.assertIsInstance(provider, AnthropicProvider)

    @override_settings(LLM_PROVIDER="ollama")
    def test_selects_ollama_provider(self):
        from .providers import get_llm_provider
        from .providers.ollama_provider import OllamaProvider

        with patch("query_understanding.providers.ollama_provider.ChatOllama") as MockChat:
            MockChat.return_value.with_structured_output.return_value = object()
            provider = get_llm_provider()
        self.assertIsInstance(provider, OllamaProvider)

    @override_settings(LLM_PROVIDER="lmstudio")
    def test_selects_lmstudio_provider(self):
        from .providers import get_llm_provider
        from .providers.lmstudio_provider import LMStudioProvider

        with patch("query_understanding.providers.lmstudio_provider.ChatOpenAI") as MockChat:
            MockChat.return_value.with_structured_output.return_value = object()
            provider = get_llm_provider()
        self.assertIsInstance(provider, LMStudioProvider)

    @override_settings(LLM_PROVIDER="not-a-real-provider")
    def test_unknown_provider_raises(self):
        from .providers import get_llm_provider
        with self.assertRaises(ValueError):
            get_llm_provider()


class FakeChatModel:
    """Stands in for a LangChain ChatModel — no real API/network calls."""

    def __init__(self, structured_result=None, text_result="an answer"):
        self._structured_result = structured_result
        self._text_result = text_result
        self.structured_prompt = None
        self.text_prompt = None

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructuredModel(self)

    def invoke(self, prompt):
        self.text_prompt = prompt
        return _FakeMessage(self._text_result)

    def stream(self, prompt):
        self.text_prompt = prompt
        for word in self._text_result.split(" "):
            yield _FakeMessage(word + " ")


class _FakeStructuredModel:
    def __init__(self, parent):
        self._parent = parent

    def invoke(self, prompt):
        self._parent.structured_prompt = prompt
        return self._parent._structured_result


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class AnthropicProviderTests(TestCase):
    def test_parse_query_and_synthesize_answer_use_injected_chat_model(self):
        from .providers.anthropic_provider import AnthropicProvider

        expected = ParsedQuery(semantic_query="boots")
        fake_chat_model = FakeChatModel(structured_result=expected, text_result="Great boots for rain.")
        provider = AnthropicProvider(chat_model=fake_chat_model)

        self.assertIs(provider.parse_query("some prompt"), expected)
        self.assertEqual(provider.synthesize_answer("some prompt"), "Great boots for rain.")

    def test_stream_answer_yields_chunks(self):
        from .providers.anthropic_provider import AnthropicProvider

        fake_chat_model = FakeChatModel(text_result="Great boots for rain.")
        provider = AnthropicProvider(chat_model=fake_chat_model)

        chunks = list(provider.stream_answer("some prompt"))
        self.assertEqual("".join(chunks).strip(), "Great boots for rain.")


class OllamaProviderTests(TestCase):
    def test_parse_query_and_synthesize_answer_use_injected_chat_model(self):
        from .providers.ollama_provider import OllamaProvider

        expected = ParsedQuery(semantic_query="boots")
        fake_chat_model = FakeChatModel(structured_result=expected, text_result="Great boots for rain.")
        provider = OllamaProvider(chat_model=fake_chat_model)

        self.assertIs(provider.parse_query("some prompt"), expected)
        self.assertEqual(provider.synthesize_answer("some prompt"), "Great boots for rain.")

    def test_stream_answer_yields_chunks(self):
        from .providers.ollama_provider import OllamaProvider

        fake_chat_model = FakeChatModel(text_result="Great boots for rain.")
        provider = OllamaProvider(chat_model=fake_chat_model)

        chunks = list(provider.stream_answer("some prompt"))
        self.assertEqual("".join(chunks).strip(), "Great boots for rain.")


class _FakeRawOpenAIClient:
    """Stands in for the raw `openai` client LMStudioProvider.parse_query()
    uses directly (bypassing LangChain) — see that method's docstring for
    why. `content`/`reasoning_content` are set per-test to exercise the
    fallback."""

    def __init__(self, content: str, reasoning_content: str = ""):
        message = type("Message", (), {"content": content, "reasoning_content": reasoning_content})()
        choice = type("Choice", (), {"message": message})()
        response = type("Response", (), {"choices": [choice]})()

        class _Completions:
            def create(_self, **kwargs):
                _self.called_with = kwargs
                return response

        self.chat = type("Chat", (), {"completions": _Completions()})()


class LMStudioProviderTests(TestCase):
    def test_parse_query_falls_back_to_reasoning_content_when_content_empty(self):
        # The Qwen3-via-LM-Studio quirk this provider works around: the
        # model's structured JSON answer lands in `reasoning_content`
        # instead of `content` — discovered live against a real
        # qwen/qwen3.5-9b instance during development.
        from .providers.lmstudio_provider import LMStudioProvider

        fake_client = _FakeRawOpenAIClient(content="", reasoning_content='{"semantic_query": "boots"}')
        provider = LMStudioProvider(client=fake_client)

        result = provider.parse_query("some prompt")

        self.assertEqual(result.semantic_query, "boots")
        self.assertEqual(fake_client.chat.completions.called_with["model"], provider.model)

    def test_parse_query_prefers_content_when_present(self):
        from .providers.lmstudio_provider import LMStudioProvider

        fake_client = _FakeRawOpenAIClient(
            content='{"semantic_query": "sandals"}',
            reasoning_content='{"semantic_query": "should not be used"}',
        )
        provider = LMStudioProvider(client=fake_client)

        result = provider.parse_query("some prompt")

        self.assertEqual(result.semantic_query, "sandals")

    def test_synthesize_answer_uses_injected_chat_model(self):
        from .providers.lmstudio_provider import LMStudioProvider

        fake_chat_model = FakeChatModel(text_result="Great boots for rain.")
        provider = LMStudioProvider(chat_model=fake_chat_model)

        self.assertEqual(provider.synthesize_answer("some prompt"), "Great boots for rain.")

    def test_stream_answer_yields_chunks(self):
        from .providers.lmstudio_provider import LMStudioProvider

        fake_chat_model = FakeChatModel(text_result="Great boots for rain.")
        provider = LMStudioProvider(chat_model=fake_chat_model)

        chunks = list(provider.stream_answer("some prompt"))
        self.assertEqual("".join(chunks).strip(), "Great boots for rain.")
