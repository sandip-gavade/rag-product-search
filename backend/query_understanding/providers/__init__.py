from django.conf import settings


def get_llm_provider():
    """Factory selecting the active LLMProvider from settings — mirrors
    ingestion.providers.get_embedding_provider.
    """
    provider_name = settings.LLM_PROVIDER

    if provider_name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider()

    if provider_name == "ollama":
        from .ollama_provider import OllamaProvider
        return OllamaProvider()

    raise ValueError(f"Unknown LLM_PROVIDER: {provider_name!r}")
