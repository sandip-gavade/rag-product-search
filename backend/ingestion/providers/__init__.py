from django.conf import settings


def get_embedding_provider():
    """Factory selecting the active EmbeddingProvider from settings.

    Single switch point for the embeddings backend — everything else
    (tasks, tests) depends on the EmbeddingProvider interface, not on
    OpenAI specifically.
    """
    provider_name = settings.EMBEDDING_PROVIDER

    if provider_name == "openai":
        from .openai_provider import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider()

    if provider_name == "lmstudio":
        from .lmstudio_provider import LMStudioEmbeddingProvider
        return LMStudioEmbeddingProvider()

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider_name!r}")
