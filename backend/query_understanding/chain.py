from catalog.products_data import CATEGORIES

from .providers import get_llm_provider
from .schema import ParsedQuery

VALID_CATEGORIES = list(CATEGORIES.keys())

PROMPT_TEMPLATE = """You are a query-understanding step for a product search engine.
Extract structured filters from the user's free-text query, and produce a
cleaned semantic_query with those filters removed (keep the descriptive
product terms, e.g. "waterproof hiking boots").

Valid categories (use exactly one of these, or null if unclear): {categories}

User query: "{query}"
"""


def understand_query(query_text: str, llm_provider=None) -> ParsedQuery:
    """Parse free text into structured filters + a cleaned semantic query.

    Falls back to treating the whole input as the semantic query (no
    filters) if the LLM call fails for any reason — bad API key, network
    error, malformed structured output from a weaker local model. Query
    understanding degrading to plain semantic search is much better than
    the search endpoint erroring out because of it.
    """
    try:
        provider = llm_provider or get_llm_provider()
        prompt = PROMPT_TEMPLATE.format(categories=", ".join(VALID_CATEGORIES), query=query_text)
        parsed = provider.parse_query(prompt)
    except Exception:
        return ParsedQuery(semantic_query=query_text)

    # Validate the category against the catalog's actual categories rather
    # than trusting the model verbatim — filtering by a category that
    # doesn't exist would silently return zero results instead of falling
    # back to unfiltered search.
    if parsed.category:
        match = next((c for c in VALID_CATEGORIES if c.lower() == parsed.category.lower()), None)
        parsed.category = match

    if not parsed.semantic_query or not parsed.semantic_query.strip():
        parsed.semantic_query = query_text

    return parsed
