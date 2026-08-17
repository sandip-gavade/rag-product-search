"""
RAG answer synthesis: turn Phase 3/4's retrieved products into a short
natural-language recommendation, grounded in the actual retrieved set.

Grounding strategy: the prompt requires the model to cite each product it
mentions by its bracketed external_id, e.g. "[p-boots-0001]" — not because
the citation itself is useful copy, but because it turns "did the model
hallucinate a product?" from a fuzzy text-matching problem into a cheap,
deterministic one: extract every bracketed ID with a regex, diff against
the IDs actually in the retrieved set. Anything left over is a hallucinated
reference, since the model was never shown any ID outside that set — it
can't cite an ID we didn't give it without inventing one.
"""

import logging
import re

from query_understanding.providers import get_llm_provider

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[([a-zA-Z0-9\-]+)\]")

PROMPT_TEMPLATE = """A user searched for: "{query}"

Here are the matched products. Reference each one you mention by its
bracketed ID exactly as written below (e.g. [{example_id}]) — do not
invent IDs, and do not mention any product that isn't in this list:

{product_list}

Write a short (2-4 sentence) natural-language recommendation for the user,
referencing specific products by name and bracketed ID. Only recommend
products from the list above.
"""


def build_grounding_prompt(query_text: str, products: list) -> str:
    lines = [f"- [{p.external_id}] {p.title} — ₹{p.price} ({p.category})" for p in products]
    return PROMPT_TEMPLATE.format(
        query=query_text,
        product_list="\n".join(lines),
        example_id=products[0].external_id,
    )


def find_ungrounded_citations(answer_text: str, products: list) -> list[str]:
    """IDs cited in `answer_text` that don't match any retrieved product.

    Empty list means every citation the model made is grounded in the
    actual retrieved set — the hallucination guard passed.
    """
    valid_ids = {p.external_id for p in products}
    cited_ids = set(CITATION_PATTERN.findall(answer_text))
    return sorted(cited_ids - valid_ids)


def synthesize_answer(query_text: str, products: list, llm_provider=None) -> dict:
    """Generate a grounded recommendation from the retrieved products
    (blocking). Used by tests and anywhere the streamed endpoint's
    incremental delivery isn't needed.

    Returns {"answer": str, "ungrounded_citations": list[str]}. A
    non-empty ungrounded_citations means the hallucination guard caught
    something — logged here so it's visible in ops, and returned to the
    caller so Phase 6's frontend can decide whether to surface it.
    """
    if not products:
        return {"answer": "No matching products were found for this search.", "ungrounded_citations": []}

    provider = llm_provider or get_llm_provider()
    prompt = build_grounding_prompt(query_text, products)
    answer = provider.synthesize_answer(prompt)

    ungrounded = find_ungrounded_citations(answer, products)
    if ungrounded:
        logger.warning("RAG answer cited product ID(s) not in the retrieved set: %s", ungrounded)

    return {"answer": answer, "ungrounded_citations": ungrounded}
