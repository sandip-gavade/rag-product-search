"""
Chunking strategy for product text.

Why one chunk per product instead of splitting into multiple chunks: RAG
chunking exists to keep each embedded piece of text focused enough that a
similarity match is meaningful, and small enough to fit a model's context
window. Product catalog entries here are short (a title + one or two
description sentences + a handful of attributes) — well under any
embedding model's limits — so splitting a single product into multiple
chunks would only fragment its meaning (e.g. separating "waterproof" from
"hiking boots") without any size benefit. One chunk per product is the
right granularity for this dataset; a catalog with long-form descriptions
would need to revisit this.
"""


def build_embedding_text(*, title: str, description: str, category: str, attributes: dict) -> str:
    """Build the single text chunk embedded for one product.

    Also used to compute the idempotency hash in ingestion.tasks, so this
    function is the single source of truth for "what changed" — if none of
    these fields differ, the product is skipped on re-ingestion.
    """
    lines = [title, f"Category: {category}", description]

    if attributes:
        attrs = ", ".join(f"{key}: {value}" for key, value in sorted(attributes.items()))
        lines.append(f"Attributes: {attrs}")

    return "\n".join(lines)
