import hashlib

from celery import shared_task
from django.contrib.postgres.search import SearchVector

from catalog.models import Product

from .providers import get_embedding_provider
from .text import build_embedding_text


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@shared_task
def embed_product(product_id: int) -> str:
    """Embed one product and refresh its search_vector. Safe to re-run.

    Returns "skipped" | "embedded" | "missing" — useful in tests and logs
    to confirm the idempotency guard actually fired instead of silently
    re-calling the embeddings API.
    """
    try:
        product = Product.objects.get(pk=product_id)
    except Product.DoesNotExist:
        return "missing"

    text = build_embedding_text(
        title=product.title,
        description=product.description,
        category=product.category,
        attributes=product.attributes,
    )
    new_hash = _content_hash(text)

    # Idempotency guard: if the text we'd embed hasn't changed since last
    # time (and an embedding already exists), skip the API call and the
    # write entirely. This is what makes re-running the bulk task a no-op
    # for unchanged products instead of re-embedding + re-billing every row.
    if product.embedding is not None and product.content_hash == new_hash:
        return "skipped"

    provider = get_embedding_provider()
    [vector] = provider.embed([text])

    product.embedding = vector
    product.content_hash = new_hash
    # Weighted so a keyword match on the product title ranks above one
    # buried in the description — read together with the fusion logic in
    # Phase 3's retrieval endpoint.
    product.search_vector = (
        SearchVector("title", weight="A", config="english")
        + SearchVector("category", weight="B", config="english")
        + SearchVector("description", weight="C", config="english")
    )
    product.save(update_fields=["embedding", "content_hash", "search_vector"])
    return "embedded"


@shared_task
def embed_all_products() -> int:
    """Enqueue embed_product for every product.

    Deliberately enqueues all rows rather than pre-filtering for
    "missing embedding" — the per-product idempotency check in
    embed_product already makes unchanged rows a cheap no-op, so a single
    dumb fan-out here is simpler than keeping two places in sync about
    what counts as "stale".
    """
    product_ids = list(Product.objects.values_list("id", flat=True))
    for product_id in product_ids:
        embed_product.delay(product_id)
    return len(product_ids)
