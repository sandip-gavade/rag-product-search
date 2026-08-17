from django.contrib.postgres.search import SearchQuery, SearchRank
from pgvector.django import CosineDistance

from catalog.models import Product
from ingestion.providers import get_embedding_provider

from .fusion import reciprocal_rank_fusion

# How many candidates each individual signal contributes to the fusion pool,
# wider than the caller's requested top_k so a product that's mediocre on
# one signal but excellent on the other still gets a chance to be pulled up
# by fusion instead of being cut before the two signals are even combined.
CANDIDATE_POOL_SIZE = 50


def _apply_filters(qs, category: str | None, price_min: float | None, price_max: float | None):
    """Structured filters from Phase 4's query understanding, applied as
    real SQL WHERE clauses before either candidate pool is ranked — so
    e.g. a price_max filter actually excludes rows from the DB query
    rather than being applied after the fact to already-fetched results.
    """
    if category:
        qs = qs.filter(category=category)
    if price_min is not None:
        qs = qs.filter(price__gte=price_min)
    if price_max is not None:
        qs = qs.filter(price__lte=price_max)
    return qs


def _vector_candidates(
    query_embedding: list[float], limit: int,
    category: str | None = None, price_min: float | None = None, price_max: float | None = None,
) -> list[tuple[int, float]]:
    """(product_id, cosine_distance) pairs, closest first.

    Products without an embedding yet (not ingested) are excluded rather
    than scored — comparing against a null vector isn't meaningful.
    """
    qs = _apply_filters(Product.objects.exclude(embedding=None), category, price_min, price_max)
    qs = qs.annotate(distance=CosineDistance("embedding", query_embedding)).order_by("distance")[:limit]
    return [(p.id, p.distance) for p in qs]


def _keyword_candidates(
    query_text: str, limit: int,
    category: str | None = None, price_min: float | None = None, price_max: float | None = None,
) -> list[tuple[int, float]]:
    """(product_id, ts_rank) pairs, highest rank first."""
    search_query = SearchQuery(query_text, config="english")
    qs = _apply_filters(Product.objects.filter(search_vector=search_query), category, price_min, price_max)
    qs = qs.annotate(rank=SearchRank("search_vector", search_query)).order_by("-rank")[:limit]
    return [(p.id, p.rank) for p in qs]


def hybrid_search(
    query_text: str, top_k: int = 10, embedding_provider=None,
    category: str | None = None, price_min: float | None = None, price_max: float | None = None,
) -> list[dict]:
    """Run vector + keyword search, fuse with RRF, return the top_k products.

    category/price_min/price_max are the structured filters Phase 4's
    query understanding extracts from the raw query — applied as SQL
    filters to both candidate pools before ranking, not as a post-filter
    on the fused results.

    Each result includes a score breakdown (vector_score, keyword_score,
    fused_score) so a caller can see *why* a product ranked where it did —
    vector_score is cosine similarity (1 - distance, higher is better);
    keyword_score is the raw Postgres ts_rank; either is None if the
    product didn't appear in that signal's candidate pool at all.
    """
    provider = embedding_provider or get_embedding_provider()
    [query_embedding] = provider.embed([query_text])

    vector_candidates = _vector_candidates(
        query_embedding, CANDIDATE_POOL_SIZE, category, price_min, price_max
    )
    keyword_candidates = _keyword_candidates(
        query_text, CANDIDATE_POOL_SIZE, category, price_min, price_max
    )

    vector_distance_by_id = dict(vector_candidates)
    keyword_rank_by_id = dict(keyword_candidates)

    vector_ranked_ids = [product_id for product_id, _ in vector_candidates]
    keyword_ranked_ids = [product_id for product_id, _ in keyword_candidates]

    fused = reciprocal_rank_fusion(vector_ranked_ids, keyword_ranked_ids)[:top_k]

    products_by_id = Product.objects.in_bulk([product_id for product_id, _ in fused])

    results = []
    for product_id, fused_score in fused:
        product = products_by_id.get(product_id)
        if product is None:
            continue
        distance = vector_distance_by_id.get(product_id)
        results.append({
            "product": product,
            "vector_score": (1 - distance) if distance is not None else None,
            "keyword_score": keyword_rank_by_id.get(product_id),
            "fused_score": fused_score,
        })
    return results
