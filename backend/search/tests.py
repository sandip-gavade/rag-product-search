from decimal import Decimal
from unittest.mock import patch

from django.contrib.postgres.search import SearchVector
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient

from catalog.models import Product

from .fusion import reciprocal_rank_fusion
from .retrieval import hybrid_search


class ReciprocalRankFusionTests(TestCase):
    """Pure function — no DB needed."""

    def test_item_ranked_first_in_both_lists_wins(self):
        fused = reciprocal_rank_fusion(["a", "b", "c"], ["a", "c", "b"])
        self.assertEqual(fused[0][0], "a")

    def test_item_appearing_in_both_lists_outranks_single_list_item(self):
        # "p1" is #1 in list_a and #2 in list_b; "p3" is #3 in list_a and
        # #1 in list_b. Appearing near the top of *both* lists should beat
        # appearing at #1 in only one.
        fused = reciprocal_rank_fusion(["p1", "p2", "p3"], ["p3", "p1"])
        ids_in_order = [item_id for item_id, _ in fused]
        self.assertEqual(ids_in_order, ["p1", "p3", "p2"])

    def test_item_only_in_one_list_is_still_included(self):
        fused = reciprocal_rank_fusion(["a"], ["b"])
        ids = {item_id for item_id, _ in fused}
        self.assertEqual(ids, {"a", "b"})

    def test_empty_lists_return_empty(self):
        self.assertEqual(reciprocal_rank_fusion([], []), [])

    def test_scores_are_descending(self):
        fused = reciprocal_rank_fusion(["a", "b", "c", "d"], ["d", "c"])
        scores = [score for _, score in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))


class FakeQueryEmbeddingProvider:
    """Returns a fixed embedding for any query text, so vector search
    ranks deterministically against hand-picked product embeddings."""

    def __init__(self, vector):
        self.vector = vector

    def embed(self, texts):
        return [self.vector for _ in texts]


def _make_embedding(hot_index: int, dimensions: int = 1536) -> list[float]:
    """A near-orthogonal-basis vector: 1.0 at `hot_index`, 0 elsewhere.
    Cosine similarity between two such vectors is 1.0 if they share the
    same hot_index, 0.0 otherwise — makes vector-search ranking exact and
    predictable in tests."""
    vec = [0.0] * dimensions
    vec[hot_index] = 1.0
    return vec


def _create_product(external_id, title, description, category, hot_index, price="999.00"):
    product = Product.objects.create(
        external_id=external_id, title=title, description=description,
        category=category, price=Decimal(price), embedding=_make_embedding(hot_index),
    )
    product.search_vector = SearchVector("title", weight="A") + SearchVector("description", weight="C")
    product.save(update_fields=["search_vector"])
    return product


class HybridSearchTests(TestCase):
    def setUp(self):
        self.boots = _create_product(
            "p-boots", "Trailhead Waterproof Hiking Boots",
            "Great for wet, rocky trails.", "Footwear", hot_index=0,
        )
        self.sandals = _create_product(
            "p-sandals", "Rockridge Waterproof Sandals",
            "Lightweight summer sandals.", "Footwear", hot_index=1,
        )
        self.speaker = _create_product(
            "p-speaker", "Voltix Portable Bluetooth Speaker",
            "Loud, compact, splash-resistant.", "Electronics", hot_index=2,
        )

    def test_top_result_matches_both_vector_and_keyword_signal(self):
        # Query embedding matches self.boots exactly (hot_index=0); the
        # query text also matches "hiking" only in self.boots' title.
        provider = FakeQueryEmbeddingProvider(_make_embedding(0))
        results = hybrid_search("waterproof hiking boots", top_k=3, embedding_provider=provider)

        self.assertEqual(results[0]["product"].id, self.boots.id)
        self.assertIsNotNone(results[0]["vector_score"])
        self.assertIsNotNone(results[0]["keyword_score"])

    def test_keyword_only_match_still_returned(self):
        # Query embedding matches nothing (all-zero vector, far from every
        # product), but "sandals" is a keyword-only match for self.sandals.
        provider = FakeQueryEmbeddingProvider([0.0] * 1536)
        results = hybrid_search("sandals", top_k=3, embedding_provider=provider)

        result_ids = [r["product"].id for r in results]
        self.assertIn(self.sandals.id, result_ids)
        sandals_result = next(r for r in results if r["product"].id == self.sandals.id)
        self.assertIsNotNone(sandals_result["keyword_score"])

    def test_products_without_embedding_are_excluded_from_vector_signal(self):
        unembedded = Product.objects.create(
            external_id="p-unembedded", title="Unembedded Product",
            description="No embedding yet.", category="Footwear", price=Decimal("100.00"),
        )
        provider = FakeQueryEmbeddingProvider(_make_embedding(0))
        results = hybrid_search("boots", top_k=10, embedding_provider=provider)

        result_ids = [r["product"].id for r in results]
        self.assertNotIn(unembedded.id, result_ids)


class SearchEndpointTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.boots = _create_product(
            "p-boots", "Trailhead Waterproof Hiking Boots",
            "Great for wet, rocky trails.", "Footwear", hot_index=0,
        )

    def test_missing_query_param_returns_400(self):
        response = self.client.get(reverse("search"))
        self.assertEqual(response.status_code, 400)

    def test_search_returns_ranked_results_with_score_breakdown(self):
        provider = FakeQueryEmbeddingProvider(_make_embedding(0))
        with patch("search.retrieval.get_embedding_provider", return_value=provider):
            response = self.client.get(reverse("search"), {"q": "waterproof hiking boots"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["query"], "waterproof hiking boots")
        self.assertGreaterEqual(len(body["results"]), 1)
        top = body["results"][0]
        self.assertEqual(top["product"]["external_id"], "p-boots")
        self.assertIn("vector_score", top)
        self.assertIn("keyword_score", top)
        self.assertIn("fused_score", top)

    def test_invalid_top_k_returns_400(self):
        response = self.client.get(reverse("search"), {"q": "boots", "top_k": "not-a-number"})
        self.assertEqual(response.status_code, 400)

    def test_embedding_provider_failure_returns_502_not_a_crash(self):
        with patch("search.retrieval.get_embedding_provider", side_effect=RuntimeError("no API key")):
            response = self.client.get(reverse("search"), {"q": "boots"})
        self.assertEqual(response.status_code, 502)
