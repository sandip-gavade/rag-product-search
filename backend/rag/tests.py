import json
from decimal import Decimal
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient

from catalog.models import Product

from .synth import build_grounding_prompt, find_ungrounded_citations, synthesize_answer


def _make_product(external_id, title, category="Footwear", price="2999.00"):
    return Product.objects.create(
        external_id=external_id, title=title, description="A great product.",
        category=category, price=Decimal(price),
    )


class BuildGroundingPromptTests(TestCase):
    def test_includes_only_retrieved_products(self):
        boots = _make_product("p-boots", "Trailhead Waterproof Hiking Boots")
        prompt = build_grounding_prompt("waterproof hiking boots", [boots])

        self.assertIn("[p-boots]", prompt)
        self.assertIn("Trailhead Waterproof Hiking Boots", prompt)
        self.assertIn(str(boots.price), prompt)

    def test_does_not_include_unretrieved_products(self):
        boots = _make_product("p-boots", "Trailhead Waterproof Hiking Boots")
        prompt = build_grounding_prompt("waterproof hiking boots", [boots])
        self.assertNotIn("p-sandals", prompt)


class FindUngroundedCitationsTests(TestCase):
    def setUp(self):
        self.boots = _make_product("p-boots", "Trailhead Waterproof Hiking Boots")

    def test_no_citations_is_grounded(self):
        self.assertEqual(find_ungrounded_citations("A nice pair of boots.", [self.boots]), [])

    def test_citation_matching_retrieved_product_is_grounded(self):
        text = "The [p-boots] are a great waterproof option."
        self.assertEqual(find_ungrounded_citations(text, [self.boots]), [])

    def test_citation_not_in_retrieved_set_is_flagged(self):
        # The classic hallucination case: the model invents a product ID
        # that was never in the retrieved set / never shown to it.
        text = "Check out the [p-boots] and also the amazing [p-imaginary-product]."
        flagged = find_ungrounded_citations(text, [self.boots])
        self.assertEqual(flagged, ["p-imaginary-product"])


class FakeStreamingProvider:
    """No real API calls — deterministic canned response, chunked."""

    def __init__(self, full_text):
        self._full_text = full_text

    def synthesize_answer(self, prompt):
        return self._full_text

    def stream_answer(self, prompt):
        for word in self._full_text.split(" "):
            yield word + " "


class SynthesizeAnswerTests(TestCase):
    def setUp(self):
        self.boots = _make_product("p-boots", "Trailhead Waterproof Hiking Boots")

    def test_returns_grounded_answer_with_no_flags(self):
        provider = FakeStreamingProvider("The [p-boots] are perfect for wet trails.")
        result = synthesize_answer("waterproof boots", [self.boots], llm_provider=provider)

        self.assertIn("p-boots", result["answer"])
        self.assertEqual(result["ungrounded_citations"], [])

    def test_flags_hallucinated_product_not_in_retrieved_set(self):
        # Fabricated LLM response naming a product that was never retrieved.
        provider = FakeStreamingProvider(
            "The [p-boots] are great, and the [p-does-not-exist] is even better."
        )
        result = synthesize_answer("waterproof boots", [self.boots], llm_provider=provider)

        self.assertEqual(result["ungrounded_citations"], ["p-does-not-exist"])

    def test_no_products_returns_fallback_without_calling_llm(self):
        provider = FakeStreamingProvider("should never be called")
        with patch.object(provider, "synthesize_answer", side_effect=AssertionError("LLM should not be called")):
            result = synthesize_answer("nonexistent product", [], llm_provider=provider)

        self.assertEqual(result["ungrounded_citations"], [])
        self.assertIn("No matching products", result["answer"])


class FakeQueryEmbeddingProvider:
    def __init__(self, vector):
        self.vector = vector

    def embed(self, texts):
        return [self.vector for _ in texts]


def _hot_vector(hot_index, dimensions=1536):
    vec = [0.0] * dimensions
    vec[hot_index] = 1.0
    return vec


class RAGAnswerEndpointTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.boots = _make_product("p-boots", "Trailhead Waterproof Hiking Boots")
        self.boots.embedding = _hot_vector(0)
        from django.contrib.postgres.search import SearchVector
        self.boots.search_vector = SearchVector("title", weight="A") + SearchVector("description", weight="C")
        self.boots.save(update_fields=["embedding", "search_vector"])

    def _collect_sse(self, response):
        body = b"".join(response.streaming_content).decode()
        events = []
        for block in body.strip().split("\n\n"):
            if not block:
                continue
            lines = block.split("\n")
            event_type = lines[0].removeprefix("event: ")
            data = json.loads(lines[1].removeprefix("data: "))
            events.append((event_type, data))
        return events

    def test_missing_query_returns_400(self):
        response = self.client.get(reverse("rag-answer"))
        self.assertEqual(response.status_code, 400)

    def test_streams_products_then_tokens_then_grounded_answer(self):
        embedding_provider = FakeQueryEmbeddingProvider(_hot_vector(0))
        llm_provider = FakeStreamingProvider("The [p-boots] are ideal for wet trails.")

        with patch("search.retrieval.get_embedding_provider", return_value=embedding_provider), \
             patch("query_understanding.chain.get_llm_provider") as mock_understand_provider, \
             patch("rag.views.get_llm_provider", return_value=llm_provider):
            # No structured filters for this test — plain semantic query.
            from query_understanding.schema import ParsedQuery
            mock_understand_provider.return_value.parse_query.return_value = ParsedQuery(
                semantic_query="waterproof hiking boots",
            )
            response = self.client.get(reverse("rag-answer"), {"q": "waterproof hiking boots"})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response["Content-Type"], "text/event-stream")
            # StreamingHttpResponse is lazy — the generator only actually
            # runs once something consumes streaming_content, so the
            # patches (esp. rag.views.get_llm_provider) must still be
            # active here, not after the `with` block exits.
            events = self._collect_sse(response)

        event_types = [e[0] for e in events]
        self.assertEqual(event_types[0], "filters")
        self.assertEqual(event_types[1], "products")
        self.assertEqual(event_types[-1], "answer_complete")
        self.assertIn("token", event_types)

        products_event = events[1][1]
        self.assertEqual(products_event["products"][0]["external_id"], "p-boots")

        final_event = events[-1][1]
        self.assertIn("p-boots", final_event["answer"])
        self.assertEqual(final_event["ungrounded_citations"], [])

    def test_hallucinated_citation_is_flagged_in_final_event(self):
        embedding_provider = FakeQueryEmbeddingProvider(_hot_vector(0))
        llm_provider = FakeStreamingProvider("The [p-boots] and the [p-fabricated] are both great.")

        with patch("search.retrieval.get_embedding_provider", return_value=embedding_provider), \
             patch("query_understanding.chain.get_llm_provider") as mock_understand_provider, \
             patch("rag.views.get_llm_provider", return_value=llm_provider):
            from query_understanding.schema import ParsedQuery
            mock_understand_provider.return_value.parse_query.return_value = ParsedQuery(
                semantic_query="waterproof hiking boots",
            )
            response = self.client.get(reverse("rag-answer"), {"q": "waterproof hiking boots"})
            events = self._collect_sse(response)

        final_event = events[-1][1]
        self.assertEqual(final_event["ungrounded_citations"], ["p-fabricated"])

    def test_no_results_short_circuits_without_calling_llm(self):
        # A category filter that doesn't match self.boots ("Footwear")
        # guarantees an empty candidate pool at the SQL level, regardless
        # of embedding similarity — Phase 3's retrieval has no similarity
        # threshold, so this is the reliable way to force zero results.
        embedding_provider = FakeQueryEmbeddingProvider(_hot_vector(0))

        with patch("search.retrieval.get_embedding_provider", return_value=embedding_provider), \
             patch("query_understanding.chain.get_llm_provider") as mock_understand_provider, \
             patch("rag.views.get_llm_provider") as mock_rag_provider:
            from query_understanding.schema import ParsedQuery
            mock_understand_provider.return_value.parse_query.return_value = ParsedQuery(
                category="Electronics", semantic_query="a product that does not exist",
            )
            response = self.client.get(reverse("rag-answer"), {"q": "a product that does not exist"})
            events = self._collect_sse(response)
            mock_rag_provider.assert_not_called()

        self.assertEqual(events[1][1]["products"], [])
        self.assertIn("No matching products", events[-1][1]["answer"])

    def test_refine_skips_query_understanding_and_applies_filters_explicitly(self):
        # Filter-chip removal: the frontend sends refine=true with the
        # remaining filters explicit, and the raw q is used verbatim as
        # the semantic query — no LLM round-trip for understanding.
        embedding_provider = FakeQueryEmbeddingProvider(_hot_vector(0))
        llm_provider = FakeStreamingProvider("The [p-boots] look great.")

        with patch("search.retrieval.get_embedding_provider", return_value=embedding_provider), \
             patch("query_understanding.chain.get_llm_provider") as mock_understand_provider, \
             patch("rag.views.get_llm_provider", return_value=llm_provider):
            response = self.client.get(reverse("rag-answer"), {
                "q": "waterproof hiking boots", "refine": "true",
                "category": "", "price_min": "", "price_max": "5000",
            })
            events = self._collect_sse(response)
            mock_understand_provider.assert_not_called()

        filters_event = events[0][1]
        self.assertIsNone(filters_event["category"])
        self.assertIsNone(filters_event["price_min"])
        self.assertEqual(filters_event["price_max"], 5000.0)
        self.assertEqual(filters_event["semantic_query"], "waterproof hiking boots")

    def test_embedding_failure_returns_502(self):
        with patch("search.retrieval.get_embedding_provider", side_effect=RuntimeError("no key")):
            response = self.client.get(reverse("rag-answer"), {"q": "boots"})
        self.assertEqual(response.status_code, 502)
