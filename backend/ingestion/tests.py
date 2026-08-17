from decimal import Decimal
from unittest.mock import patch

from django.contrib.postgres.search import SearchQuery
from django.test import TestCase, override_settings

from catalog.models import Product

from .providers import get_embedding_provider
from .providers.base import EmbeddingProvider
from .text import build_embedding_text


class BuildEmbeddingTextTests(TestCase):
    """Pure function — no DB needed."""

    def test_includes_title_category_description(self):
        text = build_embedding_text(
            title="Trailhead Waterproof Hiking Boots",
            description="Built for wet trails.",
            category="Footwear",
            attributes={},
        )
        self.assertIn("Trailhead Waterproof Hiking Boots", text)
        self.assertIn("Footwear", text)
        self.assertIn("Built for wet trails.", text)

    def test_includes_attributes_when_present(self):
        text = build_embedding_text(
            title="T", description="D", category="C",
            attributes={"color": "Black", "brand": "Trailhead"},
        )
        self.assertIn("color: Black", text)
        self.assertIn("brand: Trailhead", text)

    def test_omits_attributes_line_when_empty(self):
        text = build_embedding_text(title="T", description="D", category="C", attributes={})
        self.assertNotIn("Attributes:", text)

    def test_changing_any_field_changes_output(self):
        base = build_embedding_text(title="T", description="D", category="C", attributes={"a": 1})
        changed = build_embedding_text(title="T2", description="D", category="C", attributes={"a": 1})
        self.assertNotEqual(base, changed)


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic, zero-cost stand-in for the real API in tests."""

    dimensions = 1536

    def __init__(self):
        self.call_count = 0
        self.last_texts = None

    def embed(self, texts):
        self.call_count += 1
        self.last_texts = texts
        return [[0.1] * self.dimensions for _ in texts]


class EmbedProductTaskTests(TestCase):
    def setUp(self):
        self.product = Product.objects.create(
            external_id="test-0001",
            title="Trailhead Waterproof Hiking Boots",
            description="Built for wet, rocky trails.",
            category="Footwear",
            price=Decimal("2999.00"),
            attributes={"brand": "Trailhead", "color": "Black"},
        )

    def test_embeds_and_saves_vector(self):
        fake_provider = FakeEmbeddingProvider()
        with patch("ingestion.tasks.get_embedding_provider", return_value=fake_provider):
            from .tasks import embed_product
            result = embed_product(self.product.id)

        self.product.refresh_from_db()
        self.assertEqual(result, "embedded")
        self.assertEqual(fake_provider.call_count, 1)
        self.assertIsNotNone(self.product.embedding)
        self.assertEqual(len(self.product.embedding), 1536)
        self.assertTrue(self.product.content_hash)

    def test_search_vector_is_populated_and_matches_title_word(self):
        fake_provider = FakeEmbeddingProvider()
        with patch("ingestion.tasks.get_embedding_provider", return_value=fake_provider):
            from .tasks import embed_product
            embed_product(self.product.id)

        matched = Product.objects.filter(
            pk=self.product.pk, search_vector=SearchQuery("hiking", config="english"),
        ).exists()
        self.assertTrue(matched)

    def test_rerun_with_unchanged_content_is_a_noop(self):
        fake_provider = FakeEmbeddingProvider()
        with patch("ingestion.tasks.get_embedding_provider", return_value=fake_provider):
            from .tasks import embed_product
            first_result = embed_product(self.product.id)
            self.product.refresh_from_db()
            first_vector = list(self.product.embedding)

            second_result = embed_product(self.product.id)
            self.product.refresh_from_db()

        self.assertEqual(first_result, "embedded")
        self.assertEqual(second_result, "skipped")
        # Only the first run should have called the (expensive, billed) API.
        self.assertEqual(fake_provider.call_count, 1)
        self.assertEqual(list(self.product.embedding), first_vector)

    def test_changed_content_triggers_re_embedding(self):
        fake_provider = FakeEmbeddingProvider()
        with patch("ingestion.tasks.get_embedding_provider", return_value=fake_provider):
            from .tasks import embed_product
            embed_product(self.product.id)

            self.product.title = "Trailhead Insulated Snow Boots"
            self.product.save(update_fields=["title"])

            result = embed_product(self.product.id)

        self.assertEqual(result, "embedded")
        self.assertEqual(fake_provider.call_count, 2)

    def test_missing_product_returns_missing(self):
        from .tasks import embed_product
        result = embed_product(999999)
        self.assertEqual(result, "missing")


class EmbedAllProductsTaskTests(TestCase):
    def setUp(self):
        for i in range(3):
            Product.objects.create(
                external_id=f"bulk-{i:04d}", title=f"Product {i}", description="D",
                category="Footwear", price=Decimal("100.00"),
            )

    def test_enqueues_one_task_per_product(self):
        with patch("ingestion.tasks.embed_product.delay") as mock_delay:
            from .tasks import embed_all_products
            count = embed_all_products()

        self.assertEqual(count, 3)
        self.assertEqual(mock_delay.call_count, 3)
        enqueued_ids = {call.args[0] for call in mock_delay.call_args_list}
        self.assertEqual(enqueued_ids, set(Product.objects.values_list("id", flat=True)))


class GetEmbeddingProviderTests(TestCase):
    @override_settings(EMBEDDING_PROVIDER="openai")
    def test_defaults_to_openai_provider(self):
        from .providers.openai_provider import OpenAIEmbeddingProvider
        provider = get_embedding_provider()
        self.assertIsInstance(provider, OpenAIEmbeddingProvider)

    @override_settings(EMBEDDING_PROVIDER="not-a-real-provider")
    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            get_embedding_provider()


class OpenAIEmbeddingProviderTests(TestCase):
    def test_embed_returns_vectors_in_order(self):
        from .providers.openai_provider import OpenAIEmbeddingProvider

        class FakeResponseItem:
            def __init__(self, embedding):
                self.embedding = embedding

        class FakeResponse:
            data = [FakeResponseItem([0.1, 0.2]), FakeResponseItem([0.3, 0.4])]

        class FakeEmbeddings:
            def create(self, model, input):
                self.called_with = (model, input)
                return FakeResponse()

        class FakeClient:
            def __init__(self):
                self.embeddings = FakeEmbeddings()

        fake_client = FakeClient()
        provider = OpenAIEmbeddingProvider(client=fake_client)

        result = provider.embed(["a", "b"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(fake_client.embeddings.called_with, (provider.model, ["a", "b"]))
