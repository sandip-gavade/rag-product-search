from decimal import Decimal

from django.core.management import call_command
from django.test import TestCase

from catalog.models import Product
from catalog.products_data import generate_products


class GenerateProductsTests(TestCase):
    """Pure function tests — no DB needed, but grouped with the app's tests."""

    def test_generates_requested_count(self):
        products = generate_products(count=500, seed=42)
        self.assertEqual(len(products), 500)

    def test_external_ids_are_unique(self):
        products = generate_products(count=500, seed=42)
        external_ids = [p["external_id"] for p in products]
        self.assertEqual(len(external_ids), len(set(external_ids)))

    def test_deterministic_for_same_seed(self):
        first = generate_products(count=100, seed=7)
        second = generate_products(count=100, seed=7)
        self.assertEqual(first, second)

    def test_required_fields_present(self):
        for product in generate_products(count=50, seed=1):
            self.assertTrue(product["title"])
            self.assertTrue(product["description"])
            self.assertTrue(product["category"])
            self.assertIn("brand", product["attributes"])
            self.assertGreater(float(product["price"]), 0)


class ProductModelTests(TestCase):
    def test_create_product(self):
        product = Product.objects.create(
            external_id="test-0001",
            title="Test Waterproof Hiking Boots",
            description="A durable boot for testing.",
            category="Footwear",
            price=Decimal("2999.00"),
            attributes={"brand": "TestBrand", "color": "Black"},
        )
        self.assertIsNone(product.embedding)
        self.assertIsNone(product.search_vector)
        self.assertEqual(Product.objects.count(), 1)

    def test_external_id_is_unique(self):
        Product.objects.create(
            external_id="dup-0001", title="A", description="A",
            category="Footwear", price=Decimal("100.00"),
        )
        with self.assertRaises(Exception):
            Product.objects.create(
                external_id="dup-0001", title="B", description="B",
                category="Footwear", price=Decimal("200.00"),
            )


class SeedCatalogCommandTests(TestCase):
    def test_seed_creates_products(self):
        call_command("seed_catalog", count=50)
        self.assertEqual(Product.objects.count(), 50)

    def test_seed_is_idempotent(self):
        call_command("seed_catalog", count=50)
        first_ids = set(Product.objects.values_list("external_id", flat=True))

        call_command("seed_catalog", count=50)

        self.assertEqual(Product.objects.count(), 50)
        second_ids = set(Product.objects.values_list("external_id", flat=True))
        self.assertEqual(first_ids, second_ids)
