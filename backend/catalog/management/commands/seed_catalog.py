from django.core.management.base import BaseCommand

from catalog.models import Product
from catalog.products_data import generate_products


class Command(BaseCommand):
    help = "Seed (or refresh) the product catalog with generated products. Safe to re-run."

    def add_arguments(self, parser):
        parser.add_argument(
            "--count", type=int, default=500,
            help="Number of products to generate (default: 500).",
        )

    def handle(self, *args, **options):
        products = generate_products(count=options["count"])

        created, updated = 0, 0
        for data in products:
            # update_or_create on the stable external_id (not the DB pk) is
            # what makes this idempotent: re-running with the same seed
            # regenerates the same external_ids, so rows get refreshed in
            # place instead of duplicated. embedding/search_vector are
            # deliberately left untouched here — the Phase 2 ingestion task
            # owns those.
            _, was_created = Product.objects.update_or_create(
                external_id=data["external_id"],
                defaults={
                    "title": data["title"],
                    "description": data["description"],
                    "category": data["category"],
                    "price": data["price"],
                    "attributes": data["attributes"],
                },
            )
            created += was_created
            updated += not was_created

        self.stdout.write(self.style.SUCCESS(
            f"Seeded {len(products)} products ({created} created, {updated} updated)."
        ))
