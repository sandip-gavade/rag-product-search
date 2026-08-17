from django.core.management.base import BaseCommand

from catalog.models import Product
from ingestion.tasks import embed_product


class Command(BaseCommand):
    """Embed every product in the catalog. Idempotent — safe to re-run
    (embed_product skips unchanged products, see ingestion/tasks.py).

    Runs inline by default (no Celery worker needed — useful right after
    `docker-compose up` on a fresh clone). Pass --async to enqueue via
    Celery instead, once a worker is running.
    """

    help = "Embed all products (inline by default; --async to enqueue via Celery)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--async", action="store_true", dest="run_async",
            help="Enqueue via Celery instead of running inline.",
        )

    def handle(self, *args, **options):
        product_ids = list(Product.objects.values_list("id", flat=True))

        if options["run_async"]:
            for product_id in product_ids:
                embed_product.delay(product_id)
            self.stdout.write(self.style.SUCCESS(
                f"Enqueued {len(product_ids)} embedding tasks via Celery."
            ))
            return

        counts = {"embedded": 0, "skipped": 0, "missing": 0}
        for product_id in product_ids:
            result = embed_product(product_id)
            counts[result] = counts.get(result, 0) + 1

        self.stdout.write(self.style.SUCCESS(
            f"Done: {counts['embedded']} embedded, {counts['skipped']} unchanged "
            f"(skipped), {counts['missing']} missing."
        ))
