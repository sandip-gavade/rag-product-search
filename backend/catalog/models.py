from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.db import models

from pgvector.django import HnswIndex, VectorField

# text-embedding-3-small produces 1536-dimensional vectors. If the
# embedding provider is later swapped (see ingestion/providers/), this
# constant — and a migration to resize the column — is the single place
# that needs to change.
EMBEDDING_DIMENSIONS = 1536


class Product(models.Model):
    # Stable identifier for the source record (independent of the DB's
    # auto-incrementing pk), so `seed_catalog` can upsert on re-run instead
    # of inserting duplicates on every execution.
    external_id = models.CharField(max_length=64, unique=True)

    title = models.CharField(max_length=255)
    description = models.TextField()
    category = models.CharField(max_length=100, db_index=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)

    # Free-form product attributes (color, brand, material, size, ...).
    # Kept as JSON rather than separate columns since attribute shape
    # varies by category and this catalog isn't queried by individual
    # attribute columns directly — the LLM query-understanding step
    # (Phase 4) reads/filters on this instead of a rigid schema.
    attributes = models.JSONField(default=dict, blank=True)

    # Populated by the Phase 2 ingestion task, not at seed time — a fresh
    # seed leaves this null until the embedding worker runs, which is what
    # makes ingestion a distinct, re-runnable step rather than part of seeding.
    embedding = VectorField(dimensions=EMBEDDING_DIMENSIONS, null=True, blank=True)

    # Postgres tsvector for keyword/full-text search, also populated by
    # the Phase 2 ingestion task (kept in sync with title/description).
    search_vector = SearchVectorField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            GinIndex(fields=['search_vector'], name='product_search_vector_gin'),
            # cosine distance is what we'll query with (`<=>`), so the HNSW
            # index is built on the matching opclass. HNSW over IVFFlat
            # because it doesn't need a training pass on existing data —
            # relevant here since embeddings are backfilled asynchronously
            # after rows already exist.
            HnswIndex(
                name='product_embedding_hnsw',
                fields=['embedding'],
                m=16,
                ef_construction=64,
                opclasses=['vector_cosine_ops'],
            ),
        ]

    def __str__(self):
        return self.title
