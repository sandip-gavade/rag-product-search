from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.db import models

from pgvector.django import HnswIndex, VectorField

# Must match the ACTIVE embedding provider's output width (see
# ingestion/providers/) — the column is a fixed-width pgvector(N), and
# Postgres rejects an insert of the wrong length outright, so this can't
# vary per request. Set to 768 for the local, zero-cost default (LM
# Studio's nomic-embed-text-v1.5). Switching EMBEDDING_PROVIDER to
# something with a different output width (e.g. "openai" -> 1536, for
# text-embedding-3-small) needs three things done together: this constant
# updated, a migration to resize the column + rebuild the HNSW index, and
# ingest_catalog re-run to regenerate every product's embedding — vectors
# from two different models aren't comparable, so a partial re-embed would
# silently corrupt similarity search rather than error.
EMBEDDING_DIMENSIONS = 768


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

    # SHA-256 of the text last embedded (see ingestion.text.build_embedding_text).
    # Lets the ingestion task detect "nothing changed" and skip re-embedding
    # on re-runs without diffing every field individually.
    content_hash = models.CharField(max_length=64, blank=True, default="")

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
