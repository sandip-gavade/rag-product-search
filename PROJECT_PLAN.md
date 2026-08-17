# PROJECT_PLAN.md — RAG-Powered Product Catalog Search

Portfolio project: AI-powered intelligent search over a product catalog using
Retrieval-Augmented Generation (RAG) with hybrid (vector + keyword) search.

## Tech Stack (locked — ask before substituting)

| Layer | Choice |
|---|---|
| Backend | Django + Django REST Framework |
| Database | PostgreSQL + `pgvector` (vectors) + native `tsvector` (keyword/full-text) |
| Async/ingestion | Celery + Redis |
| Embeddings | OpenAI `text-embedding-3-small`, behind a swappable provider interface (future: local `sentence-transformers`) |
| Orchestration | LangChain (query understanding + RAG chain) |
| LLM | Anthropic Claude API (default) — structured query parsing + answer synthesis, behind a swappable `LLMProvider` interface with an Ollama/LM Studio (Qwen3) implementation for local dev, switchable via `LLM_PROVIDER` env var |
| Frontend | React (Vite), plain fetch/axios, no UI framework |
| Containerization | Docker Compose (postgres+pgvector, redis, django, react) |
| Deploy target | Render or Railway (free tier) |

## Working agreements

- Implement one phase at a time. Do not start phase *N+1* until phase *N*'s
  "done when" checklist is met and reviewed.
- One commit per phase (small, focused — not one giant commit).
- AI/RAG-specific code (chunking, embedding dimensionality, reranking, prompt
  construction) gets **comments explaining why**, since this is the layer
  being learned. Standard Django/React code follows normal conventions
  (comments only for non-obvious *why*).
- No secrets in code. All config via `.env`, with `.env.example` kept in sync
  and committed.
- Basic tests for retrieval and ingestion logic at minimum, added in the
  phase that introduces the logic (not deferred to the end).

---

## Phase 1 — Data & Models

**Goal:** A realistic product catalog persisted in Postgres, with the columns
needed for hybrid search already in place.

- Source or synthesize ~500 realistic products (title, description, category,
  price, attributes/brand/tags). Prefer an existing public dataset
  (Amazon/Flipkart-style) if a suitably licensed one is easy to get; otherwise
  synthesize with a script.
- Django app `catalog` with a `Product` model:
  - Core fields: title, description, category, price, attributes (JSONField).
  - `embedding` column (`pgvector.django.VectorField`, dim=1536 for
    `text-embedding-3-small`).
  - `search_vector` column (Postgres `tsvector`, via `django.contrib.postgres.search.SearchVectorField`).
  - Indexes: ivfflat/hnsw on `embedding`, GIN on `search_vector`.
- `manage.py seed_catalog` management command to load/import the dataset
  (idempotent — safe to re-run, upserts by a stable external ID).

**Done when:**
- [x] `docker compose up db` gives a Postgres instance with `pgvector` enabled.
- [x] `python manage.py migrate` creates the `Product` table with embedding +
      tsvector columns and indexes.
- [x] `python manage.py seed_catalog` loads ~500 products, re-running it does
      not duplicate rows.
- [x] `python manage.py test catalog` passes (model-level tests: fields,
      constraints, seed idempotency).

## Phase 2 — Ingestion Pipeline

**Goal:** Every product gets an embedding and an up-to-date `tsvector`,
computed asynchronously and safely re-runnable.

- Embedding provider interface (`EmbeddingProvider` ABC/protocol) with an
  `OpenAIEmbeddingProvider` implementation — swappable later for a local model.
- Chunking strategy for product text (likely: title + description + key
  attributes concatenated into one chunk per product, since catalog entries
  are short — comment explaining why multi-chunk-per-doc isn't needed here
  unless descriptions turn out to be long).
- Celery task `embed_product(product_id)`:
  - Builds the text to embed, calls the provider, stores the vector.
  - Updates `search_vector` via Postgres `SearchVector(...)`.
  - Idempotent: skips re-embedding if content hash unchanged (store a
    `content_hash` field to detect this).
- Bulk task to enqueue embedding for all products missing/stale embeddings.

**Done when:**
- [x] `celery -A config worker` running + triggering the bulk task embeds all
      seeded products. *(Verified with a dry run against all 500 seeded
      products: worker started, connected to Redis, `embed_all_products`
      fanned out 500 `embed_product` tasks, each correctly reached the
      OpenAI call and failed there with 401 — no `OPENAI_API_KEY` configured
      yet. Re-run with a real key once added to `.env`; no code changes
      needed.)*
- [x] Re-running the bulk task is a no-op for unchanged products (verified by
      a test asserting no duplicate API calls / no vector churn).
- [x] `search_vector` is populated and queryable via `SearchQuery`.
- [x] Tests cover: chunking/text-building function, idempotency check,
      provider interface with a mocked provider (no real API calls in tests).

## Phase 3 — Hybrid Retrieval Endpoint

**Goal:** A DRF endpoint that takes a raw query string and returns top-k
products ranked by a fused score of vector similarity + keyword relevance.

- `POST /api/search/` (or `GET` with `q` param) accepting a raw query.
- Runs both:
  - Cosine similarity via pgvector (`<=>` operator) against the query
    embedding.
  - Full-text search via `SearchQuery`/`SearchRank` against `search_vector`.
- Merge/rerank: normalize both scores and combine (e.g. weighted sum or
  Reciprocal Rank Fusion — decide and document the tradeoff in code comments:
  RRF is more robust to score-scale differences between vector and keyword
  search, which is why it's likely preferred over a naive weighted sum).
- Returns top-k serialized products with score breakdown (vector score,
  keyword score, fused score) for transparency/debuggability.

**Done when:**
- [x] Endpoint returns ranked results for a raw query with no filters.
      *(GET `/api/search/?q=...`, live-tested against the running dev
      server: missing `q` → 400; a real query with no `OPENAI_API_KEY`
      configured yet → clean 502, not a crash — matches Phase 2's status.)*
- [x] Reranking logic is unit-testable independent of the DB (pure function
      taking two ranked lists, returning fused list).
- [x] Comments explain the fusion method and why it was chosen over
      alternatives.
- [x] Tests: fusion function correctness (known inputs → known ranking),
      endpoint smoke test against seeded data.

## Phase 4 — Query Understanding (LangChain)

**Goal:** Free-text queries get parsed into structured filters + a cleaned
semantic query before hitting Phase 3's retrieval.

- `LLMProvider` interface (mirrors the embeddings provider pattern from
  Phase 2) with `parse_query()` and `synthesize_answer()` methods.
  `AnthropicProvider` (Claude, default) and `OllamaProvider` (local Qwen3 via
  Ollama/LM Studio, `langchain-ollama`'s `ChatOllama`) implementations,
  selected via `LLM_PROVIDER` env var. Comment on why local models need
  stricter output validation/retry (Qwen3 tool-calling is decent but more
  prone to malformed JSON than Claude on edge cases).
- LangChain chain using the active `LLMProvider` with structured/tool-call
  output to extract:
  - `price_min`, `price_max`, `category`, `attributes` (e.g. color, brand),
    and a `semantic_query` string (the cleaned residual text for embedding).
- Wire this in front of the Phase 3 endpoint: parse → apply filters as SQL
  `WHERE` clauses → run hybrid retrieval on `semantic_query` within the
  filtered set.
- Graceful fallback: if parsing fails or returns nothing usable, fall back to
  treating the whole input as the semantic query.

**Done when:**
- [ ] Given `"waterproof hiking boots under ₹3000"`, the chain extracts
      `price_max=3000`, `category≈footwear`, `semantic_query≈"waterproof
      hiking boots"`.
- [ ] Extracted filters are applied as real SQL filters (verified with a test
      query that only matches when the filter is respected).
- [ ] Fallback path is tested (malformed/ambiguous query still returns
      results).
- [ ] Chain works with `LLM_PROVIDER=anthropic` and `LLM_PROVIDER=ollama`
      (local Qwen3 pulled and running) — same test suite passes against both,
      or Ollama-specific flakiness is documented if structured output proves
      unreliable at the model size available locally.

## Phase 5 — RAG Answer Synthesis

**Goal:** A short natural-language recommendation referencing only the
actual retrieved products.

- Take Phase 3/4's top-k results, build a grounded prompt (product
  names/IDs/prices only from the retrieved set — comment on why we pass
  structured product data rather than letting the model free-associate, to
  prevent hallucinated product names).
- Call the active `LLMProvider` (`synthesize_answer()`) to generate a short
  recommendation paragraph.
- Post-check: verify every product name/ID mentioned in the LLM output
  exists in the retrieved set; log/flag if not (cheap hallucination guard).
- Stream the response if straightforward with the chosen endpoint framework
  (Django async view or SSE); otherwise return synchronously and note why.

**Done when:**
- [ ] Endpoint returns a natural-language answer that mentions ≥1 actual
      retrieved product by name.
- [ ] Hallucination guard test: fabricate a mock LLM response naming a
      product not in the retrieved set, assert it's flagged.
- [ ] Streaming (or documented sync fallback) works end-to-end.

## Phase 6 — React Frontend

**Goal:** Minimal, clean UI to demo the whole flow.

- Search bar → calls Phase 4/5 combined endpoint.
- Filter chips reflecting the parsed structured filters (editable/removable).
- Product result cards (title, price, category, image placeholder if no
  images in dataset).
- Streamed natural-language answer displayed at the top, above results.
- No UI framework — plain CSS, functional components, hooks only.

**Done when:**
- [ ] `npm run dev` gives a working search UI against the local backend.
- [ ] Typing a query shows filter chips, ranked product cards, and the
      synthesized answer.
- [ ] Basic loading/error states handled.

## Phase 7 — Evaluation

**Goal:** Quantify retrieval quality and latency.

- 20–30 hand-written test queries with expected relevant product IDs
  (`eval/queries.json` or similar).
- Script `eval/run_eval.py` that runs each query against the retrieval
  endpoint, computes precision@k (k=5 and k=10) and measures latency.
- Output a markdown table (query, precision@5, precision@10, latency) plus
  aggregate averages — formatted to paste directly into the README.

**Done when:**
- [ ] `python eval/run_eval.py` runs all queries and prints/saves the
      markdown table.
- [ ] Aggregate precision@k and average latency are reported.

## Phase 8 — Docker & Deploy Prep

**Goal:** One-command local bring-up, and a documented path to a free-tier
deploy.

- `docker-compose.yml`: postgres (pgvector image), redis, django (+celery
  worker), react.
- `.env.example` covering every required env var (API keys, DB creds, Redis
  URL, Django secret key, `LLM_PROVIDER`, `OLLAMA_BASE_URL`/model name for
  the local-LLM path).
- Deploy config for Render or Railway (whichever fits the free tier better at
  build time — decide and note why in the README) — build/start commands,
  env var list, migration step on release.

**Done when:**
- [ ] `docker-compose up` from a clean clone (with `.env` filled in) gives a
      fully working app — seed data, ingestion, search, frontend.
- [ ] Deploy config is present and documented, even if not actually deployed
      live (deploying is optional/user's call).

## Phase 9 — README

**Goal:** A portfolio-quality README.

- Architecture diagram (mermaid).
- Setup instructions (local Docker Compose path, and manual path).
- "Why hybrid search over pure vector search" explanation.
- Phase 7 eval results table.
- Screenshots/GIF of the UI if easy to capture.

**Done when:**
- [ ] README covers architecture, setup, design rationale, and eval results.
- [ ] A stranger could clone the repo and get it running from the README
      alone.

---

## Decisions locked in

1. **Dataset:** try a public dataset (e.g. Kaggle Amazon/Flipkart-style
   listings) first; fall back to synthesizing ~500 products if licensing/
   quality is an issue.
2. **Fusion (Phase 3):** Reciprocal Rank Fusion.
3. **LLM provider:** swappable `LLMProvider` interface, default Claude API,
   with an Ollama/LM Studio (Qwen3) implementation for local dev — see Phase
   4/5 and the tech stack table above.

Ready to start Phase 1 on your go-ahead.
