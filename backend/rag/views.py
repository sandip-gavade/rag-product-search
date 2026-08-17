import json

from django.http import JsonResponse, StreamingHttpResponse
from django.views import View

from catalog.models import Product
from query_understanding.chain import understand_query
from query_understanding.providers import get_llm_provider
from search.retrieval import hybrid_search

from .synth import build_grounding_prompt, find_ungrounded_citations


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _serialize_product(product: Product) -> dict:
    return {
        "id": product.id,
        "external_id": product.external_id,
        "title": product.title,
        "category": product.category,
        "price": str(product.price),
    }


class RAGAnswerView(View):
    """GET /api/rag/answer/?q=<query>&top_k=<int>

    The full pipeline in one call: Phase 4 query understanding → Phase 3
    hybrid retrieval → Phase 5 grounded answer synthesis — streamed as
    Server-Sent Events, since token-by-token delivery is straightforward
    here (every active LLMProvider exposes stream_answer() via LangChain's
    .stream()) and meaningfully improves perceived latency over waiting
    for the full paragraph.

    A plain Django View, not DRF's APIView: DRF performs content
    negotiation against the request's Accept header before the view runs,
    and the browser's EventSource sends `Accept: text/event-stream` —
    which no DRF renderer matches, so APIView 406s before this code would
    even execute. Nothing here needs DRF's serialization anyway (SSE
    frames are hand-built, error responses are plain JSON), so a plain
    View sidesteps the mismatch entirely instead of fighting it.

    Event sequence: "filters" (the active filters, once) → "products" (the
    retrieved set, once) → "token" (repeated, one per streamed chunk) →
    "answer_complete" (final hallucination-guard result) — or
    "synthesis_error" in place of the token stream if synthesis fails.
    (Named "synthesis_error", not "error" — "error" is a reserved event
    type on the browser's EventSource; a server-sent `event: error` is
    indistinguishable from a genuine connection failure there, so a
    distinct name is required for the frontend to tell them apart.)

    Filter-chip refinement: pass `refine=true` with `category`/`price_min`/
    `price_max` (any can be blank to mean "no filter") to re-run retrieval
    against the *same* semantic query with explicit filters, skipping
    another LLM call entirely — this is what the frontend does when a user
    removes a filter chip, so removing a chip is instant and free rather
    than round-tripping through query understanding again.
    """

    def get(self, request):
        query = request.GET.get("q", "").strip()
        if not query:
            return JsonResponse({"detail": "Query parameter 'q' is required."}, status=400)

        try:
            top_k = int(request.GET.get("top_k", 5))
        except ValueError:
            return JsonResponse({"detail": "'top_k' must be an integer."}, status=400)

        if request.GET.get("refine") == "true":
            semantic_query = query
            category = request.GET.get("category") or None
            price_min = self._parse_optional_float(request.GET.get("price_min"))
            price_max = self._parse_optional_float(request.GET.get("price_max"))
        else:
            parsed = understand_query(query)
            semantic_query = parsed.semantic_query
            category = parsed.category
            price_min = parsed.price_min
            price_max = parsed.price_max

        try:
            results = hybrid_search(
                semantic_query, top_k=top_k,
                category=category, price_min=price_min, price_max=price_max,
            )
        except Exception:
            return JsonResponse(
                {"detail": "Search is temporarily unavailable (embedding provider error)."},
                status=502,
            )

        products = [r["product"] for r in results]
        active_filters = {
            "category": category, "price_min": price_min, "price_max": price_max,
            # The cleaned semantic query, not the raw NL query — refine
            # requests (filter-chip removal) resend this as `q` verbatim,
            # so price/category phrases already stripped out by query
            # understanding don't leak back into the semantic search text.
            "semantic_query": semantic_query,
        }
        response = StreamingHttpResponse(
            self._event_stream(semantic_query, products, active_filters), content_type="text/event-stream",
        )
        # Disable buffering (nginx/proxy default) so tokens actually arrive
        # incrementally instead of all at once when the stream closes.
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response

    @staticmethod
    def _parse_optional_float(value):
        return float(value) if value else None

    def _event_stream(self, query: str, products: list[Product], active_filters: dict):
        yield _sse("filters", active_filters)
        yield _sse("products", {"products": [_serialize_product(p) for p in products]})

        if not products:
            yield _sse("answer_complete", {
                "answer": "No matching products were found for this search.",
                "ungrounded_citations": [],
            })
            return

        prompt = build_grounding_prompt(query, products)

        try:
            provider = get_llm_provider()
            full_answer = ""
            for chunk in provider.stream_answer(prompt):
                full_answer += chunk
                yield _sse("token", {"text": chunk})
        except Exception:
            yield _sse("synthesis_error", {"detail": "Answer synthesis is temporarily unavailable."})
            return

        ungrounded = find_ungrounded_citations(full_answer, products)
        yield _sse("answer_complete", {"answer": full_answer, "ungrounded_citations": ungrounded})
