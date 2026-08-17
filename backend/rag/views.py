import json

from django.http import StreamingHttpResponse
from rest_framework.response import Response
from rest_framework.views import APIView

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


class RAGAnswerView(APIView):
    """GET /api/rag/answer/?q=<query>&top_k=<int>

    The full pipeline in one call: Phase 4 query understanding → Phase 3
    hybrid retrieval → Phase 5 grounded answer synthesis — streamed as
    Server-Sent Events, since token-by-token delivery is straightforward
    here (every active LLMProvider exposes stream_answer() via LangChain's
    .stream()) and meaningfully improves perceived latency over waiting
    for the full paragraph.

    Event sequence: "products" (the retrieved set, once) → "token"
    (repeated, one per streamed chunk) → "done" (final hallucination-guard
    result) — or "error" in place of the token stream if synthesis fails.
    """

    def get(self, request):
        query = request.query_params.get("q", "").strip()
        if not query:
            return Response({"detail": "Query parameter 'q' is required."}, status=400)

        try:
            top_k = int(request.query_params.get("top_k", 5))
        except ValueError:
            return Response({"detail": "'top_k' must be an integer."}, status=400)

        parsed = understand_query(query)

        try:
            results = hybrid_search(
                parsed.semantic_query, top_k=top_k,
                category=parsed.category, price_min=parsed.price_min, price_max=parsed.price_max,
            )
        except Exception:
            return Response(
                {"detail": "Search is temporarily unavailable (embedding provider error)."},
                status=502,
            )

        products = [r["product"] for r in results]
        response = StreamingHttpResponse(
            self._event_stream(query, products), content_type="text/event-stream",
        )
        # Disable buffering (nginx/proxy default) so tokens actually arrive
        # incrementally instead of all at once when the stream closes.
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response

    def _event_stream(self, query: str, products: list[Product]):
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
            yield _sse("error", {"detail": "Answer synthesis is temporarily unavailable."})
            return

        ungrounded = find_ungrounded_citations(full_answer, products)
        yield _sse("answer_complete", {"answer": full_answer, "ungrounded_citations": ungrounded})
