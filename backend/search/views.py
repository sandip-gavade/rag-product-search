from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from query_understanding.chain import understand_query

from .retrieval import hybrid_search
from .serializers import SearchResultSerializer


class SearchView(APIView):
    """GET /api/search/?q=<query>&top_k=<int>

    Query understanding (LangChain + Claude/Ollama) parses the raw query
    into structured filters and a cleaned semantic query, then hybrid
    (vector + keyword) retrieval runs the semantic query within those
    filters, fused with RRF. See query_understanding/chain.py for the
    parse step and its fallback, and fusion.py for why RRF was chosen
    over a raw weighted-score merge.
    """

    def get(self, request):
        query = request.query_params.get("q", "").strip()
        if not query:
            return Response(
                {"detail": "Query parameter 'q' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            top_k = int(request.query_params.get("top_k", 10))
        except ValueError:
            return Response(
                {"detail": "'top_k' must be an integer."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # understand_query never raises — an LLM failure degrades to plain
        # semantic search (see chain.py) rather than blocking the request.
        parsed = understand_query(query)

        try:
            results = hybrid_search(
                parsed.semantic_query,
                top_k=top_k,
                category=parsed.category,
                price_min=parsed.price_min,
                price_max=parsed.price_max,
            )
        except Exception:
            # The one external call that *can't* degrade gracefully — the
            # query embedding is required for vector search. No API key
            # configured, rate limit, network blip: surface a clean 502
            # instead of a raw 500/stack trace.
            return Response(
                {"detail": "Search is temporarily unavailable (embedding provider error)."},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        serializer = SearchResultSerializer(results, many=True)
        return Response({
            "query": query,
            "parsed": {
                "semantic_query": parsed.semantic_query,
                "category": parsed.category,
                "price_min": parsed.price_min,
                "price_max": parsed.price_max,
                "attributes": parsed.attributes,
            },
            "results": serializer.data,
        })
