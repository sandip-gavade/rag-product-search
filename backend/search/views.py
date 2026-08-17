from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .retrieval import hybrid_search
from .serializers import SearchResultSerializer


class SearchView(APIView):
    """GET /api/search/?q=<query>&top_k=<int>

    Hybrid (vector + keyword) retrieval over the catalog, fused with RRF.
    See fusion.py for why RRF was chosen over a raw weighted-score merge.
    Query-understanding (structured filters, LLM-cleaned semantic query)
    is layered in front of this in Phase 4 — this endpoint takes the raw
    query text as-is.
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

        try:
            results = hybrid_search(query, top_k=top_k)
        except Exception:
            # The one external network call in this request (the query
            # embedding) can fail for reasons outside our control — no API
            # key configured, rate limit, network blip. Surface that as a
            # clean 502 instead of a raw 500/stack trace.
            return Response(
                {"detail": "Search is temporarily unavailable (embedding provider error)."},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        serializer = SearchResultSerializer(results, many=True)
        return Response({"query": query, "results": serializer.data})
