// Talks to the Phase 5 RAG endpoint (query understanding + hybrid
// retrieval + grounded answer synthesis, all in one streamed call).
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * Streams a search via Server-Sent Events and returns the EventSource so
 * the caller can close it (new search started, component unmounted).
 *
 * Native EventSource only supports GET with no custom headers/body, which
 * is fine here — everything the endpoint needs travels as query params.
 *
 * `filters`/`refine` implement filter-chip removal: when refine is true,
 * the backend skips another LLM call and reuses `query` verbatim as the
 * semantic query, applying `filters` directly as SQL WHERE clauses — see
 * rag/views.py's refine mode.
 */
export function streamSearch({
  query,
  filters,
  refine = false,
  onFilters,
  onProducts,
  onToken,
  onComplete,
  onSynthesisError,
  onConnectionError,
}) {
  const params = new URLSearchParams({ q: query });
  if (refine) {
    params.set("refine", "true");
    params.set("category", filters?.category ?? "");
    params.set("price_min", filters?.price_min ?? "");
    params.set("price_max", filters?.price_max ?? "");
  }

  const source = new EventSource(`${API_BASE}/api/rag/answer/?${params.toString()}`);

  source.addEventListener("filters", (event) => onFilters?.(JSON.parse(event.data)));
  source.addEventListener("products", (event) => onProducts?.(JSON.parse(event.data).products));
  source.addEventListener("token", (event) => onToken?.(JSON.parse(event.data).text));
  source.addEventListener("answer_complete", (event) => {
    onComplete?.(JSON.parse(event.data));
    source.close();
  });
  // Business-level failure the backend reported explicitly (e.g. the LLM
  // call itself failed) — distinct from the browser's own "error" event
  // below, which fires for network/connection-level failures and never
  // carries JSON in event.data.
  source.addEventListener("synthesis_error", (event) => {
    onSynthesisError?.(JSON.parse(event.data));
    source.close();
  });
  source.addEventListener("error", () => {
    onConnectionError?.();
    source.close();
  });

  return source;
}
