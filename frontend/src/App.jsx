import { useEffect, useRef, useState } from "react";

import { streamSearch } from "./api";
import AnswerBanner from "./components/AnswerBanner";
import FilterChips from "./components/FilterChips";
import ProductCard from "./components/ProductCard";
import SearchBar from "./components/SearchBar";

const CITATION_PATTERN = /\[([a-zA-Z0-9-]+)\]/g;

function citedExternalIds(answerText) {
  return new Set([...answerText.matchAll(CITATION_PATTERN)].map((match) => match[1]));
}

export default function App() {
  const [hasSearched, setHasSearched] = useState(false);
  const [filters, setFilters] = useState(null);
  const [products, setProducts] = useState(null);
  const [answerText, setAnswerText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [ungroundedCitations, setUngroundedCitations] = useState([]);
  const [error, setError] = useState(null);

  const sourceRef = useRef(null);

  useEffect(() => () => sourceRef.current?.close(), []);

  // Shared by a fresh search (refine: false, raw query text) and a
  // filter-chip removal (refine: true, cleaned semantic_query + the
  // remaining filters) — the only difference is what's passed to
  // streamSearch(); the state reset and callback wiring are identical.
  function runSearch({ query, refine, filtersForRequest }) {
    sourceRef.current?.close();

    setHasSearched(true);
    setProducts(null);
    setAnswerText("");
    setUngroundedCitations([]);
    setError(null);
    setIsStreaming(true);
    if (!refine) setFilters(null);

    sourceRef.current = streamSearch({
      query,
      filters: filtersForRequest,
      refine,
      onFilters: setFilters,
      onProducts: setProducts,
      onToken: (text) => setAnswerText((prev) => prev + text),
      onComplete: ({ answer, ungrounded_citations }) => {
        setAnswerText(answer);
        setUngroundedCitations(ungrounded_citations ?? []);
        setIsStreaming(false);
      },
      onSynthesisError: ({ detail }) => {
        setError(detail || "Answer synthesis is temporarily unavailable.");
        setIsStreaming(false);
      },
      onConnectionError: () => {
        setError("Search is temporarily unavailable. Please try again in a moment.");
        setIsStreaming(false);
        setProducts((prev) => prev ?? []);
      },
    });
  }

  function handleSearch(query) {
    runSearch({ query, refine: false, filtersForRequest: null });
  }

  function handleRemoveFilter(key) {
    if (!filters) return;
    const nextFilters = { ...filters, [key]: null };
    // Refine mode resends the cleaned semantic_query verbatim, not the
    // original free-text query — see rag/views.py's refine branch.
    runSearch({ query: nextFilters.semantic_query, refine: true, filtersForRequest: nextFilters });
  }

  const cited = answerText ? citedExternalIds(answerText) : new Set();
  const isLoadingProducts = hasSearched && products === null && !error;

  return (
    <div className="page">
      <header>
        <h1>Product Search</h1>
        <p className="subtitle">Hybrid vector + keyword search with an LLM-grounded recommendation</p>
      </header>

      <SearchBar onSearch={handleSearch} disabled={isStreaming || isLoadingProducts} />

      {hasSearched && (
        <main>
          <FilterChips filters={filters} onRemove={handleRemoveFilter} />

          <AnswerBanner
            answerText={answerText}
            isStreaming={isStreaming}
            error={error}
            ungroundedCitations={ungroundedCitations}
          />

          {isLoadingProducts && <p className="status">Searching…</p>}

          {products && products.length === 0 && !error && (
            <p className="status">No matching products found.</p>
          )}

          {products && products.length > 0 && (
            <div className="product-grid">
              {products.map((product) => (
                <ProductCard key={product.id} product={product} cited={cited.has(product.external_id)} />
              ))}
            </div>
          )}
        </main>
      )}
    </div>
  );
}
