// Displays the filters query understanding (Phase 4) extracted from the
// raw search text. Removing a chip re-runs the search in "refine" mode —
// same semantic query, that one filter cleared, no extra LLM call.
export default function FilterChips({ filters, onRemove }) {
  if (!filters) return null;

  const chips = [];
  if (filters.category) {
    chips.push({ key: "category", label: filters.category });
  }
  if (filters.price_min != null) {
    chips.push({ key: "price_min", label: `Min ₹${filters.price_min}` });
  }
  if (filters.price_max != null) {
    chips.push({ key: "price_max", label: `Max ₹${filters.price_max}` });
  }

  if (chips.length === 0) return null;

  return (
    <div className="filter-chips">
      {chips.map((chip) => (
        <span className="chip" key={chip.key}>
          {chip.label}
          <button
            type="button"
            className="chip-remove"
            onClick={() => onRemove(chip.key)}
            aria-label={`Remove ${chip.label} filter`}
          >
            ×
          </button>
        </span>
      ))}
    </div>
  );
}
