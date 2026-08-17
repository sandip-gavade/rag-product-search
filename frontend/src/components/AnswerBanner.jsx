export default function AnswerBanner({ answerText, isStreaming, error, ungroundedCitations }) {
  if (error) {
    return (
      <div className="answer-banner answer-error" role="alert">
        {error}
      </div>
    );
  }

  if (!answerText && !isStreaming) return null;

  return (
    <div className="answer-banner">
      <p>
        {answerText}
        {isStreaming && <span className="cursor" aria-hidden="true" />}
      </p>
      {ungroundedCitations?.length > 0 && (
        // Hallucination guard (Phase 5) caught something — surfaced
        // rather than silently hidden, since this is a portfolio project
        // demonstrating the guard actually works.
        <p className="answer-warning">
          Note: the answer referenced {ungroundedCitations.length === 1 ? "a product" : "products"} not
          in the search results ({ungroundedCitations.join(", ")}) — flagged by the grounding check.
        </p>
      )}
    </div>
  );
}
