// No product images in the synthesized dataset (Phase 1) — a category
// initial in a colored tile stands in, cheaper than sourcing/hosting
// placeholder images for a portfolio dataset that's generated, not real.
export default function ProductCard({ product, cited }) {
  return (
    <article className={`product-card${cited ? " cited" : ""}`}>
      <div className="product-image-placeholder" aria-hidden="true">
        {product.category.charAt(0)}
      </div>
      <div className="product-body">
        <h3>{product.title}</h3>
        <p className="product-meta">
          {product.category} · ₹{product.price}
        </p>
        {cited && <span className="cited-badge">Mentioned in answer</span>}
      </div>
    </article>
  );
}
