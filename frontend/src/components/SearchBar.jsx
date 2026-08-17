import { useState } from "react";

export default function SearchBar({ onSearch, disabled }) {
  const [value, setValue] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    const trimmed = value.trim();
    if (trimmed) onSearch(trimmed);
  }

  return (
    <form className="search-bar" onSubmit={handleSubmit}>
      <input
        type="text"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder='Try "waterproof hiking boots under ₹3000"'
        disabled={disabled}
        aria-label="Search products"
      />
      <button type="submit" disabled={disabled || !value.trim()}>
        Search
      </button>
    </form>
  );
}
