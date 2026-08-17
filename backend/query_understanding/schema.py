from typing import Optional

from pydantic import BaseModel, Field


class ParsedQuery(BaseModel):
    """Structured filters + cleaned semantic query extracted from free text.

    Bound to the LLM as a tool-call schema (see providers/) so extraction
    uses the model's native structured-output support instead of parsing
    free-form text.
    """

    price_min: Optional[float] = Field(
        default=None, description="Minimum price mentioned in the query, or null if none."
    )
    price_max: Optional[float] = Field(
        default=None,
        description="Maximum price mentioned in the query (e.g. 'under 3000' -> 3000), or null if none.",
    )
    category: Optional[str] = Field(
        default=None, description="Best-matching product category from the provided list, or null if unclear."
    )
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Extracted attributes such as color, brand, or material as key-value pairs.",
    )
    semantic_query: str = Field(
        description="The residual descriptive text with price/category filters removed, "
        "used for semantic similarity search (e.g. 'waterproof hiking boots')."
    )
