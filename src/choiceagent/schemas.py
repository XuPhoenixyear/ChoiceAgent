"""Validated public data contracts for catalog, retrieval, and application requests."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Reject unexpected inputs and non-finite numbers at trust boundaries."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class Product(StrictModel):
    """A source-backed parent product from a frozen metadata snapshot."""

    item_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=2000)
    category: str = Field(default="Office Products", max_length=300)
    category_path: list[str] = Field(default_factory=list, max_length=30)
    brand: str | None = Field(default=None, max_length=500)
    price_usd: float | None = Field(default=None, ge=0, le=1_000_000)
    features: list[str] = Field(default_factory=list, max_length=100)
    description: str = Field(default="", max_length=30000)
    attributes: dict[str, str] = Field(default_factory=dict)
    source_url: str = "https://amazon-reviews-2023.github.io/"
    source_locator: str = ""
    catalog_version: str = "office_mvp_v1"


class Constraints(StrictModel):
    """Program-enforced constraints; an unknown price never passes a price filter."""

    max_price_usd: float | None = Field(default=None, ge=0, le=1_000_000)
    min_price_usd: float | None = Field(default=None, ge=0, le=1_000_000)
    excluded_brands: list[str] = Field(default_factory=list, max_length=20)
    category: str | None = Field(default=None, max_length=200)
    required_terms: list[str] = Field(default_factory=list, max_length=20)
    excluded_terms: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_range(self) -> Constraints:
        if (
            self.min_price_usd is not None
            and self.max_price_usd is not None
            and self.min_price_usd > self.max_price_usd
        ):
            raise ValueError("min_price_usd must not exceed max_price_usd")
        for values in (self.excluded_brands, self.required_terms, self.excluded_terms):
            if any(not value.strip() or len(value) > 200 for value in values):
                raise ValueError("constraint terms must contain 1 to 200 characters")
        return self


class Evidence(StrictModel):
    evidence_id: str
    item_id: str
    field: str
    value: Any
    source_locator: str


class ProductDisplay(StrictModel):
    """Chinese presentation; source fields and evidence remain authoritative."""

    locale: Literal["zh-CN"] = "zh-CN"
    title: str = Field(min_length=1, max_length=2000)
    category: str
    features: list[str] = Field(default_factory=list, max_length=3)
    attributes: dict[str, str] = Field(default_factory=dict)
    translation_kind: Literal["model", "rule"]
    source_language: Literal["en"] = "en"


class SearchItem(StrictModel):
    item_id: str
    title: str
    category: str = "Office Products"
    brand: str | None = None
    price_usd: float | None = None
    features: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    display: ProductDisplay | None = None
    rank: int = 0
    score: float = 0
    route_ranks: dict[str, int] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    missing_attributes: list[str] = Field(default_factory=list)


class SessionRequest(StrictModel):
    session_id: str = Field(min_length=36, max_length=36, pattern=r"^[a-f0-9-]+$")


class SearchRequest(SessionRequest):
    query: str = Field(min_length=1, max_length=1000)
    constraints: Constraints = Field(default_factory=Constraints)
    limit: int = Field(default=5, ge=1, le=10)
    mode: Literal["hybrid", "lexical", "dense"] = "hybrid"
    personalize: bool = True


class CompareRequest(SessionRequest):
    item_ids: list[str] = Field(min_length=2, max_length=4)
    dimensions: list[str] = Field(
        default_factory=lambda: ["price_usd", "brand", "category"], max_length=12
    )


class FeedbackRequest(SessionRequest):
    event_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=64)
    action: Literal["like", "dislike", "save", "remove"]


class ChatRequest(SessionRequest):
    request_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)


class SearchResponse(StrictModel):
    trace_id: str
    session_id: str
    catalog_version: str
    status: Literal["ok", "no_match", "retrieval_incomplete", "degraded"]
    items: list[SearchItem]
    applied_constraints: Constraints
    profile_version: int
    warnings: list[str] = Field(default_factory=list)
    timing_ms: dict[str, float] = Field(default_factory=dict)
    total_eligible: int


class ToolAction(StrictModel):
    """Finite model action schema; every argument is validated by its tool."""

    action: Literal["search_catalog", "compare_items", "clarify", "answer"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    message: str = Field(default="", max_length=2000)


class SearchToolArguments(StrictModel):
    """Validate model-controlled search arguments before accessing a tool."""

    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=10, strict=True)
