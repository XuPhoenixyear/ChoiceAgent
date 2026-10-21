"""Source normalization tests do not require network access."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from choiceagent.ingestion import _collect, normalize_product, parse_price, sha256_file


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (12.5, 12.5),
        ("$1,234.50", 1234.5),
        (None, None),
        ("Not available", None),
        ("5-10", None),
        (float("nan"), None),
        (float("inf"), None),
        (-1, None),
        (True, None),
    ],
)
def test_price_is_a_finite_historical_usd_number(source: object, expected: float | None) -> None:
    assert parse_price(source) == expected


def test_normalization_keeps_missing_values_and_excludes_aggregate_fields() -> None:
    product = normalize_product(
        {
            "parent_asin": "B000TEST01",
            "title": " Desk   organizer ",
            "price": "unknown",
            "average_rating": 5,
            "rating_number": 900,
            "details": {"Brand": "Maker", "Material": "Metal", "Best Sellers Rank": {"Office": 3}},
            "features": ["Three trays"],
            "description": ["Original text"],
        },
        source_line=12,
        catalog_version="fixture-v1",
    )
    assert product is not None
    assert product["title"] == "Desk organizer"
    assert product["price_usd"] is None
    assert product["brand"] == "Maker"
    assert product["attributes"] == {"Brand": "Maker", "Material": "Metal"}
    assert product["source_locator"].endswith("line=12")
    assert "average_rating" not in product


def test_invalid_identity_is_rejected() -> None:
    assert normalize_product({"title": "Orphan"}, source_line=1, catalog_version="x") is None
    assert (
        normalize_product(
            {"title": "Product", "parent_asin": "unsafe/id"}, source_line=1, catalog_version="x"
        )
        is None
    )


def test_storefront_is_never_silently_used_as_brand() -> None:
    product = normalize_product(
        {"parent_asin": "B000TEST01", "title": "Product", "store": "Merchant"},
        source_line=1,
        catalog_version="test",
    )
    assert product is not None
    assert product["brand"] is None


def test_bounded_collection_deduplicates_and_stops(tmp_path: Path) -> None:
    rows = [
        {"parent_asin": "A", "title": "First", "price": 2},
        {"parent_asin": "A", "title": "Duplicate"},
        {"parent_asin": "B", "title": "Second"},
        {"parent_asin": "C", "title": "Must not read"},
    ]
    source = io.BytesIO(b"\n".join(json.dumps(row).encode() for row in rows))
    raw_path = tmp_path / "raw.jsonl"
    catalog_path = tmp_path / "catalog.jsonl"
    counts = _collect(source, raw_path=raw_path, catalog_path=catalog_path, limit=2, version="test")
    assert counts == {"products": 2, "scanned_lines": 3, "malformed_lines": 0, "priced_products": 1}
    assert len(raw_path.read_text().splitlines()) == 2
    assert [json.loads(line)["item_id"] for line in catalog_path.read_text().splitlines()] == [
        "A",
        "B",
    ]
    assert len(sha256_file(catalog_path)) == 64


def test_short_source_never_publishes_partial_catalog(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    catalog_path = tmp_path / "catalog.jsonl"
    with pytest.raises(ValueError, match="Source ended"):
        _collect(
            io.BytesIO(b""), raw_path=raw_path, catalog_path=catalog_path, limit=2, version="x"
        )
    assert not catalog_path.exists()
    assert not raw_path.exists()
    assert not list(tmp_path.glob("*.tmp"))
