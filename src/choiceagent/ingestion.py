"""Bounded, reproducible ingestion of the official Amazon Reviews 2023 catalog.

Only metadata is collected for the two-week release. No review ratings, aggregate
popularity, live inventory, or generated attributes enter the retrieval catalog.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)
OFFICIAL_SOURCE = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "meta_categories/meta_Office_Products.jsonl.gz"
)
DATASET_HOMEPAGE = "https://amazon-reviews-2023.github.io/"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest without loading a large file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: Any, *, maximum: int = 16000) -> str:
    """Normalize source whitespace, preserving the original factual wording."""
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:maximum]


def parse_price(value: Any) -> float | None:
    """Accept finite, non-negative USD numbers; reject ranges and placeholders."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().replace("$", "").replace(",", "")
        if not re.fullmatch(r"\d+(?:\.\d+)?", value):
            return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def normalize_product(
    raw: dict[str, Any], *, source_line: int, catalog_version: str
) -> dict[str, Any] | None:
    """Map one official metadata row to the stable public Product schema."""
    item_id = clean_text(raw.get("parent_asin"), maximum=80)
    title = clean_text(raw.get("title"), maximum=1500)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", item_id) or not title:
        return None
    raw_details = raw.get("details")
    details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
    attributes = {
        clean_text(key, maximum=120): clean_text(value, maximum=1000)
        for key, value in details.items()
        if value is not None
        and clean_text(value)
        and key.casefold()
        not in {"best sellers rank", "customer reviews", "average rating", "rating number"}
    }
    # `store` is a storefront, not necessarily the product brand.
    brand = clean_text(details.get("Brand"), maximum=200) or None
    raw_features = raw.get("features") or []
    if not isinstance(raw_features, list):
        raw_features = [raw_features]
    features = [clean_text(value, maximum=2000) for value in raw_features[:25] if value]
    raw_description = raw.get("description") or []
    if not isinstance(raw_description, list):
        raw_description = [raw_description]
    categories = raw.get("categories") or []
    if not isinstance(categories, list):
        categories = [categories]
    return {
        "item_id": item_id,
        "title": title,
        "category": clean_text(raw.get("main_category") or "Office Products", maximum=200),
        "category_path": [clean_text(value, maximum=200) for value in categories if value],
        "brand": brand,
        "price_usd": parse_price(raw.get("price")),
        "features": features,
        "description": clean_text(" ".join(str(value) for value in raw_description)),
        "attributes": attributes,
        "source_url": OFFICIAL_SOURCE,
        "source_locator": f"meta_Office_Products.jsonl.gz#line={source_line}",
        "catalog_version": catalog_version,
    }


def _collect(
    stream: Iterable[bytes], *, raw_path: Path, catalog_path: Path, limit: int, version: str
) -> dict[str, int]:
    seen: set[str] = set()
    scanned = 0
    malformed = 0
    priced = 0
    # Files are published atomically so a stopped download is never a valid catalog.
    raw_temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
    catalog_temporary = catalog_path.with_suffix(catalog_path.suffix + ".tmp")
    try:
        with (
            raw_temporary.open("wb") as raw_output,
            catalog_temporary.open("w", encoding="utf-8") as catalog_output,
        ):
            for source_line, line in enumerate(stream, start=1):
                scanned += 1
                try:
                    raw = json.loads(line)
                    product = normalize_product(
                        raw, source_line=source_line, catalog_version=version
                    )
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError, AttributeError):
                    malformed += 1
                    continue
                if product is None or product["item_id"] in seen:
                    continue
                seen.add(product["item_id"])
                raw_output.write(line.rstrip(b"\r\n") + b"\n")
                catalog_output.write(json.dumps(product, ensure_ascii=False) + "\n")
                priced += product["price_usd"] is not None
                if len(seen) >= limit:
                    break
        if len(seen) < limit:
            raise ValueError(f"Source ended at {len(seen)} valid products; requested {limit}.")
        raw_temporary.replace(raw_path)
        catalog_temporary.replace(catalog_path)
    finally:
        raw_temporary.unlink(missing_ok=True)
        catalog_temporary.unlink(missing_ok=True)
    return {
        "products": len(seen),
        "scanned_lines": scanned,
        "malformed_lines": malformed,
        "priced_products": priced,
    }


def prepare_catalog(
    root: Path, *, limit: int = 3000, source: str = OFFICIAL_SOURCE
) -> dict[str, Any]:
    """Stream the official compressed file and freeze a bounded prefix sample.

    Prefix sampling is deterministic and convenient for a demo, not a random or
    representative category sample. The manifest makes this selection explicit.
    """
    if not 1 <= limit <= 5000:
        raise ValueError("MVP catalog limit must be between 1 and 5000.")
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "office_products_metadata_sample.jsonl"
    catalog_path = processed_dir / "catalog.jsonl"
    version = f"amazon23-office-prefix-{limit}-v1"
    LOGGER.info("Fetching official metadata prefix: %s", source)
    with requests.get(source, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        with gzip.GzipFile(fileobj=response.raw) as stream:
            counts = _collect(
                stream,
                raw_path=raw_path,
                catalog_path=catalog_path,
                limit=limit,
                version=version,
            )
    manifest = {
        "schema_version": 1,
        "dataset": "McAuley-Lab Amazon Reviews 2023",
        "category": "Office_Products",
        "homepage": DATASET_HOMEPAGE,
        "source_url": source,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "sampling": "First N unique valid metadata rows in source order; not random.",
        "catalog_version": version,
        "price_semantics": "Historical crawl-time USD; missing values remain null.",
        "brand_semantics": "Only details.Brand; storefront is not treated as brand.",
        "excluded_fields": [
            "average_rating",
            "rating_number",
            "bought_together",
            "images",
            "videos",
            "details.Best Sellers Rank",
            "details.Customer Reviews",
        ],
        **counts,
        "raw_sample_sha256": sha256_file(raw_path),
        "catalog_sha256": sha256_file(catalog_path),
        "raw_sample_path": str(raw_path.relative_to(root)),
        "catalog_path": str(catalog_path.relative_to(root)),
    }
    manifest_path = processed_dir / "catalog.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Frozen %d products (%d priced)", counts["products"], counts["priced_products"])
    return manifest
