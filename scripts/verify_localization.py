"""Audit complete Chinese presentation and exercise the real running API.

This measures translation coverage and structural fidelity, not semantic accuracy.
Source facts, prices and evidence must survive unchanged. No model is mocked.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from choiceagent.localization import CatalogLocalizer, source_hash
from choiceagent.schemas import Product, SearchResponse

ROOT = Path(__file__).resolve().parents[1]
CHINESE = re.compile(r"[\u4e00-\u9fff]")
QUERIES = ("桌面收纳", "钢笔", "打印纸", "办公椅", "订书机", "鼠标垫", "笔记本", "文件夹", "白板")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    catalog_path = ROOT / "data/processed/catalog.jsonl"
    sidecar_path = ROOT / "data/processed/catalog.zh-CN.jsonl"
    catalog = {
        product.item_id: product
        for line in catalog_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for product in [Product.model_validate_json(line)]
    }
    rows = [json.loads(line) for line in sidecar_path.read_text(encoding="utf-8").splitlines()]
    localizer = CatalogLocalizer(sidecar_path)
    checks: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    def check(condition: bool, criterion: str) -> None:
        checks.append({"criterion": criterion, "passed": condition})

    index_manifest = json.loads((ROOT / "artifacts/indexes/manifest.json").read_text())
    check(
        sha256(catalog_path) == index_manifest["catalog_sha256"], "Indexed source catalog unchanged"
    )
    check(len(rows) == len(catalog) == 3000, "One display row for each of 3000 source products")
    check(
        {row["item_id"] for row in rows} == set(catalog),
        "Sidecar product IDs exactly match catalog",
    )
    check(
        all(row["source_hash"] == source_hash(catalog[row["item_id"]]) for row in rows),
        "All translations bound to current source content",
    )
    displays = {item_id: localizer.display(product) for item_id, product in catalog.items()}
    kinds = Counter(display["translation_kind"] for display in displays.values())
    check(kinds["model"] / len(catalog) >= 0.95, "At least 95% of titles translated by the model")
    check(
        all(CHINESE.search(display["title"]) for display in displays.values()),
        "All primary product titles contain Chinese",
    )
    check(
        all(CHINESE.search(display["category"]) for display in displays.values()),
        "All categories displayed in Chinese",
    )
    check(
        all(
            len(displays[item_id]["features"]) <= min(3, len(product.features))
            for item_id, product in catalog.items()
        ),
        "No features invented for empty source fields",
    )

    def audit_cards(items: list[dict[str, Any]]) -> None:
        for item in items:
            product = catalog[item["item_id"]]
            for field in ("title", "category", "brand", "price_usd", "features", "attributes"):
                check(
                    item[field] == getattr(product, field),
                    f"{product.item_id}: original {field} unchanged",
                )
            expected_evidence = {
                "title": product.title,
                "price_usd": product.price_usd,
                "brand": product.brand,
                "category": product.category,
                "features": product.features or None,
                **{f"features.{i}": value for i, value in enumerate(product.features)},
                **{f"attributes.{key}": value for key, value in product.attributes.items()},
            }
            check(
                all(
                    entry["value"] == expected_evidence.get(entry["field"])
                    for entry in item["evidence"]
                ),
                f"{product.item_id}: evidence retains source values",
            )
            check(
                item["display"] == displays[product.item_id],
                f"{product.item_id}: API uses current Chinese display",
            )

    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=90) as client:

        def request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
            response = client.request(method, path, json=body)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            events.append(
                {
                    "method": method,
                    "path": path,
                    "request": body,
                    "status": response.status_code,
                    "response": data,
                }
            )
            return data

        health = request("GET", "/health")
        fingerprint = hashlib.sha256()
        for path in sorted((ROOT / "src/choiceagent").glob("*.py")):
            fingerprint.update(path.name.encode())
            fingerprint.update(path.read_bytes())
        check(
            health["source_fingerprint"] == fingerprint.hexdigest(),
            "API restarted with current source fingerprint",
        )
        stats = request("GET", "/v1/catalog/stats")
        check(
            set(stats["category_labels"]) == set(stats["categories"]),
            "Every filter category has a presentation label",
        )
        check(
            all(CHINESE.search(label) for label in stats["category_labels"].values()),
            "All filter labels contain Chinese",
        )
        sid = request("POST", "/v1/sessions")["session_id"]
        for query in QUERIES:
            result = request(
                "POST",
                "/v1/search",
                {
                    "session_id": sid,
                    "query": query,
                    "mode": "hybrid" if query == "桌面收纳" else "lexical",
                    "personalize": False,
                    "limit": 5,
                },
            )
            SearchResponse.model_validate(result)
            check(bool(result["items"]), f"Chinese query returns products: {query}")
            audit_cards(result["items"])
        chat = request(
            "POST",
            "/v1/chat",
            {
                "session_id": sid,
                "request_id": str(uuid.uuid4()),
                "message": "找桌面收纳，预算35美元",
            },
        )
        audit_cards(chat["items"])
        check(bool(chat["items"]), "Real local-model Agent returns products")
        check(chat["usage"].get("total_tokens", 0) > 0, "Agent made a real local-model generation")
        check(
            all(item["display"]["title"] in chat["answer"] for item in chat["items"][:3]),
            "Agent answer uses Chinese product titles",
        )
        comparison = request(
            "POST",
            "/v1/compare",
            {
                "session_id": sid,
                "item_ids": [item["item_id"] for item in chat["items"][:2]],
                "dimensions": [
                    "title",
                    "price_usd",
                    "brand",
                    "category",
                    "Material",
                    "features",
                    "nonexistent_field",
                ],
            },
        )
        audit_cards(comparison["items"])
        check(
            all(CHINESE.search(row["label_zh"]) for row in comparison["rows"]),
            "All comparison labels contain Chinese",
        )
        check(
            all(value["display_value"] is None for value in comparison["rows"][-1]["values"]),
            "Unknown comparison attributes stay unknown",
        )
        answer = request(
            "POST",
            "/v1/chat",
            {
                "session_id": sid,
                "request_id": str(uuid.uuid4()),
                "message": "比较前两个商品的材质和尺寸",
            },
        )
        check(answer.get("comparison") is not None, "Real Agent comparison succeeds")
        if answer.get("comparison"):
            audit_cards(answer["comparison"]["items"])
            check(
                all(row["label_zh"] in answer["answer"] for row in answer["comparison"]["rows"]),
                "Agent comparison uses the requested Chinese field names",
            )

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_fingerprint": fingerprint.hexdigest(),
        "catalog_sha256": sha256(catalog_path),
        "sidecar_sha256": sha256(sidecar_path),
        "product_count": len(catalog),
        "translation_kinds": dict(kinds),
        "products_with_chinese_features": sum(
            bool(display["features"]) for display in displays.values()
        ),
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "passed": all(check["passed"] for check in checks),
        "limitations": "Coverage and source preservation checks do not measure semantic translation accuracy or retrieval relevance.",
        "checks": checks,
        "http_events": events,
    }
    output = ROOT / "reports/localization"
    output.mkdir(parents=True, exist_ok=True)
    (output / "api_acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "product_count",
                    "translation_kinds",
                    "products_with_chinese_features",
                    "checks_passed",
                    "checks_total",
                )
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
