"""Catalog integrity, FTS safety, and hard-constraint behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from choiceagent.catalog import (
    CatalogRepository,
    lexical_tokens,
    matches_query_anchor,
    rewrite_query,
)
from choiceagent.schemas import Constraints, Product


def make_repository(tmp_path: Path) -> CatalogRepository:
    rows = [
        {
            "item_id": "A",
            "title": "Metal desk organizer black",
            "category": "Office Products",
            "category_path": ["Office Products", "Desk Organizers"],
            "brand": "Maker",
            "price_usd": 20,
            "features": ["Three trays"],
            "attributes": {"Material": "Metal"},
            "catalog_version": "test-v1",
        },
        {
            "item_id": "B",
            "title": "Wood desk organizer white",
            "category": "Office Products",
            "category_path": ["Office Products", "Desk Organizers"],
            "brand": "Other",
            "price_usd": None,
            "features": [],
            "catalog_version": "test-v1",
        },
        {
            "item_id": "C",
            "title": "Printer paper letter",
            "category": "Office Products",
            "category_path": ["Office Products", "Paper"],
            "brand": "Maker",
            "price_usd": 4,
            "features": [],
            "catalog_version": "test-v1",
        },
    ]
    path = tmp_path / "catalog.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    repository = CatalogRepository(path, tmp_path / "catalog.sqlite3")
    repository.initialize()
    return repository


def test_catalog_is_idempotent_and_resolves_identity(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    repository.initialize()
    assert repository.get("A").title == "Metal desk organizer black"
    assert repository.get("missing") is None
    assert len(repository.all_products()) == 3
    assert repository.stats()["priced_count"] == 2


def test_price_filter_rejects_unknown_price(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    assert [item.item_id for item in repository.filter_products(Constraints(max_price_usd=20))] == [
        "A",
        "C",
    ]
    assert repository.filter_products(Constraints(max_price_usd=1)) == []
    assert [item.item_id for item in repository.filter_products(Constraints(min_price_usd=10))] == [
        "A"
    ]


def test_material_brand_and_category_constraints(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    assert [
        item.item_id for item in repository.filter_products(Constraints(required_terms=["金属"]))
    ] == ["A"]
    assert [
        item.item_id for item in repository.filter_products(Constraints(excluded_terms=["木质"]))
    ] == ["A", "C"]
    assert [
        item.item_id for item in repository.filter_products(Constraints(excluded_brands=["MAKER"]))
    ] == ["B"]
    assert [item.item_id for item in repository.filter_products(Constraints(category="Paper"))] == [
        "C"
    ]


def test_chinese_rewrite_and_bm25_score_direction(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    assert rewrite_query("桌面收纳架") == "desk organizer"
    rows = repository.lexical_search("桌面收纳架")
    assert {row[0] for row in rows} == {"A", "B"}
    assert [score for _, score in rows] == sorted(score for _, score in rows)


def test_fts_query_syntax_and_sql_injection_are_literal(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    assert lexical_tokens('" OR * NEAR() ; --') == ["near"]
    repository.lexical_search('"; DROP TABLE products; --')
    repository.lexical_search("(desk) NOT * OR title:paper")
    assert repository.stats()["count"] == 3


def test_unknown_terms_do_not_return_the_whole_catalog(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    assert repository.lexical_search("galacticflorble") == []
    assert repository.lexical_search("完全未知词语") == []


def test_compound_query_does_not_match_an_unrelated_generic_office_product() -> None:
    chair = Product(
        item_id="CHAIR", title="Office chair", category_path=["Office Products", "Chairs"]
    )
    pen = Product(item_id="PEN", title="Office pen", category_path=["Office Products", "Pens"])
    assert matches_query_anchor(chair, "办公椅")
    assert not matches_query_anchor(pen, "办公椅")


def test_device_compatibility_does_not_turn_a_mousepad_into_a_mouse() -> None:
    pad = Product(item_id="PAD", title="Mouse Pad for Wireless Mouse")
    mouse = Product(item_id="MOUSE", title="Wireless Mouse 2.4G Optical")
    assert not matches_query_anchor(pad, "无线鼠标")
    assert matches_query_anchor(mouse, "无线鼠标")


def test_duplicate_ids_abort_rebuild(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    lines = repository.catalog_path.read_text().splitlines()
    repository.catalog_path.write_text("\n".join([*lines, lines[0]]) + "\n")
    with pytest.raises(ValueError, match="Duplicate item"):
        repository.initialize()
