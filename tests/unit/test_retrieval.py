"""Ranking, evidence, strict filtering, and graceful degradation tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from choiceagent.catalog import CatalogRepository
from choiceagent.retrieval import SearchEngine, product_card
from choiceagent.schemas import Constraints, SearchItem


def make_engine(tmp_path: Path) -> SearchEngine:
    rows = [
        {
            "item_id": "A",
            "title": "Metal desk organizer",
            "price_usd": 20,
            "brand": "Maker",
            "features": ["Three trays"],
            "attributes": {"Material": "Metal"},
            "source_locator": "fixture#line=1",
            "catalog_version": "test",
        },
        {
            "item_id": "B",
            "title": "Wood desk organizer",
            "price_usd": 30,
            "brand": "Other",
            "features": [],
            "source_locator": "fixture#line=2",
            "catalog_version": "test",
        },
        {
            "item_id": "C",
            "title": "Printer paper",
            "price_usd": 4,
            "brand": None,
            "features": [],
            "source_locator": "fixture#line=3",
            "catalog_version": "test",
        },
        {
            "item_id": "D",
            "title": "Desk organizer unknown price",
            "price_usd": None,
            "features": [],
            "catalog_version": "test",
        },
    ]
    path = tmp_path / "catalog.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    repository = CatalogRepository(path, tmp_path / "catalog.sqlite3")
    repository.initialize()
    return SearchEngine(repository, tmp_path / "missing-index", None)


def test_lexical_cards_are_schema_valid_and_have_exact_evidence(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    result = engine.search("desk organizer", Constraints(), mode="lexical")
    assert result["status"] == "ok"
    assert all(SearchItem.model_validate(card) for card in result["items"])
    card = next(card for card in result["items"] if card["item_id"] == "A")
    evidence = {entry["evidence_id"]: entry for entry in card["evidence"]}
    assert evidence["A:attributes.Material"]["value"] == "Metal"
    assert evidence["A:price_usd"]["source_locator"] == "fixture#line=1"


def test_personalization_never_bypasses_query_or_hard_budget(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    result = engine.search(
        "desk organizer", Constraints(max_price_usd=25), mode="lexical", liked_item_ids=["C", "B"]
    )
    assert [card["item_id"] for card in result["items"]] == ["A"]
    assert result["total_eligible"] == 2
    assert "historical_usd_price_verified" in result["items"][0]["reason_codes"]


def test_empty_filter_and_empty_retrieval_are_different(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    no_match = engine.search("desk organizer", Constraints(max_price_usd=0), mode="lexical")
    incomplete = engine.search("galacticflorble", Constraints(), mode="lexical")
    assert no_match["status"] == "no_match"
    assert no_match["total_eligible"] == 0
    assert incomplete["status"] == "retrieval_incomplete"
    assert incomplete["total_eligible"] == 4


def test_low_confidence_dense_query_does_not_force_a_nearest_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(tmp_path)
    monkeypatch.setattr(engine, "_dense_results", lambda query: [("C", 0.37), ("A", 0.32)])
    result = engine.search("galacticflorble", Constraints(), mode="hybrid")
    assert result["status"] == "retrieval_incomplete"
    assert result["items"] == []


def test_hybrid_without_dense_falls_back_explicitly(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    result = engine.search("桌面收纳", Constraints(), mode="hybrid")
    assert result["status"] == "degraded"
    assert result["items"]
    assert any("语义检索暂不可用" in warning for warning in result["warnings"])
    assert all("dense" not in card["route_ranks"] for card in result["items"])


def test_rrf_routes_and_dense_relevance_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(tmp_path)
    monkeypatch.setattr(
        engine, "_dense_results", lambda query: [("B", 0.8), ("A", 0.77), ("C", 0.2), ("D", 0.75)]
    )
    result = engine.search("desk organizer", Constraints(max_price_usd=40))
    assert {card["item_id"] for card in result["items"]} == {"A", "B"}
    assert all(set(card["route_ranks"]) == {"lexical", "dense"} for card in result["items"])


def test_dense_profile_only_reranks_relevant_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(tmp_path)
    engine._vectors = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
    engine._id_positions = {"A": 0, "B": 1, "C": 2, "D": 3}
    monkeypatch.setattr(
        engine, "_dense_results", lambda query: [("A", 0.8), ("B", 0.79), ("C", 0.3)]
    )
    result = engine.search("desk organizer", Constraints(max_price_usd=40), liked_item_ids=["C"])
    assert result["items"][0]["item_id"] == "B"
    assert "C" not in {card["item_id"] for card in result["items"]}


def test_dislike_is_explicit_and_personalization_can_be_disabled(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    hidden = engine.search("desk organizer", Constraints(), mode="lexical", disliked_item_ids=["A"])
    visible = engine.search(
        "desk organizer", Constraints(), mode="lexical", disliked_item_ids=["A"], personalize=False
    )
    assert "A" not in {card["item_id"] for card in hidden["items"]}
    assert "A" in {card["item_id"] for card in visible["items"]}


def test_missing_fields_never_get_fabricated_evidence(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    card = product_card(engine.repository.get("D"))
    assert card["price_usd"] is None
    assert "price_usd" in card["missing_attributes"]
    assert "D:price_usd" not in {entry["evidence_id"] for entry in card["evidence"]}
