"""Shared service contracts: visibility, factual comparisons, and deadline safety."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

import choiceagent.services as services_module
from choiceagent.errors import AppError
from choiceagent.schemas import CompareRequest, Constraints, SearchRequest, SearchResponse
from choiceagent.services import DiscoveryService


def search_request(service: DiscoveryService, **kwargs: Any) -> SearchRequest:
    session_id = service.state.create_session()["session_id"]
    return SearchRequest(session_id=session_id, query="desk organizer", mode="lexical", **kwargs)


def test_search_records_only_returned_products_and_exposes_a_session_trace(
    sample_discovery: DiscoveryService,
) -> None:
    request = search_request(sample_discovery, constraints=Constraints(max_price_usd=25))
    result = sample_discovery.search(request)
    SearchResponse.model_validate(result)
    assert [item["item_id"] for item in result["items"]] == ["FIXTURE_A"]
    state = sample_discovery.state.get_session(request.session_id)
    assert state["visible_item_ids"] == ["FIXTURE_A"]
    assert state["last_query"] == "desk organizer"
    assert state["constraints"]["max_price_usd"] == 25
    trace = sample_discovery.state.get_trace(result["trace_id"], request.session_id)
    assert trace["item_ids"] == ["FIXTURE_A"]
    assert trace["mode"] == "lexical"


def test_search_reuses_explicit_feedback_and_honors_personalization_switch(
    sample_discovery: DiscoveryService,
) -> None:
    request = search_request(sample_discovery)
    sample_discovery.search(request)
    sample_discovery.state.feedback(request.session_id, "dislike", "FIXTURE_A", "dislike")
    personalized = sample_discovery.search(request)
    general = sample_discovery.search(request.model_copy(update={"personalize": False}))
    assert personalized["profile_version"] == 1
    assert "FIXTURE_A" not in {item["item_id"] for item in personalized["items"]}
    assert "FIXTURE_A" in {item["item_id"] for item in general["items"]}


def test_compare_unknown_values_remain_null_and_evidence_is_field_exact(
    sample_discovery: DiscoveryService,
) -> None:
    request = search_request(sample_discovery)
    sample_discovery.search(request)
    comparison = sample_discovery.compare(
        CompareRequest(
            session_id=request.session_id,
            item_ids=["FIXTURE_A", "FIXTURE_B"],
            dimensions=["price_usd", "material", "attributes.Material", "battery_life", "features"],
        )
    )
    rows = {row["dimension"]: row["values"] for row in comparison["rows"]}
    assert rows["price_usd"] == [
        {
            "item_id": "FIXTURE_A",
            "value": 20,
            "display_value": 20,
            "evidence_id": "FIXTURE_A:price_usd",
        },
        {"item_id": "FIXTURE_B", "value": None, "display_value": None, "evidence_id": None},
    ]
    assert rows["material"][0] == {
        "item_id": "FIXTURE_A",
        "value": "Metal",
        "display_value": "金属",
        "evidence_id": "FIXTURE_A:attributes.Material",
    }
    assert rows["attributes.Material"] == rows["material"]
    assert rows["features"][0] == {
        "item_id": "FIXTURE_A",
        "value": ["Three trays"],
        "display_value": "功能描述请查看英文原文",
        "evidence_id": "FIXTURE_A:features",
    }
    assert rows["features"][1] == {
        "item_id": "FIXTURE_B",
        "value": None,
        "display_value": None,
        "evidence_id": None,
    }
    assert all(
        value["value"] is None and value["evidence_id"] is None for value in rows["battery_life"]
    )
    evidence_by_item = {
        card["item_id"]: {
            evidence["evidence_id"]: evidence["value"] for evidence in card["evidence"]
        }
        for card in comparison["items"]
    }
    assert all(
        value["evidence_id"] is None
        or evidence_by_item[value["item_id"]].get(value["evidence_id"]) == value["value"]
        for row in comparison["rows"]
        for value in row["values"]
    )
    assert (
        sample_discovery.state.get_trace(comparison["trace_id"], request.session_id)["kind"]
        == "compare"
    )


def test_compare_cannot_access_products_visible_only_in_another_session(
    sample_discovery: DiscoveryService,
) -> None:
    request = search_request(sample_discovery)
    sample_discovery.search(request)
    another = sample_discovery.state.create_session()["session_id"]
    with pytest.raises(AppError) as caught:
        sample_discovery.compare(
            CompareRequest(session_id=another, item_ids=["FIXTURE_A", "FIXTURE_B"])
        )
    assert caught.value.code == "ITEM_NOT_VISIBLE"
    assert caught.value.status_code == 403


def test_duplicate_comparison_ids_are_rejected(sample_discovery: DiscoveryService) -> None:
    request = search_request(sample_discovery)
    sample_discovery.search(request)
    with pytest.raises(AppError) as caught:
        sample_discovery.compare(
            CompareRequest(session_id=request.session_id, item_ids=["FIXTURE_A", "FIXTURE_A"])
        )
    assert caught.value.status_code == 422


def test_comparison_still_checks_catalog_identity_after_visibility(
    sample_discovery: DiscoveryService,
) -> None:
    session_id = sample_discovery.state.create_session()["session_id"]
    sample_discovery.state.record_search(
        session_id, ["FIXTURE_A", "REMOVED_ITEM"], "older snapshot", {}
    )
    with pytest.raises(AppError) as caught:
        sample_discovery.compare(
            CompareRequest(session_id=session_id, item_ids=["FIXTURE_A", "REMOVED_ITEM"])
        )
    assert caught.value.code == "UNKNOWN_ITEM"
    assert caught.value.status_code == 404


def test_expired_search_deadline_does_not_make_products_visible(
    sample_discovery: DiscoveryService,
) -> None:
    request = search_request(sample_discovery)
    with pytest.raises(AppError) as caught:
        sample_discovery.search(request, deadline=time.monotonic() - 1)
    assert caught.value.code == "TURN_TIMEOUT"
    assert sample_discovery.state.get_session(request.session_id)["visible_item_ids"] == []


def test_search_finishing_after_deadline_does_not_commit_results(
    sample_discovery: DiscoveryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = search_request(sample_discovery)
    clock = [100.0]
    original = sample_discovery.engine.search

    def slow_search(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        clock[0] = 102.0
        return result

    monkeypatch.setattr(
        services_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], perf_counter=time.perf_counter),
    )
    monkeypatch.setattr(sample_discovery.engine, "search", slow_search)
    with pytest.raises(AppError) as caught:
        sample_discovery.search(request, deadline=101.0)
    assert caught.value.code == "TURN_TIMEOUT"
    state = sample_discovery.state.get_session(request.session_id)
    assert state["visible_item_ids"] == []
    assert state["last_query"] == ""


def test_unknown_session_is_rejected_before_retrieval(
    sample_discovery: DiscoveryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_search(*args: Any, **kwargs: Any) -> dict[str, Any]:
        pytest.fail("Retrieval must not run for an unknown session")

    monkeypatch.setattr(sample_discovery.engine, "search", forbidden_search)
    with pytest.raises(AppError) as caught:
        sample_discovery.search(
            SearchRequest(
                session_id="00000000-0000-0000-0000-000000000000", query="desk", mode="lexical"
            )
        )
    assert caught.value.code == "SESSION_NOT_FOUND"
