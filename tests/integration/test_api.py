"""HTTP integration tests use real catalog/state services and no model/network."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import choiceagent.localization as localization
from choiceagent.api import create_app, restrict_gpus
from choiceagent.config import Settings
from choiceagent.localization import CatalogLocalizer, source_hash
from choiceagent.schemas import SearchResponse
from choiceagent.services import DiscoveryService


@pytest.fixture
def client_factory(
    sample_discovery: DiscoveryService, offline_provider: Any, tmp_path: Path
) -> Iterator[Callable[[str | None], TestClient]]:
    with ExitStack() as stack:

        def create(api_key: str | None = None) -> TestClient:
            settings = Settings(
                _env_file=None,
                project_root=tmp_path,
                log_dir=tmp_path / "logs",
                api_key=api_key,
                gpu_ids="5",
            )
            app = create_app(settings, discovery=sample_discovery, provider=offline_provider)
            return stack.enter_context(TestClient(app, raise_server_exceptions=False))

        yield create


def new_session(client: TestClient) -> str:
    response = client.post("/v1/sessions")
    assert response.status_code == 201
    return str(response.json()["session_id"])


def search(client: TestClient, session_id: str, **values: Any) -> Any:
    return client.post(
        "/v1/search",
        json={"session_id": session_id, "query": "desk organizer", "mode": "lexical", **values},
    )


def test_health_and_stats_do_not_call_or_load_generation(
    client_factory: Callable[[str | None], TestClient], offline_provider: Any
) -> None:
    client = client_factory(None)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["catalog_count"] == 3
    assert response.json()["dense_ready"] is False
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"
    uuid.UUID(response.headers["X-Request-ID"])
    stats = client.get("/v1/catalog/stats").json()
    assert stats["count"] == 3
    assert stats["priced_count"] == 2
    assert offline_provider.calls == 0


def test_search_feedback_compare_and_trace_roundtrip(
    client_factory: Callable[[str | None], TestClient],
) -> None:
    client = client_factory(None)
    session_id = new_session(client)
    response = search(client, session_id)
    assert response.status_code == 200
    result = SearchResponse.model_validate(response.json())
    assert {item.item_id for item in result.items} == {"FIXTURE_A", "FIXTURE_B"}
    feedback = {
        "session_id": session_id,
        "event_id": str(uuid.uuid4()),
        "item_id": "FIXTURE_A",
        "action": "like",
    }
    first = client.post("/v1/feedback", json=feedback)
    replay = client.post("/v1/feedback", json=feedback)
    assert first.status_code == replay.status_code == 200
    assert first.json() == replay.json()
    assert first.json()["profile_version"] == 1
    comparison = client.post(
        "/v1/compare",
        json={
            "session_id": session_id,
            "item_ids": ["FIXTURE_A", "FIXTURE_B"],
            "dimensions": ["price_usd", "Size"],
        },
    )
    assert comparison.status_code == 200
    values = comparison.json()["rows"][0]["values"]
    assert values[1]["value"] is None and values[1]["evidence_id"] is None
    trace = client.get(f"/v1/traces/{result.trace_id}", params={"session_id": session_id})
    assert trace.status_code == 200
    assert trace.json()["kind"] == "search"
    session = client.get(f"/v1/sessions/{session_id}").json()
    assert session["liked_item_ids"] == ["FIXTURE_A"]


def test_chinese_display_survives_http_search_compare_and_agent_without_changing_evidence(
    client_factory: Callable[[str | None], TestClient],
    sample_discovery: DiscoveryService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    product = sample_discovery.engine.repository.get("FIXTURE_A")
    assert product is not None
    original = product.model_dump()
    path = tmp_path / "zh.jsonl"
    path.write_text(
        json.dumps(
            {
                "item_id": product.item_id,
                "source_hash": source_hash(product),
                "translation_version": "zh-CN-v1",
                "title": "Fixture Maker 黑色金属桌面收纳架",
                "features": ["三个托盘"],
                "attributes": {"Material": "金属", "Size": "10 英寸"},
                "translation_kind": "model",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(localization, "DEFAULT_LOCALIZER", CatalogLocalizer(path))
    client = client_factory(None)
    session_id = new_session(client)
    result = search(client, session_id).json()
    card = next(item for item in result["items"] if item["item_id"] == product.item_id)
    assert card["display"]["title"] == "Fixture Maker 黑色金属桌面收纳架"
    assert card["display"]["features"] == ["三个托盘"]
    assert card["title"] == original["title"]
    assert card["attributes"] == original["attributes"]
    assert {entry["field"]: entry["value"] for entry in card["evidence"]}["title"] == original[
        "title"
    ]
    assert (
        client.get("/v1/catalog/stats").json()["category_labels"]["Office Products"] == "办公用品"
    )
    comparison = client.post(
        "/v1/compare",
        json={
            "session_id": session_id,
            "item_ids": ["FIXTURE_A", "FIXTURE_B"],
            "dimensions": ["title", "Material", "features", "price_usd"],
        },
    ).json()
    material = comparison["rows"][1]
    assert material["label_zh"] == "材质"
    assert material["values"][0]["value"] == "Metal"
    assert material["values"][0]["display_value"] == "金属"
    assert material["values"][1]["display_value"] is None
    chat = client.post(
        "/v1/chat",
        json={
            "session_id": session_id,
            "request_id": "zh-display",
            "message": "找桌面收纳",
        },
    ).json()
    assert "Fixture Maker 黑色金属桌面收纳架" in chat["answer"]
    assert original["title"] not in chat["answer"]
    assert sample_discovery.engine.repository.get(product.item_id).model_dump() == original


def test_session_trace_and_item_visibility_are_isolated(
    client_factory: Callable[[str | None], TestClient],
) -> None:
    client = client_factory(None)
    first, second = new_session(client), new_session(client)
    result = search(client, first).json()
    trace = client.get(f"/v1/traces/{result['trace_id']}", params={"session_id": second})
    assert trace.status_code == 404
    assert trace.json()["error"]["code"] == "TRACE_NOT_FOUND"
    feedback = client.post(
        "/v1/feedback",
        json={
            "session_id": second,
            "event_id": str(uuid.uuid4()),
            "item_id": "FIXTURE_A",
            "action": "save",
        },
    )
    assert feedback.status_code == 403
    comparison = client.post(
        "/v1/compare", json={"session_id": second, "item_ids": ["FIXTURE_A", "FIXTURE_B"]}
    )
    assert comparison.status_code == 403
    assert client.get(f"/v1/sessions/{second}").json()["visible_item_ids"] == []


def test_feedback_idempotency_conflict_is_a_safe_409(
    client_factory: Callable[[str | None], TestClient],
) -> None:
    client = client_factory(None)
    session_id = new_session(client)
    search(client, session_id)
    payload = {
        "session_id": session_id,
        "event_id": str(uuid.uuid4()),
        "item_id": "FIXTURE_A",
        "action": "like",
    }
    assert client.post("/v1/feedback", json=payload).status_code == 200
    response = client.post("/v1/feedback", json={**payload, "action": "dislike"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    assert client.get(f"/v1/sessions/{session_id}").json()["liked_item_ids"] == ["FIXTURE_A"]


@pytest.mark.parametrize(
    "change",
    [
        {"limit": 0},
        {"limit": 11},
        {"query": ""},
        {"query": "x" * 1001},
        {"mode": "unsafe"},
        {"constraints": {"max_price_usd": -1}},
        {"constraints": {"min_price_usd": 50, "max_price_usd": 20}},
        {"unexpected_secret": "never-echo-this"},
    ],
)
def test_invalid_search_inputs_are_422_without_echoing_values(
    client_factory: Callable[[str | None], TestClient], change: dict[str, Any]
) -> None:
    client = client_factory(None)
    session_id = new_session(client)
    response = search(client, session_id, **change)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_ARGUMENT"
    assert "never-echo-this" not in response.text
    assert response.json()["error"]["trace_id"] == response.headers["X-Request-ID"]


def test_nonfinite_json_price_is_rejected(
    client_factory: Callable[[str | None], TestClient],
) -> None:
    client = client_factory(None)
    session_id = new_session(client)
    content = json.dumps(
        {"session_id": session_id, "query": "desk", "constraints": {"max_price_usd": float("nan")}}
    )
    response = client.post(
        "/v1/search", content=content, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_ARGUMENT"


def test_oversized_and_malformed_bodies_have_bounded_public_errors(
    client_factory: Callable[[str | None], TestClient],
) -> None:
    client = client_factory(None)
    oversized = client.post(
        "/v1/search", content=b"x" * 65537, headers={"Content-Type": "application/json"}
    )
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "REQUEST_TOO_LARGE"
    assert len(oversized.content) < 1000
    malformed = client.post(
        "/v1/search",
        content=b'{"secret":"do-not-echo",',
        headers={"Content-Type": "application/json"},
    )
    assert malformed.status_code == 422
    assert "do-not-echo" not in malformed.text


def test_optional_access_key_protects_business_endpoints_but_not_health(
    client_factory: Callable[[str | None], TestClient],
) -> None:
    client = client_factory("fixture-access-secret")
    assert client.get("/health").status_code == 200
    for header in (
        None,
        "Bearer wrong-secret",
        "Basic fixture-access-secret",
        "fixture-access-secret",
    ):
        response = client.get(
            "/v1/catalog/stats", headers={"Authorization": header} if header else {}
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"
        assert "fixture-access-secret" not in response.text
    response = client.post(
        "/v1/sessions", headers={"Authorization": "Bearer fixture-access-secret"}
    )
    assert response.status_code == 201


def test_unknown_session_and_missing_trace_are_404(
    client_factory: Callable[[str | None], TestClient],
) -> None:
    client = client_factory(None)
    unknown = str(uuid.uuid4())
    assert client.get(f"/v1/sessions/{unknown}").status_code == 404
    assert search(client, unknown).status_code == 404
    session_id = new_session(client)
    assert (
        client.get(f"/v1/traces/{uuid.uuid4()}", params={"session_id": session_id}).status_code
        == 404
    )


def test_empty_budget_result_has_no_match_not_server_error(
    client_factory: Callable[[str | None], TestClient],
) -> None:
    client = client_factory(None)
    session_id = new_session(client)
    response = search(client, session_id, constraints={"max_price_usd": 0})
    assert response.status_code == 200
    assert response.json()["status"] == "no_match"
    assert response.json()["items"] == []


def test_internal_failure_does_not_expose_paths_or_exception_text(
    client_factory: Callable[[str | None], TestClient],
    sample_discovery: DiscoveryService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = client_factory(None)
    session_id = new_session(client)

    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("secret=/private/model/path")

    monkeypatch.setattr(sample_discovery.engine, "search", fail)
    response = search(client, session_id)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "/private/model/path" not in response.text
    assert "RuntimeError" not in response.text


def test_gpu_visibility_cannot_exceed_two_explicit_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    with pytest.raises(ValueError, match="at most two"):
        restrict_gpus("5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    restrict_gpus("4,5")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "5"


def test_chat_http_replay_and_persisted_evidence(
    client_factory: Callable[[str | None], TestClient], offline_provider: Any
) -> None:
    client = client_factory(None)
    sid = new_session(client)
    payload = {"session_id": sid, "request_id": "http-chat-1", "message": "找桌面收纳，预算20美元"}
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "degraded"  # Fixture deliberately has no dense artifact.
    assert result["items"] and all(item["price_usd"] <= 20 for item in result["items"])
    assert all(item["evidence"] for item in result["items"])
    calls = offline_provider.calls
    assert client.post("/v1/chat", json=payload).json() == result
    assert offline_provider.calls == calls
    conflict = client.post("/v1/chat", json={**payload, "message": "other intent"})
    assert conflict.status_code == 409
    trace = client.get(f"/v1/traces/{result['trace_id']}", params={"session_id": sid})
    assert trace.status_code == 200 and trace.json()["llm_calls"] == 1
