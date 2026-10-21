"""Controller contract tests; scripted model outputs are adversarial test inputs."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from choiceagent.agent import AgentService, explicit_constraints, parse_action
from choiceagent.catalog import CatalogRepository
from choiceagent.config import Settings
from choiceagent.errors import AppError
from choiceagent.model_client import GenerationResult, ModelClientError
from choiceagent.retrieval import SearchEngine, product_card
from choiceagent.schemas import ChatRequest, Constraints, Product, SearchRequest
from choiceagent.services import DiscoveryService
from choiceagent.state import StateStore


def action(name: str, **arguments: Any) -> str:
    return json.dumps({"action": name, "arguments": arguments, "message": ""})


class ScriptedProvider:
    def __init__(self, outputs: list[str | Exception]) -> None:
        self.outputs = iter(outputs)
        self.calls = 0

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        deadline: float | None,
    ) -> GenerationResult:
        self.calls += 1
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return GenerationResult(value, {"completion_tokens": 10}, "scripted-unit-only", 1, "stop")


class FixtureEngine:
    """Deterministic fixture tool, independent of the real retrieval implementation tests."""

    def __init__(self, repository: CatalogRepository) -> None:
        self.repository = repository
        self.queries: list[str] = []

    def search(self, query: str, constraints: Constraints, **kwargs: Any) -> dict[str, Any]:
        self.queries.append(query)
        products = self.repository.filter_products(constraints)
        return {
            "status": "ok" if products else "no_match",
            "items": [product_card(p, rank=i + 1) for i, p in enumerate(products)],
            "warnings": [],
            "applied_constraints": constraints.model_dump(),
            "total_eligible": len(products),
        }


@pytest.fixture
def system(tmp_path: Path) -> tuple[DiscoveryService, FixtureEngine, str]:
    products = [
        Product(
            item_id="A",
            title="Wood desk organizer",
            brand="Acme",
            price_usd=10,
            features=["wood"],
            source_locator="fixture:1",
        ),
        Product(
            item_id="B",
            title="Metal desk organizer",
            brand="Beta",
            price_usd=25,
            features=["metal"],
            source_locator="fixture:2",
        ),
    ]
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("\n".join(p.model_dump_json() for p in products) + "\n")
    repository = CatalogRepository(catalog, tmp_path / "catalog.sqlite3")
    repository.initialize()
    engine = FixtureEngine(repository)
    discovery = DiscoveryService(cast(SearchEngine, engine), StateStore(tmp_path / "state.sqlite3"))
    return discovery, engine, discovery.state.create_session()["session_id"]


def controller(
    discovery: DiscoveryService, outputs: list[str | Exception], **settings: Any
) -> tuple[AgentService, ScriptedProvider]:
    provider = ScriptedProvider(outputs)
    return AgentService(discovery, provider, Settings(**settings)), provider


def request(
    session_id: str, message: str = "找桌面收纳，预算20美元", key: str = "test-1"
) -> ChatRequest:
    return ChatRequest(session_id=session_id, request_id=key, message=message)


@pytest.mark.parametrize(
    ("message", "expected_min", "expected_max"),
    [
        ("预算20美元的收纳盒", None, 20),
        ("under $30 desk organizer", None, 30),
        ("至少50美元的椅子", 50, None),
        ("50美元以上", 50, None),
        ("at least $25", 25, None),
        ("10到30美元的桌面收纳", 10, 30),
        ("不少于10美元且不超过30美元", 10, 30),
    ],
)
def test_explicit_price_intent(
    message: str, expected_min: float | None, expected_max: float | None
) -> None:
    constraints, clarification = explicit_constraints(message, Constraints())
    assert clarification is None
    assert constraints.min_price_usd == expected_min
    assert constraints.max_price_usd == expected_max


@pytest.mark.parametrize(
    "message", ["200元的键盘", "预算¥200", "预算￥200", "budget 200 yuan", "预算20欧元"]
)
def test_non_usd_is_clarified(message: str) -> None:
    original = Constraints(max_price_usd=10)
    assert explicit_constraints(message, original)[0] == original
    assert explicit_constraints(message, original)[1] is not None


def test_constraints_persist_cancel_conflict_and_brand_boundary() -> None:
    previous = Constraints(max_price_usd=20, excluded_terms=["wood"])
    assert explicit_constraints("取消预算", previous)[0].max_price_usd is None
    assert explicit_constraints("至少30美元", previous)[1] is not None
    value, error = explicit_constraints("mouse exclude brand Acme under $20", previous)
    assert error is None and value.excluded_brands == ["Acme"]
    assert value.excluded_terms == ["wood"]


def test_parse_fences_and_reject_unknown_actions() -> None:
    assert parse_action("```json\n" + action("answer") + "\n```").action == "answer"
    with pytest.raises(ValueError):
        parse_action('{"action":"run_shell","arguments":{"cmd":"rm -rf /"}}')


def test_real_tool_evidence_and_ignored_model_prose(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, _, sid = system
    malicious_answer = (
        '{"action":"answer","arguments":{},"message":"Everything costs $999.99 and is in stock"}'
    )
    agent, provider = controller(
        discovery, [action("search_catalog", query="desk organizer"), malicious_answer]
    )
    result = agent.chat(request(sid))
    assert provider.calls == 2 and result["status"] == "completed"
    assert [x["item_id"] for x in result["items"]] == ["A"]
    assert "$10.00" in result["answer"] and "999.99" not in result["answer"]
    assert result["items"][0]["evidence"]
    trace = discovery.state.get_trace(result["trace_id"], sid)
    assert trace["llm_calls"] == 2 and trace["tool_calls"][0]["ok"]


def test_receipt_replays_without_model_and_conflicting_payload_fails(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, _, sid = system
    agent, provider = controller(
        discovery, [action("search_catalog", query="desk organizer"), action("answer")]
    )
    first = agent.chat(request(sid))
    assert agent.chat(request(sid)) == first and provider.calls == 2
    with pytest.raises(AppError) as failure:
        agent.chat(request(sid, "different message"))
    assert failure.value.status_code == 409


def test_model_failure_falls_back_to_cards(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, _, sid = system
    agent, provider = controller(
        discovery, [ModelClientError("MODEL_TIMEOUT", "timeout", retryable=True)]
    )
    result = agent.chat(request(sid))
    assert provider.calls == 1 and result["status"] == "degraded"
    assert result["items"] and "MODEL_TIMEOUT" in result["warnings"]


def test_bad_json_repair_once_then_fallback(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, _, sid = system
    agent, provider = controller(discovery, ["not json", "still not json"])
    result = agent.chat(request(sid))
    assert provider.calls == 2 and result["status"] == "degraded" and result["items"]
    assert discovery.state.get_trace(result["trace_id"], sid)["repair_attempts"] == 1


def test_malformed_search_arguments_are_rejected_and_repaired(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, engine, sid = system
    agent, _ = controller(
        discovery,
        [
            action("search_catalog", query={"attack": "nested"}),
            action("search_catalog", query="desk organizer"),
            action("answer"),
        ],
    )
    result = agent.chat(request(sid))
    assert engine.queries == ["desk organizer"]
    assert result["tool_calls"][0]["error_code"] == "INVALID_TOOL_ARGUMENT"
    assert result["items"]


def test_current_budget_cannot_be_overridden_by_model_arguments(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, _, sid = system
    agent, _ = controller(
        discovery,
        [
            action("search_catalog", query="desk organizer", constraints={"max_price_usd": 999}),
            action("search_catalog", query="desk organizer"),
            action("answer"),
        ],
    )
    result = agent.chat(request(sid))
    assert [p["price_usd"] for p in result["items"]] == [10]


@pytest.mark.parametrize(
    "message", ["有什么适合我的？", "预算20美元", "比较前两个商品", "预算200元的键盘"]
)
def test_insufficient_context_asks_without_loading_model(
    system: tuple[DiscoveryService, FixtureEngine, str], message: str
) -> None:
    discovery, _, sid = system
    agent, provider = controller(discovery, [])
    assert agent.chat(request(sid, message))["status"] == "clarification"
    assert provider.calls == 0


def test_clarification_free_text_is_not_trusted(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, _, sid = system
    agent, _ = controller(discovery, ['{"action":"clarify","message":"Buy Acme for $1.23"}'])
    result = agent.chat(request(sid))
    assert result["status"] == "clarification" and "$1.23" not in result["answer"]


def test_budget_only_reuses_previous_product_query(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, engine, sid = system
    discovery.search(SearchRequest(session_id=sid, query="desk organizer"))
    agent, _ = controller(discovery, [action("search_catalog", query="chair"), action("answer")])
    result = agent.chat(request(sid, "预算改成20美元"))
    assert engine.queries[-1] == "desk organizer"
    assert [i["item_id"] for i in result["items"]] == ["A"]


def test_compare_ordinal_overrides_invented_ids_and_preserves_unknown_fields(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, _, sid = system
    discovery.search(SearchRequest(session_id=sid, query="desk organizer"))
    agent, _ = controller(
        discovery,
        [action("compare_items", item_ids=["EVIL", "B"], dimensions=["brand"]), action("answer")],
    )
    result = agent.chat(request(sid, "比较前两个商品的尺寸"))
    assert [i["item_id"] for i in result["items"]] == ["A", "B"]
    row = next(r for r in result["comparison"]["rows"] if r["dimension"] == "Product Dimensions")
    assert all(v["value"] is None and v["evidence_id"] is None for v in row["values"])
    assert "未知" in result["answer"]


def test_repeated_tool_and_call_budget_bound_work(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, engine, sid = system
    repeated = action("search_catalog", query="desk organizer")
    agent, provider = controller(discovery, [repeated] * 4)
    result = agent.chat(request(sid))
    assert len(engine.queries) == 1 and provider.calls == 2
    assert "repeated_tool_stopped" in result["warnings"]
    agent2, provider2 = controller(
        discovery, [action("search_catalog", query="desk organizer")], max_llm_calls=1
    )
    assert agent2.chat(request(sid, key="another"))["items"] and provider2.calls == 1


def test_no_matches_does_not_invent_products(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, _, sid = system
    agent, _ = controller(discovery, [action("search_catalog", query="desk organizer")])
    result = agent.chat(request(sid, "找桌面收纳，预算0美元"))
    assert result["status"] == "no_match" and result["items"] == []


def test_busy_session_rejected_before_second_generation(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, _, sid = system
    agent, _ = controller(discovery, [])
    agent._locks[sid] = threading.Lock()
    agent._locks[sid].acquire()
    try:
        with pytest.raises(AppError) as failure:
            agent.chat(request(sid))
        assert failure.value.code == "SESSION_BUSY"
    finally:
        agent._locks[sid].release()


def test_model_failure_after_search_retains_verified_cards(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, engine, sid = system
    agent, _ = controller(
        discovery,
        [
            action("search_catalog", query="desk organizer"),
            ModelClientError("MODEL_UNAVAILABLE", "down", retryable=True),
        ],
    )
    result = agent.chat(request(sid))
    assert result["status"] == "degraded" and len(result["items"]) == 1
    assert len(engine.queries) == 1 and "MODEL_UNAVAILABLE" in result["warnings"]


def test_deadline_rejects_late_model_output_before_tool(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    import time

    discovery, engine, sid = system

    class LateProvider(ScriptedProvider):
        def generate(self, *args: Any, **kwargs: Any) -> GenerationResult:
            time.sleep(0.02)
            return super().generate(*args, **kwargs)

    provider = LateProvider([action("search_catalog", query="desk organizer")])
    agent = AgentService(discovery, provider, Settings(turn_timeout_seconds=0.01))
    result = agent.chat(request(sid))
    assert result["status"] == "degraded" and engine.queries == []
    assert "TURN_TIMEOUT" in result["warnings"]


def test_llm_budget_exhaustion_is_explicit(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, _, sid = system
    agent, _ = controller(
        discovery, [action("search_catalog", query="desk organizer")], max_llm_calls=1
    )
    result = agent.chat(request(sid))
    assert result["items"] and result["status"] == "degraded"
    assert "llm_budget_exhausted" in result["warnings"]


@pytest.mark.parametrize(
    "message",
    ["找钢笔，预算50美元", "找办公椅，至少50美元", "找收纳盒20美元以内", "desk under $20"],
)
def test_short_product_intent_is_not_budget_only(message: str) -> None:
    assert not AgentService._budget_only(message)


@pytest.mark.parametrize(
    "message",
    ["预算改成20美元", "至少50美元", "50美元以上", "取消预算", "under $20", "20美元", "10到20美元"],
)
def test_true_budget_only_changes(message: str) -> None:
    assert AgentService._budget_only(message)


@pytest.mark.parametrize(
    "wrong_action",
    [
        action("compare_items", item_ids=["A", "B", "C", "D", "E"]),
        action("clarify"),
        action("answer"),
    ],
)
def test_budget_only_cannot_be_diverted_to_another_tool(
    system: tuple[DiscoveryService, FixtureEngine, str], wrong_action: str
) -> None:
    discovery, engine, sid = system
    discovery.search(SearchRequest(session_id=sid, query="desk organizer"))
    agent, _ = controller(discovery, [wrong_action, action("answer")])
    result = agent.chat(request(sid, "预算改成20美元"))
    assert engine.queries == ["desk organizer", "desk organizer"]
    assert result["items"] and all(p["price_usd"] <= 20 for p in result["items"])
    assert discovery.state.get_session(sid)["constraints"]["max_price_usd"] == 20


def test_repeated_invalid_tool_arguments_remain_explicit_failure(
    system: tuple[DiscoveryService, FixtureEngine, str],
) -> None:
    discovery, engine, sid = system
    bad = action("search_catalog", query={"bad": "input"})
    agent, _ = controller(discovery, [bad, bad])
    result = agent.chat(request(sid))
    assert result["status"] == "degraded" and engine.queries == []
    assert "INVALID_TOOL_ARGUMENT" in result["warnings"]
