"""Transactional correctness under retries, concurrent writes, and session boundaries."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from choiceagent.errors import AppError
from choiceagent.state import StateStore


def visible_session(store: StateStore, item_ids: list[str] | None = None) -> str:
    session_id = str(store.create_session()["session_id"])
    store.record_search(session_id, item_ids or ["A", "B"], "test query", {})
    return session_id


def test_concurrent_duplicate_feedback_has_exactly_one_effect(sample_state: StateStore) -> None:
    session_id = visible_session(sample_state)
    event_id = str(uuid.uuid4())
    with ThreadPoolExecutor(max_workers=12) as executor:
        responses = list(
            executor.map(
                lambda _: sample_state.feedback(session_id, event_id, "A", "like"), range(24)
            )
        )
    assert all(response == responses[0] for response in responses)
    state = sample_state.get_session(session_id)
    assert state["profile_version"] == 1
    assert state["liked_item_ids"] == ["A"]


def test_concurrent_distinct_writes_are_not_lost(sample_state: StateStore) -> None:
    items = [f"ITEM_{index}" for index in range(24)]
    session_id = visible_session(sample_state, items)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda item_id: sample_state.feedback(session_id, item_id, item_id, "like"), items
            )
        )
    state = sample_state.get_session(session_id)
    assert state["profile_version"] == len(items)
    assert set(state["liked_item_ids"]) == set(items)


def test_idempotency_conflict_rolls_back_instead_of_changing_feedback(
    sample_state: StateStore,
) -> None:
    session_id = visible_session(sample_state)
    sample_state.feedback(session_id, "event", "A", "like")
    with pytest.raises(AppError) as caught:
        sample_state.feedback(session_id, "event", "B", "save")
    assert caught.value.status_code == 409
    assert caught.value.code == "IDEMPOTENCY_CONFLICT"
    assert sample_state.get_session(session_id)["profile_version"] == 1
    assert sample_state.get_session(session_id)["shortlist_item_ids"] == []


def test_feedback_receipts_are_scoped_to_each_session(sample_state: StateStore) -> None:
    first = visible_session(sample_state)
    second = visible_session(sample_state)
    sample_state.feedback(first, "same-event", "A", "like")
    sample_state.feedback(second, "same-event", "B", "save")
    assert sample_state.get_session(first)["liked_item_ids"] == ["A"]
    assert sample_state.get_session(second)["liked_item_ids"] == []
    assert sample_state.get_session(second)["shortlist_item_ids"] == ["B"]


def test_feedback_cannot_operate_on_another_sessions_visible_product(
    sample_state: StateStore,
) -> None:
    first = visible_session(sample_state, ["A"])
    second = visible_session(sample_state, ["B"])
    sample_state.feedback(first, "first", "A", "save")
    with pytest.raises(AppError) as caught:
        sample_state.feedback(second, "forged", "A", "like")
    assert caught.value.code == "ITEM_NOT_VISIBLE"
    assert sample_state.get_session(second)["profile_version"] == 0
    assert (
        sample_state.get_receipt(second, "feedback", "forged", {"item_id": "A", "action": "like"})
        is None
    )


def test_like_dislike_are_mutually_exclusive_and_shortlist_is_independent(
    sample_state: StateStore,
) -> None:
    session_id = visible_session(sample_state)
    sample_state.feedback(session_id, "1", "A", "like")
    sample_state.feedback(session_id, "2", "A", "save")
    sample_state.feedback(session_id, "3", "A", "dislike")
    state = sample_state.get_session(session_id)
    assert state["liked_item_ids"] == []
    assert state["disliked_item_ids"] == ["A"]
    assert state["shortlist_item_ids"] == ["A"]
    sample_state.feedback(session_id, "4", "A", "remove")
    assert sample_state.get_session(session_id)["shortlist_item_ids"] == []


def test_trace_cannot_be_read_from_a_different_session(sample_state: StateStore) -> None:
    first = visible_session(sample_state)
    second = visible_session(sample_state)
    trace = {"trace_id": str(uuid.uuid4()), "kind": "search", "query": "private"}
    sample_state.save_trace(first, trace)
    assert sample_state.get_trace(trace["trace_id"], first) == trace
    with pytest.raises(AppError) as caught:
        sample_state.get_trace(trace["trace_id"], second)
    assert caught.value.status_code == 404
    assert caught.value.code == "TRACE_NOT_FOUND"


def test_concurrent_chat_completion_stores_one_receipt_and_one_message_pair(
    sample_state: StateStore,
) -> None:
    session_id = visible_session(sample_state)
    response = {"answer": "verified answer", "request_id": "request"}
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda _: sample_state.complete_chat(session_id, "request", "question", response),
                range(16),
            )
        )
    state = sample_state.get_session(session_id)
    assert state["messages"] == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "verified answer"},
    ]
    assert (
        sample_state.get_receipt(session_id, "chat", "request", {"message": "question"}) == response
    )
    with pytest.raises(AppError) as caught:
        sample_state.complete_chat(session_id, "request", "different question", response)
    assert caught.value.status_code == 409


def test_history_and_visibility_have_documented_bounds(sample_state: StateStore) -> None:
    session_id = visible_session(sample_state, [str(index) for index in range(1050)])
    assert len(sample_state.get_session(session_id)["visible_item_ids"]) == 1000
    assert "0" not in sample_state.get_session(session_id)["visible_item_ids"]
    for index in range(12):
        sample_state.complete_chat(
            session_id, str(index), f"question {index}", {"answer": f"answer {index}"}
        )
    messages = sample_state.get_session(session_id)["messages"]
    assert len(messages) == 20
    assert messages[0]["content"] == "question 2"
    assert messages[-1]["content"] == "answer 11"


def test_concurrent_session_creation_respects_capacity(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "limited.sqlite3", max_sessions=2)

    def create(_: int) -> str:
        try:
            return str(store.create_session()["session_id"])
        except AppError as error:
            assert error.status_code == 429
            return error.code

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(create, range(12)))
    assert results.count("SESSION_LIMIT") == 10
    assert len({result for result in results if result != "SESSION_LIMIT"}) == 2


def test_just_redisplayed_items_remain_visible_when_history_capacity_is_full(
    sample_state: StateStore,
) -> None:
    session_id = visible_session(sample_state, [str(index) for index in range(1000)])
    sample_state.record_search(session_id, ["0", "NEW_ITEM"], "redisplay an old favorite", {})
    state = sample_state.get_session(session_id)
    assert len(state["visible_item_ids"]) <= 1000
    assert {"0", "NEW_ITEM"}.issubset(state["visible_item_ids"])
    sample_state.feedback(session_id, "freshly-visible", "0", "save")
    assert sample_state.get_session(session_id)["shortlist_item_ids"] == ["0"]


def test_invalid_action_does_not_create_a_receipt_or_mutate_profile(
    sample_state: StateStore,
) -> None:
    session_id = visible_session(sample_state)
    with pytest.raises(AppError) as caught:
        sample_state.feedback(session_id, "invalid", "A", "execute_purchase")
    assert caught.value.status_code == 422
    assert sample_state.get_session(session_id)["profile_version"] == 0


def test_trace_retention_keeps_recent_records_without_affecting_session_state(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "bounded.sqlite3", max_traces=2)
    first, second = visible_session(store), visible_session(store)
    store.save_trace(first, {"trace_id": "old", "kind": "search"})
    store.save_trace(second, {"trace_id": "recent", "kind": "compare"})
    store.save_trace(first, {"trace_id": "newest", "kind": "search"})
    with pytest.raises(AppError) as caught:
        store.get_trace("old", first)
    assert caught.value.code == "TRACE_NOT_FOUND"
    assert store.get_trace("recent", second)["kind"] == "compare"
    assert store.get_trace("newest", first)["kind"] == "search"
    assert store.get_session(first)["visible_item_ids"] == ["A", "B"]
    with pytest.raises(AppError):
        store.get_trace("newest", second)


@pytest.mark.parametrize("limits", [{"max_sessions": 0}, {"max_traces": 0}, {"max_traces": -1}])
def test_storage_limits_must_be_positive(tmp_path: Path, limits: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        StateStore(tmp_path / "invalid.sqlite3", **limits)


def test_unknown_session_never_returns_another_sessions_state(sample_state: StateStore) -> None:
    sample_state.create_session()
    with pytest.raises(AppError) as caught:
        sample_state.get_session("' OR 1=1;--")
    assert caught.value.code == "SESSION_NOT_FOUND"
