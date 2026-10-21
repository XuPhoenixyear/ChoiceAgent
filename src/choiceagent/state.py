"""Transactional session state, idempotency receipts, and bounded trace storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from choiceagent.errors import AppError


class StateStore:
    """Each operation owns its connection; writes use an immediate transaction."""

    def __init__(self, path: Path, *, max_sessions: int = 10000, max_traces: int = 10000) -> None:
        if max_sessions < 1 or max_traces < 1:
            raise ValueError("Storage limits must be positive")
        self.path = path
        self.max_sessions = max_sessions
        self.max_traces = max_traces
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, state_json TEXT NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    session_id TEXT NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL,
                    payload_hash TEXT NOT NULL, response_json TEXT NOT NULL,
                    PRIMARY KEY(session_id,kind,key),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                    data_json TEXT NOT NULL, created_at REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS traces_created_at ON traces(created_at);
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise

    @staticmethod
    def _read(db: sqlite3.Connection, session_id: str) -> dict[str, Any]:
        row = db.execute(
            "SELECT state_json FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None:
            raise AppError("SESSION_NOT_FOUND", "会话不存在，请新建会话。", 404)
        return dict(json.loads(row[0]))

    @staticmethod
    def _write(db: sqlite3.Connection, state: dict[str, Any]) -> None:
        db.execute(
            "UPDATE sessions SET state_json=?,updated_at=? WHERE session_id=?",
            (json.dumps(state, ensure_ascii=False), time.time(), state["session_id"]),
        )

    def create_session(self) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        state: dict[str, Any] = {
            "session_id": session_id,
            "profile_version": 0,
            "liked_item_ids": [],
            "disliked_item_ids": [],
            "shortlist_item_ids": [],
            "visible_item_ids": [],
            "last_result_order": [],
            "last_query": "",
            "constraints": {},
            "messages": [],
        }
        with self._transaction() as db:
            count = int(db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
            if count >= self.max_sessions:
                raise AppError("SESSION_LIMIT", "会话容量已满。", 429)
            db.execute(
                "INSERT INTO sessions VALUES (?,?,?,?)",
                (session_id, json.dumps(state), time.time(), time.time()),
            )
        return state

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._connection() as db:
            return self._read(db, session_id)

    def record_search(
        self, session_id: str, item_ids: list[str], query: str, constraints: dict[str, Any]
    ) -> dict[str, Any]:
        with self._transaction() as db:
            state = self._read(db, session_id)
            state["last_result_order"] = item_ids
            current = set(item_ids)
            previous_ids = [x for x in state["visible_item_ids"] if x not in current]
            state["visible_item_ids"] = list(dict.fromkeys(previous_ids + item_ids))[-1000:]
            state["last_query"] = query
            state["constraints"] = constraints
            self._write(db, state)
        return state

    @staticmethod
    def payload_hash(payload: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    def _receipt(
        self,
        db: sqlite3.Connection,
        session_id: str,
        kind: str,
        key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        row = db.execute(
            "SELECT payload_hash,response_json FROM receipts WHERE session_id=? AND kind=? AND key=?",
            (session_id, kind, key),
        ).fetchone()
        if row is None:
            return None
        if row[0] != self.payload_hash(payload):
            raise AppError("IDEMPOTENCY_CONFLICT", "同一请求标识不能用于不同内容。", 409)
        return dict(json.loads(row[1]))

    def get_receipt(
        self, session_id: str, kind: str, key: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        with self._connection() as db:
            self._read(db, session_id)
            return self._receipt(db, session_id, kind, key, payload)

    def complete_chat(
        self, session_id: str, key: str, message: str, response: dict[str, Any]
    ) -> None:
        with self._transaction() as db:
            state = self._read(db, session_id)
            payload = {"message": message}
            if self._receipt(db, session_id, "chat", key, payload) is not None:
                return
            state["messages"] = (
                state["messages"]
                + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response["answer"]},
                ]
            )[-20:]
            self._write(db, state)
            db.execute(
                "INSERT INTO receipts VALUES (?,?,?,?,?)",
                (
                    session_id,
                    "chat",
                    key,
                    self.payload_hash(payload),
                    json.dumps(response, ensure_ascii=False),
                ),
            )

    def feedback(self, session_id: str, event_id: str, item_id: str, action: str) -> dict[str, Any]:
        payload = {"item_id": item_id, "action": action}
        with self._transaction() as db:
            state = self._read(db, session_id)
            previous = self._receipt(db, session_id, "feedback", event_id, payload)
            if previous is not None:
                return previous
            if item_id not in state["visible_item_ids"]:
                raise AppError("ITEM_NOT_VISIBLE", "只能操作本会话已经展示的商品。", 403)
            if action == "like":
                if item_id not in state["liked_item_ids"]:
                    state["liked_item_ids"].append(item_id)
                state["disliked_item_ids"] = [x for x in state["disliked_item_ids"] if x != item_id]
            elif action == "dislike":
                if item_id not in state["disliked_item_ids"]:
                    state["disliked_item_ids"].append(item_id)
                state["liked_item_ids"] = [x for x in state["liked_item_ids"] if x != item_id]
            elif action == "save":
                if item_id not in state["shortlist_item_ids"]:
                    state["shortlist_item_ids"].append(item_id)
            elif action == "remove":
                state["shortlist_item_ids"] = [
                    x for x in state["shortlist_item_ids"] if x != item_id
                ]
            else:
                raise AppError("INVALID_ACTION", "不支持该反馈动作。", 422)
            state["profile_version"] += 1
            self._write(db, state)
            response = {
                key: state[key]
                for key in (
                    "profile_version",
                    "liked_item_ids",
                    "disliked_item_ids",
                    "shortlist_item_ids",
                )
            }
            response["event_id"] = event_id
            db.execute(
                "INSERT INTO receipts VALUES (?,?,?,?,?)",
                (
                    session_id,
                    "feedback",
                    event_id,
                    self.payload_hash(payload),
                    json.dumps(response),
                ),
            )
        return response

    def save_trace(self, session_id: str, trace: dict[str, Any]) -> None:
        with self._transaction() as db:
            self._read(db, session_id)
            db.execute(
                "INSERT OR REPLACE INTO traces VALUES (?,?,?,?)",
                (trace["trace_id"], session_id, json.dumps(trace, ensure_ascii=False), time.time()),
            )
            db.execute(
                "DELETE FROM traces WHERE trace_id IN "
                "(SELECT trace_id FROM traces ORDER BY created_at DESC, rowid DESC LIMIT -1 OFFSET ?)",
                (self.max_traces,),
            )

    def get_trace(self, trace_id: str, session_id: str) -> dict[str, Any]:
        with self._connection() as db:
            self._read(db, session_id)
            row = db.execute(
                "SELECT data_json FROM traces WHERE trace_id=? AND session_id=?",
                (trace_id, session_id),
            ).fetchone()
        if row is None:
            raise AppError("TRACE_NOT_FOUND", "未找到该会话的执行记录。", 404)
        return dict(json.loads(row[0]))
