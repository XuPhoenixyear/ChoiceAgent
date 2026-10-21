"""Isolated CPU-only fixtures shared by service and HTTP contract tests."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from choiceagent.catalog import CatalogRepository
from choiceagent.model_client import GenerationResult
from choiceagent.retrieval import SearchEngine
from choiceagent.services import DiscoveryService
from choiceagent.state import StateStore


class OfflineProvider:
    """A predictable tool controller; never allocates a model or sends requests."""

    def __init__(self) -> None:
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
        text = (
            '{"action":"search_catalog","arguments":{"query":"desk organizer","limit":5},"message":""}'
            if self.calls % 2
            else '{"action":"answer","arguments":{},"message":""}'
        )
        return GenerationResult(
            text=text,
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            model="offline-test-controller",
        )


@pytest.fixture
def sample_state(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "state.sqlite3")


@pytest.fixture
def sample_discovery(tmp_path: Path, sample_state: StateStore) -> DiscoveryService:
    catalog_path = tmp_path / "catalog.jsonl"
    fixture = Path(__file__).resolve().parents[1] / "data/fixtures/catalog.jsonl"
    shutil.copyfile(fixture, catalog_path)
    repository = CatalogRepository(catalog_path, tmp_path / "catalog.sqlite3")
    repository.initialize()
    return DiscoveryService(
        SearchEngine(repository, tmp_path / "missing-index", None), sample_state
    )


@pytest.fixture
def offline_provider() -> OfflineProvider:
    return OfflineProvider()
