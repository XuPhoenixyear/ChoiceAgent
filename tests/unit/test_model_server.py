from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from choiceagent.model_server import (
    CompletionRequest,
    GenerationEngine,
    GenerationError,
    ModelSettings,
    _prepare_runtime,
    create_app,
)


@pytest.fixture(autouse=True)
def valid_fake_gpu_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake engines still use a valid deployment config; they never load Torch."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")


class FakeEngine(GenerationEngine):
    """Exercises admission/deadlines without loading any real model or GPU."""

    def __init__(self, settings: ModelSettings, *, delay: float = 0.0) -> None:
        super().__init__(settings)
        self.load_count = 0
        self.delay = delay
        self.started = threading.Event()
        self.active = 0
        self.max_active = 0

    def load(self) -> None:
        if self._model is None:
            self.load_count += 1
            self._model = object()

    def _decode(
        self,
        request: CompletionRequest,
        *,
        deadline: float,
        cancel: threading.Event,
        started: float,
    ) -> dict[str, Any]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            until = time.monotonic() + self.delay
            while time.monotonic() < until:
                self._check_deadline(deadline, cancel)
                time.sleep(0.005)
            return {"choices": [{"message": {"content": "fake"}}], "model": "fake"}
        finally:
            self.active -= 1


def body() -> dict[str, object]:
    return {"messages": [{"role": "user", "content": "secret-request"}], "max_tokens": 20}


def test_health_does_not_load_model(tmp_path: Path) -> None:
    settings = ModelSettings(runtime_dir=tmp_path)
    engine = FakeEngine(settings)
    with TestClient(create_app(settings, engine=engine)) as client:
        assert client.get("/health").json()["ready"] is False
        assert engine.load_count == 0
        assert client.post("/v1/chat/completions", json=body()).status_code == 200
        assert client.get("/health").json()["ready"] is True
        assert engine.load_count == 1


@pytest.mark.parametrize(
    "invalid", [{"max_tokens": 769}, {"temperature": 3.0}, {"unknown": "secret"}, {"messages": []}]
)
def test_invalid_requests_do_not_echo_prompts(tmp_path: Path, invalid: dict[str, object]) -> None:
    settings = ModelSettings(runtime_dir=tmp_path)
    with TestClient(create_app(settings, engine=FakeEngine(settings))) as client:
        response = client.post("/v1/chat/completions", json={**body(), **invalid})
    assert response.status_code == 422
    assert "secret" not in response.text


@pytest.mark.parametrize("timeout", ["nan", "inf", "-1", "invalid"])
def test_invalid_timeout_header(tmp_path: Path, timeout: str) -> None:
    settings = ModelSettings(runtime_dir=tmp_path)
    with TestClient(create_app(settings, engine=FakeEngine(settings))) as client:
        response = client.post(
            "/v1/chat/completions", json=body(), headers={"X-Request-Timeout": timeout}
        )
    assert response.status_code == 422


def test_inference_serialization_and_bounded_queue(tmp_path: Path) -> None:
    engine = FakeEngine(ModelSettings(runtime_dir=tmp_path, max_waiting=0), delay=0.1)
    request = CompletionRequest.model_validate(body())
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            engine.generate, request, deadline=time.monotonic() + 2, cancel=threading.Event()
        )
        assert engine.started.wait(timeout=1)
        with pytest.raises(GenerationError) as caught:
            engine.generate(request, deadline=time.monotonic() + 2, cancel=threading.Event())
        assert caught.value.code == "MODEL_BUSY"
        first.result(timeout=2)
    assert engine.max_active == 1
    assert engine.load_count == 1


def test_deadline_releases_inference_slot(tmp_path: Path) -> None:
    engine = FakeEngine(ModelSettings(runtime_dir=tmp_path, max_waiting=0), delay=0.2)
    request = CompletionRequest.model_validate(body())
    with pytest.raises(GenerationError) as caught:
        engine.generate(request, deadline=time.monotonic() + 0.02, cancel=threading.Event())
    assert caught.value.code == "MODEL_TIMEOUT"
    engine.delay = 0
    assert (
        engine.generate(request, deadline=time.monotonic() + 1, cancel=threading.Event())["model"]
        == "fake"
    )


def test_cancellation_releases_inference_slot(tmp_path: Path) -> None:
    engine = FakeEngine(ModelSettings(runtime_dir=tmp_path, max_waiting=0))
    request = CompletionRequest.model_validate(body())
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(GenerationError) as caught:
        engine.generate(request, deadline=time.monotonic() + 1, cancel=cancelled)
    assert caught.value.code == "MODEL_CANCELLED"
    assert (
        engine.generate(request, deadline=time.monotonic() + 1, cancel=threading.Event())["model"]
        == "fake"
    )


@pytest.mark.parametrize("mask", ["5", "0,1", " 2 , 3 "])
def test_explicit_gpu_mask_is_respected_without_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mask: str
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setenv("CHOICEAGENT_GPU_IDS", "7")
    _prepare_runtime(ModelSettings(runtime_dir=tmp_path))
    expected = ",".join(value.strip() for value in mask.split(","))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == expected
    assert os.environ["CHOICEAGENT_GPU_IDS"] == expected


def test_gpu_mask_uses_explicit_project_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CHOICEAGENT_GPU_IDS", "0,1")
    _prepare_runtime(ModelSettings(runtime_dir=tmp_path))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


@pytest.mark.parametrize("mask", ["", "0,0", "05,5", "0,1,2", "GPU-fake", "5,", "-1", "５"])
def test_invalid_gpu_mask_is_rejected_without_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mask: str
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(ValueError, match="one or two"):
        _prepare_runtime(ModelSettings(runtime_dir=tmp_path))
