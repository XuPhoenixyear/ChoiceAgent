from __future__ import annotations

import time

import httpx
import pytest

from choiceagent.model_client import LocalModelClient, ModelClientError


def payload() -> dict[str, object]:
    return {
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "model": "Qwen3-1.7B-local",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "elapsed_ms": 12.5,
    }


def test_valid_generation_and_deadline_header() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert 0 < float(request.headers["X-Request-Timeout"]) <= 2
        return httpx.Response(200, json=payload())

    with LocalModelClient(transport=httpx.MockTransport(respond)) as client:
        result = client.generate(
            [{"role": "user", "content": "hello"}],
            max_tokens=12,
            temperature=0.0,
            deadline=time.monotonic() + 2,
        )
    assert result.text == "hello"
    assert result.usage["total_tokens"] == 5


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"choices": []},
        {**payload(), "usage": {"total_tokens": -1}},
        {**payload(), "choices": [{"message": {"content": None}}]},
    ],
)
def test_invalid_provider_shape(body: object) -> None:
    with (
        LocalModelClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
        ) as client,
        pytest.raises(ModelClientError, match="invalid response") as caught,
    ):
        client.generate([{"role": "user", "content": "hello"}])
    assert caught.value.code == "MODEL_PROTOCOL_ERROR"


def test_http_timeout_is_typed() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    with (
        LocalModelClient(transport=httpx.MockTransport(fail)) as client,
        pytest.raises(ModelClientError) as caught,
    ):
        client.generate([{"role": "user", "content": "hello"}])
    assert caught.value.code == "MODEL_TIMEOUT"
    assert caught.value.retryable


def test_expired_deadline_makes_no_network_request() -> None:
    def unexpected(_request: httpx.Request) -> httpx.Response:
        pytest.fail("Expired requests must not be sent")

    with (
        LocalModelClient(transport=httpx.MockTransport(unexpected)) as client,
        pytest.raises(ModelClientError) as caught,
    ):
        client.generate([], deadline=time.monotonic() - 1)
    assert caught.value.code == "MODEL_TIMEOUT"


@pytest.mark.parametrize(
    "status,code", [(429, "MODEL_BUSY"), (503, "MODEL_UNAVAILABLE"), (504, "MODEL_TIMEOUT")]
)
def test_remote_http_errors(status: int, code: str) -> None:
    with (
        LocalModelClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(status, text="private prompt"))
        ) as client,
        pytest.raises(ModelClientError) as caught,
    ):
        client.generate([{"role": "user", "content": "hello"}])
    assert caught.value.code == code
    assert "private prompt" not in str(caught.value)


def test_remote_context_error_is_preserved() -> None:
    with (
        LocalModelClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(422, json={"error": {"code": "MODEL_CONTEXT_LIMIT"}})
            )
        ) as client,
        pytest.raises(ModelClientError) as caught,
    ):
        client.generate([{"role": "user", "content": "hello"}])
    assert caught.value.code == "MODEL_CONTEXT_LIMIT"


@pytest.mark.parametrize(
    "url", ["file:///tmp/a", "https://user:secret@host", "http://localhost?secret=value"]
)
def test_invalid_base_url(url: str) -> None:
    with pytest.raises(ValueError):
        LocalModelClient(url)
