"""Bounded synchronous client for the project's local generation service."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Provider-independent generation response; no provider side effects."""

    text: str
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    elapsed_ms: float = 0.0
    finish_reason: str = "stop"


class ModelClientError(RuntimeError):
    """Safe error surface consumed by the application fallback policy."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class LocalModelClient:
    """Call an explicit local model URL; never falls back to a paid provider."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        timeout: float = 60.0,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an explicit HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query, or fragment")
        self._timeout = timeout
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            transport=transport,
            trust_env=False,
        )

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 768,
        temperature: float = 0.0,
        deadline: float | None = None,
    ) -> GenerationResult:
        """Generate within a monotonic absolute deadline, including queue time."""
        if not 1 <= max_tokens <= 768:
            raise ValueError("max_tokens must be between 1 and 768")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        remaining = self._timeout if deadline is None else deadline - time.monotonic()
        timeout = min(self._timeout, remaining)
        if timeout <= 0:
            raise ModelClientError("MODEL_TIMEOUT", "Model request deadline expired.")
        try:
            response = self._client.post(
                "v1/chat/completions",
                json={
                    "messages": [dict(message) for message in messages],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                headers={"X-Request-Timeout": f"{timeout:.3f}"},
                timeout=httpx.Timeout(timeout),
            )
        except httpx.TimeoutException as exc:
            raise ModelClientError(
                "MODEL_TIMEOUT", "Local model request timed out.", retryable=True
            ) from exc
        except httpx.RequestError as exc:
            raise ModelClientError(
                "MODEL_UNAVAILABLE", "Local model service is unavailable.", retryable=True
            ) from exc
        if response.is_error:
            self._raise_response_error(response)
        try:
            payload: Any = response.json()
            choices = payload["choices"]
            choice = choices[0]
            content = choice["message"]["content"]
            usage = payload["usage"]
            model = payload["model"]
            elapsed_ms = payload.get("elapsed_ms", 0.0)
            finish_reason = choice.get("finish_reason", "stop")
            if not isinstance(content, str) or not isinstance(model, str):
                raise ValueError("Non-string response content or model")
            if not isinstance(usage, dict) or not all(
                isinstance(key, str) and type(value) is int and value >= 0
                for key, value in usage.items()
            ):
                raise ValueError("Invalid usage accounting")
            required = {"prompt_tokens", "completion_tokens", "total_tokens"}
            if not required.issubset(usage):
                raise ValueError("Missing usage accounting")
            if not isinstance(finish_reason, str):
                raise ValueError("Invalid finish reason")
            if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, int | float):
                raise ValueError("Invalid elapsed time")
            if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
                raise ValueError("Invalid elapsed time")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelClientError(
                "MODEL_PROTOCOL_ERROR", "Local model returned an invalid response."
            ) from exc
        return GenerationResult(
            text=content,
            usage=usage,
            model=model,
            elapsed_ms=float(elapsed_ms),
            finish_reason=finish_reason,
        )

    @staticmethod
    def _raise_response_error(response: httpx.Response) -> None:
        default_codes = {
            408: "MODEL_TIMEOUT",
            429: "MODEL_BUSY",
            503: "MODEL_UNAVAILABLE",
            504: "MODEL_TIMEOUT",
        }
        code = default_codes.get(response.status_code, "MODEL_PROTOCOL_ERROR")
        try:
            body = response.json()
            remote_code = body.get("error", {}).get("code")
            if isinstance(remote_code, str) and remote_code.startswith("MODEL_"):
                code = remote_code
        except (ValueError, AttributeError):
            pass
        raise ModelClientError(
            code,
            f"Local model request failed (HTTP {response.status_code}).",
            retryable=response.status_code in {408, 429, 503, 504},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LocalModelClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
