"""Single-GPU Qwen3 generation server with bounded admission and deadlines.

Run exactly one process: ``python -m choiceagent.model_server``. GPU visibility
is established before importing Torch. This module never downloads model files.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path("/datas/wuxi/Projects/datas/models/Qwen3-1.7B")


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Deliberately small configuration surface for deterministic deployment."""

    runtime_dir: Path = PROJECT_ROOT / "runtime"
    preload: bool = False
    max_context: int = 8192
    max_output: int = 768
    max_waiting: int = 2
    request_timeout: float = 60.0
    max_request_timeout: float = 120.0
    host: str = "127.0.0.1"
    port: int = 8001

    @classmethod
    def from_env(cls) -> ModelSettings:
        return cls(
            runtime_dir=PROJECT_ROOT / "runtime",
            preload=os.getenv("CHOICEAGENT_MODEL_PRELOAD", "0") == "1",
            request_timeout=float(os.getenv("CHOICEAGENT_MODEL_TIMEOUT", "60")),
            host=os.getenv("CHOICEAGENT_MODEL_HOST", "127.0.0.1"),
            port=int(os.getenv("CHOICEAGENT_MODEL_PORT", "8001")),
        )


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=40_000)


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    messages: list[ChatMessage] = Field(min_length=1, max_length=32)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)
    max_tokens: int = Field(default=768, ge=1, le=768)
    request_id: str | None = Field(default=None, min_length=1, max_length=100)


class GenerationError(RuntimeError):
    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def _prepare_runtime(settings: ModelSettings) -> None:
    """Constrain caches and respect the caller's explicit one/two-GPU mask."""
    selected = os.getenv("CUDA_VISIBLE_DEVICES", os.getenv("CHOICEAGENT_GPU_IDS", "5"))
    gpu_ids = [value.strip() for value in selected.split(",")]
    if not 1 <= len(gpu_ids) <= 2 or any(
        not value.isascii() or not value.isdigit() for value in gpu_ids
    ):
        raise ValueError("GPU visibility must contain one or two distinct numeric physical IDs")
    gpu_ids = [str(int(value)) for value in gpu_ids]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU visibility must contain one or two distinct numeric physical IDs")
    visibility = ",".join(gpu_ids)
    loaded_torch = sys.modules.get("torch")
    if (
        loaded_torch is not None
        and loaded_torch.cuda.is_initialized()
        and os.environ.get("CUDA_VISIBLE_DEVICES") != visibility
    ):
        raise RuntimeError(
            "CUDA was already initialized outside the required GPU visibility policy"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = visibility
    os.environ["CHOICEAGENT_GPU_IDS"] = visibility
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    directories = {
        "HF_HOME": settings.runtime_dir / "cache" / "huggingface",
        "TORCH_HOME": settings.runtime_dir / "cache" / "torch",
        "XDG_CACHE_HOME": settings.runtime_dir / "cache",
        "TMPDIR": settings.runtime_dir / "tmp",
    }
    for key, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)


class GenerationEngine:
    """A single lazy model, one active decode, and at most two queued calls."""

    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(settings.max_waiting + 1)
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._closing = threading.Event()
        self._active_cancel: threading.Event | None = None
        self._last_error: str | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None and not self._closing.is_set()

    def load(self) -> None:
        """Called only while holding the inference lock or before admission."""
        if self._model is not None:
            return
        _prepare_runtime(self.settings)
        if not MODEL_PATH.is_dir():
            raise GenerationError("MODEL_UNAVAILABLE", "Local model path is missing.", 503)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            expected_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            if not torch.cuda.is_available() or torch.cuda.device_count() != expected_devices:
                raise GenerationError(
                    "MODEL_UNAVAILABLE",
                    "CUDA devices do not match the explicit one/two-GPU mask.",
                    503,
                )
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH, local_files_only=True, trust_remote_code=False
            )
            model: Any = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            model.to("cuda:0")
            model.eval()
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._last_error = None
            LOGGER.info(
                "local_model_loaded model=Qwen3-1.7B visible_gpu_count=%d model_device=cuda:0",
                expected_devices,
            )
        except GenerationError:
            raise
        except Exception as exc:
            self._last_error = type(exc).__name__
            LOGGER.exception("local_model_load_failed error_type=%s", type(exc).__name__)
            raise GenerationError(
                "MODEL_UNAVAILABLE", "Local model could not be loaded.", 503
            ) from exc

    def generate(
        self,
        request: CompletionRequest,
        *,
        deadline: float,
        cancel: threading.Event,
    ) -> dict[str, Any]:
        started = time.monotonic()
        if self._closing.is_set():
            raise GenerationError("MODEL_UNAVAILABLE", "Model service is shutting down.", 503)
        if not self._slots.acquire(blocking=False):
            raise GenerationError("MODEL_BUSY", "Local model queue is full.", 429)
        acquired = False
        try:
            while not acquired:
                self._check_deadline(deadline, cancel)
                acquired = self._lock.acquire(
                    timeout=min(0.1, max(0.001, deadline - time.monotonic()))
                )
            self._active_cancel = cancel
            self._check_deadline(deadline, cancel)
            self.load()
            self._check_deadline(deadline, cancel)
            return self._decode(request, deadline=deadline, cancel=cancel, started=started)
        except GenerationError:
            raise
        except Exception as exc:
            LOGGER.error("local_generation_unexpected_error error_type=%s", type(exc).__name__)
            raise GenerationError(
                "MODEL_UNAVAILABLE", "Local generation could not complete.", 503
            ) from exc
        finally:
            if acquired:
                self._active_cancel = None
                self._lock.release()
            self._slots.release()

    def _check_deadline(self, deadline: float, cancel: threading.Event) -> None:
        if cancel.is_set() or self._closing.is_set():
            raise GenerationError("MODEL_CANCELLED", "Model request was cancelled.", 408)
        if time.monotonic() >= deadline:
            raise GenerationError("MODEL_TIMEOUT", "Model request deadline exceeded.", 504)

    def _decode(
        self,
        request: CompletionRequest,
        *,
        deadline: float,
        cancel: threading.Event,
        started: float,
    ) -> dict[str, Any]:
        from transformers import StoppingCriteria, StoppingCriteriaList

        engine = self

        class DeadlineStop(StoppingCriteria):
            def __call__(self, _input_ids: Any, _scores: Any, **_kwargs: Any) -> bool:
                return time.monotonic() >= deadline or cancel.is_set() or engine._closing.is_set()

        messages = [message.model_dump() for message in request.messages]
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        # The chat template already inserts special tokens; do not add them again.
        encoded = self._tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        if prompt_tokens + request.max_tokens > self.settings.max_context:
            raise GenerationError(
                "MODEL_CONTEXT_LIMIT",
                "Prompt plus output exceeds the 8192 token context limit.",
                422,
            )
        self._check_deadline(deadline, cancel)
        encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        options: dict[str, Any] = {
            "max_new_tokens": min(request.max_tokens, self.settings.max_output),
            "do_sample": request.temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
            "stopping_criteria": StoppingCriteriaList([DeadlineStop()]),
            "use_cache": True,
        }
        if request.temperature > 0:
            options.update(temperature=request.temperature, top_p=0.9, top_k=20)
        try:
            with self._torch.inference_mode():
                output = self._model.generate(**encoded, **options)
            self._check_deadline(deadline, cancel)
            new_tokens = output[0, prompt_tokens:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            completion_tokens = int(new_tokens.shape[0])
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            LOGGER.info(
                "local_generation_completed prompt_tokens=%d completion_tokens=%d elapsed_ms=%.2f",
                prompt_tokens,
                completion_tokens,
                elapsed_ms,
            )
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "Qwen3-1.7B-local",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "length"
                        if completion_tokens >= request.max_tokens
                        else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "elapsed_ms": elapsed_ms,
            }
        except GenerationError:
            raise
        except self._torch.cuda.OutOfMemoryError as exc:
            self._torch.cuda.empty_cache()
            raise GenerationError(
                "MODEL_UNAVAILABLE", "Insufficient GPU memory for generation.", 503
            ) from exc
        except Exception as exc:
            LOGGER.exception("local_generation_failed error_type=%s", type(exc).__name__)
            raise GenerationError("MODEL_UNAVAILABLE", "Local generation failed.", 503) from exc
        finally:
            del encoded

    def close(self) -> None:
        self._closing.set()
        if self._active_cancel is not None:
            self._active_cancel.set()


def create_app(
    settings: ModelSettings | None = None,
    *,
    engine: GenerationEngine | None = None,
) -> FastAPI:
    resolved = settings or ModelSettings.from_env()
    generator = engine or GenerationEngine(resolved)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        _prepare_runtime(resolved)
        if resolved.preload:
            await asyncio.to_thread(generator.load)
        yield
        generator.close()

    application = FastAPI(title="ChoiceAgent local Qwen3 generator", lifespan=lifespan)
    application.state.engine = generator

    @application.exception_handler(GenerationError)
    async def generation_error(_request: Request, exc: GenerationError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status, content={"error": {"code": exc.code, "message": str(exc)}}
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default includes the rejected input (possibly the whole prompt).
        return JSONResponse(
            status_code=422,
            content={
                "error": {"code": "MODEL_INVALID_REQUEST", "message": "Invalid completion request."}
            },
        )

    @application.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ready" if generator.ready else "not_loaded",
            "ready": generator.ready,
            "model": "Qwen3-1.7B-local",
            "model_path": str(MODEL_PATH),
            "max_context_tokens": resolved.max_context,
            "max_output_tokens": resolved.max_output,
            "max_waiting": resolved.max_waiting,
            "visible_gpu": os.getenv("CUDA_VISIBLE_DEVICES", os.getenv("CHOICEAGENT_GPU_IDS", "5")),
            "model_device": "cuda:0",
            "thinking": False,
        }

    @application.post("/v1/chat/completions")
    async def completion(
        body: CompletionRequest,
        request: Request,
        x_request_timeout: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        timeout = resolved.request_timeout
        if x_request_timeout is not None:
            try:
                timeout = float(x_request_timeout)
            except ValueError as exc:
                raise GenerationError(
                    "MODEL_INVALID_REQUEST", "Invalid request timeout.", 422
                ) from exc
            if not math.isfinite(timeout) or timeout <= 0:
                raise GenerationError("MODEL_INVALID_REQUEST", "Invalid request timeout.", 422)
        timeout = min(timeout, resolved.max_request_timeout)
        cancel = threading.Event()
        deadline = time.monotonic() + timeout
        task = asyncio.create_task(
            asyncio.to_thread(generator.generate, body, deadline=deadline, cancel=cancel)
        )
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.05)
                if await request.is_disconnected():
                    cancel.set()
            return await task
        except asyncio.CancelledError:
            cancel.set()

            # The stopping criterion releases the inference lock on the next token.
            def consume_result(done: asyncio.Task[dict[str, Any]]) -> None:
                if not done.cancelled():
                    with contextlib.suppress(Exception):
                        done.result()

            task.add_done_callback(consume_result)
            raise

    return application


app = create_app()


def main() -> None:
    import uvicorn

    settings = ModelSettings.from_env()
    _prepare_runtime(settings)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        workers=1,
        limit_concurrency=16,
        backlog=16,
        log_level="info",
    )


if __name__ == "__main__":
    main()
