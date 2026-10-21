"""FastAPI application factory and HTTP-only error/security boundaries."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from choiceagent import __version__
from choiceagent.agent import AgentService, GenerationProvider
from choiceagent.catalog import CatalogRepository
from choiceagent.config import Settings
from choiceagent.errors import AppError
from choiceagent.localization import category_label
from choiceagent.logging_config import configure_logging
from choiceagent.model_client import LocalModelClient
from choiceagent.retrieval import SearchEngine
from choiceagent.schemas import (
    ChatRequest,
    CompareRequest,
    FeedbackRequest,
    SearchRequest,
    SearchResponse,
)
from choiceagent.services import DiscoveryService
from choiceagent.state import StateStore

LOGGER = logging.getLogger(__name__)


def restrict_gpus(gpu_ids: str) -> None:
    """Respect a narrower caller mask, but reject visibility wider than two cards."""
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", gpu_ids)
    ids = [value.strip() for value in selected.split(",") if value.strip()]
    if len(ids) > 2 or any(not value.isdigit() for value in ids):
        raise ValueError("Use at most two explicit numeric GPU IDs")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class RequestBoundaryMiddleware(BaseHTTPMiddleware):
    """Bound request body size and record status/timing without body contents."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id
        started = time.perf_counter()
        if request.method in {"POST", "PUT", "PATCH"}:
            chunks: list[bytes] = []
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > 65536:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": "REQUEST_TOO_LARGE",
                                "message": "请求内容过大。",
                                "trace_id": trace_id,
                            }
                        },
                    )
                chunks.append(chunk)
            request._body = b"".join(
                chunks
            )  # Starlette's cached request wrapper replays this body.
        response = await call_next(request)
        response.headers["X-Request-ID"] = trace_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        LOGGER.info(
            "http_request",
            extra={
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        )
        return response


def create_app(
    settings: Settings | None = None,
    *,
    discovery: DiscoveryService | None = None,
    provider: GenerationProvider | None = None,
) -> FastAPI:
    """Build an injectable application; heavy models load only inside their services."""
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal discovery, provider
        restrict_gpus(cfg.gpu_ids)
        configure_logging(cfg.log_dir)
        if discovery is None:
            repository = CatalogRepository(cfg.catalog_path, cfg.catalog_db)
            repository.initialize()
            engine = SearchEngine(
                repository, cfg.index_dir, cfg.embedding_model_path, device=cfg.embedding_device
            )
            discovery = DiscoveryService(
                engine, StateStore(cfg.state_db, max_sessions=cfg.max_sessions)
            )
        owns_provider = provider is None
        if provider is None:
            provider = LocalModelClient(cfg.model_base_url, timeout=cfg.model_timeout_seconds)
        fingerprint = hashlib.sha256()
        for source in sorted(Path(__file__).parent.glob("*.py")):
            fingerprint.update(source.name.encode())
            fingerprint.update(source.read_bytes())
        app.state.source_fingerprint = fingerprint.hexdigest()
        app.state.discovery = discovery
        app.state.agent = AgentService(discovery, provider, cfg)
        yield
        discovery.engine.close()
        if owns_provider and isinstance(provider, LocalModelClient):
            provider.close()

    app = FastAPI(
        title="ChoiceAgent",
        version=__version__,
        lifespan=lifespan,
        description="Evidence-grounded local product discovery MVP. Historical USD catalog.",
    )
    app.add_middleware(RequestBoundaryMiddleware)

    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "trace_id": getattr(request.state, "trace_id", ""),
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Do not include rejected user values or full bodies in public errors.
        fields = [".".join(str(part) for part in item["loc"]) for item in exc.errors()]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "INVALID_ARGUMENT",
                    "message": "请求参数不符合要求。",
                    "fields": fields,
                    "trace_id": getattr(request.state, "trace_id", ""),
                }
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", "")
        LOGGER.exception("unexpected_failure", extra={"trace_id": trace_id})
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "服务暂时无法完成请求。",
                    "trace_id": trace_id,
                }
            },
        )

    def require_key(request: Request) -> None:
        if cfg.api_key is not None:
            authorization = request.headers.get("Authorization", "")
            supplied = authorization.removeprefix("Bearer ")
            if not authorization.startswith("Bearer ") or not hmac.compare_digest(
                supplied, cfg.api_key
            ):
                raise AppError("UNAUTHORIZED", "需要有效访问密钥。", 401)

    dependencies: list[Any] = [Depends(require_key)]

    def svc(request: Request) -> DiscoveryService:
        return request.app.state.discovery

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        service = svc(request)
        return {
            "status": "ok",
            "version": __version__,
            "source_fingerprint": request.app.state.source_fingerprint,
            "catalog_count": service.engine.repository.stats()["count"],
            "dense_ready": service.engine.dense_ready,
        }

    @app.post("/v1/sessions", dependencies=dependencies, status_code=201)
    def create_session(request: Request) -> dict[str, Any]:
        state = svc(request).state.create_session()
        return {"session_id": state["session_id"], "profile_version": state["profile_version"]}

    @app.get("/v1/sessions/{session_id}", dependencies=dependencies)
    def get_session(session_id: str, request: Request) -> dict[str, Any]:
        return svc(request).state.get_session(session_id)

    @app.get("/v1/catalog/stats", dependencies=dependencies)
    def stats(request: Request) -> dict[str, Any]:
        result = svc(request).engine.repository.stats()
        return {
            **result,
            "category_labels": {name: category_label(name) for name in result["categories"]},
        }

    @app.post("/v1/search", response_model=SearchResponse, dependencies=dependencies)
    def search(payload: SearchRequest, request: Request) -> dict[str, Any]:
        return svc(request).search(payload)

    @app.post("/v1/compare", dependencies=dependencies)
    def compare(payload: CompareRequest, request: Request) -> dict[str, Any]:
        return svc(request).compare(payload)

    @app.post("/v1/feedback", dependencies=dependencies)
    def feedback(payload: FeedbackRequest, request: Request) -> dict[str, Any]:
        return svc(request).state.feedback(
            payload.session_id, payload.event_id, payload.item_id, payload.action
        )

    @app.post("/v1/chat", dependencies=dependencies)
    def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
        return request.app.state.agent.chat(payload)

    @app.get("/v1/traces/{trace_id}", dependencies=dependencies)
    def get_trace(
        trace_id: str, request: Request, session_id: str = Query(min_length=36, max_length=36)
    ) -> dict[str, Any]:
        return svc(request).state.get_trace(trace_id, session_id)

    return app


# Uvicorn --factory imports create_app without allocating a model at module import.
