"""Shared business service for normal search and model-selected tools."""

from __future__ import annotations

import time
import uuid
from typing import Any

from choiceagent.errors import AppError
from choiceagent.localization import field_label, localize_value
from choiceagent.retrieval import SearchEngine, product_card
from choiceagent.schemas import CompareRequest, Constraints, SearchRequest
from choiceagent.state import StateStore


class DiscoveryService:
    def __init__(self, engine: SearchEngine, state: StateStore) -> None:
        self.engine = engine
        self.state = state

    @staticmethod
    def check_deadline(deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise AppError("TURN_TIMEOUT", "本次处理已超时，请重试。", 504)

    def search(self, request: SearchRequest, *, deadline: float | None = None) -> dict[str, Any]:
        self.check_deadline(deadline)
        state = self.state.get_session(request.session_id)
        started = time.perf_counter()
        result = self.engine.search(
            request.query,
            request.constraints,
            limit=request.limit,
            mode=request.mode,
            liked_item_ids=state["liked_item_ids"],
            disliked_item_ids=state["disliked_item_ids"],
            personalize=request.personalize,
        )
        self.check_deadline(deadline)
        self.state.record_search(
            request.session_id,
            [item["item_id"] for item in result["items"]],
            request.query,
            request.constraints.model_dump(),
        )
        result.update(
            trace_id=str(uuid.uuid4()),
            session_id=request.session_id,
            profile_version=state["profile_version"],
        )
        result.setdefault("catalog_version", self.engine.repository.stats()["catalog_version"])
        result.setdefault("timing_ms", {})["total"] = round(
            (time.perf_counter() - started) * 1000, 3
        )
        self.state.save_trace(
            request.session_id,
            {
                "trace_id": result["trace_id"],
                "kind": "search",
                "query": request.query,
                "constraints": request.constraints.model_dump(),
                "mode": request.mode,
                "personalize": request.personalize,
                "profile_version": state["profile_version"],
                "status": result["status"],
                "item_ids": [x["item_id"] for x in result["items"]],
                "timing_ms": result["timing_ms"],
                "warnings": result["warnings"],
            },
        )
        return result

    def compare(self, request: CompareRequest, *, deadline: float | None = None) -> dict[str, Any]:
        self.check_deadline(deadline)
        state = self.state.get_session(request.session_id)
        if len(set(request.item_ids)) != len(request.item_ids):
            raise AppError("INVALID_ARGUMENT", "请选择不同商品进行比较。", 422)
        products = []
        for item_id in request.item_ids:
            if item_id not in state["visible_item_ids"]:
                raise AppError("ITEM_NOT_VISIBLE", "只能比较本会话已经展示的商品。", 403)
            product = self.engine.repository.get(item_id)
            if product is None:
                raise AppError("UNKNOWN_ITEM", "商品不属于当前目录。", 404)
            products.append(product)
        cards = [product_card(product, rank=i + 1) for i, product in enumerate(products)]
        rows: list[dict[str, Any]] = []
        for dimension in dict.fromkeys(request.dimensions):
            if len(dimension) > 200:
                raise AppError("INVALID_ARGUMENT", "比较维度过长。", 422)
            values: list[dict[str, Any]] = []
            for product, card in zip(products, cards, strict=True):
                field = dimension
                if dimension in {"price_usd", "brand", "category", "title", "features"}:
                    value = getattr(product, dimension)
                    if dimension == "features" and not value:
                        value = None
                else:
                    attribute_key = dimension.removeprefix("attributes.")
                    matching = next(
                        (
                            key
                            for key in product.attributes
                            if key.casefold() == attribute_key.casefold()
                        ),
                        None,
                    )
                    value = product.attributes.get(matching) if matching else None
                    field = f"attributes.{matching}" if matching else dimension
                display = card["display"]
                translated = localize_value(field, value)
                if value is not None:
                    if dimension in {"title", "category"}:
                        translated = display[dimension]
                    elif dimension == "features":
                        translated = display["features"] or "功能描述请查看英文原文"
                    elif field.startswith("attributes."):
                        translated = display["attributes"].get(field_label(field), translated)
                values.append(
                    {
                        "item_id": product.item_id,
                        "value": value,
                        "display_value": translated,
                        "evidence_id": f"{product.item_id}:{field}" if value is not None else None,
                    }
                )
            rows.append(
                {"dimension": dimension, "label_zh": field_label(dimension), "values": values}
            )
        self.check_deadline(deadline)
        result = {
            "trace_id": str(uuid.uuid4()),
            "items": cards,
            "rows": rows,
            "warnings": ["historical_price_usd", "missing_attributes_are_unknown"],
        }
        self.state.save_trace(
            request.session_id,
            {
                "trace_id": result["trace_id"],
                "kind": "compare",
                "item_ids": request.item_ids,
                "dimensions": request.dimensions,
            },
        )
        return result

    def constraints(self, session_id: str) -> Constraints:
        return Constraints.model_validate(self.state.get_session(session_id)["constraints"])
