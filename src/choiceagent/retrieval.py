"""Exact dense + BM25 retrieval with bounded, relevance-first personalization."""

from __future__ import annotations

import gc
import json
import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from choiceagent.catalog import (
    CatalogRepository,
    lexical_tokens,
    matches_lexical_coverage,
    matches_query_anchor,
    product_text,
    rewrite_query,
)
from choiceagent.ingestion import sha256_file
from choiceagent.localization import display_for
from choiceagent.schemas import Constraints, Product

LOGGER = logging.getLogger(__name__)
QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
MAX_LENGTH = 384
# A conservative development-set gate, not a calibrated relevance probability.
# Keep this parameter separate from measured retrieval quality claims.
MIN_DENSE_COSINE = 0.42


def model_signature(model_path: Path) -> dict[str, Any]:
    """Fingerprint configs and record weight file sizes without rehashing gigabytes."""
    files: dict[str, Any] = {}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        path = model_path / name
        if path.exists():
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    weight_files = {
        path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in sorted(model_path.glob("*.safetensors"))
    }
    return {
        "path": str(model_path.resolve()),
        "configuration_files": files,
        "weights_size_mtime": weight_files,
    }


class LocalSemanticEncoder:
    """Lazy Qwen3 embedding encoder using last-token pooling and unit vectors.

    CUDA visibility is set by the process launcher, never widened here. The
    deployment uses physical GPU 5, exposed to this process as cuda:0.
    """

    def __init__(self, model_path: Path, device: str = "cpu", max_length: int = MAX_LENGTH) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.max_length = max_length
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local embedding model does not exist: {self.model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, padding_side="left"
        )
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._model = (
            AutoModel.from_pretrained(
                self.model_path,
                local_files_only=True,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
            .to(self.device)
            .eval()
        )
        LOGGER.info("Loaded local embedding model on %s", self.device)

    def encode(
        self, texts: Sequence[str], *, query: bool = False, batch_size: int = 8
    ) -> np.ndarray:
        """Encode a sequence under one inference lock; no network access occurs."""
        import torch
        import torch.nn.functional as functional

        if not texts:
            raise ValueError("At least one embedding text is required.")
        with self._lock:
            self._load()
            vectors: list[np.ndarray] = []
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                if query:
                    batch = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}" for text in batch]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                with torch.inference_mode():
                    hidden = self._model(**encoded).last_hidden_state
                    mask = encoded["attention_mask"]
                    if bool((mask[:, -1] == 1).all()):
                        pooled = hidden[:, -1]
                    else:
                        indices = mask.sum(dim=1) - 1
                        pooled = hidden[
                            torch.arange(hidden.shape[0], device=hidden.device), indices
                        ]
                    normalized = functional.normalize(pooled.float(), p=2, dim=1)
                vectors.append(normalized.cpu().numpy())
                if len(texts) > batch_size:
                    LOGGER.info(
                        "Embedded %d/%d catalog products",
                        min(start + batch_size, len(texts)),
                        len(texts),
                    )
            return np.ascontiguousarray(np.concatenate(vectors), dtype=np.float32)

    def close(self) -> None:
        """Release model ownership; allocator cache release is best-effort."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            gc.collect()
            if self.device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()


def build_dense_index(
    repository: CatalogRepository,
    index_dir: Path,
    model_path: Path,
    *,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """Build the exact cosine baseline and its content/identity manifest."""
    import faiss

    products = repository.all_products()
    ids = [product.item_id for product in products]
    index_dir.mkdir(parents=True, exist_ok=True)
    encoder = LocalSemanticEncoder(model_path, device=device, max_length=max_length)
    started = time.perf_counter()
    try:
        vectors = encoder.encode(
            [product_text(product) for product in products], batch_size=batch_size
        )
    finally:
        encoder.close()
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_temporary = index_dir / "catalog.faiss.tmp"
    vectors_temporary = index_dir / "embeddings.npy.tmp"
    ids_temporary = index_dir / "item_ids.json.tmp"
    faiss.write_index(index, str(index_temporary))
    with vectors_temporary.open("wb") as stream:
        np.save(stream, vectors)
    ids_temporary.write_text(json.dumps(ids) + "\n", encoding="utf-8")
    index_temporary.replace(index_dir / "catalog.faiss")
    vectors_temporary.replace(index_dir / "embeddings.npy")
    ids_temporary.replace(index_dir / "item_ids.json")
    manifest = {
        "schema_version": 1,
        "index_type": "IndexFlatIP",
        "metric": "cosine_on_unit_vectors",
        "count": len(ids),
        "dimension": vectors.shape[1],
        "dtype": "float32",
        "catalog_version": products[0].catalog_version,
        "catalog_sha256": sha256_file(repository.catalog_path),
        "item_ids_sha256": sha256_file(index_dir / "item_ids.json"),
        "index_sha256": sha256_file(index_dir / "catalog.faiss"),
        "embeddings_sha256": sha256_file(index_dir / "embeddings.npy"),
        "model": model_signature(model_path),
        "pooling": "last_token",
        "normalized": True,
        "max_length": max_length,
        "batch_size": batch_size,
        "query_instruction": QUERY_INSTRUCTION,
        "document_text": "title category category_path brand features description attributes",
        "build_elapsed_seconds": round(time.perf_counter() - started, 3),
        "device": device,
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def product_card(
    product: Product,
    *,
    rank: int = 0,
    score: float = 0.0,
    route_ranks: dict[str, int] | None = None,
    reason_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Construct a card whose factual values have deterministic source evidence."""
    facts: list[tuple[str, Any]] = [
        ("title", product.title),
        ("price_usd", product.price_usd),
        ("brand", product.brand),
        ("category", product.category),
        ("features", product.features if product.features else None),
    ]
    facts.extend((f"features.{index}", value) for index, value in enumerate(product.features))
    facts.extend((f"attributes.{key}", value) for key, value in product.attributes.items())
    evidence = [
        {
            "evidence_id": f"{product.item_id}:{field}",
            "item_id": product.item_id,
            "field": field,
            "value": value,
            "source_locator": product.source_locator,
        }
        for field, value in facts
        if value is not None
    ]
    return {
        "item_id": product.item_id,
        "title": product.title,
        "category": product.category,
        "brand": product.brand,
        "price_usd": product.price_usd,
        "features": product.features,
        "attributes": product.attributes,
        "display": display_for(product),
        "rank": rank,
        "score": float(score),
        "route_ranks": route_ranks or {},
        "reason_codes": reason_codes or [],
        "evidence": evidence,
        "missing_attributes": [
            field for field in ("price_usd", "brand") if getattr(product, field) is None
        ],
    }


class SearchEngine:
    """Deterministic shared ranking backend; it never mutates session memory."""

    def __init__(
        self,
        repository: CatalogRepository,
        index_dir: Path,
        embedding_model_path: Path | None,
        device: str = "cpu",
    ) -> None:
        self.repository = repository
        self.index_dir = Path(index_dir)
        self.embedding_model_path = Path(embedding_model_path) if embedding_model_path else None
        self.device = device
        self._index: Any = None
        self._vectors: np.ndarray | None = None
        self._ids: list[str] = []
        self._id_positions: dict[str, int] = {}
        self._encoder: LocalSemanticEncoder | None = None
        self._encoder_lock = threading.Lock()
        self._dense_error: str | None = None
        self._manifest: dict[str, Any] = {}
        self._load_index()

    @property
    def dense_ready(self) -> bool:
        """Whether a validated dense artifact is available (without model loading)."""
        return self._index is not None and self.embedding_model_path is not None

    def _load_index(self) -> None:
        import faiss

        try:
            manifest = json.loads((self.index_dir / "manifest.json").read_text(encoding="utf-8"))
            if manifest["catalog_sha256"] != sha256_file(self.repository.catalog_path):
                raise ValueError("Dense catalog checksum mismatch; rebuild the index.")
            for filename, checksum_key in (
                ("catalog.faiss", "index_sha256"),
                ("item_ids.json", "item_ids_sha256"),
                ("embeddings.npy", "embeddings_sha256"),
            ):
                if sha256_file(self.index_dir / filename) != manifest[checksum_key]:
                    raise ValueError(f"Dense artifact checksum mismatch: {filename}")
            if (
                self.embedding_model_path is not None
                and model_signature(self.embedding_model_path) != manifest["model"]
            ):
                raise ValueError("Dense encoder fingerprint changed; rebuild the index.")
            ids = json.loads((self.index_dir / "item_ids.json").read_text(encoding="utf-8"))
            index = faiss.read_index(str(self.index_dir / "catalog.faiss"))
            vectors = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
            catalog_ids = {product.item_id for product in self.repository.all_products()}
            if (
                len(ids) != len(set(ids))
                or set(ids) != catalog_ids
                or index.ntotal != len(ids)
                or vectors.shape != (len(ids), index.d)
                or manifest["dimension"] != index.d
                or manifest["count"] != len(ids)
            ):
                raise ValueError("Dense artifact ID/count/dimension mismatch.")
            if (
                not isinstance(index, faiss.IndexFlatIP)
                or index.metric_type != faiss.METRIC_INNER_PRODUCT
            ):
                raise ValueError("Dense index must be the exact inner-product baseline.")
            if (
                manifest["pooling"] != "last_token"
                or manifest["normalized"] is not True
                or manifest["query_instruction"] != QUERY_INSTRUCTION
                or not 1 <= manifest["max_length"] <= 8192
                or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
            ):
                raise ValueError("Dense encoder contract or unit-vector invariant mismatch.")
            self._ids, self._index, self._vectors = ids, index, vectors
            self._id_positions = {item_id: position for position, item_id in enumerate(ids)}
            self._manifest = manifest
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            self._dense_error = str(error)
            LOGGER.warning("Dense retrieval unavailable; lexical fallback: %s", error)

    def _dense_results(self, query: str) -> list[tuple[str, float]]:
        if not self.dense_ready or self.embedding_model_path is None:
            raise RuntimeError(self._dense_error or "Dense model/index is not configured.")
        with self._encoder_lock:
            if self._encoder is None:
                self._encoder = LocalSemanticEncoder(
                    self.embedding_model_path, self.device, max_length=self._manifest["max_length"]
                )
            query_vector = self._encoder.encode([query], query=True)
        distances, positions = self._index.search(query_vector, len(self._ids))
        return [
            (self._ids[int(position)], float(score))
            for position, score in zip(positions[0], distances[0], strict=True)
            if position >= 0
        ]

    def search(
        self,
        query: str,
        constraints: Constraints,
        limit: int = 5,
        mode: str = "hybrid",
        liked_item_ids: Sequence[str] = (),
        disliked_item_ids: Sequence[str] = (),
        personalize: bool = True,
    ) -> dict[str, Any]:
        """Filter first, fuse route ranks, and apply only a bounded soft preference."""
        if mode not in {"hybrid", "lexical", "dense"}:
            raise ValueError("Unsupported retrieval mode.")
        if not query.strip() or not 1 <= limit <= 10:
            raise ValueError("Search requires a nonempty query and limit in [1, 10].")
        started = time.perf_counter()
        eligible = {
            product.item_id: product for product in self.repository.filter_products(constraints)
        }
        warnings: list[str] = []
        timing: dict[str, float] = {}
        base: dict[str, Any] = {
            "items": [],
            "applied_constraints": constraints.model_dump(),
            "total_eligible": len(eligible),
            "warnings": warnings,
            "timing_ms": timing,
        }
        if not eligible:
            return {
                **base,
                "status": "no_match",
                "timing_ms": {"total": round((time.perf_counter() - started) * 1000, 3)},
            }
        relevant_ids = {
            item_id for item_id, product in eligible.items() if matches_query_anchor(product, query)
        }
        routes: dict[str, list[str]] = {}
        lexical_started = time.perf_counter()
        lexical_results = self.repository.lexical_search(query, limit=5000)
        lexical_results = [
            (item_id, score)
            for item_id, score in lexical_results
            if item_id in relevant_ids and matches_lexical_coverage(eligible[item_id], query)
        ]
        if mode in {"lexical", "hybrid"}:
            routes["lexical"] = [
                item_id for item_id, _ in lexical_results if item_id in relevant_ids
            ][:200]
        timing["lexical"] = round((time.perf_counter() - lexical_started) * 1000, 3)
        degraded = False
        if mode in {"hybrid", "dense"}:
            dense_started = time.perf_counter()
            try:
                dense_results = [
                    (item_id, score)
                    for item_id, score in self._dense_results(query)
                    if item_id in relevant_ids
                ]
                # An explicit floor and relative gate prevent nearest-neighbor
                # retrieval from always inventing a match for unrelated queries.
                threshold = (
                    max(MIN_DENSE_COSINE, dense_results[0][1] - 0.12) if dense_results else 1.0
                )
                routes["dense"] = [
                    item_id for item_id, score in dense_results if score >= threshold
                ][:200]
            except (OSError, ValueError, RuntimeError, ImportError) as error:
                degraded = True
                LOGGER.warning("Dense query failed; using lexical fallback: %s", error)
                warnings.append("语义检索暂不可用，已使用词面检索。")
                routes["lexical"] = [
                    item_id for item_id, _ in lexical_results if item_id in relevant_ids
                ][:200]
            timing["dense"] = round((time.perf_counter() - dense_started) * 1000, 3)
        route_ranks: dict[str, dict[str, int]] = {}
        scores: dict[str, float] = {}
        for route, item_ids in routes.items():
            for rank, item_id in enumerate(item_ids, start=1):
                route_ranks.setdefault(item_id, {})[route] = rank
                scores[item_id] = scores.get(item_id, 0.0) + 61.0 / (60.0 + rank)
        disliked = set(disliked_item_ids) if personalize else set()
        for item_id in disliked:
            scores.pop(item_id, None)
        personalized: set[str] = set()
        if personalize and liked_item_ids and scores:
            liked_products = [
                product
                for item_id in liked_item_ids
                if (product := self.repository.get(item_id)) is not None
            ]
            valid_likes = [product.item_id for product in liked_products]
            if valid_likes:
                if self._vectors is not None:
                    positions = [
                        self._id_positions[item_id]
                        for item_id in valid_likes
                        if item_id in self._id_positions
                    ]
                    if positions:
                        profile = np.asarray(self._vectors[positions]).mean(axis=0)
                        norm = float(np.linalg.norm(profile))
                        if norm > 0:
                            profile /= norm
                            for item_id in scores:
                                similarity = max(
                                    0.0,
                                    float(
                                        np.dot(self._vectors[self._id_positions[item_id]], profile)
                                    ),
                                )
                                scores[item_id] += 0.15 * similarity
                                personalized.add(item_id)
                else:
                    liked_terms = set().union(
                        *(set(lexical_tokens(product_text(product))) for product in liked_products)
                    )
                    for item_id in scores:
                        terms = set(lexical_tokens(product_text(eligible[item_id])))
                        similarity = len(terms & liked_terms) / max(1, len(terms | liked_terms))
                        scores[item_id] += 0.15 * similarity
                        if similarity:
                            personalized.add(item_id)
        ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))[:limit]
        cards = []
        for rank, item_id in enumerate(ordered, start=1):
            reasons = [f"{route}_match" for route in route_ranks[item_id]]
            if item_id in personalized:
                reasons.append("explicit_like_content_similarity")
            if constraints.max_price_usd is not None or constraints.min_price_usd is not None:
                reasons.append("historical_usd_price_verified")
            cards.append(
                product_card(
                    eligible[item_id],
                    rank=rank,
                    score=scores[item_id],
                    route_ranks=route_ranks[item_id],
                    reason_codes=reasons,
                )
            )
        if rewrite_query(query) != query.casefold():
            warnings.append("中文词面检索使用受控词表改写；语义检索保留原始查询。")
        if not cards:
            warnings.append("未检索到足够相关的候选；这不证明目录中不存在匹配商品。")
        timing["total"] = round((time.perf_counter() - started) * 1000, 3)
        return {
            **base,
            "items": cards,
            "status": "retrieval_incomplete" if not cards else ("degraded" if degraded else "ok"),
        }

    def close(self) -> None:
        """Release the lazy local encoder when the application shuts down."""
        if self._encoder is not None:
            self._encoder.close()
