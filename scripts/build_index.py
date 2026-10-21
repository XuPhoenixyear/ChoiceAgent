#!/usr/bin/env python3
"""Build the exact normalized FAISS catalog index from a local embedding model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from choiceagent.catalog import CatalogRepository
from choiceagent.retrieval import build_dense_index


def main() -> None:
    """Prepare SQLite and dense artifacts; GPU visibility is launcher-controlled."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--model", type=Path, default=Path("/datas/wuxi/Projects/datas/models/Qwen3-Embedding-4B")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = args.root.resolve()
    repository = CatalogRepository(
        root / "data/processed/catalog.jsonl", root / "runtime/catalog.sqlite3"
    )
    repository.initialize()
    manifest = build_dense_index(
        repository,
        root / "artifacts/indexes",
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
