#!/usr/bin/env python3
"""Fetch only a bounded prefix of the official Office_Products metadata."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from choiceagent.ingestion import prepare_catalog


def main() -> None:
    """Run the deterministic catalog preparation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--limit", type=int, default=3000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = prepare_catalog(args.root.resolve(), limit=args.limit)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
