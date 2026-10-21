"""Immutable catalog repository with safe lexical retrieval and strict filters."""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from choiceagent.schemas import Constraints, Product

LOGGER = logging.getLogger(__name__)

# Deliberately small, auditable bridge for an English-source / Chinese-UI demo.
# Dense retrieval handles paraphrases outside this vocabulary; it is not a
# general-purpose translator. Phrases are longest-first to avoid partial overlap.
CHINESE_TERMS: dict[str, str] = {
    "桌面收纳架": "desk organizer",
    "桌面收纳": "desk organizer",
    "桌面整理": "desk organizer",
    "文件收纳": "file organizer",
    "文件整理": "file organizer",
    "显示器支架": "monitor stand",
    "笔记本支架": "laptop stand",
    "笔记本电脑": "laptop",
    "无线键盘": "wireless keyboard",
    "无线鼠标": "wireless mouse",
    "打印机": "printer",
    "打印纸": "printer paper",
    "复印纸": "copy paper",
    "办公椅": "office chair",
    "订书机": "stapler",
    "订书钉": "staples",
    "标签打印": "label printer",
    "标签纸": "labels",
    "便利贴": "sticky notes",
    "记事本": "notebook",
    "笔记本": "notebook",
    "收纳盒": "storage box",
    "收纳架": "organizer",
    "文件夹": "folder",
    "文件柜": "file cabinet",
    "回形针": "paper clips",
    "橡皮筋": "rubber bands",
    "橡皮擦": "eraser",
    "圆珠笔": "ballpoint pen",
    "中性笔": "gel pen",
    "荧光笔": "highlighter",
    "碎纸机": "shredder",
    "计算器": "calculator",
    "鼠标垫": "mouse pad",
    "显示器": "monitor",
    "键盘": "keyboard",
    "鼠标": "mouse",
    "台灯": "desk lamp",
    "胶带": "tape",
    "剪刀": "scissors",
    "白板": "whiteboard",
    "铅笔": "pencil",
    "钢笔": "fountain pen",
    "书立": "bookend",
    "笔筒": "pencil holder",
    "桌垫": "desk pad",
    "金属": "metal",
    "木质": "wood",
    "木制": "wood",
    "塑料": "plastic",
    "竹制": "bamboo",
    "黑色": "black",
    "白色": "white",
    "蓝色": "blue",
    "红色": "red",
    "绿色": "green",
    "透明": "clear",
    "无线": "wireless",
    "可调节": "adjustable",
    "办公": "office",
    "书桌": "desk",
    "桌面": "desk",
    "文件": "file",
    "收纳": "organizer",
    "纸张": "paper",
}
_ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "find",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "the",
    "to",
    "want",
    "with",
    "would",
    "show",
    "under",
    "usd",
    "dollar",
    "dollars",
    "budget",
}


_PRODUCT_ANCHORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("printer paper", ("paper",)),
    ("copy paper", ("paper",)),
    ("desk organizer", ("organizer", "organiser", "storage", "caddy", "holder", "shelf")),
    ("file organizer", ("organizer", "organiser", "file", "storage")),
    ("office chair", ("chair", "seat")),
    ("monitor stand", ("stand", "riser", "mount")),
    ("laptop stand", ("stand", "riser", "mount")),
    ("mouse pad", ("pad", "mat")),
    ("wireless mouse", ("wireless mouse", "mouse wireless")),
    ("wireless keyboard", ("wireless keyboard", "keyboard wireless")),
    ("sticky notes", ("note", "memo")),
    ("pencil holder", ("holder", "organizer", "cup")),
)


def rewrite_query(query: str) -> str:
    """Expand known Chinese product terms into English for lexical retrieval."""
    rewritten = query.casefold()
    for phrase in sorted(CHINESE_TERMS, key=len, reverse=True):
        rewritten = rewritten.replace(phrase, f" {CHINESE_TERMS[phrase]} ")
    return re.sub(r"\s+", " ", rewritten).strip()


def lexical_tokens(query: str) -> list[str]:
    """Return bounded literal tokens, never arbitrary SQLite FTS syntax."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{0,49}", rewrite_query(query))
    return list(dict.fromkeys(word for word in words if word not in _ENGLISH_STOPWORDS))[:20]


def product_text(product: Product) -> str:
    """Serialize only known catalog facts for retrieval and matching."""
    return " ".join(
        [
            product.title,
            product.category,
            " ".join(product.category_path),
            product.brand or "",
            " ".join(product.features),
            product.description,
            " ".join(f"{key} {value}" for key, value in product.attributes.items()),
        ]
    )


def matches_query_anchor(product: Product, query: str) -> bool:
    """Gate known compound product types using title/category evidence.

    This narrow safeguard prevents generic words like `office` from dominating
    a compound query such as `office chair`. Unknown types retain dense recall.
    """
    rewritten = rewrite_query(query)
    identity_text = " ".join([product.title, *product.category_path]).casefold()
    # Accessory titles can mention a supported device ("pad for wireless
    # mouse"). They are not evidence that the item itself is that device.
    if "wireless mouse" in rewritten and any(
        accessory in identity_text for accessory in ("mouse pad", "mousepad", "wrist rest")
    ):
        return False
    if "wireless keyboard" in rewritten and any(
        accessory in identity_text for accessory in ("wrist rest", "keyboard cover", "keyboard pad")
    ):
        return False
    for phrase, anchors in _PRODUCT_ANCHORS:
        if phrase in rewritten:
            return any(anchor in identity_text for anchor in anchors)
    return True


def matches_lexical_coverage(product: Product, query: str) -> bool:
    """Reject long OR queries matched only by an incidental single source word."""
    tokens = lexical_tokens(query)
    if len(tokens) < 3:
        return True
    text = product_text(product).casefold()
    return sum(token in text for token in tokens) >= 2


def matches_constraints(product: Product, constraints: Constraints) -> bool:
    """Apply hard constraints before candidate ranking, including unknown price."""
    price = product.price_usd
    if constraints.max_price_usd is not None and (
        price is None or price > constraints.max_price_usd
    ):
        return False
    if constraints.min_price_usd is not None and (
        price is None or price < constraints.min_price_usd
    ):
        return False
    if product.brand and product.brand.casefold() in {
        value.casefold() for value in constraints.excluded_brands
    }:
        return False
    text = product_text(product).casefold()
    if constraints.category:
        category_text = " ".join([product.category, *product.category_path]).casefold()
        if rewrite_query(constraints.category) not in category_text:
            return False
    for value in constraints.required_terms:
        translated = rewrite_query(value)
        if translated not in text:
            return False
    for value in constraints.excluded_terms:
        translated = rewrite_query(value)
        if translated in text:
            return False
    return True


class CatalogRepository:
    """Read-only-at-runtime SQLite repository built from a versioned JSONL file."""

    def __init__(self, catalog_path: Path, db_path: Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.db_path = Path(db_path)

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Idempotently rebuild SQLite when the exact input checksum changes."""
        digest = hashlib.sha256(self.catalog_path.read_bytes()).hexdigest()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=30)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            previous = connection.execute(
                "SELECT value FROM metadata WHERE key='catalog_sha256'"
            ).fetchone()
            if previous and previous[0] == digest:
                return
            products: list[Product] = []
            seen: set[str] = set()
            with self.catalog_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    product = Product.model_validate_json(line)
                    if product.item_id in seen:
                        raise ValueError(f"Duplicate item ID at catalog line {line_number}")
                    seen.add(product.item_id)
                    products.append(product)
            if not products:
                raise ValueError("Catalog is empty.")
            versions = {product.catalog_version for product in products}
            if len(versions) != 1:
                raise ValueError("Catalog contains mixed versions.")
            with connection:
                connection.execute("DROP TABLE IF EXISTS products_fts")
                connection.execute("DROP TABLE IF EXISTS products")
                connection.execute(
                    "CREATE TABLE products (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE VIRTUAL TABLE products_fts USING fts5(item_id UNINDEXED, title, body, brand, category, tokenize='unicode61')"
                )
                for product in products:
                    connection.execute(
                        "INSERT INTO products VALUES (?, ?)",
                        (product.item_id, product.model_dump_json()),
                    )
                    connection.execute(
                        "INSERT INTO products_fts(item_id,title,body,brand,category) VALUES (?,?,?,?,?)",
                        (
                            product.item_id,
                            product.title,
                            product_text(product),
                            product.brand or "",
                            " ".join(product.category_path),
                        ),
                    )
                connection.execute(
                    "INSERT OR REPLACE INTO metadata VALUES ('catalog_sha256', ?)", (digest,)
                )
                connection.execute(
                    "INSERT OR REPLACE INTO metadata VALUES ('catalog_version', ?)",
                    (next(iter(versions)),),
                )
            LOGGER.info("Initialized catalog database with %d products", len(products))
        finally:
            connection.close()

    def get(self, item_id: str) -> Product | None:
        """Look up one validated product by its parent ASIN."""
        with self._read() as connection:
            row = connection.execute(
                "SELECT payload FROM products WHERE item_id=?", (item_id,)
            ).fetchone()
        return Product.model_validate_json(row[0]) if row else None

    def all_products(self) -> list[Product]:
        """Return products in deterministic item-ID order."""
        with self._read() as connection:
            rows = connection.execute("SELECT payload FROM products ORDER BY item_id").fetchall()
        return [Product.model_validate_json(row[0]) for row in rows]

    def filter_products(self, constraints: Constraints) -> list[Product]:
        """Return the full eligible universe, including products outside top-K."""
        return [
            product for product in self.all_products() if matches_constraints(product, constraints)
        ]

    def lexical_search(self, query: str, limit: int = 200) -> list[tuple[str, float]]:
        """Return FTS5 BM25 ascending (smaller is better), with safe OR tokens."""
        tokens = lexical_tokens(query)
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{token}"' for token in tokens)
        bounded_limit = max(1, min(limit, 5000))
        with self._read() as connection:
            rows = connection.execute(
                "SELECT item_id, bm25(products_fts, 0.0, 5.0, 1.0, 2.0, 1.0) AS score "
                "FROM products_fts WHERE products_fts MATCH ? ORDER BY score ASC, item_id ASC LIMIT ?",
                (fts_query, bounded_limit),
            ).fetchall()
        return [(row[0], float(row[1])) for row in rows]

    def stats(self) -> dict[str, Any]:
        """Return a small, source-aware health summary without model allocation."""
        products = self.all_products()
        category_counts = Counter(product.category for product in products)
        brand_counts = Counter(product.brand for product in products if product.brand)
        return {
            "count": len(products),
            "priced_count": sum(product.price_usd is not None for product in products),
            "categories": dict(category_counts.most_common()),
            "brands": dict(brand_counts.most_common(50)),
            "catalog_version": products[0].catalog_version if products else "unknown",
            "source": "McAuley-Lab Amazon Reviews 2023 / Office_Products; historical metadata",
        }
