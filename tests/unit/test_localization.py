"""Chinese presentation keeps source facts and invalidates stale translations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from choiceagent.localization import (
    TRANSLATION_VERSION,
    CatalogLocalizer,
    category_label,
    field_label,
    localize_value,
    source_hash,
)
from choiceagent.schemas import Product


@pytest.fixture
def product() -> Product:
    return Product(
        item_id="LOCALIZATION_FIXTURE",
        title="Black Model A-12 Metal Desk Organizer 12 inches",
        category="Office Products",
        category_path=["Office Products", "Desk Accessories & Workspace Organizers"],
        brand="Black",
        price_usd=12.5,
        features=[
            "Metal construction",
            "12 inches wide",
            "Two compartments",
            "Extra source detail",
        ],
        attributes={
            "Brand": "Black",
            "Manufacturer": "Natural",
            "Item model number": "Blue-Medium-12",
            "Material": "Metal",
            "Color": "Black",
            "Product Dimensions": "12 x 8 x 4 inches",
        },
    )


def translation(product: Product) -> dict[str, Any]:
    """Build a local sidecar row without invoking any model or external service."""
    return {
        "item_id": product.item_id,
        "source_hash": source_hash(product),
        "title": "Black A-12 金属桌面收纳架 12 英寸",
        "category": "办公用品",
        "features": ["金属结构", "宽度 12 英寸", "两个隔层"],
        "attributes": {"Material": "金属", "Color": "黑色"},
        "translation_kind": "model",
        "model": "offline-fixture",
        "translation_version": TRANSLATION_VERSION,
    }


def write_sidecar(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_hash_matches_mapping_and_ignores_identity_and_untranslated_fields(
    product: Product,
) -> None:
    original = product.model_dump()
    equivalent = dict(reversed(list(original.items())))
    equivalent["attributes"] = dict(reversed(list(original["attributes"].items())))
    equivalent["item_id"] = "ANOTHER_ID"
    equivalent["price_usd"] = 999
    equivalent["source_locator"] = "a-different-source-locator"
    equivalent["description"] = "Not included in the translated content"
    equivalent["features"] = [*product.features[:3], "Changed fourth source detail"]
    assert source_hash(product) == source_hash(original) == source_hash(equivalent)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "Different source title"),
        ("category", "Computers"),
        ("category_path", ["Different source path"]),
        ("brand", "Different Brand"),
        ("features", ["A changed first feature"]),
        ("attributes", {"Material": "Wood"}),
    ],
)
def test_hash_changes_when_translated_source_changes(
    product: Product, field: str, value: Any
) -> None:
    changed = product.model_dump()
    changed[field] = value
    assert source_hash(changed) != source_hash(product)


@pytest.mark.parametrize(
    "category",
    [
        "Office Products",
        "Amazon Home",
        "All Electronics",
        "Industrial & Scientific",
        "Tools & Home Improvement",
        "Computers",
        "AMAZON FASHION",
        "Arts, Crafts & Sewing",
        "Books",
        "Toys & Games",
        "Health & Personal Care",
        "Sports & Outdoors",
        "Cell Phones & Accessories",
        "Camera & Photo",
        "Home Audio & Theater",
        "All Beauty",
        "Baby",
        "Automotive",
        "Portable Audio & Accessories",
        "Digital Music",
        "Video Games",
        "Musical Instruments",
        "Handmade",
    ],
)
def test_all_frozen_catalog_main_categories_have_specific_chinese_labels(category: str) -> None:
    label = category_label(category)
    assert label != category
    assert label != "其他商品"
    assert any("\u4e00" <= char <= "\u9fff" for char in label)


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("Size", "Medium", "中号"),
        ("attributes.Point Type", "Medium", "中等笔尖"),
        ("Color", "Natural", "本色"),
        ("Material", "Natural", "Natural"),
        ("Theme", "Natural", "Natural"),
        ("Brand", "Black", "Black"),
        ("Manufacturer", "Natural", "Natural"),
        ("Item model number", "Medium", "Medium"),
        ("Manufacturer Part Number", "12 inches", "12 inches"),
        ("Material Type", "Metal", "金属"),
        ("Color", "Black", "黑色"),
        ("Is Discontinued By Manufacturer", "No", "否"),
        ("Theme", "No", "No"),
        ("Mystery Attribute", "Black", "Black"),
        ("Sheet Size", "Letter", "美式 Letter 规格"),
        ("Product Dimensions", "12 x 8 x 4 inches", "12 x 8 x 4 英寸"),
        ("Item Weight", "2.5 pounds", "2.5 磅"),
        ("price_usd", 12.5, 12.5),
        ("Material", None, None),
        ("features", ["12 inches wide", "Black"], ["12 英寸 wide", "Black"]),
    ],
)
def test_attribute_translation_uses_field_context_and_preserves_identifiers_and_amounts(
    field: str, raw: Any, expected: Any
) -> None:
    assert localize_value(field, raw) == expected


def test_field_labels_accept_source_paths_and_existing_chinese() -> None:
    assert field_label("attributes.Material Type") == "材质类型"
    assert field_label("商品重量") == "商品重量"
    assert field_label("No Such Attribute") == "其他属性"


def test_missing_sidecar_provides_rule_fallback_without_fabricated_features(
    tmp_path: Path, product: Product
) -> None:
    display = CatalogLocalizer(tmp_path / "missing.jsonl").display(product)
    assert display["translation_kind"] == "rule"
    assert display["locale"] == "zh-CN"
    assert display["features"] == []
    assert display["attributes"]["材质"] == "金属"
    assert display["attributes"]["品牌"] == product.brand
    assert "evidence" not in display


def test_valid_translation_keeps_source_facts_and_identifiers_immutable(
    tmp_path: Path, product: Product
) -> None:
    source = product.model_dump()
    source["evidence"] = [{"field": "title", "value": product.title}]
    before = copy.deepcopy(source)
    row = translation(product)
    row["attributes"].update(
        {
            "Brand": "黑牌",
            "Manufacturer": "天然制造商",
            "Item model number": "蓝色中号",
            "Invented Attribute": "不存在的参数",
        }
    )
    row["category"] = "不应覆盖源分类的缓存标签"
    path = tmp_path / "sidecar.jsonl"
    write_sidecar(path, [row])
    display = CatalogLocalizer(path).display(source)
    assert display["translation_kind"] == "model"
    assert display["title"] == row["title"]
    assert display["features"] == row["features"]
    assert display["category"] == "办公用品"
    assert display["attributes"]["品牌"] == "Black"
    assert display["attributes"]["制造商"] == "Natural"
    assert display["attributes"]["商品型号"] == "Blue-Medium-12"
    assert "不存在的参数" not in display["attributes"].values()
    assert "evidence" not in display
    assert source == before
    display["attributes"]["品牌"] = "Changed UI value"
    assert source == before


@pytest.mark.parametrize("changed_field", ["title", "features", "attributes"])
def test_stale_translation_is_not_applied_to_changed_source(
    tmp_path: Path, product: Product, changed_field: str
) -> None:
    path = tmp_path / "sidecar.jsonl"
    write_sidecar(path, [translation(product)])
    changed = product.model_dump()
    changed[changed_field] = {
        "title": "New source desk organizer",
        "features": ["New first feature"],
        "attributes": {"Material": "Wood"},
    }[changed_field]
    display = CatalogLocalizer(path).display(changed)
    assert display["translation_kind"] == "rule"
    assert display["title"] != translation(product)["title"]
    assert display["features"] == []


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("translation_version", "older-version"),
        ("source_hash", "stale-hash"),
        ("title", "English-only result"),
        ("title", None),
    ],
)
def test_invalid_translation_row_falls_back(
    tmp_path: Path, product: Product, field: str, invalid: Any
) -> None:
    row = translation(product)
    row[field] = invalid
    path = tmp_path / "sidecar.jsonl"
    write_sidecar(path, [row])
    assert CatalogLocalizer(path).display(product)["translation_kind"] == "rule"


@pytest.mark.parametrize("source_feature_count", [0, 1, 2, 3, 4])
def test_translation_cannot_add_feature_rows_beyond_known_source(
    tmp_path: Path, product: Product, source_feature_count: int
) -> None:
    product = product.model_copy(update={"features": product.features[:source_feature_count]})
    original_features = list(product.features)
    row = translation(product)
    row["features"] = ["中文描述一", "中文描述二", "中文描述三", "多余的中文描述"]
    path = tmp_path / "sidecar.jsonl"
    write_sidecar(path, [row])
    display = CatalogLocalizer(path).display(product)
    assert display["features"] == row["features"][: min(3, source_feature_count)]
    assert product.features == original_features


def test_corrupt_sidecar_is_nonfatal_and_removal_clears_cached_translation(
    tmp_path: Path, product: Product
) -> None:
    path = tmp_path / "sidecar.jsonl"
    path.write_text("{not-valid-json}\n", encoding="utf-8")
    localizer = CatalogLocalizer(path)
    assert localizer.display(product)["translation_kind"] == "rule"
    write_sidecar(path, [translation(product)])
    assert localizer.display(product)["translation_kind"] == "model"
    path.unlink()
    assert localizer.display(product)["translation_kind"] == "rule"


def test_localizer_reloads_atomically_replaced_translation_file(
    tmp_path: Path, product: Product
) -> None:
    path = tmp_path / "sidecar.jsonl"
    row = translation(product)
    write_sidecar(path, [row])
    localizer = CatalogLocalizer(path)
    assert localizer.display(product)["title"] == row["title"]
    replacement = tmp_path / "replacement.jsonl"
    row["title"] = "更新后的中文桌面收纳架标题"
    write_sidecar(replacement, [row])
    replacement.replace(path)
    assert localizer.display(product)["title"] == row["title"]


def test_nontranslated_extra_features_do_not_expire_valid_cache(
    tmp_path: Path, product: Product
) -> None:
    path = tmp_path / "sidecar.jsonl"
    write_sidecar(path, [translation(product)])
    changed = product.model_copy(update={"features": [*product.features[:3], "New fourth detail"]})
    assert CatalogLocalizer(path).display(changed)["translation_kind"] == "model"


def test_rule_sidecar_does_not_replace_known_product_type_with_generic_main_category(
    tmp_path: Path,
) -> None:
    product = Product(
        item_id="B08PQK4WF5",
        title=(
            "Marbrasse Rose Gold Desk Organizer | Set of 2 Desktop Organizer, Rose Gold Desk "
            "Accessories for Women, Large Capacity Metal Supplies Organizer with Pencil Holder "
            "for Home & Office Desktop Organization & Decor"
        ),
        brand="Marbrasse",
        category="Tools & Home Improvement",
    )
    row = translation(product)
    row.update(title="Marbrasse 工具与家装", features=[], translation_kind="rule")
    path = tmp_path / "sidecar.jsonl"
    write_sidecar(path, [row])
    display = CatalogLocalizer(path).display(product)
    assert display["title"] == "Marbrasse · 桌面收纳架"
    assert display["translation_kind"] == "rule"
    assert display["category"] == "工具与家装"
