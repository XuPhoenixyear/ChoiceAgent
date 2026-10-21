"""CPU-only safety boundaries for accepting offline translation model output."""

from __future__ import annotations

import builtins
import copy
import importlib.util
import json
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/localize_catalog.py"


@pytest.fixture
def builder(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import only parsing helpers, failing before any model package can load."""
    original_import = builtins.__import__

    def import_without_models(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", 1)[0] in {"torch", "transformers", "sentence_transformers"}:
            raise AssertionError(f"CPU parser test attempted to import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_models)
    spec = importlib.util.spec_from_file_location("localization_builder_cpu_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_item() -> dict[str, Any]:
    return {
        "item_id": "BUILDER_FIXTURE",
        "title": "Acme 12.5 inch metal ruler",
        "brand": "Acme",
        "category": "Office Products",
        "category_path": ["Office Products", "Rulers"],
        "price_usd": None,
        "features": ["12.5 inches long", "Made from metal", "Pack of 1,000 sheets"],
        "attributes": {
            "Brand": "Acme",
            "Manufacturer": "Natural",
            "Item model number": "Black-Medium-A7",
            "Color": "Blue",
            "Material": "Metal",
            "Product Dimensions": "12.5 inches",
        },
        "source_locator": "self-authored CPU test fixture",
    }


def test_parser_helpers_import_and_run_without_model_packages(
    builder: ModuleType, source_item: dict[str, Any]
) -> None:
    parsed, errors = builder.parse_translation('{"title":"12.5英寸金属直尺"}', source_item)
    assert parsed["title"] == "Acme 12.5英寸金属直尺"
    assert not errors
    assert "torch" not in vars(builder)
    assert "transformers" not in vars(builder)


def test_numeric_values_compare_decimal_values_and_thousands_without_unit_conversion(
    builder: ModuleType,
) -> None:
    assert builder.numeric_values("12.50 inches / 1,000 sheets / 1000.0 sheets") == {
        Decimal("12.5"),
        Decimal("1000"),
    }


@pytest.mark.parametrize("added_number", ["24", "125", "1.25", "1001"])
def test_changed_or_added_numeric_title_is_rejected(
    builder: ModuleType, source_item: dict[str, Any], added_number: str
) -> None:
    raw = json.dumps({"title": f"Acme {added_number}英寸金属直尺"}, ensure_ascii=False)
    parsed, errors = builder.parse_translation(raw, source_item)
    assert "title_added_number" in errors
    assert parsed == builder.rule_fallback(source_item)


def test_numeric_reformatting_is_accepted_without_introducing_a_new_value(
    builder: ModuleType, source_item: dict[str, Any]
) -> None:
    raw = json.dumps(
        {"title": "Acme 12.50英寸金属直尺", "features": ["长度12.50英寸", "金属材质", "1000张"]},
        ensure_ascii=False,
    )
    parsed, errors = builder.parse_translation(raw, source_item)
    assert parsed["features"] == ["长度12.50英寸", "金属材质", "1000张"]
    assert not errors


def test_feature_cannot_borrow_numbers_from_another_source_feature(
    builder: ModuleType, source_item: dict[str, Any]
) -> None:
    raw = json.dumps(
        {"title": "Acme 金属直尺", "features": ["长度12.5英寸", "1000张金属材质", "1000张"]},
        ensure_ascii=False,
    )
    parsed, errors = builder.parse_translation(raw, source_item)
    assert parsed["features"] == ["长度12.5英寸", "1000张"]
    assert "feature_added_number" in errors


@pytest.mark.parametrize("source_count", [0, 1, 2])
def test_generated_features_cannot_fill_absent_source_rows(
    builder: ModuleType, source_item: dict[str, Any], source_count: int
) -> None:
    item = {**source_item, "features": source_item["features"][:source_count]}
    raw = json.dumps(
        {"title": "Acme 金属直尺", "features": ["长度可查看原文", "金属材质", "纸张包装"]},
        ensure_ascii=False,
    )
    parsed, errors = builder.parse_translation(raw, item)
    assert parsed["features"] == ["长度可查看原文", "金属材质", "纸张包装"][:source_count]
    assert errors


def test_truncated_json_keeps_complete_escaped_strings_and_discards_open_string(
    builder: ModuleType, source_item: dict[str, Any]
) -> None:
    title = 'Acme "轻便"12.5英寸直尺'
    feature = '长度12.5英寸，带"刻度"'
    raw = (
        '{"title":'
        + json.dumps(title, ensure_ascii=False)
        + ',"features":['
        + json.dumps(feature, ensure_ascii=False)
        + ',"尚未结束的说明\\'
    )
    parsed, errors = builder.parse_translation(raw, source_item)
    assert parsed["title"] == title
    assert parsed["features"] == [feature]
    assert "recovered_truncated_json" in errors


def test_incomplete_title_is_never_repaired_into_a_successful_translation(
    builder: ModuleType, source_item: dict[str, Any]
) -> None:
    parsed, errors = builder.parse_translation('{"title":"Acme 12.5英寸金属直尺', source_item)
    assert parsed == builder.rule_fallback(source_item)
    assert "invalid_json" in errors


def test_truncated_object_can_keep_only_its_closed_title(
    builder: ModuleType, source_item: dict[str, Any]
) -> None:
    parsed, errors = builder.parse_translation('{"title":"Acme 金属直尺",', source_item)
    assert parsed["title"] == "Acme 金属直尺"
    assert parsed["features"] == []
    assert "recovered_truncated_json" in errors


def test_model_attributes_cannot_override_identity_or_invent_numeric_facts(
    builder: ModuleType, source_item: dict[str, Any]
) -> None:
    original = copy.deepcopy(source_item)
    raw = json.dumps(
        {
            "title": "金属直尺",
            "features": ["长度12.5英寸"],
            "category": "不存在的中文分类",
            "attributes": {
                "Brand": "黑牌",
                "Manufacturer": "原木厂商",
                "Item model number": "蓝色中号",
                "Product Dimensions": "24英寸",
                "Material": "木材",
                "Invented": "新增事实",
            },
        },
        ensure_ascii=False,
    )
    parsed, errors = builder.parse_translation(raw, source_item)
    assert parsed["title"] == "Acme 金属直尺"
    assert parsed["category"] == "办公用品"
    assert parsed["attributes"]["Material"] == "金属"
    assert parsed["attributes"]["Product Dimensions"] == "12.5 英寸"
    for key in ("Brand", "Manufacturer", "Item model number"):
        assert (
            key not in parsed["attributes"]
            or parsed["attributes"][key] == original["attributes"][key]
        )
    assert "Invented" not in parsed["attributes"]
    assert errors
    assert source_item == original


@pytest.mark.parametrize("brand", ["Acme", "晨光", "A.C.M.E（文具）"])
def test_english_title_is_not_success_even_with_chinese_brand_or_chinese_features(
    builder: ModuleType, source_item: dict[str, Any], brand: str
) -> None:
    item = {**source_item, "brand": brand, "title": f"{brand} metal ruler"}
    raw = json.dumps({"title": item["title"], "features": ["金属直尺"]}, ensure_ascii=False)
    parsed, errors = builder.parse_translation(raw, item)
    assert parsed == builder.rule_fallback(item)
    assert "non_chinese_title" in errors


def test_rejected_output_does_not_mutate_source_data(
    builder: ModuleType, source_item: dict[str, Any]
) -> None:
    original = copy.deepcopy(source_item)
    parsed, errors = builder.parse_translation('{"title":"Acme 99英寸金属直尺"}', source_item)
    assert errors
    parsed["title"] = "Changed derived presentation"
    parsed["features"].append("后续界面修改")
    assert source_item == original
