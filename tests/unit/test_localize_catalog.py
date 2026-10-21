"""Translation validation prevents display-only generation from changing numbers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from choiceagent.localization import source_hash

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/localize_catalog.py"
SPEC = importlib.util.spec_from_file_location("catalog_translator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
translator: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(translator)


def product() -> dict[str, Any]:
    return {
        "item_id": "A",
        "title": "Acme 12 inch ruler",
        "brand": "Acme",
        "category": "Office Products",
        "category_path": ["Office Products", "Rulers"],
        "features": ["12 inch ruler", "Made from metal"],
        "attributes": {"Color": "Blue"},
    }


def test_rejects_added_title_number() -> None:
    value, errors = translator.parse_translation(
        json.dumps({"title": "Acme 24英寸直尺"}), product()
    )
    assert errors == ["title_added_number"]
    assert value["title"] == "Acme 办公用品"


def test_feature_numbers_are_bound_to_corresponding_source_fact() -> None:
    raw = {
        "title": "Acme 12英寸直尺",
        "features": ["12英寸直尺", "12英寸金属材质"],
        "attributes": {"Color": "蓝色", "Width": "12英寸"},
    }
    value, errors = translator.parse_translation(json.dumps(raw), product())
    assert value["features"] == ["12英寸直尺"]
    assert value["attributes"] == {"Color": "蓝色"}
    assert errors == ["feature_added_number", "ignored_model_attributes"]


def test_malformed_features_are_not_iterated_as_strings() -> None:
    value, errors = translator.parse_translation(
        '{"title":"12英寸直尺","features":"金属"}', product()
    )
    assert value["title"] == "Acme 12英寸直尺"
    assert value["features"] == []
    assert errors == ["invalid_features_shape"]


def test_missing_chinese_title_and_invalid_json_use_actual_category() -> None:
    item = {**product(), "category": "Amazon Home"}
    for raw in ['{"title":"Acme ruler"}', "broken json"]:
        value, errors = translator.parse_translation(raw, item)
        assert errors
        assert value["title"] == "Acme 家居用品"
        assert value["features"] == []


def test_source_hash_tracks_exact_source_not_price() -> None:
    item = product()
    assert source_hash(item) == source_hash({**item, "price_usd": 99})
    assert source_hash(item) != source_hash({**item, "features": ["12 inches"]})


def test_truncation_recovers_only_complete_json_strings() -> None:
    raw = '{"title":"Acme 12英寸直尺","features":["12英寸直尺","金属材质","未写完'
    value, errors = translator.parse_translation(raw, product())
    assert value["title"] == "Acme 12英寸直尺"
    assert value["features"] == ["12英寸直尺", "金属材质"]
    assert errors == ["recovered_truncated_json"]


def test_assistant_review_changes_only_failed_source_bound_title() -> None:
    item = product()
    row = {**translator.rule_fallback(item), "translation_kind": "rule", "model": "Qwen3-1.7B"}
    review = {
        "item_id": item["item_id"],
        "source_hash": source_hash(item),
        "title": "12英寸金属直尺",
        "translation_version": "zh-CN-v1",
        "model": "coding-assistant-source-review",
        "features": ["新增内容不能复制"],
    }
    assert translator.apply_assistant_review(item, row, review)
    assert row["title"] == "Acme 12英寸金属直尺"
    assert row["model"] == "coding-assistant-source-review"
    assert row["features"] == []


def test_assistant_review_cannot_override_success_or_mismatch_source() -> None:
    item = product()
    base_review = {
        "item_id": item["item_id"],
        "source_hash": source_hash(item),
        "title": "12英寸金属直尺",
        "translation_version": "zh-CN-v1",
        "model": "coding-assistant-source-review",
    }
    for changes in [
        {"source_hash": "outdated"},
        {"item_id": "other"},
        {"translation_version": "old"},
        {"model": "Qwen3-1.7B"},
        {"title": "24英寸金属直尺"},
        {"title": "Acme English ruler"},
    ]:
        row = {**translator.rule_fallback(item), "translation_kind": "rule", "model": "Qwen3-1.7B"}
        assert not translator.apply_assistant_review(item, row, {**base_review, **changes})
        assert row["model"] == "Qwen3-1.7B"
    row = {"title": "原有成功译文", "translation_kind": "model", "model": "Qwen3-1.7B"}
    assert not translator.apply_assistant_review(item, row, base_review)
    assert row["title"] == "原有成功译文"


def test_compartment_and_remaining_english_corrections_require_matching_source() -> None:
    item = {
        **product(),
        "title": "5 compartment organizer",
        "features": ["Place anywhere on a vanity table"],
    }
    raw = "5层收纳盒及15层架，vanity桌，放置 anywhere"
    assert translator.correct_terms(raw, item) == "5格收纳盒及15层架，梳妆台，可放在任意位置"
    assert (
        translator.correct_terms("5层收纳盒", {**item, "title": "5 compartment 2-tier organizer"})
        == "5层收纳盒"
    )
    assert (
        translator.correct_terms("5层收纳盒", {**item, "title": "5 compartment layer organizer"})
        == "5层收纳盒"
    )
    long_exclusion = {**item, "features": ["x" * 250 + " layer organizer"]}
    assert translator.correct_terms("5层收纳盒", long_exclusion) == "5层收纳盒"
    assert translator.correct_terms(raw, product()) == raw


def test_coaster_organizer_and_flannel_corrections_do_not_override_explicit_source() -> None:
    item = {
        **product(),
        "title": "Desk organizer storage box with coaster",
        "features": ["PU Leather & Flannel"],
    }
    raw = "办公文件夹带座垫，皮质与棉质"
    assert translator.correct_terms(raw, item) == "桌面收纳盒带杯垫，皮质与绒布"
    protected = {**item, "features": ["Seat cushion, cotton, file folder"]}
    assert translator.correct_terms(raw, protected) == raw
    assert translator.correct_terms(raw, product()) == raw


def test_stapler_translation_protects_real_punches_and_staple_guns() -> None:
    item = {
        **product(),
        "title": "Mini office stapler with 10# staples",
        "category_path": ["Office Products", "Staplers & Punches", "Desktop Staplers"],
        "features": [],
    }
    raw = "迷你打孔器配10#打孔针"
    assert translator.correct_terms(raw, item) == "迷你订书机配10#订书钉"
    for excluded in [
        "hole-punch",
        "paper punches",
        "staple gun",
        "stapler gun",
        "nail gun",
        "tacker",
    ]:
        protected = {**item, "features": ["x" * 250 + " " + excluded]}
        assert translator.correct_terms(raw, protected) == raw
    assert translator.correct_terms(raw, {**item, "attributes": {"Style": "Punch"}}) == raw
    assert translator.correct_terms(raw, {**item, "category_path": ["Paper Punches"]}) == raw
    assert translator.correct_terms(raw, product()) == raw
