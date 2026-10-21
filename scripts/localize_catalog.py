"""Build a resumable Chinese display cache using the exclusively local Qwen model.

This never changes the source catalog. Translations are display aids, not evidence.
Run a 32-item trial first, then resume the full catalog with the same cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from choiceagent.localization import category_label, localize_value, source_hash

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/datas/wuxi/Projects/datas/models/Qwen3-1.7B")
VERSION = "zh-CN-v1"
CHINESE = re.compile(r"[\u4e00-\u9fff]")
NUMBERS = re.compile(r"\d+(?:[.,]\d+)*")
ATTRIBUTE_KEYS = frozenset(
    {
        "Color",
        "Material",
        "Material Type",
        "Size",
        "Style",
        "Pattern",
        "Shape",
        "Ink Color",
        "Point Type",
        "Point Size",
        "Number of Items",
        "Sheet Size",
        "Item Weight",
        "Product Dimensions",
        "Item Dimensions LxWxH",
        "Unit Count",
    }
)
PROMPT = """请把英文商品资料翻译为简体中文，只输出JSON：{"title":"中文标题","features":["中文要点","中文要点","中文要点"]}。
标题必须翻译商品类型和描述词，不要复制英文原句，最多45个汉字；仅品牌、型号可保留原文。
features逐条对应原文，最多3条，每条摘译约20个汉字。允许删减，不得猜测或新增原文没有的事实。
保留原数字，不得新增数字。不要编号，不要属性，不要解释。"""


def numeric_values(value: str) -> set[Decimal]:
    result: set[Decimal] = set()
    for match in NUMBERS.findall(value):
        try:
            result.add(Decimal(match.replace(",", "")))
        except InvalidOperation:
            continue
    return result


def attributes_for(item: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value)[:180]
        for key, value in item.get("attributes", {}).items()
        if key in ATTRIBUTE_KEYS
    }


def translation_input(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "title": item["title"],
        "brand": item["brand"],
        "features": [str(value)[:220] for value in item.get("features", [])[:3]],
    }


def rule_fallback(item: Mapping[str, Any]) -> dict[str, Any]:
    """Use only the known source category and brand when translation is untrustworthy."""
    brand = str(item.get("brand") or "").strip()
    category = category_label(str(item.get("category") or "Unknown"))
    return {
        "title": f"{brand} {category}".strip(),
        "category": category,
        "features": [],
        "attributes": {},
    }


def has_chinese_description(title: str, item: Mapping[str, Any]) -> bool:
    brand = str(item.get("brand") or "").strip()
    description = re.sub(re.escape(brand), "", title, flags=re.IGNORECASE) if brand else title
    return bool(CHINESE.search(description))


def correct_terms(value: str, item: Mapping[str, Any]) -> str:
    """Resolve known lexical errors only when the exact English source supports it."""
    source = json.dumps(translation_input(item), ensure_ascii=False)
    source_full = json.dumps(
        {
            **translation_input(item),
            "features": list(item.get("features", []))[:3],
            "attributes": item.get("attributes", {}),
        },
        ensure_ascii=False,
    )
    staple_source = json.dumps(
        {
            "title": item.get("title", ""),
            "brand": item.get("brand"),
            "category": item.get("category", ""),
            # This shared ancestor names two families; leaf/product evidence decides.
            "category_path": [
                part
                for part in item.get("category_path", [])
                if str(part).casefold() != "staplers & punches"
            ],
            "features": list(item.get("features", []))[:3],
            "attributes": item.get("attributes", {}),
        },
        ensure_ascii=False,
    )
    if re.search(
        r"\b(?:staplers?|staples)\b", staple_source, flags=re.IGNORECASE
    ) and not re.search(
        r"\b(?:hole[- ]*)?punch(?:es|ers?|ing|ed)?\b|\b(?:stapler?|nail|tack)[- ]*guns?\b|\b(?:tackers?|nailers?)\b",
        staple_source,
        flags=re.IGNORECASE,
    ):
        value = value.replace("打孔器", "订书机").replace("打孔针", "订书钉")
    if re.search(r"\bcompartments?\b", source_full, flags=re.IGNORECASE) and not re.search(
        r"\b(?:tier|layer|level)s?\b", source_full, flags=re.IGNORECASE
    ):
        counts = re.findall(
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*[- ]?\s*compartments?\b", source_full, flags=re.IGNORECASE
        )
        for count in counts:
            value = re.sub(rf"(?<![\d.]){re.escape(count)}\s*层", count + "格", value)
    if re.search(r"\bcoasters?\b", source_full, flags=re.IGNORECASE) and not re.search(
        r"\b(?:seat|cushion)s?\b", source_full, flags=re.IGNORECASE
    ):
        value = value.replace("座垫", "杯垫")
    if (
        re.search(r"\bdesk organizer\b", source_full, flags=re.IGNORECASE)
        and re.search(r"\bcoasters?\b", source_full, flags=re.IGNORECASE)
        and not re.search(r"\bfolders?\b", source_full, flags=re.IGNORECASE)
    ):
        organizer = (
            "桌面收纳盒"
            if re.search(r"\bstorage box\b", source_full, flags=re.IGNORECASE)
            else "桌面收纳用品"
        )
        value = value.replace("办公文件夹", organizer)
    if (
        re.search(r"\bflannel\b", source_full, flags=re.IGNORECASE)
        and re.search(r"\bdesk organizer\b", source_full, flags=re.IGNORECASE)
        and re.search(r"\bcoasters?\b", source_full, flags=re.IGNORECASE)
        and not re.search(r"\bcotton\b", source_full, flags=re.IGNORECASE)
    ):
        value = value.replace("棉质", "绒布")
    if re.search(r"\bvanity\b", source_full, flags=re.IGNORECASE):
        value = re.sub(
            r"(?<![A-Za-z])vanity\s*(?:table|桌|台)(?![A-Za-z])",
            "梳妆台",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"(?<![A-Za-z])vanity(?![A-Za-z])", "梳妆台", value, flags=re.IGNORECASE)
    if re.search(r"\banywhere\b", source_full, flags=re.IGNORECASE):
        value = re.sub(r"(?<![A-Za-z])anywhere(?![A-Za-z])", "任意位置", value, flags=re.IGNORECASE)
        value = re.sub(r"放置\s*任意位置", "可放在任意位置", value)
    if re.search(r"\bfile bands\b", source, flags=re.IGNORECASE):
        value = value.replace("非乳胶色带带", "非乳胶文件橡皮筋")
    if (
        str(item.get("title", "")).casefold() == "star mica - bulk mineral"
        and value == "星玛 - 矿物粉末"
    ):
        value = "Star Mica 散装矿物"
    if re.search(r"\btoner cartridges?\b", source, flags=re.IGNORECASE) and not re.search(
        r"\bink\b", source, flags=re.IGNORECASE
    ):
        value = value.replace("墨盒", "碳粉盒")
    if re.search(r"\bgel pens?\b", source, flags=re.IGNORECASE):
        value = value.replace("胶笔", "中性笔")
    if re.search(r"2[- ]way speakerphone", source, flags=re.IGNORECASE):
        value = value.replace("双扬声器", "双向免提通话")
    if "assorted bright colors and sizes" in source.casefold():
        value = re.sub(
            "assorted bright colors and sizes", "多种亮色和尺寸", value, flags=re.IGNORECASE
        )
    # Known brand names must not be invented while translating a different brand.
    for chinese, source_brand in {
        "佳能": r"\bcanon\b|佳能",
        "爱普生": r"\bepson\b|爱普生",
        "惠普": r"\bhp(?=\d|\b)|惠普",
        "兄弟": r"\bbrother\b|兄弟",
        "三星": r"\bsamsung\b|三星",
        "得力": r"\bdeli\b|得力",
        "晨光": r"\bm&g\b|晨光",
        "百乐": r"\bpilot\b|百乐",
        "三菱": r"\bmitsubishi\b|三菱",
        "斑马": r"\bzebra\b|斑马",
    }.items():
        if chinese in value and not re.search(source_brand, source, flags=re.IGNORECASE):
            value = value.replace(chinese, "")
    if re.search(r"\brhinestones?\b", source, flags=re.IGNORECASE) and not re.search(
        r"\bdiamonds?\b", source, flags=re.IGNORECASE
    ):
        value = value.replace("钻石", "水钻")
    if re.search(r"\blegal\b", source, flags=re.IGNORECASE):
        value = value.replace("法律尺寸", "Legal规格").replace("法律文件", "Legal规格文件")
    if re.search(r"\bletter\b", source, flags=re.IGNORECASE) and re.search(
        r"\blegal\b", source, flags=re.IGNORECASE
    ):
        value = value.replace("信纸", "Letter规格纸张")
    return value


def parse_translation(raw: str, item: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        obj = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        # A token limit may truncate the final array or object. Reuse only complete
        # JSON strings already emitted; never invent a missing string or delimiter.
        title_match = re.search(r'"title"\s*:\s*("(?:[^"\\]|\\.)*")', raw)
        if title_match is None:
            return rule_fallback(item), ["invalid_json"]
        obj = {"title": json.loads(title_match.group(1)), "features": []}
        feature_match = re.search(r'"features"\s*:\s*\[', raw)
        if feature_match:
            remaining = raw[feature_match.end() :].lstrip()
            decoder = json.JSONDecoder()
            while remaining.startswith('"') and len(obj["features"]) < 3:
                try:
                    value, consumed = decoder.raw_decode(remaining)
                except json.JSONDecodeError:
                    break
                obj["features"].append(value)
                remaining = remaining[consumed:].lstrip().removeprefix(",").lstrip()
        errors.append("recovered_truncated_json")
    if not isinstance(obj, dict) or not isinstance(obj.get("title"), str):
        return rule_fallback(item), ["invalid_shape"]
    title = str(obj["title"]).strip()
    source = json.dumps(translation_input(item), ensure_ascii=False)
    source_numbers = numeric_values(source)
    if not has_chinese_description(title, item) or len(title) > 240:
        return rule_fallback(item), ["non_chinese_title"]
    if numeric_values(title) - source_numbers:
        return rule_fallback(item), ["title_added_number"]
    brand = str(item.get("brand") or "").strip()
    if brand and brand.casefold() not in title.casefold():
        title = f"{brand} {title}"
    category = category_label(str(item.get("category") or "Unknown"))
    if not CHINESE.search(category) or numeric_values(category) - source_numbers:
        category = category_label(str(item.get("category") or "Unknown"))
    features: list[str] = []
    candidate_features = obj.get("features", [])
    if not isinstance(candidate_features, list):
        candidate_features = []
        errors.append("invalid_features_shape")
    for index, value in enumerate(candidate_features[:3]):
        if not isinstance(value, str) or not CHINESE.search(value):
            errors.append("non_chinese_feature")
            continue
        # Each summary may only use numeric values from its corresponding source fact.
        originals = item.get("features", [])[:3]
        if index >= len(originals) or numeric_values(value) - numeric_values(str(originals[index])):
            errors.append("feature_added_number")
            continue
        features.append(value.strip())
    attributes: dict[str, str] = {
        key: str(localize_value(key, value)) for key, value in attributes_for(item).items()
    }
    if obj.get("attributes"):
        errors.append("ignored_model_attributes")
    return {
        "title": correct_terms(title, item),
        "category": category,
        "features": [correct_terms(value, item) for value in features],
        "attributes": attributes,
    }, errors


class Translator:
    def __init__(self, *, max_input: int, max_output: int) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_OFFLINE"] = "1"
        for name, directory in {
            "HF_HOME": ROOT / "runtime/cache/huggingface",
            "TORCH_HOME": ROOT / "runtime/cache/torch",
            "XDG_CACHE_HOME": ROOT / "runtime/cache",
            "CUDA_CACHE_PATH": ROOT / "runtime/cache/cuda",
            "TRITON_CACHE_DIR": ROOT / "runtime/cache/triton",
            "TMPDIR": ROOT / "runtime/tmp",
        }.items():
            directory.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(directory)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(4)
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Offline translation must see exactly physical GPU 5")
        self.torch = torch
        self.max_input, self.max_output = max_input, max_output
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            MODEL_PATH, local_files_only=True, trust_remote_code=False
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        move_to_device: Any = self.model.to
        move_to_device("cuda:0")
        self.model.eval()
        torch.cuda.reset_peak_memory_stats()

    def translate(
        self, items: list[dict[str, Any]], *, title_only: bool = False
    ) -> tuple[list[str], int]:
        prompt = (
            "请将下面英文商品标题翻译成简体中文。商品类型和描述词必须译成中文，品牌型号保持原文。"
            "可省略冗长的型号清单，不得添加任何原文没有的信息或数字。只输出中文标题文本，不要JSON。"
            if title_only
            else PROMPT
        )
        texts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": str(item["title"])
                        if title_only
                        else json.dumps(translation_input(item), ensure_ascii=False),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for item in items
        ]
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input
        ).to("cuda:0")
        length = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=128 if title_only else self.max_output,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        decoded = self.tokenizer.batch_decode(output[:, length:], skip_special_tokens=True)
        return list(decoded), int(output[:, length:].numel())


def load_cache(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("translation_version") == VERSION:
                    cache[(row["item_id"], row["source_hash"])] = row
            except (json.JSONDecodeError, KeyError):
                continue  # An interrupted last append is harmless; that item is retried.
    return cache


def apply_assistant_review(
    item: Mapping[str, Any], row: dict[str, Any], review: Mapping[str, Any] | None
) -> bool:
    """Accept an independently labelled, source-bound review of a failed Qwen title."""
    if row.get("translation_kind") != "rule" or review is None:
        return False
    if (
        review.get("item_id") != item["item_id"]
        or review.get("source_hash") != source_hash(item)
        or review.get("translation_version") != VERSION
        or review.get("model") != "coding-assistant-source-review"
    ):
        return False
    title = review.get("title")
    if not isinstance(title, str) or not has_chinese_description(title, item) or len(title) > 240:
        return False
    if numeric_values(title) - numeric_values(
        json.dumps(translation_input(item), ensure_ascii=False)
    ):
        return False
    brand = str(item.get("brand") or "").strip()
    if brand and brand.casefold() not in title.casefold():
        title = brand + " " + title
    row.update(
        title=correct_terms(title, item),
        translation_kind="model",
        model="coding-assistant-source-review",
    )
    return True


def publish_snapshot(
    items: list[dict[str, Any]], cache: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    """Atomically publish all source rows, using conservative placeholders while building."""
    rows: list[dict[str, Any]] = []
    cached_count = 0
    reviews = load_cache(ROOT / "data/processed/catalog.zh-CN.review.jsonl")
    reviews.update(load_cache(ROOT / "runtime/localization/assistant_review.jsonl"))
    review_accepted = 0
    for item in items:
        cached = cache.get((item["item_id"], source_hash(item)))
        if cached is not None:
            row = dict(cached)
            cached_count += 1
        else:
            row = {
                "item_id": item["item_id"],
                "source_hash": source_hash(item),
                **rule_fallback(item),
                "translation_kind": "rule",
                "model": str(MODEL_PATH),
                "translation_version": VERSION,
            }
        if not has_chinese_description(row["title"], item):
            row.update(rule_fallback(item))
            row["translation_kind"] = "rule"
        review_accepted += apply_assistant_review(
            item, row, reviews.get((item["item_id"], source_hash(item)))
        )
        row["title"] = correct_terms(row["title"], item)
        row["features"] = [correct_terms(value, item) for value in row["features"]]
        row["attributes"] = {
            key: str(localize_value(key, value)) for key, value in attributes_for(item).items()
        }
        rows.append(row)
    output = ROOT / "data/processed/catalog.zh-CN.jsonl"
    temporary = output.with_suffix(f".snapshot-{os.getpid()}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(output)
    progress = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_count": len(items),
        "cached_count": cached_count,
        "assistant_review_accepted_count": review_accepted,
        "qwen_title_count": sum(
            row["model"] == str(MODEL_PATH) and row["translation_kind"] == "model" for row in rows
        ),
        "kinds": dict(Counter(row["translation_kind"] for row in rows)),
        "complete": cached_count == len(items),
    }
    (ROOT / "runtime/localization/progress.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return progress


def finalize_report() -> None:
    """CPU-only integrity audit after a completed GPU build, preserving its timings."""
    source_path = ROOT / "data/processed/catalog.jsonl"
    items = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()]
    cache = load_cache(ROOT / "runtime/localization/cache.jsonl")
    if any((item["item_id"], source_hash(item)) not in cache for item in items):
        raise RuntimeError("Full source-bound translation cache is not complete")
    review_path = ROOT / "data/processed/catalog.zh-CN.review.jsonl"
    reviews = load_cache(review_path)
    reviews.update(load_cache(ROOT / "runtime/localization/assistant_review.jsonl"))
    accepted_reviews: list[dict[str, Any]] = []
    for item in items:
        key = (item["item_id"], source_hash(item))
        row = dict(cache[key])
        if apply_assistant_review(item, row, reviews.get(key)):
            accepted_reviews.append(
                {
                    "item_id": item["item_id"],
                    "source_hash": key[1],
                    "title": row["title"],
                    "model": "coding-assistant-source-review",
                    "translation_kind": "model",
                    "translation_version": VERSION,
                }
            )
    review_temporary = review_path.with_suffix(".jsonl.tmp")
    review_temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted_reviews),
        encoding="utf-8",
    )
    review_temporary.replace(review_path)
    publish_snapshot(items, cache)
    output = ROOT / "data/processed/catalog.zh-CN.jsonl"
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    report_path = ROOT / "reports/localization_build.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    numeric_issues: list[dict[str, str]] = []
    for item, row in zip(items, rows, strict=True):
        source_numbers = numeric_values(json.dumps(item, ensure_ascii=False))
        if numeric_values(row["title"]) - source_numbers:
            numeric_issues.append({"item_id": item["item_id"], "field": "title"})
        for feature in row["features"]:
            if numeric_values(feature) - numeric_values(" ".join(item.get("features", [])[:3])):
                numeric_issues.append({"item_id": item["item_id"], "field": "features"})
    source_feature_count = sum(len(item.get("features", [])[:3]) for item in items)
    feature_count = sum(len(row["features"]) for row in rows)
    report.update(
        {
            "finalized_at": datetime.now(UTC).isoformat(),
            "item_count": len(rows),
            "source_hash_verified_count": sum(
                row["source_hash"] == source_hash(item)
                for item, row in zip(items, rows, strict=True)
            ),
            "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "assistant_review_artifact": "data/processed/catalog.zh-CN.review.jsonl",
            "assistant_review_artifact_sha256": hashlib.sha256(
                review_path.read_bytes()
            ).hexdigest(),
            "finalizer_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "qwen_raw_generation_kinds": dict(
                Counter(row["translation_kind"] for row in cache.values())
            ),
            "qwen_title_count": sum(
                row["model"] == str(MODEL_PATH) and row["translation_kind"] == "model"
                for row in rows
            ),
            "assistant_review_title_count": sum(
                row["model"] == "coding-assistant-source-review" for row in rows
            ),
            "assistant_review_note": "AI coding-assistant source-title review of failed Qwen titles; not human review and not Qwen generation. Runtime model remains the user's local Qwen3-1.7B.",
            "kinds": dict(Counter(row["translation_kind"] for row in rows)),
            "chinese_title_count": sum(
                has_chinese_description(row["title"], item)
                for item, row in zip(items, rows, strict=True)
            ),
            "source_feature_count_first_three": source_feature_count,
            "feature_count": feature_count,
            "features_omitted_or_rejected_count": max(source_feature_count - feature_count, 0),
            "source_items_without_features": sum(not item.get("features") for item in items),
            "items_with_features": sum(bool(row["features"]) for row in rows),
            "items_without_translated_features": sum(not row["features"] for row in rows),
            "final_numeric_addition_issues": numeric_issues,
            "attributes_policy": "Source fields only; controlled field-specific translations; model attributes ignored.",
            "source_bound_term_corrections": [
                "rhinestone -> 水钻 (unless diamond also appears)",
                "toner cartridge -> 碳粉盒 (unless ink also appears)",
                "gel pen -> 中性笔",
                "2-way speakerphone -> 双向免提通话",
                "Letter/Legal retain US paper-format names",
                "assorted bright colors and sizes -> 多种亮色和尺寸",
                "file bands -> 文件橡皮筋",
                "Star Mica - Bulk Mineral -> Star Mica 散装矿物",
                "remove invented common Chinese brand names unsupported by the original",
            ],
            "samples": [
                {"item_id": item["item_id"], "original_title": item["title"], "translation": row}
                for item, row in list(zip(items, rows, strict=True))[:8]
            ],
            "quality_scope": "All 3000 rows checked for source identity/hash, Chinese description and Arabic numeric additions; semantic translation fidelity and Chinese-number paraphrases are not proven.",
        }
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "item_count",
                    "kinds",
                    "feature_count",
                    "artifact_sha256",
                    "source_hash_verified_count",
                )
            }
        ),
        flush=True,
    )


def watch_cache() -> None:
    """CPU-only publishing helper; never imports Torch or starts inference."""
    items = [
        json.loads(line)
        for line in (ROOT / "data/processed/catalog.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cache_path = ROOT / "runtime/localization/cache.jsonl"
    last_count = -240
    while True:
        cache = load_cache(cache_path)
        count = len(cache)
        if count - last_count >= 240 or count == len(items):
            progress = publish_snapshot(items, cache)
            print(json.dumps(progress), flush=True)
            last_count = count
        if count == len(items):
            return
        time.sleep(20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finalize-report", action="store_true", help="CPU-only final artifact audit"
    )
    parser.add_argument(
        "--watch-cache", action="store_true", help="CPU-only periodic atomic publishing"
    )
    parser.add_argument("--batch-size", type=int, default=8, choices=[1, 2, 4, 8, 16, 32, 48])
    parser.add_argument("--trial-size", type=int, default=0)
    parser.add_argument("--max-input", type=int, default=1024)
    parser.add_argument("--max-output", type=int, default=192)
    args = parser.parse_args()
    if not 0 <= args.trial_size <= 3000:
        parser.error("--trial-size must be between 0 and 3000")
    if not 128 <= args.max_input <= 8064 or not 16 <= args.max_output <= 768:
        parser.error("--max-input must be 128..8064 and --max-output must be 16..768")
    if args.max_input + max(args.max_output, 128) > 8192:
        parser.error("Input and maximum output must fit the 8192-token context")
    if args.finalize_report:
        finalize_report()
        return
    if args.watch_cache:
        watch_cache()
        return
    started = time.monotonic()
    source_path = ROOT / "data/processed/catalog.jsonl"
    all_items = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()]
    items = all_items
    if args.trial_size:
        indices = (
            [
                index * (len(all_items) - 1) // (args.trial_size - 1)
                for index in range(args.trial_size)
            ]
            if args.trial_size > 1
            else [0]
        )
        items = [all_items[index] for index in indices]
    cache_path = ROOT / "runtime/localization/cache.jsonl"
    raw_path = ROOT / "runtime/localization/raw.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    pending = [item for item in items if (item["item_id"], source_hash(item)) not in cache]
    errors: Counter[str] = Counter()
    generated_tokens = 0
    model: Translator | None = None
    if pending:
        model = Translator(max_input=args.max_input, max_output=args.max_output)
    batch_metrics: list[dict[str, Any]] = []
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        batch_started = time.monotonic()
        assert model is not None
        raw_texts, tokens = model.translate(batch)
        generated_tokens += tokens
        parsed = [parse_translation(raw, item) for raw, item in zip(raw_texts, batch, strict=True)]
        retry_indices = [
            index
            for index, (translated, issues) in enumerate(parsed)
            if translated["title"] == rule_fallback(batch[index])["title"] and issues
        ]
        if retry_indices:
            retry_texts, retry_tokens = model.translate(
                [batch[index] for index in retry_indices], title_only=True
            )
            generated_tokens += retry_tokens
            for index, retry_raw in zip(retry_indices, retry_texts, strict=True):
                title = retry_raw.strip().strip('"').strip()
                item = batch[index]
                issues = parsed[index][1]
                if (
                    has_chinese_description(title, item)
                    and len(title) <= 240
                    and not (
                        numeric_values(title)
                        - numeric_values(str(item["title"]) + str(item["brand"]))
                    )
                ):
                    brand = str(item.get("brand") or "").strip()
                    if brand and brand.casefold() not in title.casefold():
                        title = brand + " " + title
                    parsed[index][0]["title"] = correct_terms(title, item)
                    issues.append("title_retry_success")
                else:
                    issues.append("title_retry_failed")
                raw_texts[index] += "\nTITLE_RETRY: " + retry_raw
        with cache_path.open("a", encoding="utf-8") as stream:
            for item, raw_text, (translated, issues) in zip(batch, raw_texts, parsed, strict=True):
                errors.update(issues)
                with raw_path.open("a", encoding="utf-8") as raw_stream:
                    raw_stream.write(
                        json.dumps(
                            {
                                "item_id": item["item_id"],
                                "source_hash": source_hash(item),
                                "raw_text": raw_text,
                                "issues": issues,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                row = {
                    "item_id": item["item_id"],
                    "source_hash": source_hash(item),
                    **translated,
                    "translation_kind": "rule"
                    if not translated["features"]
                    and issues
                    and translated["title"] == rule_fallback(item)["title"]
                    else "model",
                    "model": str(MODEL_PATH),
                    "translation_version": VERSION,
                }
                cache[(item["item_id"], row["source_hash"])] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        metric = {
            "offset": offset,
            "count": len(batch),
            "elapsed_seconds": round(time.monotonic() - batch_started, 3),
            "padded_generated_tokens": tokens,
        }
        batch_metrics.append(metric)
        print(
            json.dumps(
                {
                    "progress": min(offset + len(batch), len(pending)),
                    "total_pending": len(pending),
                    **metric,
                }
            ),
            flush=True,
        )
    rows = [cache[(item["item_id"], source_hash(item))] for item in items]
    for item, row in zip(items, rows, strict=True):
        if not has_chinese_description(row["title"], item):
            row.update(rule_fallback(item))
            row["translation_kind"] = "rule"
        row["attributes"] = {
            key: str(localize_value(key, value)) for key, value in attributes_for(item).items()
        }
        row["title"] = correct_terms(row["title"], item)
        row["features"] = [correct_terms(value, item) for value in row["features"]]
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": str(MODEL_PATH),
        "translation_version": VERSION,
        "source_catalog_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_hash_fields": [
            "title",
            "category",
            "category_path",
            "brand",
            "features[:3]",
            "attributes",
        ],
        "source_hash_serialization": 'json.dumps(ensure_ascii=False,sort_keys=True,separators=(",",":"))',
        "trial": bool(args.trial_size),
        "item_count": len(rows),
        "newly_translated": len(pending),
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input,
        "max_output_tokens": args.max_output,
        "temperature": 0,
        "enable_thinking": False,
        "local_files_only": True,
        "physical_gpu": 5,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "padded_generated_tokens": generated_tokens,
        "kinds": dict(Counter(row["translation_kind"] for row in rows)),
        "chinese_title_count": sum(bool(CHINESE.search(row["title"])) for row in rows),
        "feature_count": sum(len(row["features"]) for row in rows),
        "items_with_features": sum(bool(row["features"]) for row in rows),
        "validation_issues": dict(errors),
        "batches": batch_metrics,
        "translation_prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "samples": [
            {"item_id": item["item_id"], "original_title": item["title"], "translation": row}
            for item, row in list(zip(items, rows, strict=True))[:8]
        ],
        "limitations": [
            "Machine translations are display aids, never source evidence.",
            "Features are concise translations of at most the first 220 characters of each of the first three source facts.",
            "Numeric additions are rejected, but factual fidelity still requires human review.",
            "Rule fallbacks retain only the known brand and the actual source category.",
        ],
    }
    if model is not None:
        report["peak_allocated_gpu_mib"] = round(
            model.torch.cuda.max_memory_allocated() / 1024**2, 1
        )
        report["peak_reserved_gpu_mib"] = round(model.torch.cuda.max_memory_reserved() / 1024**2, 1)
    output = ROOT / (
        "runtime/localization/trial.jsonl"
        if args.trial_size
        else "data/processed/catalog.zh-CN.jsonl"
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(output)
    report_path = ROOT / (
        "reports/localization_trial.json" if args.trial_size else "reports/localization_build.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "item_count",
                    "newly_translated",
                    "elapsed_seconds",
                    "kinds",
                    "feature_count",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
