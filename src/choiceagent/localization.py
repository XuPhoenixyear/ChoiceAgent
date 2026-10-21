"""Source-bound Chinese presentation, separate from retrieval facts and evidence."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from choiceagent.config import PROJECT_ROOT
from choiceagent.schemas import Product

LOGGER = logging.getLogger(__name__)
TRANSLATION_VERSION = "zh-CN-v1"
DEFAULT_PATH = PROJECT_ROOT / "data/processed/catalog.zh-CN.jsonl"
CATEGORY_LABELS = {
    "Office Products": "办公用品",
    "Amazon Home": "家居用品",
    "All Electronics": "电子产品",
    "Tools & Home Improvement": "工具与家装",
    "Arts, Crafts & Sewing": "艺术、手工与缝纫",
    "Industrial & Scientific": "工业与科研用品",
    "Computers": "电脑及配件",
    "Home & Kitchen": "家居与厨房",
    "Toys & Games": "玩具与游戏",
    "Cell Phones & Accessories": "手机及配件",
    "Electronics": "电子产品",
    "Sports & Outdoors": "运动与户外",
    "Automotive": "汽车用品",
    "Health & Personal Care": "健康与个护",
    "Beauty & Personal Care": "美容与个护",
    "Books": "图书",
    "Pet Supplies": "宠物用品",
    "Camera & Photo": "摄影器材",
    "Musical Instruments": "乐器",
    "Grocery & Gourmet Food": "食品与饮料",
    "Clothing, Shoes & Jewelry": "服饰鞋靴与珠宝",
    "Software": "软件",
    "Video Games": "电子游戏",
    "Baby": "母婴用品",
    "Appliances": "家用电器",
    "Patio, Lawn & Garden": "庭院与园艺",
    "Handmade Products": "手工制品",
    "Amazon Devices": "亚马逊设备",
    "Gift Cards": "礼品卡",
    "Unique Finds": "特色商品",
    "Unknown": "未分类商品",
    "Unknown Category": "未分类商品",
    "Other": "其他商品",
    "AMAZON FASHION": "服饰时尚",
    "Home Audio & Theater": "家庭影音",
    "All Beauty": "美容个护",
    "Portable Audio & Accessories": "便携音频及配件",
    "Digital Music": "数字音乐",
    "Handmade": "手工制品",
}
FIELD_LABELS = {
    "title": "商品名称",
    "price_usd": "历史价格（美元）",
    "brand": "品牌",
    "category": "类别",
    "features": "功能描述",
    "material": "材质",
    "color": "颜色",
    "colour": "颜色",
    "size": "尺寸",
    "style": "款式",
    "product dimensions": "商品尺寸",
    "item dimensions": "商品尺寸",
    "item dimensions lxwxh": "商品尺寸（长×宽×高）",
    "item weight": "商品重量",
    "manufacturer": "制造商",
    "manufacturer part number": "制造商零件编号",
    "item model number": "商品型号",
    "model name": "型号名称",
    "model number": "型号",
    "number of items": "件数",
    "number of pieces": "数量",
    "unit count": "单位数量",
    "sheet size": "纸张尺寸",
    "sheet count": "纸张数量",
    "paper finish": "纸张表面",
    "number of drawers": "抽屉数量",
    "number of compartments": "隔层数量",
    "special feature": "特点",
    "special features": "特点",
    "finish type": "表面处理",
    "shape": "形状",
    "mounting type": "安装方式",
    "assembly required": "是否需要组装",
    "recommended uses for product": "建议用途",
    "specific uses for product": "适用用途",
    "product care instructions": "养护说明",
    "included components": "所含部件",
    "closure type": "闭合方式",
    "binding": "装订方式",
    "ruling": "内页格线",
    "point type": "笔尖类型",
    "ink color": "墨水颜色",
    "writing instrument form": "书写工具类型",
    "pencil lead degree (hardness)": "铅芯硬度",
    "grip type": "握柄类型",
    "country of origin": "原产国",
    "is discontinued by manufacturer": "是否已停产",
    "batteries required?": "是否需要电池",
    "batteries included?": "是否包含电池",
    "date first available": "首次上架日期",
    "package dimensions": "包装尺寸",
    "item package quantity": "每包数量",
    "capacity": "容量",
    "pattern": "图案",
    "room type": "适用空间",
    "item length": "长度",
    "item width": "宽度",
    "item height": "高度",
    "manufacturer recommended age": "制造商建议年龄",
    "connector type": "接口类型",
    "connectivity technology": "连接方式",
    "compatible devices": "兼容设备",
    "power source": "供电方式",
    "material type": "材质类型",
    "closure": "闭合或收合方式",
    "age range (description)": "适用年龄",
    "theme": "主题",
    "target audience": "适用人群",
    "format": "格式",
    "year": "年份",
    "occasion": "适用场景",
    "cover material": "封面材质",
    "compatibility options": "兼容选项",
    "line size": "线宽",
    "page yield": "标称打印页数",
}
VALUE_LABELS = {
    "black": "黑色",
    "white": "白色",
    "blue": "蓝色",
    "red": "红色",
    "green": "绿色",
    "pink": "粉色",
    "yellow": "黄色",
    "orange": "橙色",
    "purple": "紫色",
    "gray": "灰色",
    "grey": "灰色",
    "silver": "银色",
    "gold": "金色",
    "brown": "棕色",
    "beige": "米色",
    "clear": "透明",
    "transparent": "透明",
    "multicolor": "多色",
    "multi-color": "多色",
    "assorted": "混色",
    "natural": "本色",
    "navy": "藏青色",
    "rose gold": "玫瑰金色",
    "wood": "木材",
    "wooden": "木制",
    "plastic": "塑料",
    "metal": "金属",
    "bamboo": "竹材",
    "acrylic": "亚克力",
    "steel": "钢",
    "stainless steel": "不锈钢",
    "aluminum": "铝",
    "aluminium": "铝",
    "paper": "纸",
    "leather": "皮革",
    "faux leather": "人造皮革",
    "pu leather": "PU人造皮革",
    "mesh": "网状材质",
    "rubber": "橡胶",
    "silicone": "硅胶",
    "fabric": "织物",
    "polyester": "聚酯纤维",
    "cardboard": "纸板",
    "glass": "玻璃",
    "ceramic": "陶瓷",
    "nylon": "尼龙",
    "yes": "是",
    "no": "否",
    "true": "是",
    "false": "否",
    "small": "小号",
    "medium": "中号",
    "large": "大号",
    "standard": "标准款",
    "classic": "经典款",
    "modern": "现代款",
    "rectangular": "长方形",
    "rectangle": "长方形",
    "round": "圆形",
    "square": "方形",
    "freestanding": "独立放置",
    "tabletop": "桌面放置",
    "desktop": "桌面",
    "wall mount": "壁挂",
    "wall mounted": "壁挂",
    "matte": "哑光",
    "glossy": "亮面",
    "fine": "细尖",
    "medium point": "中等笔尖",
    "extra fine": "特细尖",
    "ballpoint pen": "圆珠笔",
    "gel pen": "中性笔",
    "fountain pen": "钢笔",
    "mechanical pencil": "自动铅笔",
    "china": "中国",
    "united states": "美国",
    "japan": "日本",
    "germany": "德国",
    "india": "印度",
    "vietnam": "越南",
    "taiwan": "中国台湾",
    "korea": "韩国",
}
# The same English word has different meanings across attributes. Restrict
# controlled values to their source field instead of applying global rewrites.
_COLOR_VALUES = frozenset(
    [
        "black",
        "white",
        "blue",
        "red",
        "green",
        "pink",
        "yellow",
        "orange",
        "purple",
        "gray",
        "grey",
        "silver",
        "gold",
        "brown",
        "beige",
        "clear",
        "transparent",
        "multicolor",
        "multi-color",
        "assorted",
        "natural",
        "navy",
        "rose gold",
    ]
)
_MATERIAL_VALUES = frozenset(
    [
        "wood",
        "wooden",
        "plastic",
        "metal",
        "bamboo",
        "acrylic",
        "steel",
        "stainless steel",
        "aluminum",
        "aluminium",
        "paper",
        "leather",
        "faux leather",
        "pu leather",
        "mesh",
        "rubber",
        "silicone",
        "fabric",
        "polyester",
        "cardboard",
        "glass",
        "ceramic",
        "nylon",
    ]
)
_IDENTITY_FIELDS = frozenset(
    {
        "brand",
        "manufacturer",
        "item model number",
        "model name",
        "model number",
        "manufacturer part number",
        "asin",
    }
)
_BOOLEAN_FIELDS = frozenset(
    {
        "is discontinued by manufacturer",
        "assembly required",
        "batteries required?",
        "batteries included?",
        "batteries required",
        "batteries included",
    }
)
_UNIT_FIELDS = frozenset(
    {
        "size",
        "sheet size",
        "product dimensions",
        "item dimensions",
        "item dimensions lxwxh",
        "package dimensions",
        "item weight",
        "item length",
        "item width",
        "item height",
        "line size",
        "capacity",
        "unit count",
        "features",
    }
)
_FIELD_VALUE_LABELS: dict[str, dict[str, str]] = {
    **{
        field: {key: VALUE_LABELS[key] for key in _COLOR_VALUES}
        for field in ("color", "colour", "ink color")
    },
    **{
        field: {key: VALUE_LABELS[key] for key in _MATERIAL_VALUES}
        for field in ("material", "material type", "cover material")
    },
    **{field: {"yes": "是", "no": "否", "true": "是", "false": "否"} for field in _BOOLEAN_FIELDS},
    "size": {
        "small": "小号",
        "medium": "中号",
        "large": "大号",
        "standard": "标准规格",
        "one size": "统一规格",
        "letter": "美式 Letter 规格",
    },
    "sheet size": {"letter": "美式 Letter 规格"},
    "point type": {
        "fine": "细尖",
        "medium": "中等笔尖",
        "medium point": "中等笔尖",
        "extra fine": "特细尖",
        "broad": "粗尖",
    },
    "style": {"standard": "标准款", "classic": "经典款", "modern": "现代款"},
    "shape": {key: VALUE_LABELS[key] for key in ("rectangular", "rectangle", "round", "square")},
    "mounting type": {
        key: VALUE_LABELS[key]
        for key in ("freestanding", "tabletop", "desktop", "wall mount", "wall mounted")
    },
    "finish type": {"matte": "哑光", "glossy": "亮面", "natural": "本色"},
    "writing instrument form": {
        key: VALUE_LABELS[key]
        for key in ("ballpoint pen", "gel pen", "fountain pen", "mechanical pencil")
    },
    "country of origin": {
        key: VALUE_LABELS[key]
        for key in (
            "china",
            "united states",
            "japan",
            "germany",
            "india",
            "vietnam",
            "taiwan",
            "korea",
        )
    },
}
# Longest/specific expressions first; these are source-title labels, not predicted facts.
PRODUCT_TYPES = [
    (r"\b(?:chair casters?|chair wheels?)\b", "办公椅脚轮"),
    (r"\b(?:armrest|arm rest) pads?\b", "扶手垫"),
    (r"\b(?:mouse ?pads?|mouse mats?)\b", "鼠标垫"),
    (r"\b(?:ink cartridges?|ink refills?)\b", "墨水替换装"),
    (r"\b(?:desk|desktop) organizer", "桌面收纳架"),
    (r"\b(?:pencil|pen) (?:holder|cup|organizer)", "笔筒与文具收纳"),
    (r"\b(?:file|letter|paper) (?:organizer|sorter|tray)", "文件收纳架"),
    (r"\boffice chair\b", "办公椅"),
    (r"\b(?:fountain|gel|ballpoint) pens?\b", "书写笔"),
    (r"\b(?:copy|printer|printing) paper\b", "打印纸"),
    (r"\b(?:staplers?|staples)\b", "订书用品"),
    (r"\b(?:notebooks?|journals?)\b", "笔记本"),
    (r"\b(?:whiteboards?|dry erase board)\b", "白板"),
    (r"\bfolders?\b", "文件夹"),
    (r"\b(?:binders?|binding)\b", "装订用品"),
    (r"\b(?:pencils?|pens?)\b", "书写文具"),
    (r"\b(?:labels?|stickers?)\b", "标签贴纸"),
    (r"\b(?:tapes?|adhesive)\b", "胶带与粘贴用品"),
    (r"\b(?:envelopes?|mailers?)\b", "信封与邮寄用品"),
    (r"\b(?:storage|organizer|holder|tray)\b", "收纳用品"),
]


def _source(product: Product | Mapping[str, Any]) -> dict[str, Any]:
    data = product.model_dump() if isinstance(product, Product) else product
    return {
        "title": data.get("title", ""),
        "category": data.get("category", "Office Products"),
        "category_path": data.get("category_path", []),
        "brand": data.get("brand"),
        "features": list(data.get("features", []))[:3],
        "attributes": data.get("attributes", {}),
    }


def source_hash(product: Product | Mapping[str, Any]) -> str:
    """Hash exactly the source fields covered by the versioned display sidecar."""
    payload = json.dumps(
        _source(product), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def category_label(raw: str) -> str:
    return CATEGORY_LABELS.get(raw, raw if re.search(r"[\u4e00-\u9fff]", raw) else "其他商品")


def field_label(raw: str) -> str:
    name = raw.removeprefix("attributes.").strip()
    if re.search(r"[\u4e00-\u9fff]", name):
        return name
    return FIELD_LABELS.get(name.casefold(), "其他属性")


def localize_value(field: str, value: Any) -> Any:
    """Translate controlled labels/units; preserve amounts, brand names and model IDs."""
    if value is None:
        return None
    normalized_field = field.removeprefix("attributes.").strip().casefold()
    if normalized_field in _IDENTITY_FIELDS:
        return value
    if isinstance(value, list):
        return [localize_value(field, part) for part in value]
    if not isinstance(value, str):
        return value
    if normalized_field == "category":
        return category_label(value)
    dictionary = _FIELD_VALUE_LABELS.get(normalized_field, {})
    if value.casefold().strip() in dictionary:
        return dictionary[value.casefold().strip()]
    if normalized_field not in _UNIT_FIELDS:
        return value
    localized = value
    for english, chinese in (
        (r"\binches\b|\binch\b", "英寸"),
        (r"\bpounds?\b", "磅"),
        (r"\bounces?\b", "盎司"),
        (r"\bcentimeters?\b", "厘米"),
        (r"\bmillimeters?\b", "毫米"),
        (r"\bgrams?\b", "克"),
    ):
        localized = re.sub(english, chinese, localized, flags=re.I)
    return localized


def fallback_display(product: Product | Mapping[str, Any]) -> dict[str, Any]:
    """Give a conservative Chinese title using only matched original title terms."""
    source = _source(product)
    title = str(source["title"])
    kind = next(
        (label for expression, label in PRODUCT_TYPES if re.search(expression, title, re.I)), None
    )
    chinese_title = (
        title
        if re.search(r"[\u4e00-\u9fff]", title)
        else kind or category_label(source["category"])
    )
    brand = source["brand"]
    if brand and brand.casefold() not in chinese_title.casefold():
        chinese_title = f"{brand} · {chinese_title}"
    return {
        "locale": "zh-CN",
        "title": chinese_title,
        "category": category_label(source["category"]),
        "features": [],
        "attributes": {
            field_label(key): str(localize_value(key, value))
            for key, value in source["attributes"].items()
            if field_label(key) != "其他属性"
        },
        "translation_kind": "rule",
        "source_language": "en",
    }


class CatalogLocalizer:
    """Hot-reload an atomically written translation sidecar, with safe stale fallback."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_PATH
        self._records: dict[str, dict[str, Any]] = {}
        self._signature: tuple[int, int] | None = None
        self._lock = threading.Lock()

    def _reload(self) -> None:
        try:
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            self._records = {}
            self._signature = None
            return
        if signature == self._signature:
            return
        try:
            records = {}
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict) and isinstance(row.get("item_id"), str):
                        records[row["item_id"]] = row
        except (OSError, ValueError):
            LOGGER.warning("Ignoring unreadable localization sidecar")
            return
        self._records, self._signature = records, signature

    def display(self, product: Product | Mapping[str, Any]) -> dict[str, Any]:
        fallback = fallback_display(product)
        data = product.model_dump() if isinstance(product, Product) else product
        with self._lock:
            self._reload()
            row = self._records.get(str(data.get("item_id", "")))
            if (
                not row
                or row.get("source_hash") != source_hash(product)
                or row.get("translation_version") != TRANSLATION_VERSION
                or row.get("translation_kind") == "rule"
            ):
                return fallback
            title = row.get("title")
            if (
                not isinstance(title, str)
                or not re.search(r"[\u4e00-\u9fff]", title)
                or len(title) > 2000
            ):
                return fallback
            features = row.get("features", [])
            if not isinstance(features, list) or any(
                not isinstance(value, str) or not re.search(r"[\u4e00-\u9fff]", value)
                for value in features
            ):
                features = []
            feature_limit = min(3, len(data.get("features", [])))
            attributes = dict(fallback["attributes"])
            for key, value in (
                row.get("attributes", {}).items() if isinstance(row.get("attributes"), dict) else []
            ):
                if (
                    key in data.get("attributes", {})
                    and isinstance(value, str)
                    and field_label(key) != "其他属性"
                ):
                    # Brand/model identifiers must remain the original exact values.
                    raw = data["attributes"][key]
                    attributes[field_label(key)] = (
                        str(localize_value(key, raw))
                        if key.casefold() in _IDENTITY_FIELDS
                        else value
                    )
            return {
                **fallback,
                "title": title,
                "category": category_label(str(data.get("category", "Office Products"))),
                "features": features[:feature_limit],
                "attributes": attributes,
                "translation_kind": row.get("translation_kind")
                if row.get("translation_kind") in {"model", "rule"}
                else "model",
            }


DEFAULT_LOCALIZER = CatalogLocalizer()


def display_for(product: Product | Mapping[str, Any]) -> dict[str, Any]:
    return DEFAULT_LOCALIZER.display(product)
