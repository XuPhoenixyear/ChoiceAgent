"""Chinese, HTTP-only Streamlit client. Run with ``streamlit run``.

The UI deliberately imports no retrieval, generation, or GPU modules. Network
requests are confined to ApiClient and happen only at connection or user actions.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"
FIELD_NAMES = {
    "price_usd": "历史美元价格",
    "brand": "品牌",
    "category": "类别",
    "title": "商品名称",
    "features": "商品特点",
    "description": "商品说明",
}
MODE_NAMES = {"综合匹配": "hybrid", "关键词匹配": "lexical", "语义匹配": "dense"}
QUICK_QUERIES = {
    "桌面收纳": "桌面收纳盒",
    "便携打印机": "便携打印机",
    "记笔记用的本子": "记笔记用的笔记本",
}
CATEGORY_NAMES = {
    "Office Products": "办公用品",
    "Amazon Home": "家居用品",
    "All Electronics": "电子产品",
    "Industrial & Scientific": "工业与科研用品",
    "Tools & Home Improvement": "工具与家装",
    "Computers": "电脑及配件",
    "AMAZON FASHION": "服饰",
    "Arts, Crafts & Sewing": "艺术手工与缝纫",
    "Books": "图书",
    "Toys & Games": "玩具与游戏",
    "Health & Personal Care": "健康与个人护理",
    "Sports & Outdoors": "运动与户外",
    "Cell Phones & Accessories": "手机及配件",
    "Camera & Photo": "摄影摄像",
    "Home Audio & Theater": "家庭音响与影音",
    "All Beauty": "美容护理",
    "Baby": "母婴用品",
    "Automotive": "汽车用品",
    "Portable Audio & Accessories": "便携音频及配件",
    "Digital Music": "数字音乐",
    "Video Games": "电子游戏",
    "Musical Instruments": "乐器",
    "Handmade": "手工制品",
}
CSS = """
<style>
.stApp {background: #f7f9fc; color: #18283e;}
[data-testid="stHeader"] {background: #f7f9fc;}
[data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stSidebar"] {background: #eef3f9;}
[data-testid="stMainBlockContainer"] {max-width: 1180px; padding-top: 2.4rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
.stButton button {border-radius: 9px;}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"]
{background: #365fe4; border-color: #365fe4;}
[data-testid="stChatMessage"] {background: #fff; border-radius: 14px;}
h1, h2, h3 {letter-spacing: -.025em;}
</style>
"""


class ApiError(Exception):
    """A public, safe error: never contains backend messages or request bodies."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def api_base_url(value: str) -> str:
    """Validate without ever echoing an environment value that may hold secrets."""
    value = value.strip().rstrip("/")
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
        )
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("服务地址配置有误，请联系维护者检查。")
    return value


@dataclass
class ApiClient:
    base_url: str

    def request(
        self, method: str, path: str, *, payload: dict | None = None, params: dict | None = None
    ) -> dict[str, Any]:
        # No implicit retries: uncertain POSTs are retried explicitly with the
        # original event_id/request_id retained in Streamlit session state.
        timeout = httpx.Timeout(
            120.0 if path == "/v1/chat" else 40.0, connect=3.0, write=10.0, pool=3.0
        )
        try:
            with httpx.Client(
                base_url=self.base_url, timeout=timeout, trust_env=False, follow_redirects=False
            ) as client:
                response = client.request(method, path, json=payload, params=params)
        except httpx.TimeoutException as exc:
            raise ApiError("等待回复超时。可以稍后重试，已输入的内容仍会保留。") from exc
        except httpx.HTTPError as exc:
            raise ApiError("暂时连接不上选物服务。请稍后重试，或请维护者检查服务。") from exc
        if response.status_code >= 300:
            messages = {
                400: "这次请求无法处理，请调整输入后重试。",
                401: "服务暂时无法验证当前连接，请联系维护者。",
                403: "无法访问这条会话或资料，请重新开始一次会话。",
                404: "这条会话或商品已不可用，请刷新会话，必要时重新开始。",
                409: "这条请求仍在处理或发生冲突，请稍后重试原请求。",
                422: "输入不符合要求，请缩短内容或检查筛选条件。",
                429: "服务正忙，请稍后重试。",
                503: "部分服务还未准备好，请稍后再试。",
            }
            raise ApiError(
                messages.get(response.status_code, "服务暂时没有完成这次请求，请稍后重试。"),
                status=response.status_code,
            )
        try:
            result = response.json()
        except (ValueError, UnicodeError) as exc:
            raise ApiError("收到的回复不完整，请稍后重试。") from exc
        if not isinstance(result, dict):
            raise ApiError("收到的回复格式不正确，请稍后重试。")
        return result

    def session(self, session_id: str) -> dict:
        return self.request("GET", f"/v1/sessions/{session_id}")


def format_price(value: Any) -> str:
    """Do not turn missing or invalid catalog prices into an attractive $0."""
    if value is None or isinstance(value, bool):
        return "价格未提供"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "价格未提供"
    if not math.isfinite(number) or number < 0:
        return "价格未提供"
    return f"US${number:,.2f} · 历史价格"


def facet_options(value: Any) -> list[str]:
    """Accept count maps and the common structured stats representations."""
    values: list[Any]
    if isinstance(value, Mapping):
        values = list(value.keys())
    elif isinstance(value, list):
        values = [
            entry.get("name", entry.get("value", entry.get("category", entry.get("brand"))))
            if isinstance(entry, dict)
            else entry
            for entry in value
        ]
    else:
        values = []
    return sorted({str(x).strip() for x in values if x is not None and str(x).strip()})


def split_terms(value: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for x in re.split(r"[,，;；\n]", value) if x.strip()))


def build_constraints(
    *,
    budget_enabled: bool,
    minimum: float,
    maximum: float,
    category: str,
    excluded_brands: list[str],
    required: str,
    excluded: str,
) -> dict:
    if budget_enabled and (minimum < 0 or maximum < 0 or minimum > maximum):
        raise ValueError("最低预算不能高于最高预算，金额也不能为负。")
    return {
        "min_price_usd": minimum if budget_enabled and minimum > 0 else None,
        "max_price_usd": maximum if budget_enabled else None,
        "excluded_brands": excluded_brands,
        "category": category if category != "全部类别" else None,
        "required_terms": split_terms(required),
        "excluded_terms": split_terms(excluded),
    }


def display_value(value: Any, dimension: str = "") -> str:
    if dimension == "price_usd":
        return format_price(value)
    if value is None or value == "" or value == []:
        return "资料未提供"
    if isinstance(value, list):
        return "；".join(str(x) for x in value)
    if isinstance(value, dict):
        return "；".join(f"{k}：{v}" for k, v in value.items())
    return str(value)


def chinese_text(value: Any) -> str | None:
    """Accept supplied Chinese text; never synthesize translated product facts."""
    if isinstance(value, str) and re.search(r"[\u3400-\u9fff]", value):
        return value.strip()
    return None


def product_display(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = item.get("display")
    if isinstance(value, Mapping) and value.get("locale") == "zh-CN":
        return value
    return {}


def category_label(raw: str, labels: Any = None) -> str:
    if raw == "全部类别":
        return raw
    provided = labels.get(raw) if isinstance(labels, Mapping) else None
    return chinese_text(provided) or CATEGORY_NAMES.get(raw) or chinese_text(raw) or "其他商品类别"


def product_title(item: Mapping[str, Any]) -> str:
    return chinese_text(product_display(item).get("title")) or "商品信息（原文可查看）"


def product_category(item: Mapping[str, Any]) -> str:
    return chinese_text(product_display(item).get("category")) or category_label(
        str(item.get("category") or "")
    )


def product_features(item: Mapping[str, Any]) -> list[str]:
    values = product_display(item).get("features")
    if not isinstance(values, list):
        return []
    return [text for value in values if (text := chinese_text(value))]


def comparison_value(entry: dict, dimension: str, item: Mapping[str, Any]) -> str:
    raw = entry.get("value")
    if raw is None or raw == "" or raw == [] or dimension in {"price_usd", "brand"}:
        return display_value(raw, dimension)
    translated = entry.get("display_value")
    if translated is not None and translated != "":
        return display_value(translated)
    if dimension == "title":
        return product_title(item)
    if dimension == "category":
        return chinese_text(product_display(item).get("category")) or category_label(str(raw))
    if isinstance(raw, (int, float)) or chinese_text(raw):
        return display_value(raw)
    return "原文可查看"


def comparison_table(
    comparison: dict, cards: dict[str, dict], *, original: bool = False
) -> list[dict]:
    rows = []
    # Prefix by position so two products with identical titles remain distinct.
    order = list(
        dict.fromkeys(
            str(v["item_id"])
            for row in comparison.get("rows", [])
            for v in row.get("values", [])
            if v.get("item_id")
        )
    )
    labels = {
        item_id: f"商品 {i + 1} · "
        + (
            str(cards.get(item_id, {}).get("title") or "商品名称未提供")[:48]
            if original
            else product_title(cards.get(item_id, {}))[:48]
        )
        for i, item_id in enumerate(order)
    }
    for row in comparison.get("rows", []):
        dimension = row.get("dimension", "")
        label = (
            dimension.removeprefix("attributes.")
            if original
            else chinese_text(row.get("label_zh")) or FIELD_NAMES.get(dimension) or "其他规格"
        )
        output = {"比较项目": label}
        values = {str(v.get("item_id")): v for v in row.get("values", [])}
        output.update(
            {
                labels[item_id]: (
                    display_value(values.get(item_id, {}).get("value"), dimension)
                    if original
                    else comparison_value(
                        values.get(item_id, {}), dimension, cards.get(item_id, {})
                    )
                )
                for item_id in order
            }
        )
        rows.append(output)
    return rows


def redact_debug(value: Any) -> Any:
    """Redact credentials/session tokens even in explicitly requested traces."""
    sensitive = re.compile(
        r"(^|[_-])(authorization|cookie|password|secret|api[_-]?key|"
        r"access[_-]?token|refresh[_-]?token|session[_-]?id|token)($|[_-])",
        re.I,
    )
    if isinstance(value, Mapping):
        return {
            str(k): "[已隐藏]" if sensitive.search(str(k)) else redact_debug(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_debug(x) for x in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)(bearer\s+)[\w.\-]+", r"\1[已隐藏]", value)
        value = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[已隐藏]@", value)
        return re.sub(
            r"(?i)([?&](?:api_key|token|key|secret|password)=)[^\s&#]+", r"\1[已隐藏]", value
        )
    return value


def init_state() -> None:
    defaults: dict[str, Any] = {
        "session_id": None,
        "connection_attempted": False,
        "profile": {},
        "stats": {},
        "cards": {},
        "search_result": None,
        "last_query": "",
        "turns": [],
        "pending_chat": None,
        "pending_feedback": None,
        "selected": [],
        "comparison": None,
        "trace_id": None,
        "trace_cache": {},
        "notice": None,
    }
    for key, value in defaults.items():
        if f"ca_{key}" not in st.session_state:
            st.session_state[f"ca_{key}"] = value


def remember_result(result: dict) -> None:
    for item in result.get("items", []):
        if isinstance(item, dict) and item.get("item_id"):
            st.session_state.ca_cards[str(item["item_id"])] = item
    if result.get("trace_id"):
        st.session_state.ca_trace_id = result["trace_id"]


def notice(kind: str, text: str) -> None:
    st.session_state.ca_notice = (kind, text)


def connect(client: ApiClient) -> None:
    st.session_state.ca_connection_attempted = True
    try:
        health = client.request("GET", "/health")
        result = client.request("POST", "/v1/sessions", payload={})
        if not result.get("session_id"):
            raise ApiError("暂时无法创建会话，请稍后重试。")
        st.session_state.ca_session_id = str(result["session_id"])
        st.session_state.ca_profile = result
        st.session_state.ca_notice = None
        try:
            st.session_state.ca_stats = client.request("GET", "/v1/catalog/stats")
        except ApiError:
            st.session_state.ca_stats = {"count": health.get("catalog_count")}
    except ApiError as exc:
        notice("error", str(exc))


def send_feedback(client: ApiClient, item_id: str | None = None, action: str | None = None) -> None:
    pending = st.session_state.ca_pending_feedback
    if pending is None:
        pending = {
            "session_id": st.session_state.ca_session_id,
            "event_id": str(uuid4()),
            "item_id": item_id,
            "action": action,
        }
        st.session_state.ca_pending_feedback = pending
    try:
        result = client.request("POST", "/v1/feedback", payload=pending)
        if result.get("event_id") != pending["event_id"]:
            raise ApiError("偏好记录尚未确认，请重试这次操作。")
        st.session_state.ca_profile.update(result)
        st.session_state.ca_pending_feedback = None
        notice("success", "偏好已记录。开启“参考我的偏好”后，新搜索会使用这些选择。")
    except ApiError as exc:
        notice("warning", str(exc))


def send_chat(client: ApiClient, message: str | None = None) -> None:
    pending = st.session_state.ca_pending_chat
    if pending is None:
        pending = {
            "session_id": st.session_state.ca_session_id,
            "request_id": str(uuid4()),
            "message": message,
        }
        st.session_state.ca_pending_chat = pending
        st.session_state.ca_turns.append(
            {"request_id": pending["request_id"], "message": message, "response": None}
        )
    try:
        result = client.request("POST", "/v1/chat", payload=pending)
        if result.get("request_id") != pending["request_id"]:
            raise ApiError("这条回复尚未确认，请重新获取原来的回复。")
        for turn in st.session_state.ca_turns:
            if turn["request_id"] == pending["request_id"]:
                turn["response"] = result
                break
        remember_result(result)
        st.session_state.ca_pending_chat = None
        st.session_state.ca_notice = None
    except ApiError as exc:
        notice("warning", str(exc))


def select_comparison(item_id: str) -> None:
    selected = st.session_state.ca_selected
    if item_id in selected:
        selected.remove(item_id)
    else:
        selected.append(item_id)
        st.session_state.ca_selected = selected[-2:]
    st.session_state.ca_comparison = None


def status_message(result: dict) -> None:
    messages = {
        "no_match": "没有找到符合条件的商品。试试放宽预算、减少限制或换个关键词。",
        "retrieval_incomplete": "这次只能检索部分资料，结果可能不完整。你可以稍后再试。",
        "degraded": "部分功能暂时不可用，先展示已经找到的结果。",
    }
    if result.get("status") in messages:
        st.info(messages[result["status"]])
    elif result.get("warnings"):
        st.caption("部分资料或功能有限制，请结合商品信息和资料依据判断。")


def render_cards(items: list[dict], scope: str, client: ApiClient) -> None:
    profile = st.session_state.ca_profile
    locked = st.session_state.ca_pending_feedback is not None
    for index, item in enumerate(items):
        item_id = str(item.get("item_id", ""))
        if not item_id:
            continue
        key = f"{scope}_{index}_{item_id}"
        with st.container(border=True):
            st.subheader(product_title(item), anchor=False)
            st.markdown(f"**{format_price(item.get('price_usd'))}**")
            st.caption(
                " · ".join(str(x) for x in [item.get("brand"), product_category(item)] if x)
                or "品牌与类别暂无记录"
            )
            features = product_features(item)
            for feature in features[:3]:
                st.write(f"• {feature}")
            if not features:
                st.caption("中文商品说明暂未准备好，可展开查看英文原文。")
            if item.get("missing_attributes"):
                st.caption("部分规格未提供，不能据此推断商品不具备相应功能。")
            columns = st.columns(4)
            for column, action, label, field in [
                (columns[0], "like", "喜欢", "liked_item_ids"),
                (columns[1], "dislike", "不感兴趣", "disliked_item_ids"),
                (columns[2], "save", "收藏", "shortlist_item_ids"),
            ]:
                marked = item_id in profile.get(field, [])
                if column.button(
                    ("已" if marked else "") + label,
                    key=f"{key}_{action}",
                    disabled=locked or marked,
                    use_container_width=True,
                ):
                    send_feedback(client, item_id, action)
                    st.rerun()
            selected = item_id in st.session_state.ca_selected
            if columns[3].button(
                "移出对比" if selected else "加入对比",
                key=f"{key}_compare",
                use_container_width=True,
            ):
                select_comparison(item_id)
                st.rerun()
            if item_id in profile.get("shortlist_item_ids", []) and st.button(
                "移出清单", key=f"{key}_remove", disabled=locked
            ):
                send_feedback(client, item_id, "remove")
                st.rerun()
            with st.expander("查看英文原文"):
                st.write("原始名称：" + str(item.get("title") or "资料未提供"))
                st.write("原始类别：" + str(item.get("category") or "资料未提供"))
                for feature in item.get("features") or []:
                    st.write(f"• {feature}")
                attributes = item.get("attributes")
                if isinstance(attributes, Mapping):
                    for field, value in attributes.items():
                        st.write(f"{field}：{display_value(value)}")
            with st.expander("查看资料依据"):
                evidence = item.get("evidence") or []
                if not evidence:
                    st.caption("本次回复未附逐项依据，请先核对商品资料。")
                for entry in evidence:
                    field = str(entry.get("field", ""))
                    label = FIELD_NAMES.get(
                        field,
                        "商品特点"
                        if field.startswith("features.")
                        else field.removeprefix("attributes."),
                    )
                    st.write(f"{label}：{display_value(entry.get('value'), field)}")
                    if entry.get("source_locator"):
                        st.caption(f"原始记录位置：{entry['source_locator']}")


def render_comparison(result: dict) -> None:
    table = comparison_table(result, st.session_state.ca_cards)
    if table:
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption("缺失信息保留为空缺说明；历史美元价不含实时库存、运费或税费。")
        with st.expander("查看比较原文"):
            st.dataframe(
                comparison_table(result, st.session_state.ca_cards, original=True),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.info("本次没有返回可比较的资料。")
    if result.get("warnings"):
        st.caption("部分比较信息有限，请结合商品的资料依据判断。")


def render_sidebar(client: ApiClient) -> dict:
    with st.sidebar:
        st.title("你的选物条件")
        stats = st.session_state.ca_stats
        count = stats.get("count")
        if isinstance(count, int):
            st.caption(f"资料库收录 {count:,} 件商品")
        st.subheader("搜索筛选")
        st.caption("以下条件用于搜索；聊天时请在消息中告诉助手你的预算和要求。")
        enabled = st.toggle("设置历史价格预算", key="ca_budget_enabled")
        lower = st.number_input(
            "最低价格（美元）",
            min_value=0.0,
            value=0.0,
            step=5.0,
            disabled=not enabled,
            key="ca_minimum",
        )
        upper = st.number_input(
            "最高价格（美元）",
            min_value=0.0,
            value=50.0,
            step=5.0,
            disabled=not enabled,
            key="ca_maximum",
        )
        st.caption("仅按资料中的历史美元价筛选，非当前售价。")
        category = st.selectbox(
            "类别",
            ["全部类别", *facet_options(stats.get("categories"))],
            key="ca_category",
            format_func=lambda value: category_label(value, stats.get("category_labels")),
        )
        brands = st.multiselect(
            "排除品牌",
            facet_options(stats.get("brands")),
            key="ca_brands",
            placeholder="选择要排除的品牌",
        )
        with st.expander("更多筛选"):
            required = st.text_input(
                "必须包含的词", max_chars=200, key="ca_required", help="多个词用逗号分隔。"
            )
            excluded = st.text_input("不要出现的词", max_chars=200, key="ca_excluded")
        mode = st.selectbox("匹配方式", list(MODE_NAMES), key="ca_mode")
        personalize = st.toggle("参考我的偏好", value=True, key="ca_personalize")
        st.caption("喜欢、不感兴趣和收藏会保存在本次会话。")
        profile = st.session_state.ca_profile
        st.caption(
            f"已喜欢 {len(profile.get('liked_item_ids', []))} 件 · "
            f"已收藏 {len(profile.get('shortlist_item_ids', []))} 件"
        )
        if st.button("刷新已保存的偏好", use_container_width=True):
            try:
                st.session_state.ca_profile = client.session(st.session_state.ca_session_id)
                notice("success", "已更新本次会话保存的偏好。")
            except ApiError as exc:
                notice("warning", str(exc))
            st.rerun()
        if st.button(
            "开始新的会话",
            use_container_width=True,
            help="开始独立的选择记录。旧会话不会被删除，但此页面不再显示它。",
        ):
            for key in [
                "session_id",
                "profile",
                "cards",
                "search_result",
                "turns",
                "pending_chat",
                "pending_feedback",
                "selected",
                "comparison",
                "trace_id",
                "trace_cache",
                "notice",
            ]:
                st.session_state.pop(f"ca_{key}", None)
            st.session_state.ca_connection_attempted = False
            st.rerun()
        return dict(
            budget_enabled=enabled,
            minimum=lower,
            maximum=upper,
            category=category,
            excluded_brands=brands,
            required=required,
            excluded=excluded,
            mode=MODE_NAMES[mode],
            personalize=personalize,
        )


def render_search(client: ApiClient, filters: dict) -> None:
    st.caption("也可以从一个简单需求开始：")
    for column, (label, query) in zip(st.columns(3), QUICK_QUERIES.items(), strict=True):
        if column.button(label, key=f"ca_quick_{query}", use_container_width=True):
            st.session_state.ca_query = query
    with st.form("ca_search_form"):
        query = st.text_input(
            "想找什么？", placeholder="例如：小桌面用的收纳盒", max_chars=500, key="ca_query"
        )
        submitted = st.form_submit_button("找一找", type="primary", use_container_width=True)
    if submitted:
        if not query.strip():
            st.warning("先输入想找的商品或用途。")
        else:
            try:
                options = {k: v for k, v in filters.items() if k not in {"mode", "personalize"}}
                constraints = build_constraints(**options)
                with st.spinner("正在找合适的商品…"):
                    result = client.request(
                        "POST",
                        "/v1/search",
                        payload={
                            "session_id": st.session_state.ca_session_id,
                            "query": query.strip(),
                            "constraints": constraints,
                            "limit": 5,
                            "mode": filters["mode"],
                            "personalize": filters["personalize"],
                        },
                    )
                st.session_state.ca_search_result = result
                st.session_state.ca_last_query = query.strip()
                st.session_state.ca_notice = None
                remember_result(result)
            except (ValueError, ApiError) as exc:
                st.warning(str(exc))
    result = st.session_state.ca_search_result
    if result is not None:
        st.caption(f"最近一次搜索：{st.session_state.ca_last_query}")
        status_message(result)
        render_cards(result.get("items", []), "search", client)
    else:
        st.info("输入用途和关键词开始挑选。喜欢或收藏几件商品后，可以再搜索看看。")


def render_chat(client: ApiClient) -> None:
    if not st.session_state.ca_turns:
        st.info("告诉我用途、预算和偏好，例如：我想买一个适合小桌面的收纳盒，预算20美元。")
    for turn in st.session_state.ca_turns:
        with st.chat_message("user"):
            st.write(turn["message"])
        response = turn.get("response")
        if response is not None:
            with st.chat_message("assistant"):
                st.markdown(
                    str(response.get("answer") or "这次未返回文字说明，请参考下面的商品资料。")
                )
                status_message(response)
                render_cards(response.get("items", []), f"chat_{turn['request_id']}", client)
                if response.get("comparison"):
                    render_comparison(response["comparison"])
    if st.session_state.ca_pending_chat:
        st.warning("上一条回复还没有确认。重试会继续同一条消息。")
        if st.button("重新获取这条回复", key="ca_retry_chat"):
            with st.spinner("正在继续获取回复…"):
                send_chat(client)
            st.rerun()
    message = st.chat_input(
        "说说你的用途、预算和偏好…",
        max_chars=2000,
        disabled=st.session_state.ca_pending_chat is not None,
    )
    if message and message.strip():
        with st.spinner("正在理解你的需求…"):
            send_chat(client, message.strip())
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="ChoiceAgent · 选物助手", page_icon="🧭", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    st.title("ChoiceAgent · 选物助手")
    st.write("把需求说清楚，把选择看明白。")
    st.caption("商品信息与美元价格来自历史资料，不代表实时售价或库存。")
    st.caption("中文商品资料为辅助译文，可展开核对英文原文。")
    try:
        client = ApiClient(api_base_url(os.getenv("CHOICEAGENT_API_URL", DEFAULT_API_URL)))
    except ValueError as exc:
        st.error(str(exc))
        return
    if st.session_state.ca_session_id is None:
        if not st.session_state.ca_connection_attempted:
            with st.spinner("正在连接选物服务…"):
                connect(client)
        if st.session_state.ca_session_id is None:
            st.info("选物服务暂时不可用，连接恢复后即可开始。")
            if st.session_state.ca_notice:
                st.warning(st.session_state.ca_notice[1])
            if st.button("重新连接", type="primary"):
                connect(client)
                st.rerun()
            return
    filters = render_sidebar(client)
    # Keep the tabs' position stable when notices appear/disappear. Otherwise
    # Streamlit can recreate the tab group and switch a completed chat to Search.
    with st.container():
        if st.session_state.ca_notice:
            kind, text = st.session_state.ca_notice
            getattr(st, kind)(text)
        if st.session_state.ca_pending_feedback:
            st.warning("还有一条偏好记录尚未确认，其他偏好按钮暂时停用。")
            if st.button("重试保存这条偏好", key="ca_retry_feedback"):
                send_feedback(client)
                st.rerun()
    search_tab, chat_tab = st.tabs(["搜索商品", "聊聊需求"])
    with search_tab:
        render_search(client, filters)
    with chat_tab:
        render_chat(client)
    st.divider()
    st.subheader("放在一起比一比")
    selected = st.session_state.ca_selected
    if not selected:
        st.caption("在商品卡片上选择“加入对比”，最多同时比较两件。")
    else:
        for item_id in selected:
            st.write("• " + product_title(st.session_state.ca_cards.get(item_id, {})))
        st.caption("保留最近选择的两件商品；继续选择会替换最早的一件。")
        if st.button("比较这两件", disabled=len(selected) != 2, key="ca_run_compare"):
            try:
                result = client.request(
                    "POST",
                    "/v1/compare",
                    payload={
                        "session_id": st.session_state.ca_session_id,
                        "item_ids": list(selected),
                        "dimensions": ["price_usd", "brand", "category"],
                    },
                )
                remember_result(result)
                st.session_state.ca_comparison = result
            except ApiError as exc:
                st.warning(str(exc))
        if st.session_state.ca_comparison:
            render_comparison(st.session_state.ca_comparison)
    with st.expander("高级：查看本次运行记录"):
        st.caption("用于核对检索与对话过程，默认不会读取；身份标识和凭证会隐藏。")
        trace_id = st.session_state.ca_trace_id
        if st.button("读取最近一次运行记录", disabled=not trace_id, key="ca_load_trace"):
            try:
                trace = client.request(
                    "GET",
                    f"/v1/traces/{trace_id}",
                    params={"session_id": st.session_state.ca_session_id},
                )
                st.session_state.ca_trace_cache[trace_id] = redact_debug(trace)
            except ApiError as exc:
                st.warning(str(exc))
        if trace_id in st.session_state.ca_trace_cache:
            st.json(st.session_state.ca_trace_cache[trace_id])


if __name__ == "__main__":
    main()
