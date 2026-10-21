"""Value boundaries and Chinese display never replace underlying product facts."""

from copy import deepcopy

import pytest

from choiceagent.ui import (
    QUICK_QUERIES,
    api_base_url,
    build_constraints,
    category_label,
    comparison_table,
    facet_options,
    format_price,
    product_category,
    product_features,
    product_title,
    redact_debug,
    split_terms,
)


@pytest.mark.parametrize("value", [None, "", "unknown", float("nan"), float("inf"), -1, True])
def test_missing_or_invalid_price_never_becomes_zero(value):
    assert format_price(value) == "价格未提供"


def test_genuine_zero_and_historical_price_remain_distinguishable():
    assert format_price(0) == "US$0.00 · 历史价格"
    assert format_price("12.5") == "US$12.50 · 历史价格"


def test_budget_validation_does_not_silently_invert_bounds():
    options = dict(
        budget_enabled=True,
        minimum=30,
        maximum=20,
        category="全部类别",
        excluded_brands=[],
        required="",
        excluded="",
    )
    with pytest.raises(ValueError, match="最低预算"):
        build_constraints(**options)
    options["budget_enabled"] = False
    constraints = build_constraints(**options)
    assert constraints["min_price_usd"] is None
    assert constraints["max_price_usd"] is None


def test_user_terms_and_stats_shapes():
    assert split_terms("a，b; a\nc") == ["a", "b", "c"]
    assert facet_options({"Acme": 5, "Beta": 3}) == ["Acme", "Beta"]
    assert facet_options([{"name": "Acme", "count": 2}, {"brand": "Beta"}, None]) == [
        "Acme",
        "Beta",
    ]


def test_comparison_keeps_identical_titles_and_missing_evidence_separate():
    comparison = {
        "rows": [
            {
                "dimension": "price_usd",
                "values": [{"item_id": "a", "value": 12}, {"item_id": "b", "value": None}],
            }
        ]
    }
    rows = comparison_table(comparison, {"a": {"title": "Same"}, "b": {"title": "Same"}})
    assert len(rows[0]) == 3
    assert rows[0]["商品 1 · 商品信息（原文可查看）"] == "US$12.00 · 历史价格"
    assert rows[0]["商品 2 · 商品信息（原文可查看）"] == "价格未提供"


def test_product_display_prefers_supplied_chinese_and_preserves_original():
    item = {
        "title": "Desk Organizer",
        "category": "Office Products",
        "features": ["Three trays"],
        "brand": "Acme",
        "attributes": {"Color": "Blue"},
        "display": {
            "locale": "zh-CN",
            "title": "桌面收纳架",
            "category": "办公用品",
            "features": ["三层托盘"],
            "attributes": {"颜色": "蓝色"},
            "translation_kind": "model",
            "source_language": "en",
        },
    }
    original = deepcopy(item)
    assert product_title(item) == "桌面收纳架"
    assert product_category(item) == "办公用品"
    assert product_features(item) == ["三层托盘"]
    assert item == original


@pytest.mark.parametrize(
    "display",
    [
        None,
        {},
        {"locale": "en", "title": "Desk Organizer"},
        {"locale": "zh-CN", "title": "Desk Organizer", "features": ["Three trays"]},
    ],
)
def test_old_or_incomplete_cards_never_put_english_in_the_main_title(display):
    item = {
        "title": "An Extremely Long English Product Title",
        "features": ["English facts"],
        "display": display,
    }
    assert product_title(item) == "商品信息（原文可查看）"
    assert product_features(item) == []


def test_category_labels_do_not_change_filter_identity_and_quick_queries_are_chinese():
    assert category_label("Office Products", {"Office Products": "办公文具"}) == "办公文具"
    assert category_label("Office Products") == "办公用品"
    assert category_label("全部类别") == "全部类别"
    constraints = build_constraints(
        budget_enabled=False,
        minimum=0,
        maximum=10,
        category="Office Products",
        excluded_brands=[],
        required="",
        excluded="",
    )
    assert constraints["category"] == "Office Products"
    assert QUICK_QUERIES["桌面收纳"] == "桌面收纳盒"
    assert all(
        any("\u3400" <= char <= "\u9fff" for char in query) for query in QUICK_QUERIES.values()
    )


def test_comparison_uses_chinese_labels_and_values_without_overwriting_raw_facts():
    cards = {"a": {"title": "Organizer", "display": {"locale": "zh-CN", "title": "桌面收纳架"}}}
    comparison = {
        "rows": [
            {
                "dimension": "attributes.Color",
                "label_zh": "颜色",
                "values": [{"item_id": "a", "value": "Blue", "display_value": "蓝色"}],
            },
            {
                "dimension": "brand",
                "label_zh": "品牌",
                "values": [{"item_id": "a", "value": "Acme", "display_value": "不应替换官方品牌"}],
            },
            {
                "dimension": "price_usd",
                "label_zh": "历史美元价格",
                "values": [{"item_id": "a", "value": None, "display_value": "免费"}],
            },
        ]
    }
    before = deepcopy(comparison)
    translated = comparison_table(comparison, cards)
    original = comparison_table(comparison, cards, original=True)
    assert translated[0] == {"比较项目": "颜色", "商品 1 · 桌面收纳架": "蓝色"}
    assert translated[1]["商品 1 · 桌面收纳架"] == "Acme"
    assert translated[2]["商品 1 · 桌面收纳架"] == "价格未提供"
    assert original[0] == {"比较项目": "Color", "商品 1 · Organizer": "Blue"}
    assert comparison == before


def test_chinese_ui_preserves_query_filter_identity_and_collapses_originals(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from choiceagent import ui

    items = [
        {
            "item_id": "a",
            "title": "ORIGINAL ENGLISH TITLE",
            "category": "Office Products",
            "brand": "Acme",
            "price_usd": 12,
            "features": ["ORIGINAL ENGLISH FEATURE"],
            "display": {
                "locale": "zh-CN",
                "title": "桌面收纳架",
                "category": "办公用品",
                "features": ["三层收纳空间"],
            },
        },
        {
            "item_id": "b",
            "title": "OLD ENGLISH TITLE",
            "category": "Office Products",
            "price_usd": None,
            "features": ["OLD ENGLISH FEATURE"],
        },
    ]
    calls = []

    def request(self, method, path, *, payload=None, params=None):
        calls.append((method, path, payload))
        if path == "/health":
            return {"catalog_count": 2}
        if path == "/v1/sessions":
            return {"session_id": "unit-session", "profile_version": 0}
        if path == "/v1/catalog/stats":
            return {
                "count": 2,
                "categories": {"Office Products": 2},
                "category_labels": {"Office Products": "办公文具"},
                "brands": {"Acme": 2},
            }
        if path == "/v1/search":
            return {"items": items, "status": "ok", "trace_id": "test-trace"}
        if path == "/v1/compare":
            return {
                "items": items,
                "rows": [
                    {
                        "dimension": "category",
                        "label_zh": "类别",
                        "values": [
                            {
                                "item_id": "a",
                                "value": "Office Products",
                                "display_value": "办公文具",
                            },
                            {
                                "item_id": "b",
                                "value": "Office Products",
                                "display_value": "办公文具",
                            },
                        ],
                    }
                ],
            }
        raise AssertionError(path)

    monkeypatch.setattr(ui.ApiClient, "request", request)
    app = AppTest.from_string("from choiceagent.ui import main\nmain()", default_timeout=15).run()
    app.button(key="ca_quick_桌面收纳盒").click().run()
    assert app.text_input(key="ca_query").value == "桌面收纳盒"
    app.selectbox(key="ca_category").select("Office Products").run()
    next(button for button in app.button if button.label == "找一找").click().run()
    assert not app.exception
    assert calls[-1][2]["query"] == "桌面收纳盒"
    assert calls[-1][2]["constraints"]["category"] == "Office Products"
    titles = [element.value for element in app.subheader]
    assert "桌面收纳架" in titles and "商品信息（原文可查看）" in titles
    assert not any("ENGLISH" in title for title in titles)
    originals = [expander for expander in app.expander if expander.label == "查看英文原文"]
    assert len(originals) == 2 and not originals[0].proto.expanded
    assert "原始名称：ORIGINAL ENGLISH TITLE" in [
        element.value for element in originals[0].markdown
    ]
    app.button(key="search_0_a_compare").click().run()
    app.button(key="search_1_b_compare").click().run()
    app.button(key="ca_run_compare").click().run()
    assert not app.exception and len(app.dataframe) == 2
    assert "商品 1 · 桌面收纳架" in app.dataframe[0].value.columns
    assert "商品 1 · ORIGINAL ENGLISH TITLE" in app.dataframe[1].value.columns


def test_debug_redaction_nested_credentials_preserves_useful_token_counts():
    redacted = redact_debug(
        {
            "session_id": "private-session",
            "headers": {"Authorization": "secret"},
            "usage": {"prompt_tokens": 10},
            "steps": [
                {
                    "api_key": "private-key",
                    "url": "http://name:password@example.com/?token=abc",
                    "message": "Authorization: Bearer private-bearer",
                }
            ],
        }
    )
    assert redacted["usage"]["prompt_tokens"] == 10
    assert "private" not in str(redacted)
    assert "name:password" not in str(redacted)
    assert "token=abc" not in str(redacted)


@pytest.mark.parametrize(
    "value",
    [
        "file:///tmp/api",
        "http://name:secret@localhost",
        "http://localhost/?key=secret",
        "http://localhost:invalid",
        "not-a-url",
    ],
)
def test_service_address_cannot_expose_credentials(value):
    with pytest.raises(ValueError) as error:
        api_base_url(value)
    assert value not in str(error.value)
