"""Record the actual local UI and API/model interaction, without mocks.

Install the browser into this project before running::

    PLAYWRIGHT_BROWSERS_PATH=runtime/browser python -m playwright install chromium
    python scripts/capture_demo.py

The API, local generator and Streamlit must already be running. This script does
not modify responses, load models, or enable hardware GPU rendering.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"


def ready(api_url: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        try:
            response = httpx.get(api_url.rstrip("/") + "/health", timeout=5, trust_env=False)
            response.raise_for_status()
            health = response.json()
            if health.get("status") in {"ok", "ready"} or health.get("catalog_count", 0) > 0:
                return health
        except (httpx.HTTPError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("The real API did not become healthy before the wait deadline.")
        time.sleep(2)


def prices(text: str) -> list[float]:
    return [
        float(value.replace(",", "")) for value in re.findall(r"US\$([\d,.]+)\s*·\s*历史价格", text)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui-url", default="http://127.0.0.1:8501")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-wait-seconds", type=float, default=180)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "demo" / "final")
    parser.add_argument("--min-video-seconds", type=float, default=70)
    parser.add_argument("--max-video-seconds", type=float, default=180)
    parser.add_argument(
        "--check-chinese",
        action="store_true",
        help="Verify Chinese product display and original-text access",
    )
    parser.add_argument(
        "--chat-profile",
        choices=("original", "simple"),
        default="original",
        help="Keep the original detailed requests or separately exercise short supported requests",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if ROOT not in output.parents:
        parser.error("Keep demo output inside the project directory.")
    output.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "tmp").mkdir(parents=True, exist_ok=True)
    (RUNTIME / "cache").mkdir(parents=True, exist_ok=True)
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(RUNTIME / "browser")
    os.environ["XDG_CACHE_HOME"] = str(RUNTIME / "cache")
    os.environ["TMPDIR"] = str(RUNTIME / "tmp")
    videos = output / "original_video"
    videos.mkdir(exist_ok=True)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "real_ui": True,
        "mocked_responses": False,
        "hardware_gpu_disabled": True,
        "ui_url": args.ui_url,
        "api_url": args.api_url,
        "chat_profile": args.chat_profile,
        "checks": [],
        "screenshots": [],
        "page_errors": [],
        "console_errors": [],
        "chat_turns": [],
    }

    def check(name: str, ok: bool, detail: object = "") -> None:
        report["checks"].append({"name": name, "passed": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}: {name}", flush=True)
        if not ok and detail:
            print(str(detail)[:1600], flush=True)

    try:
        report["api_health"] = ready(args.api_url, args.api_wait_seconds)
        model_health = httpx.get("http://127.0.0.1:8001/health", timeout=5, trust_env=False)
        model_health.raise_for_status()
        report["model_health"] = model_health.json()
        check("real_api_and_local_model_ready", report["model_health"].get("ready") is True)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-gpu",
                    "--disable-gpu-compositing",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-software-rasterizer",
                    "--disk-cache-dir=" + str(RUNTIME / "browser-cache"),
                ],
            )
            context = browser.new_context(
                viewport={"width": 1440, "height": 1000},
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
                color_scheme="light",
                record_video_dir=str(videos),
                record_video_size={"width": 1440, "height": 1000},
            )
            page = context.new_page()
            video = page.video
            if video is None:
                raise RuntimeError("Browser did not enable the requested video recording.")
            page.set_default_timeout(15000)
            page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
            page.on(
                "console",
                lambda message: (
                    report["console_errors"].append(message.text)
                    if message.type == "error"
                    else None
                ),
            )
            started = time.monotonic()

            def pause(seconds: float = 3) -> None:
                page.wait_for_timeout(seconds * 1000)

            def capture(name: str) -> None:
                path = output / f"{name}.png"
                page.screenshot(path=str(path), full_page=False)
                report["screenshots"].append(str(path.relative_to(ROOT)))

            def failure(name: str, error: Exception) -> None:
                check(name, False, str(error)[:1600])
                capture(name + "_error")

            try:
                page.goto(args.ui_url, wait_until="domcontentloaded")
                expect(
                    page.get_by_role("heading", name="ChoiceAgent · 选物助手", exact=True)
                ).to_be_visible()
                expect(page.get_by_role("textbox", name="想找什么？", exact=True)).to_be_visible(
                    timeout=30000
                )
                check("initial_page", True)
                pause(4)
                capture("01_home")

                try:
                    if args.check_chinese:
                        stats = httpx.get(
                            args.api_url.rstrip("/") + "/v1/catalog/stats",
                            timeout=5,
                            trust_env=False,
                        ).json()
                        label = stats["category_labels"]["Office Products"]
                        page.get_by_role("combobox", name="类别", exact=True).click()
                        page.get_by_role("option", name=label, exact=True).click()
                        report["selected_category"] = {"raw": "Office Products", "display": label}
                    # Streamlit's switch uses an overlaid label. Keyboard Space
                    # activates the real focused control without a forced click.
                    budget_switch = page.get_by_role("switch", name="设置历史价格预算", exact=True)
                    budget_switch.focus()
                    budget_switch.press("Space")
                    expect(budget_switch).to_be_checked()
                    maximum = page.get_by_role("spinbutton", name="最高价格（美元）", exact=True)
                    expect(maximum).to_be_enabled()
                    maximum.fill("35")
                    maximum.press("Enter")
                    expect(maximum).to_have_value("35.00")
                    pause(1)
                    page.get_by_role("textbox", name="想找什么？", exact=True).fill(
                        "适合小桌面的收纳盒"
                    )
                    page.get_by_role("button", name="找一找", exact=True).click()
                    expect(page.get_by_role("button", name="喜欢", exact=True).first).to_be_visible(
                        timeout=120000
                    )
                    content = page.locator("body").inner_text()
                    found_prices = prices(content)
                    report["search_prices_usd"] = found_prices
                    check(
                        "chinese_search_and_35_usd_budget",
                        bool(found_prices) and max(found_prices) <= 35,
                        {"query": "适合小桌面的收纳盒", "prices_usd": found_prices},
                    )
                    pause(5)
                    capture("02_search_budget")

                    if args.check_chinese:
                        search_panel = page.get_by_role("tabpanel").nth(0)
                        titles = search_panel.get_by_role("heading", level=3).all_text_contents()
                        report["search_titles"] = titles
                        check(
                            "product_titles_chinese",
                            bool(titles)
                            and all(re.search(r"[\u3400-\u9fff]", title) for title in titles),
                            titles,
                        )
                        original = (
                            search_panel.get_by_test_id("stExpander")
                            .filter(has=page.get_by_text("查看英文原文", exact=True))
                            .first
                        )
                        summary = original.locator("summary")
                        check(
                            "english_original_collapsed_by_default",
                            summary.evaluate("node => node.parentElement.open") is False,
                        )
                        summary.click()
                        original_title = original.get_by_text(re.compile(r"^原始名称：")).first
                        expect(original_title).to_be_visible()
                        report["original_title"] = original_title.inner_text()
                        check(
                            "english_original_available",
                            bool(re.search(r"[A-Za-z]", report["original_title"])),
                        )
                        pause(3)
                        capture("02b_english_original")
                        summary.click()

                        trace_expander = (
                            page.get_by_test_id("stExpander")
                            .filter(has=page.get_by_text("高级：查看本次运行记录", exact=True))
                            .first
                        )
                        trace_expander.locator("summary").click()
                        page.get_by_role("button", name="读取最近一次运行记录", exact=True).click()
                        trace = page.get_by_test_id("stJson").first
                        expect(trace).to_be_visible()
                        trace_text = trace.inner_text()
                        check(
                            "chinese_category_submits_original_id",
                            "Office Products" in trace_text and "constraints" in trace_text,
                        )
                        trace_expander.locator("summary").click()

                    page.get_by_role("button", name="喜欢", exact=True).first.click()
                    expect(
                        page.get_by_role("button", name="已喜欢", exact=True).first
                    ).to_be_visible()
                    check("like_persisted_in_ui", True)
                    if args.check_chinese:
                        page.get_by_role("button", name="收藏", exact=True).first.click()
                        expect(
                            page.get_by_role("button", name="已收藏", exact=True).first
                        ).to_be_visible()
                        check("save_control_and_product_title_chinese", True)
                    pause(3)
                    page.get_by_role("button", name="加入对比", exact=True).first.click()
                    expect(page.get_by_role("button", name="移出对比", exact=True)).to_have_count(1)
                    page.get_by_role("button", name="加入对比", exact=True).first.click()
                    expect(page.get_by_role("button", name="移出对比", exact=True)).to_have_count(2)
                    page.get_by_role("button", name="比较这两件", exact=True).click()
                    expect(page.get_by_test_id("stDataFrame").first).to_be_visible(timeout=30000)
                    check("compare_two_visible_products", True)
                    page.get_by_role(
                        "heading", name="放在一起比一比", exact=True
                    ).scroll_into_view_if_needed()
                    pause(6)
                    capture("03_comparison")
                except Exception as error:
                    failure("search_feedback_comparison", error)

                page.get_by_role("tab", name="聊聊需求", exact=True).click()
                messages = [
                    (
                        "帮我选小桌面用的收纳用品，预算不超过35美元。请按历史美元价格推荐，并说明依据。",
                        35,
                        "04_chat_first_turn",
                    ),
                    (
                        "把预算改到20美元以内，仍然用于小桌面。请重新筛选，不要推荐超过预算的商品。",
                        20,
                        "05_chat_budget_update",
                    ),
                ]
                if args.chat_profile == "simple":
                    messages = [
                        ("找桌面收纳，预算35美元", 35, "04_chat_first_turn"),
                        ("预算改到20美元", 20, "05_chat_budget_update"),
                    ]
                for index, (message, budget, screenshot) in enumerate(messages):
                    try:
                        chatbox = page.get_by_placeholder("说说你的用途、预算和偏好…", exact=True)
                        chatbox.scroll_into_view_if_needed()
                        chatbox.fill(message)
                        pause(2)
                        chatbox.press("Enter")
                        expect(page.get_by_test_id("stChatMessage")).to_have_count(
                            (index + 1) * 2, timeout=160000
                        )
                        assistant = page.get_by_test_id("stChatMessage").nth(index * 2 + 1)
                        response_text = assistant.inner_text()
                        if args.check_chinese:
                            answer = assistant.get_by_test_id(
                                "stMarkdownContainer"
                            ).first.inner_text()
                            check(
                                f"chat_turn_{index + 1}_answer_chinese",
                                bool(re.search(r"[\u3400-\u9fff]", answer))
                                and not re.search(
                                    r"(?:[A-Za-z][A-Za-z'-]*\s+){12,}[A-Za-z]", answer
                                ),
                                answer[:1200],
                            )
                        response_prices = prices(response_text)
                        report["chat_turns"].append(
                            {
                                "message": message,
                                "displayed_response": response_text,
                                "prices_usd": response_prices,
                                "budget_usd": budget,
                            }
                        )
                        check(f"chat_turn_{index + 1}_response_rendered", bool(response_text))
                        check(
                            f"chat_turn_{index + 1}_budget",
                            bool(response_prices) and max(response_prices) <= budget,
                            {"prices_usd": response_prices, "budget_usd": budget},
                        )
                        expect(
                            page.get_by_role("tab", name="聊聊需求", exact=True)
                        ).to_have_attribute("aria-selected", "true")
                        check(f"chat_turn_{index + 1}_tab_remains_active", True)
                        # Scroll the actual request and answer into the recording;
                        # never alter the document or replace backend content.
                        user_turn = page.get_by_test_id("stChatMessage").nth(index * 2)
                        user_turn.evaluate("""node => {
                            node.scrollIntoView({block: 'start'});
                            const main = document.querySelector('[data-testid="stMain"]');
                            if (main) main.scrollTop -= 100;
                        }""")
                        pause(6)
                        capture(screenshot)
                    except Exception as error:
                        failure(f"chat_turn_{index + 1}", error)
                        break

                page.get_by_role("tab", name="搜索商品", exact=True).click()
                page.get_by_role(
                    "textbox", name="想找什么？", exact=True
                ).scroll_into_view_if_needed()
                remaining = max(0, args.min_video_seconds - (time.monotonic() - started))
                while remaining > 0:
                    pause(min(remaining, 10))
                    remaining = max(0, args.min_video_seconds - (time.monotonic() - started))
                check("no_browser_page_errors", not report["page_errors"], report["page_errors"])
            except Exception as error:
                failure("ui_run", error)
            finally:
                report["recording_elapsed_seconds"] = round(time.monotonic() - started, 2)
                original_video = Path(video.path())
                context.close()
                browser.close()
                report["original_video"] = str(original_video.relative_to(ROOT))

        if shutil.which("ffmpeg") and original_video.exists():
            duration = float(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=nw=1:nk=1",
                        str(original_video),
                    ],
                    text=True,
                ).strip()
            )
            speed = max(1.0, duration / max(1, args.max_video_seconds))
            mp4 = output / "choiceagent_demo.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(original_video),
                    "-an",
                    "-vf",
                    f"setpts=PTS/{speed:.6f}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(mp4),
                ],
                check=True,
                capture_output=True,
            )
            report["mp4_video"] = str(mp4.relative_to(ROOT))
            report["original_video_seconds"] = duration
            report["playback_speed"] = speed
    except Exception as error:
        check("setup", False, str(error)[:1600])
    finally:
        report["passed"] = bool(report["checks"]) and all(row["passed"] for row in report["checks"])
        (output / "ui_acceptance.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        lines = [
            "# 真实浏览器界面验收",
            "",
            f"生成时间：{report['created_at_utc']}",
            "",
            "使用真实 Streamlit、真实 API 与本地模型；未拦截、替换或模拟接口回复。浏览器禁用硬件 GPU。",
            "",
            f"对话请求组：{args.chat_profile}。完整请求与实际回复保留在 JSON 报告中。",
            "",
        ]
        for row in report["checks"]:
            lines.append(
                f"- {'通过' if row['passed'] else '未通过'}：{row['name']}。"
                + (f" {row['detail']}" if row["detail"] else "")
            )
        lines += ["", "## 素材", ""] + [f"- `{value}`" for value in report["screenshots"]]
        for key in ["mp4_video", "original_video"]:
            if key in report:
                lines.append(f"- `{report[key]}`")
        if report.get("playback_speed", 1) > 1:
            lines.append(
                f"- MP4 将完整原始录制以 {report['playback_speed']:.2f} 倍速播放；未删改界面内容。"
            )
        lines += [
            "",
            "## 浏览器错误",
            "",
            json.dumps(
                {"page_errors": report["page_errors"], "console_errors": report["console_errors"]},
                ensure_ascii=False,
                indent=2,
            ),
        ]
        (output / "ui_acceptance.md").write_text("\n".join(lines) + "\n")
        print(f"Report: {output / 'ui_acceptance.json'}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
