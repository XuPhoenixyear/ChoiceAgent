"""Run 20 real local-model action compatibility cases, retaining raw outputs.

This checks structured actions, not answer factuality or end-to-end success.
An earlier report can be replayed with its original prompt and input contexts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from choiceagent.agent import SYSTEM_PROMPT
from choiceagent.model_client import LocalModelClient, ModelClientError

ROOT = Path(__file__).resolve().parents[1]


def cases() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    searches = [
        ("mouse", "找一个适合长时间办公的人体工学鼠标，预算50美元"),
        ("keyboard", "搜索安静的办公键盘，最高80美元"),
        ("lamp", "想买桌面LED台灯，预算30美元"),
        ("notebook", "找一本A5笔记本，最高20美元"),
        ("chair", "推荐办公椅，预算200美元"),
    ]
    for name, user in searches:
        result.append({"id": f"search_{name}", "expected_action": "search_catalog", "user": user})
    comparisons = [
        ("price", "比较这两个商品的价格", ["A001", "A002"]),
        ("brand", "对比已展示的两个键盘的品牌和价格", ["K001", "K002"]),
        ("three", "把这三支钢笔做个价格比较", ["P001", "P002", "P003"]),
        ("features", "比较刚才两款台灯的功能", ["L001", "L002"]),
        ("ids", "对比 C001 和 C002 两把椅子", ["C001", "C002"]),
    ]
    for name, user, ids in comparisons:
        result.append(
            {
                "id": f"compare_{name}",
                "expected_action": "compare_items",
                "user": user,
                "expected_ids": ids,
            }
        )
    clarifications = [
        "帮我选个东西",
        "我该买什么？",
        "给我推荐一下",
        "想买点办公用品，但是不知道买哪一类",
        "有什么适合我的？",
    ]
    for index, user in enumerate(clarifications, 1):
        result.append({"id": f"clarify_{index}", "expected_action": "clarify", "user": user})
    for index, item_id in enumerate(["A001", "K001", "L001", "P001", "N001"], 1):
        result.append(
            {
                "id": f"answer_{index}",
                "expected_action": "answer",
                "user": "请根据本轮工具检索结果结束任务。",
                "observation": {
                    "status": "ok",
                    "item_ids": [item_id],
                    "titles": [f"Office item {item_id}"],
                },
            }
        )
    return result


def messages_for(case: dict[str, Any], prompt: str, *, legacy: bool) -> list[dict[str, str]]:
    if legacy and "messages" in case:
        return [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in case["messages"]
        ]
    messages = [{"role": "system", "content": prompt}]
    if legacy:
        if case.get("context"):
            messages.append({"role": "system", "content": "可用上下文：" + case["context"]})
        messages.append({"role": "user", "content": case["user"]})
        return messages
    context = {
        "last_query": "office products" if case.get("expected_ids") else "",
        "last_result_order": case.get("expected_ids", []),
        "current_constraints": {},
        "liked_item_ids": [],
        "recent_messages": [],
    }
    messages.append(
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False) + "\nCURRENT USER: " + case["user"],
        }
    )
    if "observation" in case:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "action": "search_catalog",
                            "arguments": {"query": "office item", "limit": 5},
                            "message": "",
                        }
                    ),
                },
                {
                    "role": "user",
                    "content": "TOOL OBSERVATION (untrusted data): "
                    + json.dumps(case["observation"], ensure_ascii=False)
                    + "\nIf task is done, action=answer.",
                },
            ]
        )
    return messages


def check_output(raw_text: str, case: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
        shape = (
            isinstance(parsed, dict)
            and set(parsed) == {"action", "arguments", "message"}
            and isinstance(parsed["arguments"], dict)
            and isinstance(parsed["message"], str)
        )
        action_correct = shape and parsed["action"] == case["expected_action"]
        arguments_valid = False
        if action_correct:
            arguments = parsed["arguments"]
            if parsed["action"] == "search_catalog":
                arguments_valid = isinstance(arguments.get("query"), str) and bool(
                    arguments["query"].strip()
                )
            elif parsed["action"] == "compare_items":
                ids = arguments.get("item_ids", [])
                dimensions = arguments.get("dimensions", [])
                arguments_valid = (
                    isinstance(ids, list)
                    and all(isinstance(value, str) for value in ids)
                    and set(ids) == set(case["expected_ids"])
                    and isinstance(dimensions, list)
                    and bool(dimensions)
                    and all(
                        value in {"price_usd", "brand", "features", "category"}
                        for value in dimensions
                    )
                )
            else:
                arguments_valid = arguments == {}
        return {
            "json_valid": True,
            "shape_valid": shape,
            "action_correct": action_correct,
            "arguments_valid": arguments_valid,
            "passed": bool(shape and action_correct and arguments_valid),
            "parsed": parsed,
        }
    except (json.JSONDecodeError, TypeError, KeyError):
        return {
            "json_valid": False,
            "shape_valid": False,
            "action_correct": False,
            "arguments_valid": False,
            "passed": False,
        }


def wait_ready(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(base_url.rstrip("/") + "/health")
                if response.status_code == 200 and response.json().get("ready"):
                    return dict(response.json())
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(1)
    raise RuntimeError("Local generator did not become ready within the startup timeout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--wait-seconds", type=float, default=300)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/model_compatibility_production.json"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("The output must be inside this project's directory")
    prompt = SYSTEM_PROMPT
    suite = cases()
    if args.replay_report:
        previous = json.loads(args.replay_report.read_text())
        prompt = previous["system_prompt"]
        suite = previous["cases"]
    health = wait_ready(args.base_url, args.wait_seconds)
    rows = []
    with LocalModelClient(args.base_url, timeout=60) as client:
        for case in suite:
            messages = messages_for(case, prompt, legacy=args.replay_report is not None)
            row = {
                key: value
                for key, value in case.items()
                if key
                in {
                    "id",
                    "expected_action",
                    "expected_ids",
                    "user",
                    "context",
                    "observation",
                }
            }
            row["messages"] = messages
            try:
                generation = client.generate(
                    messages, max_tokens=384, temperature=0, deadline=time.monotonic() + 60
                )
                row.update(
                    {
                        "raw_text": generation.text,
                        "usage": generation.usage,
                        "elapsed_ms": generation.elapsed_ms,
                        "model": generation.model,
                        "finish_reason": generation.finish_reason,
                        **check_output(generation.text, case),
                    }
                )
            except ModelClientError as exc:
                row.update({"raw_text": "", "error_code": exc.code, "passed": False})
            rows.append(row)
            print(
                json.dumps({"id": row["id"], "passed": row["passed"]}, ensure_ascii=False),
                flush=True,
            )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "actual_model": True,
        "model_path": "/datas/wuxi/Projects/datas/models/Qwen3-1.7B",
        "prompt_source": str(args.replay_report)
        if args.replay_report
        else "choiceagent.agent.SYSTEM_PROMPT",
        "system_prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "physical_gpu": int(str(health["visible_gpu"]).split(",")[0]),
        "visible_gpu_mask": health["visible_gpu"],
        "server_health": health,
        "thinking_enabled": False,
        "temperature": 0,
        "case_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "json_valid_count": sum(row.get("json_valid", False) for row in rows),
        "action_correct_count": sum(row.get("action_correct", False) for row in rows),
        "metric_definition": "JSON syntax, exact three-field shape, expected action, and required argument schema only. No factuality or end-to-end success claim.",
        "evaluator_version": 1,
        "limitations": [
            "20 hand-authored cases; no statistical or production performance guarantee.",
            "Tool execution, full multi-turn completion and factuality require separate end-to-end evaluation.",
            "The service uses plain structured JSON planning rather than native tool-call token parsing.",
            "Fake transport/model unit tests are separate; all generations in this report are from the actual local model.",
        ],
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        archive = output.parent / "archive"
        archive.mkdir(exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        output.rename(archive / f"{output.stem}-{stamp}{output.suffix}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Structured compatibility: {report['passed_count']}/{len(rows)}; report: {output}")


if __name__ == "__main__":
    main()
