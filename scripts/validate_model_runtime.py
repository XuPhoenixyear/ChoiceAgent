"""Validate real local-model context, decode deadline, and bounded queue guards."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/model_runtime_validation.json"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("The output must be inside this project's directory")
    health_response = httpx.get(args.base_url.rstrip("/") + "/health", timeout=5, trust_env=False)
    health_response.raise_for_status()
    health = health_response.json()
    if not health.get("ready"):
        parser.error("Preload the local model and wait for ready=true first")

    def call(content: str, max_tokens: int = 128, timeout_header: str = "30") -> dict[str, Any]:
        started = time.monotonic()
        response = httpx.post(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            headers={"X-Request-Timeout": timeout_header},
            timeout=35,
            trust_env=False,
        )
        return {
            "http_status": response.status_code,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            "response": response.json(),
        }

    context = call("x " * 9000, 32)
    expired = call("请连续写出从1到10000的整数，不要停止，不要解释。", 768, "0.05")
    recovery = call("只回复 OK", 8)
    barrier = threading.Barrier(6)

    def worker(index: int) -> dict[str, Any]:
        barrier.wait(timeout=10)
        return {"worker": index, **call("请连续写出从1到200的整数，不要停止，不要解释。", 128)}

    with ThreadPoolExecutor(max_workers=6) as pool:
        queued = list(pool.map(worker, range(6)))
    assertions = {
        "context_rejected_with_correct_code": context["http_status"] == 422
        and context["response"].get("error", {}).get("code") == "MODEL_CONTEXT_LIMIT",
        "decode_deadline_enforced": expired["http_status"] == 504
        and expired["response"].get("error", {}).get("code") == "MODEL_TIMEOUT",
        "timeout_recovery": recovery["http_status"] == 200,
        "queue_is_bounded": sum(row["http_status"] == 429 for row in queued) >= 3,
        "at_most_three_admitted": sum(row["http_status"] == 200 for row in queued) <= 3,
    }
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "actual_model": True,
        "model_path": "/datas/wuxi/Projects/datas/models/Qwen3-1.7B",
        "physical_gpu": int(str(health["visible_gpu"]).split(",")[0]),
        "visible_gpu_mask": health["visible_gpu"],
        "server_health": health,
        "context_limit": context,
        "deadline_50ms": expired,
        "post_timeout_recovery": recovery,
        "concurrent_six_requests": queued,
        "assertions": assertions,
        "limitations": [
            "Run when no other client is using the generator; admission counts otherwise include their requests.",
            "One warm-run observation, not a production latency guarantee.",
            "Deadline stopping occurs between decode steps; prefill, one kernel, and scheduling can overrun the exact duration.",
            "Context overflow is rejected, never silently truncated.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(assertions, ensure_ascii=False))
    if not all(assertions.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
