#!/usr/bin/env python3
"""Run reproducible CPU quality gates and retain their unabridged logs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "reports/quality"
    output.mkdir(parents=True, exist_ok=True)
    commands = [
        ("format", ["-m", "ruff", "format", "--check", "src", "scripts", "tests"]),
        ("lint", ["-m", "ruff", "check", "src", "scripts", "tests"]),
        ("types", ["-m", "mypy", "src", "scripts"]),
        (
            "tests",
            [
                "-m",
                "pytest",
                "-q",
                "--junitxml=reports/quality/junit.xml",
                "--cov=choiceagent",
                "--cov-fail-under=80",
                "--cov-report=term-missing",
                "--cov-report=json:reports/coverage.json",
            ],
        ),
        ("dependencies", ["-m", "pip", "check"]),
        ("package", ["-m", "build", "--no-isolation"]),
    ]
    environment = os.environ.copy()
    environment.update(
        PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", TMPDIR=str(ROOT / "runtime/tmp")
    )
    (ROOT / "runtime/tmp").mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for name, arguments in commands:
        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, *arguments],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=300,
        )
        (output / f"{name}.log").write_text(process.stdout)
        result = {
            "name": name,
            "command": [sys.executable, *arguments],
            "exit_code": process.returncode,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        results.append(result)
        print(f"{name}: {'PASS' if process.returncode == 0 else 'FAIL'}", flush=True)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "python": sys.executable,
        "gpu_required": False,
        "results": results,
        "passed": all(r["exit_code"] == 0 for r in results),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
