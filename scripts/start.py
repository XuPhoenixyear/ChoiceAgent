"""Start or stop this project's three services with bounded readiness checks."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]


class LauncherError(RuntimeError):
    """An actionable startup/shutdown failure without a traceback."""


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    module: str
    port: int
    health_path: str
    arguments: tuple[str, ...]


SPECS = (
    ServiceSpec(
        "model", "choiceagent.model_server", 8001, "/health", ("-m", "choiceagent.model_server")
    ),
    ServiceSpec(
        "api",
        "uvicorn",
        8000,
        "/health",
        (
            "-m",
            "uvicorn",
            "choiceagent.api:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--workers",
            "1",
        ),
    ),
    ServiceSpec(
        "ui",
        "streamlit",
        8501,
        "/_stcore/health",
        (
            "-m",
            "streamlit",
            "run",
            "src/choiceagent/ui.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
            "--theme.base",
            "light",
            "--browser.gatherUsageStats",
            "false",
        ),
    ),
)


@dataclass(frozen=True)
class ProcessIdentity:
    argv: tuple[str, ...]
    cwd: Path
    executable: Path
    start_ticks: int


def process_identity(pid: int) -> ProcessIdentity | None:
    """Read Linux birth time as well as argv/cwd to reject recycled PIDs."""
    if pid <= 0:
        return None
    proc = Path(f"/proc/{pid}")
    try:
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] in {"Z", "X"}:
            return None
        return ProcessIdentity(
            argv=tuple(
                value.decode(errors="replace")
                for value in (proc / "cmdline").read_bytes().split(b"\0")
                if value
            ),
            cwd=(proc / "cwd").resolve(strict=True),
            executable=(proc / "exe").resolve(strict=True),
            start_ticks=int(stat[19]),
        )
    except (OSError, ValueError, IndexError):
        return None


def is_owned(
    identity: ProcessIdentity | None, entry: dict[str, Any], record: dict[str, Any]
) -> bool:
    if identity is None:
        return False
    spec = next((value for value in SPECS if value.name == entry.get("name")), None)
    argv = entry.get("argv")
    return bool(
        spec is not None
        and isinstance(argv, list)
        and argv
        and entry.get("module") == spec.module
        and tuple(argv[1:]) == spec.arguments
        and identity.argv == tuple(argv)
        and identity.start_ticks == entry.get("start_ticks")
        and identity.cwd == ROOT.resolve()
        and record.get("root") == str(ROOT.resolve())
        and identity.executable == Path(str(record.get("python", ""))).resolve()
    )


@contextlib.contextmanager
def launch_lock(path: Path) -> Iterator[None]:
    """Serialize two launcher invocations, including stop while starting."""
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LauncherError("Another project launcher operation is still running.") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def check_ports_free(specs: Sequence[ServiceSpec]) -> None:
    """Bind locally without starting listeners or contacting existing services."""
    with contextlib.ExitStack() as stack:
        for spec in specs:
            handle = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                handle.bind(("127.0.0.1", spec.port))
            except OSError as error:
                raise LauncherError(
                    f"Port {spec.port} ({spec.name}) is unavailable; no services were started. "
                    "Stop its owner separately or use --stop for this launcher's recorded services."
                ) from error


def healthy(spec: ServiceSpec, response: httpx.Response) -> bool:
    if response.status_code != 200:
        return False
    if spec.name == "ui":
        return response.text.strip() == "ok"
    try:
        payload = response.json()
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    if spec.name == "model":
        return payload.get("status") == "ready" and payload.get("ready") is True
    count = payload.get("catalog_count")
    return (
        payload.get("status") == "ok"
        and isinstance(payload.get("dense_ready"), bool)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
    )


def wait_ready(
    running: Sequence[tuple[ServiceSpec, subprocess.Popen[bytes]]], timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    states = {spec.name: "waiting" for spec, _ in running}
    degraded_announced = False
    with httpx.Client(trust_env=False, follow_redirects=False) as client:
        while True:
            for spec, process in running:
                if process.poll() is not None:
                    raise LauncherError(
                        f"{spec.name} exited during startup; see logs/{spec.name}.log."
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LauncherError(
                        f"Readiness timed out after {timeout:g}s: {states}. See logs/."
                    )
                try:
                    response = client.get(
                        f"http://127.0.0.1:{spec.port}{spec.health_path}",
                        timeout=min(2.0, remaining),
                    )
                    states[spec.name] = "ready" if healthy(spec, response) else "not ready"
                    if (
                        spec.name == "api"
                        and states[spec.name] == "ready"
                        and response.json().get("dense_ready") is False
                        and not degraded_announced
                    ):
                        print(
                            "API ready in degraded mode: semantic index unavailable; "
                            "keyword search and the UI remain available.",
                            flush=True,
                        )
                        degraded_announced = True
                except httpx.HTTPError:
                    states[spec.name] = "unreachable"
            if all(value == "ready" for value in states.values()):
                if any(process.poll() is not None for _, process in running):
                    raise LauncherError("A service exited while readiness was being confirmed.")
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LauncherError(f"Readiness timed out after {timeout:g}s: {states}. See logs/.")
            time.sleep(min(0.25, remaining))


def write_record(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(path)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)


def remove_own_record(path: Path, launch_id: str) -> None:
    try:
        saved = json.loads(path.read_text())
        if isinstance(saved, dict) and saved.get("launch_id") == launch_id:
            path.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


def cleanup_children(running: Sequence[tuple[ServiceSpec, subprocess.Popen[bytes]]]) -> None:
    """Only Popen children from this failed invocation may be force-cleaned."""
    for _, process in running:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
    deadline = time.monotonic() + 5.0
    for _, process in running:
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)


def wait_child_identity(
    spec: ServiceSpec, process: subprocess.Popen[bytes], argv: Sequence[str], timeout: float = 2.0
) -> ProcessIdentity:
    """Allow a short exec transition while requiring the complete child identity."""
    started = time.monotonic()
    deadline = started + timeout
    expected_argv = tuple(argv)
    expected_executable = Path(sys.executable).resolve()
    last_mismatch = "process metadata unavailable"
    attempts = 0
    while True:
        attempts += 1
        return_code = process.poll()
        if return_code is not None:
            raise LauncherError(
                f"{spec.name} exited with code {return_code} before identity verification; "
                f"see logs/{spec.name}.log."
            )
        identity = process_identity(process.pid)
        if identity is not None:
            mismatches = [
                name
                for name, matches in (
                    ("argv", identity.argv == expected_argv),
                    ("cwd", identity.cwd == ROOT.resolve()),
                    ("executable", identity.executable == expected_executable),
                    ("start time", identity.start_ticks > 0),
                )
                if not matches
            ]
            if not mismatches:
                if attempts > 1:
                    print(
                        f"{spec.name} identity verified after {time.monotonic() - started:.3f}s "
                        f"(initially {last_mismatch}).",
                        flush=True,
                    )
                return identity
            last_mismatch = ", ".join(mismatches)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LauncherError(
                f"Unable to verify newly started {spec.name} within {timeout:g}s "
                f"({last_mismatch}); see logs/{spec.name}.log."
            )
        time.sleep(min(0.02, remaining))


def start_services(gpu_ids: list[str], timeout: float, pid_path: Path) -> dict[str, Any]:
    if not (ROOT / "data/processed/catalog.jsonl").exists():
        raise LauncherError("Prepare the catalog first: python scripts/prepare_catalog.py --help")
    if pid_path.exists():
        raise LauncherError("Service record exists; use --stop before restarting.")
    check_ports_free(SPECS)
    runtime = ROOT / "runtime"
    (ROOT / "logs").mkdir(exist_ok=True)
    for directory in ("tmp", "cache/huggingface", "cache/torch", "cache/matplotlib"):
        (runtime / directory).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
            "CHOICEAGENT_GPU_IDS": ",".join(gpu_ids),
            "CHOICEAGENT_MODEL_PRELOAD": "1",
            "CHOICEAGENT_MODEL_HOST": "127.0.0.1",
            "CHOICEAGENT_MODEL_PORT": "8001",
            "CHOICEAGENT_MODEL_BASE_URL": "http://127.0.0.1:8001",
            "CHOICEAGENT_API_URL": "http://127.0.0.1:8000",
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "TMPDIR": str(runtime / "tmp"),
            "HF_HOME": str(runtime / "cache/huggingface"),
            "TORCH_HOME": str(runtime / "cache/torch"),
            "MPLCONFIGDIR": str(runtime / "cache/matplotlib"),
            "PYTHONPATH": str(ROOT / "src")
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    record: dict[str, Any] = {
        "launch_id": str(uuid4()),
        "python": str(Path(sys.executable).resolve()),
        "root": str(ROOT.resolve()),
        "gpu_ids": gpu_ids,
        "started_at": time.time(),
        "status": "starting",
        "services": [],
    }
    running: list[tuple[ServiceSpec, subprocess.Popen[bytes]]] = []
    try:
        for spec in SPECS:
            argv = [sys.executable, *spec.arguments]
            service_env = env.copy()
            if spec.name == "ui":
                service_env["CUDA_VISIBLE_DEVICES"] = ""
            with (ROOT / f"logs/{spec.name}.log").open("ab") as output:
                process = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    env=service_env,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            running.append((spec, process))
            identity = wait_child_identity(spec, process, argv, timeout=min(2.0, timeout))
            record["services"].append(
                {
                    "name": spec.name,
                    "module": spec.module,
                    "pid": process.pid,
                    "argv": argv,
                    "start_ticks": identity.start_ticks,
                }
            )
        write_record(pid_path, record)
        wait_ready(running, timeout)
        record["status"] = "ready"
        record["ready_at"] = time.time()
        write_record(pid_path, record)
    except BaseException:
        try:
            cleanup_children(running)
        finally:
            remove_own_record(pid_path, record["launch_id"])
        raise
    return record


def signal_verified(entry: dict[str, Any], record: dict[str, Any]) -> bool:
    """Prefer pidfds; older Python builds recheck full identity before kill."""
    pid = entry.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if not is_owned(process_identity(pid), entry, record):
        return False
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        # Some Conda builds omit pidfd_open despite a recent Linux kernel.
        # Recheck birth time, argv, cwd and interpreter immediately before kill.
        if not is_owned(process_identity(pid), entry, record):
            return False
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except ProcessLookupError:
            return False
    try:
        descriptor = os.pidfd_open(pid, 0)
    except ProcessLookupError:
        return False
    try:
        if not is_owned(process_identity(pid), entry, record):
            return False
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(descriptor)


def stop_services(pid_path: Path, timeout: float) -> None:
    if not pid_path.exists():
        print("No recorded services.")
        return
    try:
        record = json.loads(pid_path.read_text())
    except (OSError, ValueError) as error:
        raise LauncherError("Cannot read service record; no process was signalled.") from error
    if (
        not isinstance(record, dict)
        or record.get("root") != str(ROOT.resolve())
        or not isinstance(record.get("services"), list)
    ):
        raise LauncherError(
            "Invalid or different-project service record; no process was signalled."
        )
    pending = []
    skipped = 0
    for entry in record["services"]:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        try:
            if signal_verified(entry, record):
                pending.append(entry)
            else:
                skipped += 1
        except OSError as error:
            if is_owned(process_identity(entry.get("pid", -1)), entry, record):
                pending.append(entry)
            print(
                f"Could not signal recorded {entry.get('name', 'service')}: {error}",
                file=sys.stderr,
            )
    deadline = time.monotonic() + timeout
    while pending:
        pending = [
            entry for entry in pending if is_owned(process_identity(entry["pid"]), entry, record)
        ]
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    if pending:
        record["services"] = pending
        record["status"] = "stopping"
        write_record(pid_path, record)
        survivors = ", ".join(f"{entry['name']} (PID {entry['pid']})" for entry in pending)
        raise LauncherError(
            f"Stop timed out after {timeout:g}s: {survivors}. Record retained at {pid_path}. "
            f"Inspect {ROOT / 'logs'} and retry: python scripts/start.py --stop --timeout 180."
        )
    pid_path.unlink(missing_ok=True)
    print(
        f"No verified recorded services remain; removed the record. Skipped {skipped} exited or unverified PID(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", default="5", help="One or two physical GPU IDs; default: 5")
    parser.add_argument(
        "--timeout",
        type=float,
        default=90,
        help="Bounded readiness/stop wait in seconds; default: 90",
    )
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not 1 <= len(ids) <= 2 or any(not value.isascii() or not value.isdigit() for value in ids):
        parser.error("--gpu-ids must contain one or two distinct numeric IDs")
    ids = [str(int(value)) for value in ids]
    if len(ids) != len(set(ids)):
        parser.error("--gpu-ids must contain one or two distinct numeric IDs")
    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    try:
        with launch_lock(runtime / "launcher.lock"):
            if args.stop:
                stop_services(runtime / "services.json", args.timeout)
            else:
                record = start_services(ids, args.timeout, runtime / "services.json")
                print(json.dumps(record, indent=2))
                print("Ready: UI http://127.0.0.1:8501 | API docs http://127.0.0.1:8000/docs")
    except (LauncherError, OSError) as error:
        print(f"Launcher: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except KeyboardInterrupt:
        print("Launcher interrupted; its startup children were cleaned up.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
