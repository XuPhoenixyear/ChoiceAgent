"""Launcher failure/ownership tests never start services or use occupied ports."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from scripts import start as launcher


@pytest.fixture
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    (tmp_path / "runtime").mkdir()
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/processed/catalog.jsonl").write_text("{}\n")
    return tmp_path


def process(pid: int) -> MagicMock:
    child = MagicMock(spec=subprocess.Popen)
    child.pid = pid
    child.poll.return_value = None
    child.wait.return_value = 0
    return child


def record_entry(
    root: Path, pid: int, spec: launcher.ServiceSpec
) -> tuple[dict[str, Any], launcher.ProcessIdentity]:
    argv = [sys.executable, *spec.arguments]
    identity = launcher.ProcessIdentity(
        tuple(argv), root.resolve(), Path(sys.executable).resolve(), pid * 100
    )
    entry = {
        "name": spec.name,
        "module": spec.module,
        "pid": pid,
        "argv": argv,
        "start_ticks": identity.start_ticks,
    }
    return entry, identity


def saved_record(root: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "root": str(root.resolve()),
        "python": str(Path(sys.executable).resolve()),
        "launch_id": "unit-launch",
        "services": entries,
        "status": "ready",
    }


def fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    clock = [0.0]
    monkeypatch.setattr(launcher.time, "monotonic", lambda: clock[0])

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(launcher.time, "sleep", sleep)
    return clock


@pytest.mark.parametrize(
    ("service", "response", "expected"),
    [
        ("model", httpx.Response(200, json={"status": "not_loaded", "ready": False}), False),
        ("model", httpx.Response(200, json={"status": "ready", "ready": "true"}), False),
        ("model", httpx.Response(200, json={"status": "ready", "ready": True}), True),
        (
            "api",
            httpx.Response(200, json={"status": "ok", "dense_ready": False, "catalog_count": 20}),
            True,
        ),
        (
            "api",
            httpx.Response(200, json={"status": "ok", "dense_ready": "false", "catalog_count": 20}),
            False,
        ),
        (
            "api",
            httpx.Response(200, json={"status": "ok", "catalog_count": 20}),
            False,
        ),
        (
            "api",
            httpx.Response(200, json={"status": "ok", "dense_ready": True, "catalog_count": 0}),
            False,
        ),
        (
            "api",
            httpx.Response(200, json={"status": "ok", "dense_ready": True, "catalog_count": 20}),
            True,
        ),
        ("ui", httpx.Response(200, text="starting"), False),
        ("ui", httpx.Response(200, text="ok"), True),
        ("model", httpx.Response(200, text="not JSON"), False),
        ("model", httpx.Response(503, json={"status": "ready", "ready": True}), False),
    ],
)
def test_http_200_is_not_readiness(service: str, response: httpx.Response, expected: bool) -> None:
    spec = next(spec for spec in launcher.SPECS if spec.name == service)
    assert launcher.healthy(spec, response) is expected


def test_occupied_port_prevents_any_launch(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = MagicMock()
    handle.__enter__.return_value = handle
    handle.bind.side_effect = OSError(98, "Address already in use")
    monkeypatch.setattr(launcher.socket, "socket", MagicMock(return_value=handle))
    spawn = MagicMock()
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    pid_path = isolated_root / "runtime/services.json"
    with pytest.raises(launcher.LauncherError, match="Port 8001"):
        launcher.start_services(["5"], 90, pid_path)
    spawn.assert_not_called()
    assert not pid_path.exists()


def test_readiness_timeout_is_bounded_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = fake_clock(monkeypatch)
    client = MagicMock()
    client.__enter__.return_value = client
    client.get.return_value = httpx.Response(200, json={"status": "not_loaded", "ready": False})
    monkeypatch.setattr(launcher.httpx, "Client", MagicMock(return_value=client))
    with pytest.raises(launcher.LauncherError, match="timed out"):
        launcher.wait_ready([(launcher.SPECS[0], process(101))], 0.6)
    assert clock[0] == pytest.approx(0.6)
    assert client.get.call_count == 3


def test_dead_child_cannot_be_masked_by_healthy_other_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = process(101)
    child.poll.side_effect = [None, 1]
    client = MagicMock()
    client.__enter__.return_value = client
    client.get.return_value = httpx.Response(200, json={"status": "ready", "ready": True})
    monkeypatch.setattr(launcher.httpx, "Client", MagicMock(return_value=client))
    with pytest.raises(launcher.LauncherError, match="exited"):
        launcher.wait_ready([(launcher.SPECS[0], child)], 1)


def test_lexical_ready_api_is_accepted_with_degraded_notice(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.get.return_value = httpx.Response(
        200, json={"status": "ok", "dense_ready": False, "catalog_count": 20}
    )
    monkeypatch.setattr(launcher.httpx, "Client", MagicMock(return_value=client))
    child = process(101)
    launcher.wait_ready([(launcher.SPECS[1], child)], 1)
    assert "degraded mode" in capsys.readouterr().out
    child.terminate.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_launch_waits_and_rolls_back_its_own_record(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    monkeypatch.setattr(launcher, "check_ports_free", MagicMock())
    children = [process(100 + index) for index in range(3)]
    spawn = MagicMock(side_effect=children)
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    identities = {
        child.pid: record_entry(isolated_root, child.pid, spec)[1]
        for spec, child in zip(launcher.SPECS, children, strict=True)
    }
    monkeypatch.setattr(launcher, "process_identity", lambda pid: identities[pid])
    wait = MagicMock(side_effect=launcher.LauncherError("not ready") if fail else None)
    monkeypatch.setattr(launcher, "wait_ready", wait)
    pid_path = isolated_root / "runtime/services.json"
    if fail:
        with pytest.raises(launcher.LauncherError, match="not ready"):
            launcher.start_services(["5"], 9, pid_path)
        assert not pid_path.exists()
        for child in children:
            child.terminate.assert_called_once()
            child.wait.assert_called_once()
    else:
        record = launcher.start_services(["5"], 9, pid_path)
        assert record["status"] == "ready"
        assert json.loads(pid_path.read_text())["services"] == record["services"]
        for child in children:
            child.terminate.assert_not_called()
    wait.assert_called_once()
    assert wait.call_args.args[1] == 9
    for index, call in enumerate(spawn.call_args_list):
        assert call.kwargs["env"]["CHOICEAGENT_MODEL_PRELOAD"] == "1"
        assert call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ("" if index == 2 else "5")
        assert Path(call.kwargs["stdout"].name).parent == isolated_root / "logs"


def test_rollback_never_removes_a_different_launch_record(tmp_path: Path) -> None:
    path = tmp_path / "services.json"
    path.write_text(json.dumps({"launch_id": "another-launch"}))
    launcher.remove_own_record(path, "this-launch")
    assert path.exists()


def test_partial_spawn_failure_cleans_only_started_children(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = process(123)
    _, identity = record_entry(isolated_root, child.pid, launcher.SPECS[0])
    monkeypatch.setattr(launcher, "check_ports_free", MagicMock())
    monkeypatch.setattr(launcher, "process_identity", lambda _pid: identity)
    spawn = MagicMock(side_effect=[child, OSError("spawn failed")])
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    path = isolated_root / "runtime/services.json"
    with pytest.raises(OSError, match="spawn failed"):
        launcher.start_services(["5"], 1, path)
    assert spawn.call_count == 2
    child.terminate.assert_called_once()
    child.wait.assert_called_once()
    assert not path.exists()


def test_failed_start_kills_only_its_unresponsive_child() -> None:
    stuck, exited = process(100), process(101)
    stuck.wait.side_effect = [subprocess.TimeoutExpired("unit child", 5), 0]
    launcher.cleanup_children([(launcher.SPECS[0], stuck), (launcher.SPECS[1], exited)])
    stuck.kill.assert_called_once()
    exited.kill.assert_not_called()


@pytest.mark.parametrize("initial", ["missing", "old_argv", "old_cwd", "old_executable"])
def test_new_child_identity_waits_for_exec_transition(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial: str,
) -> None:
    clock = fake_clock(monkeypatch)
    entry, identity = record_entry(isolated_root, 123, launcher.SPECS[0])
    first = {
        "missing": None,
        "old_argv": replace(identity, argv=(sys.executable, "scripts/start.py")),
        "old_cwd": replace(identity, cwd=isolated_root / "old"),
        "old_executable": replace(identity, executable=Path("/old/python")),
    }[initial]
    reads = MagicMock(side_effect=[first, identity])
    monkeypatch.setattr(launcher, "process_identity", reads)
    assert launcher.wait_child_identity(launcher.SPECS[0], process(123), entry["argv"]) == identity
    assert reads.call_count == 2
    assert clock[0] == pytest.approx(0.02)


def test_new_child_identity_timeout_is_bounded(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = fake_clock(monkeypatch)
    monkeypatch.setattr(launcher, "process_identity", lambda _pid: None)
    with pytest.raises(launcher.LauncherError, match=r"within 0\.1s"):
        launcher.wait_child_identity(launcher.SPECS[0], process(123), [sys.executable], timeout=0.1)
    assert clock[0] == pytest.approx(0.1)


def test_new_child_exit_is_detected_before_identity_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    child = process(123)
    child.poll.return_value = 2
    identity_read = MagicMock()
    monkeypatch.setattr(launcher, "process_identity", identity_read)
    with pytest.raises(launcher.LauncherError, match="exited with code 2"):
        launcher.wait_child_identity(launcher.SPECS[0], child, [sys.executable])
    identity_read.assert_not_called()


@pytest.mark.parametrize("change", ["birth", "cwd", "argv", "python"])
def test_stop_rejects_reused_or_different_process(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    entry, identity = record_entry(isolated_root, 123, launcher.SPECS[0])
    variants = {
        "birth": replace(identity, start_ticks=identity.start_ticks + 1),
        "cwd": replace(identity, cwd=isolated_root / "other"),
        "argv": replace(identity, argv=(sys.executable, "another_task.py")),
        "python": replace(identity, executable=Path("/other/python")),
    }
    monkeypatch.setattr(launcher, "process_identity", lambda _pid: variants[change])
    monkeypatch.setattr(launcher.os, "pidfd_open", MagicMock(return_value=77), raising=False)
    close = MagicMock()
    monkeypatch.setattr(launcher.os, "close", close)
    send = MagicMock()
    monkeypatch.setattr(launcher.signal, "pidfd_send_signal", send)
    assert not launcher.signal_verified(entry, saved_record(isolated_root, [entry]))
    send.assert_not_called()
    close.assert_not_called()


def test_stop_uses_pidfd_then_prunes_exited_record(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, identity = record_entry(isolated_root, 123, launcher.SPECS[0])
    pid_path = isolated_root / "runtime/services.json"
    pid_path.write_text(json.dumps(saved_record(isolated_root, [entry])))
    monkeypatch.setattr(
        launcher, "process_identity", MagicMock(side_effect=[identity, identity, None])
    )
    monkeypatch.setattr(launcher.os, "pidfd_open", MagicMock(return_value=77), raising=False)
    monkeypatch.setattr(launcher.os, "close", MagicMock())
    send = MagicMock()
    monkeypatch.setattr(launcher.signal, "pidfd_send_signal", send)
    launcher.stop_services(pid_path, 1)
    send.assert_called_once_with(77, launcher.signal.SIGTERM)
    assert not pid_path.exists()


def test_stop_timeout_retains_verified_survivor(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = fake_clock(monkeypatch)
    entry, identity = record_entry(isolated_root, 123, launcher.SPECS[0])
    pid_path = isolated_root / "runtime/services.json"
    pid_path.write_text(json.dumps(saved_record(isolated_root, [entry])))
    monkeypatch.setattr(launcher, "signal_verified", MagicMock(return_value=True))
    monkeypatch.setattr(launcher, "process_identity", lambda _pid: identity)
    with pytest.raises(launcher.LauncherError, match="Stop timed out"):
        launcher.stop_services(pid_path, 0.3)
    assert clock[0] == pytest.approx(0.3)
    saved = json.loads(pid_path.read_text())
    assert saved["status"] == "stopping"
    assert saved["services"] == [entry]


def test_exited_or_unverified_entries_are_cleaned_without_signals(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, _ = record_entry(isolated_root, 123, launcher.SPECS[0])
    pid_path = isolated_root / "runtime/services.json"
    pid_path.write_text(json.dumps(saved_record(isolated_root, [entry])))
    monkeypatch.setattr(launcher, "process_identity", lambda _pid: None)
    monkeypatch.setattr(launcher.os, "pidfd_open", MagicMock(return_value=77), raising=False)
    monkeypatch.setattr(launcher.os, "close", MagicMock())
    send = MagicMock()
    monkeypatch.setattr(launcher.signal, "pidfd_send_signal", send)
    launcher.stop_services(pid_path, 1)
    send.assert_not_called()
    assert not pid_path.exists()


def test_current_process_identity_can_be_read_without_signalling() -> None:
    identity = launcher.process_identity(os.getpid())
    assert identity is not None
    assert identity.start_ticks > 0
    assert identity.executable == Path(sys.executable).resolve()


@pytest.mark.parametrize("reused", [False, True])
def test_stop_without_pidfd_rechecks_identity_before_kill(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    reused: bool,
) -> None:
    entry, identity = record_entry(isolated_root, 123, launcher.SPECS[0])
    second_read = replace(identity, start_ticks=identity.start_ticks + 1) if reused else identity
    monkeypatch.delattr(launcher.os, "pidfd_open", raising=False)
    monkeypatch.setattr(
        launcher, "process_identity", MagicMock(side_effect=[identity, second_read])
    )
    kill = MagicMock()
    monkeypatch.setattr(launcher.os, "kill", kill)
    assert launcher.signal_verified(entry, saved_record(isolated_root, [entry])) is not reused
    if reused:
        kill.assert_not_called()
    else:
        kill.assert_called_once_with(123, launcher.signal.SIGTERM)


def test_concurrent_launcher_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "launcher.lock"
    with (
        launcher.launch_lock(path),
        pytest.raises(launcher.LauncherError, match="Another project launcher"),
        launcher.launch_lock(path),
    ):
        pytest.fail("The second launcher acquired the same lock")
