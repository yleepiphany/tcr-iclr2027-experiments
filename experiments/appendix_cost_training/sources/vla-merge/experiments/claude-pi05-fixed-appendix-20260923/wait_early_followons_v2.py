#!/usr/bin/env python3
"""Start the frozen early-request build and formal queue after content acceptance.

The waiter performs no GPU work. It gives each dependency one attempt, preserves
failure evidence, and never changes an active or frozen experiment.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
ROOT = WORK / "vla-merge-runtime/experiments/claude-pi05-fixed-appendix-20260923"
CAPTURE = ROOT / "early-capture-attempt-04"
BUILD = ROOT / "early-build-attempt-02"
EVAL = ROOT / "early-formal-eval-attempt-02"
RUN = ROOT / "early-followons-waiter-v2"
AUDITOR = WORK / "coordination/2026-09-23/audit-appendix-early-pools.py"
AUDIT = WORK / "coordination/2026-09-23/appendix-early-pools-content-audit.json"
PYTHON = WORK / "pi05_lora_finetune_v2_20260826/.venv/bin/python"
BUILD_ENTRY = HERE / "run_early_build_v8.py"
EVAL_ENTRY = HERE / "run_early_eval_v8.py"
POLL_SECONDS = 120


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


def owner_alive(receipt: Path) -> bool:
    if not receipt.is_file():
        return False
    row = json.loads(receipt.read_text())
    if row.get("host") != socket.gethostname():
        return False
    proc = Path(f"/proc/{row['pid']}/stat")
    return proc.is_file() and int(proc.read_text().split()[21]) == row["kernel_starttime"]


def ensure_capture_accepted() -> bool:
    terminal = CAPTURE / "queue-ended.json"
    if not terminal.is_file():
        if not owner_alive(CAPTURE / "waiter-started.json"):
            raise RuntimeError("early capture has no live owner or terminal receipt")
        return False
    row = json.loads(terminal.read_text())
    if row.get("status") != "complete" or row.get("stopped") is not False:
        raise RuntimeError("early capture did not complete cleanly")
    return True


def launch(entry: Path, label: str) -> dict:
    log_path = RUN / f"{label}.log"
    with log_path.open("x") as log:
        child = subprocess.Popen([str(PYTHON), "-u", str(entry), "run"],
                                 cwd=WORK, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env={**os.environ,
                                 "PYTHONDONTWRITEBYTECODE": "1"})
    time.sleep(1)
    proc = Path(f"/proc/{child.pid}/stat")
    starttime = int(proc.read_text().split()[21]) if proc.is_file() else None
    receipt = {"entry": str(entry), "entry_sha256": sha(entry),
               "pid": child.pid, "host": socket.gethostname(),
               "kernel_starttime": starttime,
               "started_at": datetime.now(timezone.utc).isoformat(),
               "immediate_returncode": child.poll()}
    save(RUN / f"{label}-launch.json", receipt)
    if child.poll() is not None:
        raise RuntimeError(f"{label} supervisor exited immediately: {child.returncode}")
    return receipt


def run() -> None:
    if RUN.exists() or BUILD.exists() or EVAL.exists() or AUDIT.exists():
        raise FileExistsError("early follow-on attempt or audit already exists")
    RUN.mkdir(parents=True, exist_ok=False)
    save(RUN / "started.json", {"pid": os.getpid(), "host": socket.gethostname(),
         "kernel_starttime": int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]),
         "started_at": datetime.now(timezone.utc).isoformat(),
         "auditor_sha256": sha(AUDITOR), "build_entry_sha256": sha(BUILD_ENTRY),
         "eval_entry_sha256": sha(EVAL_ENTRY)})
    stopping = False

    def on_signal(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        while not ensure_capture_accepted():
            if stopping:
                raise RuntimeError("operator stopped dependency waiter before capture")
            save(RUN / "state.json", {"phase": "waiting_capture", "gpu_work_started": False,
                 "updated_at": datetime.now(timezone.utc).isoformat()})
            time.sleep(POLL_SECONDS)
        if stopping:
            raise RuntimeError("operator stopped dependency waiter before audit")
        save(RUN / "state.json", {"phase": "auditing_capture", "gpu_work_started": False,
             "updated_at": datetime.now(timezone.utc).isoformat()})
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        with (RUN / "content-audit.log").open("x") as log:
            subprocess.run([str(PYTHON), str(AUDITOR), "--run", str(CAPTURE),
                            "--output", str(AUDIT)], cwd=WORK, env=env,
                           stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        if json.loads(AUDIT.read_text()).get("accepted") is not True:
            raise RuntimeError("early capture content audit rejected")
        if stopping:
            raise RuntimeError("operator stopped dependency waiter before follow-ons")
        with (RUN / "build-prepare.log").open("x") as log:
            subprocess.run([str(PYTHON), str(BUILD_ENTRY), "prepare"], cwd=WORK,
                           env=env, stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        with (RUN / "eval-prepare.log").open("x") as log:
            subprocess.run([str(PYTHON), str(EVAL_ENTRY), "prepare"], cwd=WORK,
                           env=env, stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        build = launch(BUILD_ENTRY, "build")
        for _ in range(15):
            if (BUILD / "started.json").is_file():
                break
            if not owner_alive(RUN / "build-launch.json"):
                raise RuntimeError("early build supervisor died before start receipt")
            time.sleep(1)
        else:
            raise RuntimeError("early build supervisor produced no start receipt")
        evaluation = launch(EVAL_ENTRY, "eval")
        save(RUN / "ended.json", {"complete": True, "capture_audit_sha256": sha(AUDIT),
             "build_plan_sha256": sha(BUILD / "plan.json"),
             "eval_plan_sha256": sha(EVAL / "plan.json"),
             "build_supervisor": build, "eval_supervisor": evaluation,
             "ended_at": datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        save(RUN / "ended.json", {"complete": False,
             "error": f"{type(error).__name__}: {error}",
             "ended_at": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    run()
