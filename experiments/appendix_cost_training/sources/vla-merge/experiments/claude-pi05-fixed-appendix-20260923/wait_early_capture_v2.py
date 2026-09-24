#!/usr/bin/env python3
"""Admit the frozen early-request collector after Fast-WAM gets first claim.

This waiter never retries an experiment. A zero-card admission probe is only a
wait; once any capture worker starts, its single queue result is terminal.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "vla-merge-runtime/experiments/claude-pi05-fixed-appendix-20260923/early-capture-attempt-04"
FAST = ROOT / "vla-merge-runtime/experiments/claude-fastwam-tcr-20260923"
sys.path.insert(0, str(HERE))
import run_early_capture_v4 as capture  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def priority_ready() -> tuple[bool, str]:
    pipeline = FAST / "pipeline-v5/state.json"
    if not pipeline.is_file():
        return False, "waiting_fastwam_v5_state"
    state = json.loads(pipeline.read_text())
    phase = state.get("phase")
    if phase in ("two_pass_build", "native_reload_gate") and state.get("active"):
        return True, phase
    if phase == "complete":
        evaluation = FAST / "evaluation-queue-v5/state.json"
        if evaluation.is_file():
            current = json.loads(evaluation.read_text())
            if current.get("active") or (FAST / "evaluation-queue-v5/ended.json").exists():
                return True, "fastwam_eval_active_or_terminal"
        return False, "waiting_fastwam_eval_first_claim"
    return False, f"waiting_fastwam_v5_{phase}"


def run() -> dict:
    started = RUN / "waiter-started.json"
    if started.exists() or (RUN / "queue-ended.json").exists():
        raise FileExistsError("Early capture waiter may start only once")
    plan = json.loads((RUN / "plan.json").read_text())
    if (plan.get("schema") != "pi05_appendix_early_capture_v1" or
            plan.get("hostname") != socket.gethostname() or len(plan.get("jobs", [])) != 8 or
            plan["source_hashes"].get(str(Path(capture.__file__).resolve())) != sha(Path(capture.__file__))):
        raise ValueError("Frozen early capture plan differs")
    save(started, {"pid": os.getpid(), "kernel_starttime": int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]),
                   "host": socket.gethostname(), "started_at": datetime.now(timezone.utc).isoformat(),
                   "capture_wrapper_sha256": sha(Path(capture.__file__))})
    stopping = False

    def on_signal(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    while True:
        if stopping:
            result = {"complete": False, "reason": "waiter operator stop before GPU admission"}
            save(RUN / "waiter-ended.json", result)
            return result
        ready, reason = priority_ready()
        save(RUN / "waiter-state.json", {"phase": "waiting_safe_gpu" if ready else "waiting_priority",
             "reason": reason, "updated_at": datetime.now(timezone.utc).isoformat(), "gpu_work_started": False})
        if not ready:
            time.sleep(60)
            continue
        safe = [row for row in capture.gpu_rows() if
                capture.MIN_FREE_MIB <= row["free_mib"] < capture.MAX_FREE_MIB_FOR_EARLY]
        if not safe:
            time.sleep(60)
            continue
        try:
            result = capture.launch(RUN, 30)
        except RuntimeError as error:
            if str(error) == "No admissible and lockable GPU is currently available":
                time.sleep(60)
                continue
            raise
        save(RUN / "waiter-ended.json", {"complete": result.get("status") == "complete",
             "queue_terminal": result, "ended_at": datetime.now(timezone.utc).isoformat()})
        return result


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
