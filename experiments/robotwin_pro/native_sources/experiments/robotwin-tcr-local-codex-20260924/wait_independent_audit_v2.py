#!/usr/bin/env python3
"""Run independent RoboTwin support-recovery audit once both new queues end cleanly."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import time


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
BASE = WORK / "vla-merge-runtime/experiments/robotwin-tcr-local-codex-20260924"
WAIT = BASE / "independent-audit-waiter-v2"
BUILD_END = BASE / "tcr-build-attempt-02/queue-ended.json"
FORMAL_END = BASE / "tcr-formal-attempt-03/queue-ended.json"
AUDITOR = WORK / "coordination/2026-09-23/audit-robotwin-tcr-local-v2.py"
SOURCES = (AUDITOR, Path(__file__), BASE / "transfer-accept.json",
           BASE / "tcr-build-attempt-02/plan.json",
           BASE / "tcr-formal-attempt-03/plan.json")
POLL_SECONDS = 300


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(32 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def auditor_module():
    spec = importlib.util.spec_from_file_location("robotwin_local_independent_audit", AUDITOR)
    require(spec is not None and spec.loader is not None, "RoboTwin independent audit unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare() -> dict:
    require(not WAIT.exists(), "RoboTwin independent audit waiter already exists")
    provenance = auditor_module().preflight()
    plan = {"schema": "robotwin_local_independent_audit_waiter_v1",
            "created_at": now(), "host": socket.gethostname(),
            "no_gpu": True, "no_retry": True, "poll_seconds": POLL_SECONDS,
            "source_sha256": {str(path): digest(path) for path in SOURCES},
            "preflight": provenance,
            "output": str(WAIT / "independent-table-data.json")}
    WAIT.mkdir(parents=True, exist_ok=False)
    save(WAIT / "plan.json", plan)
    return {"plan": str(WAIT / "plan.json"), "preflight": provenance}


def run() -> None:
    require(not (WAIT / "started.json").exists(), "No waiter restart in same attempt")
    plan = read(WAIT / "plan.json")
    require(plan.get("schema") == "robotwin_local_independent_audit_waiter_v1"
            and plan.get("host") == socket.gethostname()
            and plan.get("no_gpu") is True and plan.get("no_retry") is True,
            "Frozen audit waiter plan differs")
    for name, expected in plan["source_sha256"].items():
        require(digest(Path(name)) == expected, f"Frozen audit input changed: {name}")
    require(auditor_module().preflight() == plan["preflight"],
            "RoboTwin audit preflight changed")
    pid = os.getpid()
    save(WAIT / "started.json", {"pid": pid, "host": socket.gethostname(),
         "kernel_starttime": int(Path(f"/proc/{pid}/stat").read_text().split()[21]),
         "started_at": now()})
    stopping = False

    def on_signal(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    while not stopping:
        if BUILD_END.exists() and FORMAL_END.exists():
            build = read(BUILD_END)
            formal = read(FORMAL_END)
            if build.get("complete") is not True or formal.get("complete") is not True:
                save(WAIT / "ended.json", {"complete": False,
                     "reason": "build_or_formal_queue_not_clean",
                     "build_terminal_sha256": digest(BUILD_END),
                     "formal_terminal_sha256": digest(FORMAL_END),
                     "ended_at": now()})
                return
            try:
                report = auditor_module().audit()
                output = Path(plan["output"])
                require(not output.exists(), "Independent table output already exists")
                save(output, report)
                save(WAIT / "ended.json", {"complete": True,
                     "table_data_path": str(output), "table_data_sha256": digest(output),
                     "episodes": report["episodes"], "ended_at": now()})
            except BaseException as error:
                save(WAIT / "ended.json", {"complete": False,
                     "reason": f"{type(error).__name__}: {error}", "ended_at": now()})
                raise
            return
        save(WAIT / "state.json", {"status": "waiting_clean_terminals",
             "build_terminal": BUILD_END.exists(), "formal_terminal": FORMAL_END.exists(),
             "updated_at": now()})
        time.sleep(POLL_SECONDS)
    save(WAIT / "ended.json", {"complete": False, "reason": "signal", "ended_at": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(), sort_keys=True))
    else:
        run()
