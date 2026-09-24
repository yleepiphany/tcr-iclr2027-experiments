#!/usr/bin/env python3
"""Run the frozen independent PRO Soup audit once the full GPU queue ends."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
RUN = WORK / "vla-merge-runtime/experiments/pi05-pro-soup-codex-20260924/formal-v1"
WAITER = RUN / "audit-waiter-v1"
AUDITOR = HERE / "audit_soup_formal.py"
PYTHON = WORK / "vla-merge-runtime/envs/iclr2027-libero-pro-py312-v1/bin/python"
FORMAL_PLAN_SHA = "6ba9c3450d4f09a15eda940ea6c8398445dad3ef08a0606c58160a2ff426b7a0"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def process_start_ticks(pid: int) -> int:
    content = Path(f"/proc/{pid}/stat").read_text()
    return int(content.rsplit(") ", 1)[1].split()[19])


def prepare() -> dict:
    if WAITER.exists():
        raise FileExistsError(WAITER)
    formal = json.loads((RUN / "started.json").read_text())
    if sha(RUN / "plan.json") != FORMAL_PLAN_SHA or formal["plan_sha256"] != FORMAL_PLAN_SHA:
        raise ValueError("Formal plan/start identity mismatch")
    ticks = process_start_ticks(formal["pid"])
    plan = {"schema": "pi05_pro_soup_independent_audit_waiter_v1",
            "created_at": now(), "gpu_use": False, "no_retry": True,
            "formal_plan_sha256": FORMAL_PLAN_SHA,
            "formal_pid": formal["pid"], "formal_start_ticks": ticks,
            "auditor_sha256": sha(AUDITOR), "waiter_sha256": sha(Path(__file__)),
            "poll_seconds": 300}
    WAITER.mkdir(parents=True, exist_ok=False)
    write(WAITER / "plan.json", plan)
    return {"path": str(WAITER), "plan_sha256": sha(WAITER / "plan.json")}


def run() -> None:
    plan = json.loads((WAITER / "plan.json").read_text())
    if (plan.get("schema") != "pi05_pro_soup_independent_audit_waiter_v1"
            or plan.get("formal_plan_sha256") != FORMAL_PLAN_SHA
            or plan.get("auditor_sha256") != sha(AUDITOR)
            or plan.get("waiter_sha256") != sha(Path(__file__))
            or sha(RUN / "plan.json") != FORMAL_PLAN_SHA
            or (WAITER / "started.json").exists()):
        raise ValueError("Frozen waiter identity mismatch or duplicate start")
    write(WAITER / "started.json", {"pid": os.getpid(), "started_at": now(),
                                     "plan_sha256": sha(WAITER / "plan.json")})
    while not (RUN / "queue-ended.json").exists():
        try:
            alive = process_start_ticks(plan["formal_pid"]) == plan["formal_start_ticks"]
        except FileNotFoundError:
            alive = False
        if not alive:
            write(WAITER / "ended.json", {"complete": False,
                                           "reason": "formal supervisor missing without terminal receipt",
                                           "ended_at": now()})
            return
        write(WAITER / "state.json", {"status": "waiting_formal_terminal", "updated_at": now()})
        time.sleep(plan["poll_seconds"])
    terminal = json.loads((RUN / "queue-ended.json").read_text())
    if terminal.get("complete") is not True:
        write(WAITER / "ended.json", {"complete": False,
                                       "reason": "formal queue ended incomplete",
                                       "formal_terminal_sha256": sha(RUN / "queue-ended.json"),
                                       "ended_at": now()})
        return
    command = [str(PYTHON), str(AUDITOR), "results"]
    with (WAITER / "audit.log").open("x") as log:
        code = subprocess.call(command, cwd=WORK, stdout=log, stderr=subprocess.STDOUT)
    data = RUN / "independent-table-data-v1.json"
    write(WAITER / "ended.json", {"complete": code == 0 and data.is_file(),
                                   "returncode": code, "command": command,
                                   "formal_terminal_sha256": sha(RUN / "queue-ended.json"),
                                   "table_data_sha256": sha(data) if data.is_file() else None,
                                   "ended_at": now()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(), indent=2))
    else:
        run()


if __name__ == "__main__":
    main()
