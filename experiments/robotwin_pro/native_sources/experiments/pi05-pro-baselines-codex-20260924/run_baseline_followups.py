#!/usr/bin/env python3
"""Queue two independent PRO baselines after the active TIES lane releases GPU 7."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
ROOT = WORK / "vla-merge-runtime/experiments/pi05-pro-baselines-codex-20260924"
CHAIN = ROOT / "followup-chain-v1"
PYTHON = Path(sys.executable)
RUNNER = HERE / "run_baseline_formal.py"
AUDITOR = HERE / "audit_baseline_formal.py"
HOST = "dsw-967394-56ffd4897d-42wft"
PLAN_SHA = {
    "ties_merging": "eec53167c4204960155b903bea8f564f44ee79387bef1212ab3f3191576107cf",
    "regmean_pp": "697ed9707b6fb6d73657d0abcaab8cd59b923a1bcb5cd4398360cab52a9d39db",
    "featcal": "25f04c5076c3195efb0bb73b3eb11beb0a3785c5f59c55ae0b9b080460b18b24",
}
ORDER = ("ties_merging", "regmean_pp", "featcal")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_ticks(pid: int) -> int | None:
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
        fields = data.rsplit(") ", 1)[1].split()
        return int(fields[19]) if fields[0] != "Z" else None
    except FileNotFoundError:
        return None


def run() -> None:
    if socket.gethostname() != HOST:
        raise ValueError("Wrong host")
    if CHAIN.exists():
        raise FileExistsError("Followup chain already exists; no retry")
    for method, expected in PLAN_SHA.items():
        run_dir = ROOT / method / "formal-v1"
        if sha(run_dir / "plan.json") != expected or read(run_dir / "CLAIM.json")["plan_sha256"] != expected:
            raise ValueError(f"Frozen {method} plan or claim changed")
        subprocess.run([str(PYTHON), str(AUDITOR), "preflight", "--method", method],
                       cwd=WORK, check=True, stdout=subprocess.DEVNULL)
    ties_dir = ROOT / "ties_merging/formal-v1"
    ties = read(ties_dir / "started.json")
    if (ties["host"] != HOST or ties["plan_sha256"] != PLAN_SHA["ties_merging"]
            or start_ticks(ties["pid"]) is None):
        raise ValueError("TIES active supervisor identity differs")
    ties_ticks = start_ticks(ties["pid"])
    CHAIN.mkdir(parents=True, exist_ok=False)
    write(CHAIN / "started.json", {"host": HOST, "pid": os.getpid(),
                                  "started_at": now(), "ties_pid": ties["pid"],
                                  "ties_start_ticks": ties_ticks,
                                  "plan_sha256": PLAN_SHA,
                                  "runner_sha256": sha(RUNNER),
                                  "auditor_sha256": sha(AUDITOR),
                                  "chain_sha256": sha(Path(__file__)),
                                  "no_retry": True, "gpu_range": [7]})
    state = {"status": "waiting_ties_terminal", "completed": [],
             "active": None, "pending": list(ORDER[1:]), "updated_at": now()}
    write(CHAIN / "state.json", state)
    while not (ties_dir / "queue-ended.json").exists():
        if start_ticks(ties["pid"]) != ties_ticks:
            state.update(status="stopped_ties_without_terminal", updated_at=now())
            write(CHAIN / "state.json", state)
            return
        time.sleep(60)
    state["ties_terminal_sha256"] = sha(ties_dir / "queue-ended.json")
    for method in ORDER[1:]:
        run_dir = ROOT / method / "formal-v1"
        if ((run_dir / "started.json").exists() or (run_dir / "queue-ended.json").exists()
                or sha(run_dir / "plan.json") != PLAN_SHA[method]
                or sha(RUNNER) != read(CHAIN / "started.json")["runner_sha256"]):
            state.update(status=f"stopped_{method}_identity_or_duplicate", updated_at=now())
            write(CHAIN / "state.json", state)
            return
        state.update(status="running", active=method, pending=[m for m in ORDER[1:] if m != method and m not in state["completed"]], updated_at=now())
        write(CHAIN / "state.json", state)
        log_path = CHAIN / f"{method}.supervisor.log"
        with log_path.open("x") as log:
            child = subprocess.Popen([str(PYTHON), "-u", str(RUNNER), "run", "--method", method],
                                     cwd=WORK, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            write(CHAIN / f"{method}.launch.json", {"pid": child.pid, "started_at": now(),
                                                     "plan_sha256": PLAN_SHA[method]})
            code = child.wait()
        terminal = run_dir / "queue-ended.json"
        write(CHAIN / f"{method}.exit.json", {"returncode": code, "finished_at": now(),
                                               "terminal_sha256": sha(terminal) if terminal.exists() else None})
        state["completed"].append(method)
        state["active"] = None
        state["updated_at"] = now()
        if not terminal.exists():
            state["status"] = f"stopped_{method}_without_terminal"
            write(CHAIN / "state.json", state)
            return
        state["pending"] = [m for m in ORDER[1:] if m not in state["completed"]]
        write(CHAIN / "state.json", state)
    state.update(status="all_terminals_recorded", updated_at=now())
    write(CHAIN / "state.json", state)


if __name__ == "__main__":
    run()
