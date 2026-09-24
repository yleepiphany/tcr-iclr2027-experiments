#!/usr/bin/env python3
"""Run independent PRO baseline audits only after each clean formal terminal."""
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
WAIT = ROOT / "independent-audit-waiter-v1"
AUDITOR = HERE / "audit_baseline_formal.py"
METHODS = ("ties_merging", "regmean_pp", "featcal")
HOST = "dsw-967394-56ffd4897d-42wft"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    if socket.gethostname() != HOST or WAIT.exists():
        raise ValueError("Wrong host or audit waiter already exists")
    for method in METHODS:
        subprocess.run([sys.executable, str(AUDITOR), "preflight", "--method", method],
                       cwd=WORK, check=True, stdout=subprocess.DEVNULL)
    WAIT.mkdir(parents=True, exist_ok=False)
    auditor_sha = sha(AUDITOR)
    write(WAIT / "started.json", {"host": HOST, "pid": os.getpid(),
                                  "started_at": now(), "auditor_sha256": auditor_sha,
                                  "waiter_sha256": sha(Path(__file__)),
                                  "methods": list(METHODS), "no_retry": True,
                                  "gpu_used": False})
    pending = set(METHODS)
    accepted: dict[str, dict] = {}
    rejected: dict[str, str] = {}
    while pending:
        if sha(AUDITOR) != auditor_sha:
            rejected["auditor"] = "Frozen independent auditor source changed"
            break
        for method in tuple(pending):
            run = ROOT / method / "formal-v1"
            terminal_path = run / "queue-ended.json"
            if not terminal_path.exists():
                continue
            pending.remove(method)
            terminal = read(terminal_path)
            if terminal.get("complete") is not True or terminal.get("stopped") is not False:
                rejected[method] = "Formal queue lacks clean terminal"
                continue
            log_path = WAIT / f"{method}.audit.log"
            with log_path.open("x") as stream:
                proc = subprocess.run([sys.executable, str(AUDITOR), "results", "--method", method],
                                      cwd=WORK, stdin=subprocess.DEVNULL,
                                      stdout=stream, stderr=subprocess.STDOUT)
            output = run / "independent-table-data-v1.json"
            if proc.returncode == 0 and output.exists() and read(output).get("accepted") is True:
                accepted[method] = {"terminal_sha256": sha(terminal_path),
                                    "table_data_sha256": sha(output), "audited_at": now()}
            else:
                rejected[method] = f"Independent audit exit {proc.returncode}"
        write(WAIT / "state.json", {"status": "waiting_formal_terminals" if pending else "complete",
                                    "accepted": accepted, "rejected": rejected,
                                    "pending": sorted(pending), "updated_at": now()})
        if pending:
            time.sleep(60)
    write(WAIT / "queue-ended.json", {"complete": not pending and not rejected,
                                      "accepted": accepted, "rejected": rejected,
                                      "pending": sorted(pending), "ended_at": now()})


if __name__ == "__main__":
    main()
