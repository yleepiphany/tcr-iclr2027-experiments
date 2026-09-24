#!/usr/bin/env python3
"""Wait for one accepted ordered build, then start its paired development-400."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time

import run_ordered_candidate as build
import run_ordered_development400 as development


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    build.base.BASE_SAVE(path, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    build_run, eval_run = args.build.resolve(), args.evaluation.resolve()
    plan = json.loads((build_run / "plan.json").read_text())
    if plan.get("schema") != "openvla_r2_ordered_aux_build_v1" or plan.get("host") != socket.gethostname():
        raise ValueError("Wrong ordered build plan or host")
    if eval_run.exists():
        raise FileExistsError("Development attempt already exists")
    save(build_run / "pipeline-started.json", {
        "pid": os.getpid(), "host": socket.gethostname(), "started_at": now(),
        "build": str(build_run), "evaluation": str(eval_run),
        "development_episodes": 400, "no_retry": True})
    try:
        while not (build_run / "queue-ended.json").exists():
            time.sleep(60)
        terminal = json.loads((build_run / "queue-ended.json").read_text())
        if (terminal.get("status") != "complete" or terminal.get("accepted_jobs") != 1
                or terminal.get("stopped") is not False):
            save(build_run / "pipeline-ended.json", {
                "status": "build_incomplete_no_evaluation", "ended_at": now()})
            return
        build.base.assert_unchanged(json.loads((build_run / "identities.json").read_text())["files"])
        build.verify(plan["jobs"][0])
        development.prepare(eval_run)
        development.run(eval_run)
        eval_terminal = json.loads((eval_run / "queue-ended.json").read_text())
        save(build_run / "pipeline-ended.json", {
            "status": "evaluation_terminal", "ended_at": now(),
            "evaluation_status": eval_terminal.get("status"),
            "accepted_jobs": eval_terminal.get("accepted_jobs", 0)})
    except BaseException as exc:
        save(build_run / "pipeline-ended.json", {
            "status": "failed_no_retry", "ended_at": now(),
            "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
