#!/usr/bin/env python3
"""Isolated complement after the v2 native request-cap failure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

import audit_capture
import capture_contract as contract
import run_capture_queue as queue


HERE = Path(__file__).resolve().parent
ROOT = contract.CAPTURE_ROOT
SOURCE = ROOT / "formal-capture-attempt-02"
PROTOCOL = ROOT / "reset-preflight-attempt-03/candidate-protocol.json"
RUNNER = HERE / "run_capture_v3.py"
RUN = ROOT / "formal-capture-attempt-03"
OUTPUT_ROOT = ROOT / "capture-v3/jobs"
REQUEST_CAP = 256


def source_acceptance() -> tuple[dict, dict, list[dict]]:
    source_plan = contract.read(SOURCE / "plan.json")
    ended = contract.read(SOURCE / "queue-ended.json")
    if (ended.get("status"), ended.get("stopped"), ended.get("completed_jobs")) != ("incomplete", True, 5):
        raise ValueError("Source attempt terminal receipt differs")
    if (source_plan.get("protocol"), source_plan.get("protocol_sha256")) != (
        str(PROTOCOL), contract.sha256_file(PROTOCOL)
    ):
        raise ValueError("Source attempt protocol differs")
    if contract.sha256_file(Path(source_plan["runner"])) != source_plan["runner_sha256"]:
        raise ValueError("Source runner changed")
    protocol = contract.read(PROTOCOL)
    audit_capture.audit_amended_bank(protocol, PROTOCOL)
    success = {row["id"] for row in ended["finished"] if row.get("outcome") == "success" and row.get("returncode") == 0}
    if success != {"r01-A-coordination", "r01-A-receptacle", "r01-A-precision", "r01-B-coordination"}:
        raise ValueError("Source accepted-job set differs")
    accepted = [audit_capture.audit_artifact(row) for row in protocol["jobs"] if row["id"] in success]
    if len(accepted) != 4:
        raise ValueError("Source strict audit is incomplete")
    return source_plan, protocol, accepted


def request_cap_gate(protocol: dict) -> dict:
    import yaml

    parent = contract.read(Path(protocol["jobs"][0]["parent"]["path"]))
    horizon_path = Path(parent["robotwin"]["root"]) / "task_config/_eval_step_limit.yml"
    horizons = yaml.safe_load(horizon_path.read_text())
    requested = [int(horizons[task["task"]]) for job in protocol["jobs"] for task in job["tasks"]]
    if len(requested) != 180 or max(requested) != 1700:
        raise ValueError("RoboTwin horizon contract differs")
    # Frozen PI05 n_action_steps=10; the native loop consumes one ten-action
    # queue before its next model request.  Keep room above the longest horizon.
    maximum_requests = max((steps + 9) // 10 for steps in requested)
    if maximum_requests >= REQUEST_CAP:
        raise ValueError("Capture cap does not cover full native trajectory")
    source = RUNNER.read_text()
    if source.count('"PI05_BLOCK_REGMEANPP_REQUESTS_PER_EPISODE": "256",') != 1:
        raise ValueError("Recovery runner request cap differs")
    if source.replace('"PI05_BLOCK_REGMEANPP_REQUESTS_PER_EPISODE": "256",',
                      '"PI05_BLOCK_REGMEANPP_REQUESTS_PER_EPISODE": "128",') != (HERE / "run_capture.py").read_text():
        raise ValueError("Recovery runner changes more than the capture capacity")
    return {"horizon": contract.bind(horizon_path), "max_steps": max(requested),
            "max_native_requests": maximum_requests, "capture_request_cap": REQUEST_CAP}


def prepare() -> dict:
    if RUN.exists() or OUTPUT_ROOT.exists():
        raise FileExistsError("Recovery output already exists")
    source_plan, protocol, accepted = source_acceptance()
    cap_gate = request_cap_gate(protocol)
    accepted_ids = {row["id"] for row in accepted}
    pending = [row for row in protocol["jobs"] if row["id"] not in accepted_ids]
    if len(pending) != 14 or len(accepted_ids) + len(pending) != 18:
        raise ValueError("Recovery complement differs")
    jobs = []
    for row in pending:
        output = OUTPUT_ROOT / row["id"]
        if output.exists():
            raise FileExistsError(output)
        jobs.append({"id": row["id"], "mode": "formal", "output": str(output),
                     "command": [str(queue.PYTHON), "-u", str(RUNNER), "run",
                                 "--protocol", str(PROTOCOL), "--job-id", row["id"],
                                 "--output", str(output)],
                     "expected_episodes": 10, "expected_requests": 50, "expected_flow_rows": 150})
    plan = {"schema": "robotwin_tcr_capture_recovery_v3", "created_at": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(), "run": str(RUN), "mode": "formal",
            "protocol": str(PROTOCOL), "protocol_sha256": contract.sha256_file(PROTOCOL),
            "runner": str(RUNNER), "runner_sha256": contract.sha256_file(RUNNER),
            "source_attempt": contract.bind(SOURCE / "queue-ended.json"),
            "source_plan": contract.bind(SOURCE / "plan.json"),
            "source_accepted": accepted, "request_cap_gate": cap_gate,
            "jobs": jobs, "gpus": list(queue.GPUS), "max_workers": 2,
            "min_free_mib": queue.MIN_FREE_MIB, "reserve_mib": queue.RESERVE_MIB,
            "no_retry": True, "no_score_based_scheduling_or_stopping": True,
            "openvla_formal_remains_cancelled": True}
    RUN.mkdir(parents=True, exist_ok=False)
    queue.save(RUN / "plan.json", plan)
    return plan


def run() -> None:
    plan = contract.read(RUN / "plan.json")
    if plan.get("schema") != "robotwin_tcr_capture_recovery_v3" or len(plan.get("jobs", [])) != 14:
        raise ValueError("Recovery plan differs")
    if plan.get("source_attempt") != contract.bind(SOURCE / "queue-ended.json"):
        raise ValueError("Source terminal receipt changed")
    if plan.get("source_plan") != contract.bind(SOURCE / "plan.json"):
        raise ValueError("Source plan changed")
    _, protocol, accepted = source_acceptance()
    if accepted != plan["source_accepted"] or request_cap_gate(protocol) != plan["request_cap_gate"]:
        raise ValueError("Source acceptance or capture cap changed")
    if {row["id"] for row in plan["jobs"]} != {row["id"] for row in protocol["jobs"]} - {row["id"] for row in accepted}:
        raise ValueError("Recovery jobs are not the exact complement")
    queue.RUNNER = RUNNER
    queue.run_queue(plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.action == "prepare":
        plan = prepare()
        print(json.dumps({"run": plan["run"], "jobs": len(plan["jobs"]),
                          "accepted_source": len(plan["source_accepted"])}, sort_keys=True))
    else:
        run()


if __name__ == "__main__":
    main()
