#!/usr/bin/env python3
"""Independent complete-bank gate before Fast-WAM TCR pass A or B."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_capture import audit

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "vla-merge-runtime/experiments/claude-fastwam-tcr-20260923"
BANK = BASE / "calibration-bank-v1.json"
QUEUE = BASE / "capture-queue-v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def audit_all() -> dict:
    plan = json.loads(BANK.read_text())
    ended = json.loads((QUEUE / "ended.json").read_text())
    state = json.loads((QUEUE / "state.json").read_text())
    expected = {job["id"] for job in plan["jobs"]}
    if (plan.get("schema") != "fastwam_four_expert_ab_calibration_bank_v1" or
            len(expected) != 80 or ended.get("complete") is not True or
            ended.get("accepted") != 80 or ended.get("failed") or
            state.get("stop_latch") is not False or
            set(state.get("accepted", [])) != expected or state.get("pending") or
            state.get("active")):
        raise ValueError("Fast-WAM A/B calibration queue is not complete and accepted")
    rows = []
    selected_total = 0
    for job in plan["jobs"]:
        episode = BASE / "calibration-v1" / job["id"]
        result = audit(BANK, episode, job["id"])
        prior = json.loads((QUEUE / "audits" / f"{job['id']}.json").read_text())
        if result != prior:
            raise ValueError(f"Independent re-audit differs: {job['id']}")
        selected = result["selected_request_indices"]
        selected_total += len(selected)
        rows.append({"id": job["id"], "suite": job["suite"], "round": job["round"],
                     "expert": job["expert"], "task_id": job["task_id"],
                     "request_count": result["requests"],
                     "selected_request_indices": selected,
                     "episode_receipt_sha256": result["episode_receipt_sha256"],
                     "request_file_sha256": result["request_files"]})
    if {row["round"] for row in rows} != {"A", "B"} or \
            any(sum(row["round"] == round_name and row["expert"] == expert for row in rows) != 10
                for round_name in ("A", "B") for expert in ("spatial", "object", "goal", "long")):
        raise ValueError("A/B expert task coverage differs")
    return {"schema": "fastwam_ab_calibration_acceptance_v1", "accepted": True,
            "jobs": 80, "episodes": 80, "selected_requests": selected_total,
            "per_expert_per_round_episodes": 10, "success_filtering": False,
            "bank_sha256": digest(BANK), "queue_plan_sha256": digest(QUEUE / "plan.json"),
            "jobs_detail": rows, "evaluation_episodes": 0,
            "tcr_checkpoint_built": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"accepted": result["accepted"], "jobs": result["jobs"],
                      "selected_requests": result["selected_requests"]}))
