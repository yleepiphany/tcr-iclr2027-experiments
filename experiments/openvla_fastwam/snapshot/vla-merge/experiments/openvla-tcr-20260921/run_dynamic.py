#!/usr/bin/env python3
"""User-authorized replacement of the unlaunched GPU3-only OFT queue.

GPU2 is admitted whenever enough memory is free; GPU3 waits for the existing
cap24 batch. Reuse immutable expert identity audit, not the old process. Smoke
and evaluation are separate phases: all four smoke tests precede any rollout.
"""
import argparse
import json
from pathlib import Path
import socket
import os
import time
import run_local as base

formal = base.formal
HERE = Path(__file__).resolve().parent
UUIDS = {2: "GPU-05093da4-f46b-ea6f-d964-d0a393deb6d8", 3: base.UUID}


def environment(gpu):
    if gpu not in (0, 2, 3):
        raise ValueError("Only local GPU2/3 are authorized")
    env = base.environment(3)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


class AvailableCards:
    def __iter__(self):
        terminal = base.DEPENDENCY / "queue-ended.json"
        if (base.DEPENDENCY / "batch-stop-report.json").exists():
            raise RuntimeError("Prior local batch stopped; stop new queue for review")
        value = json.loads(terminal.read_text()) if terminal.exists() else None
        if value is not None and (value.get("status") != "complete" or value.get("completed_jobs") != 4):
            raise RuntimeError("Prior local batch ended unsuccessfully")
        yield 2
        if value is not None:
            yield 3


def prepare(run, previous):
    if (previous / "started.json").exists():
        raise RuntimeError("Old queue launched workers; cannot reassign its jobs")
    audit = json.loads((previous / "identities.json").read_text())
    base.assert_unchanged(audit["files"])
    audit["reused_identity_audit"] = {"path": str(previous / "identities.json"),
                                      "sha256": formal.sha(previous / "identities.json")}
    audit["files"][str(Path(__file__).resolve())] = base.file_identity(Path(__file__).resolve())
    run.mkdir(parents=True, exist_ok=False)
    formal.save(run / "identities.json", audit)
    old = json.loads((previous / "plan.json").read_text())
    phases = []
    for name, smoke in (("smoke", True), ("eval", False)):
        phase = run / name
        jobs = []
        for job in old["jobs"]:
            if job["smoke"] != smoke:
                continue
            model = Path(audit["experts"][job["expert"]]["local_path"])
            out = phase / "jobs" / job["id"]
            jobs.append({**job, "output": str(out), "identity": str(run / "identities.json"),
                         "command": base.command(model, job["suite"], out, smoke)})
        plan = {**old, "schema": "oft_dynamic_23_phase_v1", "run": str(phase), "jobs": jobs,
                "gpus": [2, 3], "gpu_uuids": UUIDS, "max_workers": 2,
                "episodes": 0 if smoke else 400, "identity_audit": str(run / "identities.json"),
                "identity_audit_sha256": formal.sha(run / "identities.json"),
                "environment_contract": {**old["environment_contract"], "CUDA_VISIBLE_DEVICES": "selected GPU 2 or 3"},
                "dynamic_runner_sha256": formal.sha(Path(__file__).resolve()),
                "authorization": "User: local GPU2/3 dynamically based on free memory"}
        formal.save(phase / "plan.json", plan)
        phases.append(str(phase))
    formal.save(run / "CLAIM.json", {"owner": "Codex", "host": base.HOST, "gpus": [2, 3],
                 "uuid_map": UUIDS, "phases": phases, "expert_episodes": 400,
                 "supersedes_unlaunched_queue": str(previous),
                 "no_tcr_result": True, "no_retry": True})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    args = p.parse_args()
    run, previous = args.run.resolve(), args.previous.resolve()
    if socket.gethostname() != base.HOST or any(formal.gpu_row(g)["uuid"] != u for g, u in UUIDS.items()):
        raise ValueError("Wrong host/card identity")
    if args.prepare:
        prepare(run, previous)
        return
    if not (previous / "dependency-stop.json").exists() or (previous / "started.json").exists():
        raise RuntimeError("Previous waiting queue must be stopped before replacement starts")
    if (run / "started.json").exists():
        raise FileExistsError("No automatic restart")
    base.assert_unchanged(json.loads((run / "identities.json").read_text())["files"])
    formal.save(run / "started.json", {"pid": os.getpid(), "kernel_starttime": formal.start_time(os.getpid()),
                "host": socket.gethostname(), "started_unix": time.time()})
    formal.GPUS = AvailableCards()
    formal.FLOOR_MIB = 12288
    formal.EVALUATOR = HERE / "evaluate.py"
    formal.IDENTITIES = run / "identities.json"
    formal.scientific_environment = environment
    formal.verify_job = base.verify
    formal.save = base.save_with_actual_counts
    total_episodes = 0
    for phase in ("smoke", "eval"):
        plan = json.loads((run / phase / "plan.json").read_text())
        formal.run_queue(plan)
        terminal = json.loads((run / phase / "queue-ended.json").read_text())
        total_episodes += terminal["episodes"]
        if terminal["status"] != "complete":
            base.BASE_SAVE(run / "queue-ended.json", {"status": "incomplete", "phase": phase,
                         "episodes": total_episodes, "tcr_evaluated": False,
                         "reason": terminal.get("reason")})
            return
    # Use original save: aggregate receipt has no per-job 'finished' list.
    base.BASE_SAVE(run / "queue-ended.json", {"status": "complete", "episodes": total_episodes,
                   "expert_episodes": total_episodes, "tcr_evaluated": False})


if __name__ == "__main__":
    main()
