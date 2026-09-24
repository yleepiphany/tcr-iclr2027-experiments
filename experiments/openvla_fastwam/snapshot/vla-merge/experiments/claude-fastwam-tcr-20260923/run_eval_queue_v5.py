#!/usr/bin/env python3
"""Dependent, no-retry Fast-WAM TCR development then formal evaluation queue."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNTIME = ROOT / "vla-merge-runtime"
BASE = RUNTIME / "experiments/claude-fastwam-tcr-20260923"
BANK = BASE / "evaluation-bank-v1.json"
PIPELINE = BASE / "pipeline-v5"
RUN = BASE / "evaluation-queue-v5"
PYTHON = RUNTIME / "envs/mergevla/bin/python"
GPUS = tuple(range(8))
ADMISSION_MIB = 40 * 1024
RUNTIME_FLOOR_MIB = 8 * 1024
MAX_WORKERS = 2

sys.path.insert(0, str(ROOT / "vla-merge/experiments/claude-firstpass-cause-20260920"))
import card_flock  # noqa: E402
from audit_eval_job import audit  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def gpu_rows() -> dict[int, dict]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used",
        "--format=csv,noheader,nounits"], text=True)
    result = {}
    for line in output.strip().splitlines():
        index, uuid, free, used = [part.strip() for part in line.split(",")]
        result[int(index)] = {"gpu": int(index), "uuid": uuid,
                              "free_mib": int(free), "used_mib": int(used)}
    return result


def source_hashes() -> dict[str, str]:
    return {name: sha(HERE / name) for name in
            ("freeze_eval_bank.py", "run_eval_job.py", "audit_eval_job.py",
             "run_eval_queue_v5.py")}


def prepare() -> dict:
    if RUN.exists():
        raise FileExistsError(RUN)
    bank = json.loads(BANK.read_text())
    jobs = bank.get("jobs", [])
    if (bank.get("schema") != "fastwam_tcr_eval_bank_v1" or len(jobs) != 80
            or sum(len(j["stock_indices"]) for j in jobs if j["stage"] == "development") != 40
            or sum(len(j["stock_indices"]) for j in jobs if j["stage"] == "formal") != 400):
        raise ValueError("Frozen evaluation bank has wrong coverage")
    plan = {"schema": "fastwam_tcr_eval_queue_v1", "host": socket.gethostname(),
            "bank": str(BANK), "bank_sha256": sha(BANK),
            "pipeline_plan_sha256": sha(PIPELINE / "plan.json"),
            "native_eval_sha256": sha(ROOT / "vla-merge_table4/Fast-WAM/source/experiments/libero/eval_libero_single.py"),
            "source_sha256": source_hashes(),
            "job_ids": [j["id"] for stage in ("development", "formal")
                        for j in jobs if j["stage"] == stage],
            "dev_gate": "all 40 dev jobs strictly accepted and at least 1/40 succeeds; no parameter/seed selection",
            "formal_selection": "stock reset indices 1..10, one frozen repeat; never use interim formal scores to schedule",
            "max_workers": MAX_WORKERS, "permitted_gpus": list(GPUS),
            "admission_min_free_mib": ADMISSION_MIB,
            "runtime_floor_mib": RUNTIME_FLOOR_MIB, "no_retry": True}
    save(RUN / "plan.json", plan)
    save(RUN / "state.json", {"status": "prepared", "accepted": [], "failed": [],
                              "active": [], "pending": plan["job_ids"],
                              "stop_latch": False})
    return {"plan": str(RUN / "plan.json"), "jobs": len(jobs),
            "formal_episodes": 400, "dependency": str(PIPELINE / "ended.json")}


def check_plan(plan: dict) -> None:
    if (plan.get("schema") != "fastwam_tcr_eval_queue_v1"
            or plan.get("host") != socket.gethostname()
            or plan.get("bank_sha256") != sha(BANK)
            or plan.get("pipeline_plan_sha256") != sha(PIPELINE / "plan.json")
            or plan.get("source_sha256") != source_hashes()
            or plan.get("native_eval_sha256") != sha(ROOT / "vla-merge_table4/Fast-WAM/source/experiments/libero/eval_libero_single.py")):
        raise ValueError("Frozen evaluation source or bank changed")


def run() -> dict:
    plan = json.loads((RUN / "plan.json").read_text())
    check_plan(plan)
    with (RUN / "coordinator.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((RUN / "state.json").read_text())
        if state.get("status") != "prepared" or state.get("active"):
            raise ValueError("Evaluation queue already started")
        jobs = {j["id"]: j for j in json.loads(BANK.read_text())["jobs"]}
        pending = list(state["pending"])
        accepted, failed, active = [], [], {}
        stop_latch = False
        stop_reason = None
        binding = None

        def stop_handler(signum, _frame):
            nonlocal stop_latch, stop_reason
            stop_latch = True
            stop_reason = f"supervisor signal {signum}"

        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)
        try:
            while pending or active:
                check_plan(plan)
                if binding is None and not stop_latch:
                    ended_path = PIPELINE / "ended.json"
                    if ended_path.exists():
                        ended = json.loads(ended_path.read_text())
                        final_path = BASE / "tcr-build-v5/final-manifest.json"
                        if ended.get("complete") is not True or not final_path.is_file():
                            stop_latch = True
                            stop_reason = "TCR build dependency ended incomplete"
                        else:
                            final = json.loads(final_path.read_text())
                            if (final.get("complete") is not True or final.get("is_tcr") is not True
                                    or final.get("native_reload_action_exact") is not True
                                    or final.get("checkpoint") != ended.get("checkpoint")):
                                stop_latch = True
                                stop_reason = "TCR build final manifest disagrees with pipeline"
                            else:
                                checkpoint = Path(final["checkpoint"]["path"])
                                if sha(checkpoint) != final["checkpoint"]["sha256"]:
                                    stop_latch = True
                                    stop_reason = "TCR checkpoint hash differs"
                                else:
                                    binding = {"checkpoint": str(checkpoint),
                                               "checkpoint_sha256": final["checkpoint"]["sha256"],
                                               "pipeline_ended_sha256": sha(ended_path),
                                               "final_manifest_sha256": sha(final_path)}
                                    save(RUN / "build-binding.json", binding)
                    else:
                        pipeline_state = json.loads((PIPELINE / "state.json").read_text())
                        if pipeline_state.get("phase") == "failed":
                            stop_latch = True
                            stop_reason = "TCR build dependency failed"

                rows = gpu_rows() if binding is not None else {}
                for gpu, item in list(active.items()):
                    process, job, output, lease, legacy, log = item
                    exitcode = process.poll()
                    if exitcode is None and rows[gpu]["free_mib"] < RUNTIME_FLOOR_MIB:
                        stop_latch = True
                        stop_reason = f"GPU {gpu} crossed {RUNTIME_FLOOR_MIB} MiB runtime floor"
                    if exitcode is None:
                        continue
                    log.close()
                    lease.release()
                    legacy.release()
                    del active[gpu]
                    if exitcode:
                        failed.append({"id": job["id"], "gpu": gpu, "exitcode": exitcode})
                        stop_latch = True
                        stop_reason = stop_reason or f"worker failed: {job['id']}"
                        continue
                    try:
                        result = audit(BANK, output, job["id"], binding["checkpoint_sha256"])
                        save(RUN / "audits" / f"{job['id']}.json", result)
                    except Exception as exc:
                        failed.append({"id": job["id"], "gpu": gpu,
                                       "reason": f"strict audit: {exc!r}"})
                        stop_latch = True
                        stop_reason = stop_reason or f"audit failed: {job['id']}"
                    else:
                        accepted.append(job["id"])
                if stop_latch:
                    for process, *_ in active.values():
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                elif binding is not None:
                    dev_complete = sum(j.startswith("development-") for j in accepted) == 40
                    if dev_complete and any(j.startswith("formal-") for j in pending):
                        successes = 0
                        for job_id in accepted:
                            if job_id.startswith("development-"):
                                path = RUN / "jobs" / job_id / "episodes.jsonl"
                                successes += sum(json.loads(line)["success"]
                                                 for line in path.read_text().splitlines() if line)
                        save(RUN / "development-gate.json",
                             {"accepted_jobs": 40, "episodes": 40, "successes": successes,
                              "pass": successes >= 1,
                              "rule": plan["dev_gate"]})
                        if successes == 0:
                            stop_latch = True
                            stop_reason = "predeclared catastrophic development gate: 0/40"
                    for gpu in sorted(GPUS, key=lambda i: rows[i]["free_mib"], reverse=True):
                        if stop_latch or len(active) >= MAX_WORKERS or not pending:
                            break
                        job = jobs[pending[0]]
                        if job["stage"] == "formal" and not dev_complete:
                            break
                        row = rows[gpu]
                        if gpu in active or row["free_mib"] < ADMISSION_MIB:
                            continue
                        lease = card_flock.take_card(gpu, row["uuid"], job["id"], "fastwam-tcr-eval")
                        if lease is None:
                            continue
                        legacy = card_flock._try_lock(
                            RUNTIME / "resource-leases" / socket.gethostname() / f"gpu-{gpu}.lock",
                            job["id"], {"gpu": gpu, "job": job["id"]})
                        if legacy is None:
                            lease.release()
                            continue
                        again = gpu_rows()[gpu]
                        if again["uuid"] != row["uuid"] or again["free_mib"] < ADMISSION_MIB:
                            legacy.release()
                            lease.release()
                            continue
                        pending.pop(0)
                        output = RUN / "jobs" / job["id"]
                        if output.exists():
                            raise FileExistsError(output)
                        log_path = RUN / "logs" / f"{job['id']}.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log = log_path.open("x")
                        command = [str(PYTHON), "-u", str(HERE / "run_eval_job.py"),
                                   "--bank", str(BANK), "--job-id", job["id"],
                                   "--checkpoint", binding["checkpoint"],
                                   "--gpu", str(gpu), "--output", str(output)]
                        env = os.environ.copy()
                        env.update({"FASTWAM_INHERITED_GPU_INDEX": str(gpu),
                                    "FASTWAM_INHERITED_GPU_UUID": row["uuid"],
                                    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                                    "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
                        process = subprocess.Popen(
                            command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            pass_fds=(lease.handle, legacy.handle))
                        active[gpu] = (process, job, output, lease, legacy, log)
                        save(RUN / "launches" / f"{job['id']}.json",
                             {"id": job["id"], "gpu": again, "pid": process.pid,
                              "command": command,
                              "time_utc": datetime.now(timezone.utc).isoformat()})
                status = ("stopping" if stop_latch else "waiting_build" if binding is None
                          else "running" if active else "waiting_safe_gpu" if pending else "complete")
                save(RUN / "state.json", {
                    "status": status, "supervisor_pid": os.getpid(),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "accepted": accepted, "failed": failed, "pending": pending,
                    "active": [{"gpu": gpu, "id": item[1]["id"], "pid": item[0].pid}
                               for gpu, item in active.items()],
                    "stop_latch": stop_latch, "stop_reason": stop_reason})
                if stop_latch and not active:
                    break
                if pending or active:
                    time.sleep(30 if binding is not None else 60)
        finally:
            for process, job, output, lease, legacy, log in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                log.close()
                lease.release()
                legacy.release()
        result = {"complete": not pending and not failed and not stop_latch,
                  "accepted": len(accepted), "failed": failed,
                  "pending": len(pending), "stop_reason": stop_reason}
        save(RUN / "ended.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    print(json.dumps(prepare() if args.stage == "prepare" else run(), sort_keys=True))
