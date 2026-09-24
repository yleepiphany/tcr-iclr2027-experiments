#!/usr/bin/env python3
"""Dynamic, no-retry Fast-WAM A/B expert trajectory queue.

The plan is frozen before launch. It leases one GPU per worker, admits only
cards with at least 40 GiB free, keeps an 8 GiB runtime floor, and audits each
completed episode before counting it. It never filters on episode success.
"""
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
BANK = BASE / "calibration-bank-v1.json"
RUN = BASE / "capture-queue-v1"
PYTHON = RUNTIME / "envs/mergevla/bin/python"
MAX_WORKERS = 3
MIN_FREE_MIB = 40 * 1024
RUNTIME_FLOOR_MIB = 8 * 1024
POLL_SECONDS = 30
GPUS = tuple(range(8))

sys.path.insert(0, str(ROOT / "vla-merge/experiments/claude-firstpass-cause-20260920"))
import card_flock  # noqa: E402
from audit_capture import audit  # noqa: E402


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_rows() -> dict[int, dict]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used",
        "--format=csv,noheader,nounits"], text=True)
    rows = {}
    for line in output.strip().splitlines():
        index, uuid, free, used = [part.strip() for part in line.split(",")]
        rows[int(index)] = {"gpu": int(index), "uuid": uuid,
                            "free_mib": int(free), "used_mib": int(used)}
    return rows


def prepare() -> dict:
    if RUN.exists():
        raise FileExistsError(RUN)
    bank = json.loads(BANK.read_text())
    if bank.get("schema") != "fastwam_four_expert_ab_calibration_bank_v1" or \
            len(bank.get("jobs", [])) != 80:
        raise ValueError("Fast-WAM A/B bank is not the frozen 80-job matrix")
    first = bank["jobs"][0]
    if first["id"] != "A-spatial-task00":
        raise ValueError("First accepted pilot does not match frozen bank")
    pilot = BASE / "calibration-v1/A-spatial-task00"
    accepted = audit(BANK, pilot, first["id"])
    RUN.mkdir(parents=True)
    plan = {"schema": "fastwam_calibration_queue_v1", "host": socket.gethostname(),
            "bank": str(BANK), "bank_sha256": sha(BANK),
            "source_native_gate": str(BASE / "native-parity-spatial.json"),
            "preaccepted_job": first["id"],
            "preaccepted_receipt_sha256": accepted["episode_receipt_sha256"],
            "job_ids": [job["id"] for job in bank["jobs"]],
            "permitted_gpus": list(GPUS), "max_workers": MAX_WORKERS,
            "admission_min_free_mib": MIN_FREE_MIB,
            "runtime_floor_mib": RUNTIME_FLOOR_MIB,
            "no_retry": True, "success_blind": True}
    save(RUN / "plan.json", plan)
    save(RUN / "state.json", {"status": "prepared", "accepted": [first["id"]],
                              "failed": [], "active": [],
                              "pending": [job["id"] for job in bank["jobs"][1:]],
                              "stop_latch": False})
    save(RUN / "audits" / f"{first['id']}.json", accepted)
    return {"plan": str(RUN / "plan.json"), "accepted": 1, "pending": 79}


def run() -> dict:
    plan = json.loads((RUN / "plan.json").read_text())
    if plan.get("host") != socket.gethostname() or plan.get("bank_sha256") != sha(BANK):
        raise ValueError("Queue host or frozen bank changed")
    with (RUN / "coordinator.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((RUN / "state.json").read_text())
        if state.get("status") != "prepared" or state.get("active"):
            raise ValueError("Queue has already started")
        bank = json.loads(BANK.read_text())
        jobs = {row["id"]: row for row in bank["jobs"]}
        pending = list(state["pending"])
        accepted = list(state["accepted"])
        failed = []
        active = {}
        stop_latch = False
        stop_reason = None

        def stop_handler(signum, _frame):
            nonlocal stop_latch, stop_reason
            stop_latch = True
            stop_reason = f"supervisor signal {signum}"

        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)
        try:
            while pending or active:
                rows = gpu_rows()
                for gpu, item in list(active.items()):
                    process, job, output, lease, legacy, log = item
                    exitcode = process.poll()
                    if exitcode is None and rows[gpu]["free_mib"] < RUNTIME_FLOOR_MIB:
                        stop_latch = True
                        stop_reason = f"GPU {gpu} below {RUNTIME_FLOOR_MIB} MiB runtime floor"
                    if exitcode is None:
                        continue
                    log.close()
                    lease.release()
                    legacy.release()
                    del active[gpu]
                    if exitcode:
                        failed.append({"id": job["id"], "gpu": gpu,
                                       "exitcode": exitcode, "reason": "worker nonzero exit"})
                        stop_latch = True
                        stop_reason = stop_reason or f"worker failed: {job['id']}"
                        continue
                    try:
                        result = audit(BANK, output, job["id"])
                        save(RUN / "audits" / f"{job['id']}.json", result)
                    except Exception as exc:
                        failed.append({"id": job["id"], "gpu": gpu,
                                       "exitcode": 0, "reason": f"strict audit: {exc!r}"})
                        stop_latch = True
                        stop_reason = stop_reason or f"audit failed: {job['id']}"
                    else:
                        accepted.append(job["id"])
                if stop_latch:
                    for process, job, output, lease, legacy, log in active.values():
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                else:
                    for gpu in sorted(GPUS, key=lambda index: rows[index]["free_mib"], reverse=True):
                        if len(active) >= MAX_WORKERS or not pending:
                            break
                        row = rows[gpu]
                        if gpu in active or row["free_mib"] < MIN_FREE_MIB:
                            continue
                        lease = card_flock.take_card(gpu, row["uuid"], pending[0],
                                                    "fastwam-calibration")
                        if lease is None:
                            continue
                        legacy = card_flock._try_lock(
                            RUNTIME / "resource-leases" / socket.gethostname() / f"gpu-{gpu}.lock",
                            pending[0], {"job": pending[0], "gpu": gpu})
                        if legacy is None:
                            lease.release()
                            continue
                        again = gpu_rows()[gpu]
                        if again["uuid"] != row["uuid"] or again["free_mib"] < MIN_FREE_MIB:
                            lease.release()
                            legacy.release()
                            continue
                        job = jobs[pending.pop(0)]
                        output = BASE / "calibration-v1" / job["id"]
                        if output.exists():
                            raise FileExistsError(output)
                        log_path = RUN / "logs" / f"{job['id']}.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log = log_path.open("x")
                        command = [str(PYTHON), "-u", str(HERE / "run_capture_episode.py"),
                                   "--gpu", str(gpu), "--expert", job["expert"],
                                   "--task", str(job["task_id"]),
                                   "--demo-index", str(job["demo_index"]),
                                   "--round", job["round"], "--seed", str(job["seed"]),
                                   "--min-free-mib", str(MIN_FREE_MIB),
                                   "--plan", str(BANK), "--output", str(output),
                                   "--inherited-lease"]
                        env = os.environ.copy()
                        env.update({"FASTWAM_INHERITED_GPU_INDEX": str(gpu),
                                    "FASTWAM_INHERITED_GPU_UUID": row["uuid"],
                                    "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2",
                                    "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"})
                        process = subprocess.Popen(
                            command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            pass_fds=(lease.handle, legacy.handle))
                        active[gpu] = (process, job, output, lease, legacy, log)
                        save(RUN / "launches" / f"{job['id']}.json",
                             {"id": job["id"], "gpu": again, "pid": process.pid,
                              "command": command, "time_utc": datetime.now(timezone.utc).isoformat()})
                status = "stopping" if stop_latch else ("running" if active else
                         "waiting_for_free_gpu" if pending else "complete")
                save(RUN / "state.json", {
                    "status": status, "supervisor_pid": os.getpid(),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "accepted": accepted, "failed": failed, "pending": pending,
                    "active": [{"gpu": gpu, "id": item[1]["id"], "pid": item[0].pid}
                               for gpu, item in active.items()],
                    "stop_latch": stop_latch, "stop_reason": stop_reason,
                    "gpu_rows": rows})
                if stop_latch and not active:
                    break
                if pending or active:
                    time.sleep(POLL_SECONDS)
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
