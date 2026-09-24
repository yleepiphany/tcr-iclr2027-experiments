#!/usr/bin/env python3
"""Resume frozen PI0.5 LIBERO-PRO Soup/TIES jobs without repeating receipts.

Each resumed job keeps its original selection, seed, model and job ID, but writes
to a new attempt directory. The original formal attempt remains immutable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
RUNTIME = WORK / "vla-merge-runtime"
HOST = "dsw-967394-56ffd4897d-42wft"
ROOT = RUNTIME / "experiments/pi05-pro-resume-codex-20260924"
GPU_UUIDS = {
    5: "GPU-059c0ed7-bcd0-3a71-69c1-2282316cddc8",
    7: "GPU-e027de95-4c80-d8ec-d650-30da9b6eb135",
}
MIN_FREE_MIB = 40 * 1024
FLOOR_MIB = 12 * 1024

sys.path.insert(0, str(HERE))
import run_soup_formal as soup  # noqa: E402
import run_baseline_formal as baseline  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def source(method: str):
    if method == "soup":
        return soup, soup.RUN, soup.SELECTION, soup.POLICY, soup.POLICY_SHA
    baseline.select_method("ties_merging")
    return baseline, baseline.RUN, baseline.SELECTION, baseline.POLICY, baseline.POLICY_SHA


def gpu_row(gpu: int) -> dict:
    line = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=uuid,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    uuid, free, used = (item.strip() for item in line)
    return {"uuid": uuid, "free_mib": int(free), "used_mib": int(used)}


def live_matching_pid(pid: int, output: Path) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return str(output) in command


def original_verified(module, original_run: Path, original_plan: dict) -> dict[str, dict]:
    state = read(original_run / "state.json")
    accepted = state.get("accepted", {})
    jobs = {job["id"]: job for job in original_plan["jobs"]}
    if not isinstance(accepted, dict) or not set(accepted) <= set(jobs):
        raise ValueError("Original accepted-job index is invalid")
    verified = {}
    for key, old_artifact in accepted.items():
        job = jobs[key]
        artifact = module.verify_job(job, original_plan)
        exit_row = read(original_run / "exits" / f"{key}.json")
        if artifact != old_artifact or exit_row.get("accepted") is not True:
            raise ValueError(f"Original accepted receipt differs: {key}")
        verified[key] = artifact
    # A terminated supervisor can miss the final state write after a child exits.
    for key, job in jobs.items():
        if key in verified:
            continue
        output = Path(job["output"])
        if (output / "extension_eval_receipt.json").exists() and (output / "eval_info.json").exists():
            artifact = module.verify_job(job, original_plan)
            exit_path = original_run / "exits" / f"{key}.json"
            if exit_path.exists() and read(exit_path).get("accepted") is True:
                verified[key] = artifact
    return verified


def validate_original(method: str, gpu: int):
    module, old_run, selection, policy, policy_sha = source(method)
    if socket.gethostname() != HOST:
        raise ValueError("Wrong host for frozen PRO source")
    if gpu_row(gpu)["uuid"] != GPU_UUIDS[gpu]:
        raise ValueError("Physical GPU UUID differs")
    original_plan_path = old_run / "plan.json"
    original_plan = read(original_plan_path)
    claim = read(old_run / "CLAIM.json")
    plan_sha = module.sha256_file(original_plan_path)
    if claim.get("plan_sha256") != plan_sha:
        raise ValueError("Original frozen plan SHA differs")
    if (original_plan.get("jobs_expected") != 480
            or original_plan.get("episodes_expected") != 4800
            or len(original_plan.get("jobs", [])) != 480
            or len({job["id"] for job in original_plan["jobs"]}) != 480
            or original_plan.get("model_sha256") != policy_sha
            or original_plan.get("source_manifest_sha256") != module.sha256_file(selection / "manifest.json")
            or original_plan.get("evaluator_sha256") != module.sha256_file(module.EVALUATOR)
            or original_plan.get("native_runner_sha256") != module.sha256_file(module.NATIVE_RUNNER)
            or original_plan.get("runner_sha256") != module.sha256_file(Path(module.__file__))):
        raise ValueError("Original formal plan identity differs")
    if module.sha256_file(policy / "model.safetensors") != policy_sha:
        raise ValueError("Original 10k model SHA differs")
    verified = original_verified(module, old_run, original_plan)
    return module, old_run, original_plan, plan_sha, verified


def attempt_path(method: str) -> Path:
    return ROOT / method / "attempt-01"


def prepare(method: str, gpu: int) -> dict:
    module, old_run, original_plan, plan_sha, verified = validate_original(method, gpu)
    run = attempt_path(method)
    if run.exists():
        raise FileExistsError(run)
    pending = []
    for job in original_plan["jobs"]:
        if job["id"] in verified:
            continue
        launch_path = old_run / "launches" / f"{job['id']}.json"
        if launch_path.exists():
            old_pid = read(launch_path).get("pid")
            if isinstance(old_pid, int) and live_matching_pid(old_pid, Path(job["output"])):
                raise RuntimeError(f"Original job still runs: {job['id']} PID {old_pid}")
        pending.append(job["id"])
    plan = {
        "schema": "pi05_pro_receipt_dedup_resume_v1", "created_at": now(),
        "method": method, "host": HOST, "gpu": gpu, "gpu_uuid": GPU_UUIDS[gpu],
        "original_run": str(old_run), "original_plan_sha256": plan_sha,
        "selection_sha256": original_plan["source_manifest_sha256"],
        "model_sha256": original_plan["model_sha256"],
        "evaluator_sha256": original_plan["evaluator_sha256"],
        "accepted_original": verified, "pending_job_ids": pending,
        "episodes_per_job": 10, "expected_total_jobs": 480,
        "resource_mode": "global_gpu_lease_plus_40gib_free_12gib_floor",
    }
    write(run / "plan.json", plan)
    write(run / "state.json", {"status": "prepared", "accepted_original": len(verified),
                                "accepted_resumed": {}, "failed": {},
                                "pending": pending, "active": None, "updated_at": now()})
    return {"run": str(run), "gpu": gpu, "accepted_original": len(verified),
            "pending": len(pending), "plan_sha256": module.sha256_file(run / "plan.json")}


def new_job(job: dict, run: Path) -> dict:
    output = run / "results" / job["id"]
    command = [arg for arg in job["command"] if not arg.startswith("--output_dir=")]
    if len(command) != len(job["command"]) - 1:
        raise ValueError(f"Output argument not unique: {job['id']}")
    command.append(f"--output_dir={output}")
    return {**job, "output": str(output), "command": command}


def run(method: str, gpu: int) -> None:
    module, old_run, original_plan, original_sha, originals = validate_original(method, gpu)
    attempt = attempt_path(method)
    plan_path = attempt / "plan.json"
    plan = read(plan_path)
    if (plan.get("method") != method or plan.get("gpu") != gpu
            or plan.get("gpu_uuid") != GPU_UUIDS[gpu]
            or plan.get("original_plan_sha256") != original_sha
            or plan.get("model_sha256") != original_plan["model_sha256"]
            or plan.get("selection_sha256") != original_plan["source_manifest_sha256"]
            or plan.get("accepted_original") != originals
            or len(plan.get("pending_job_ids", [])) + len(originals) != 480):
        raise ValueError("Resume plan/source identity differs")
    job_by_id = {job["id"]: job for job in original_plan["jobs"]}
    own_lock = (attempt / "run.lock").open("a+")
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read(attempt / "state.json")
    accepted = state.get("accepted_resumed", {})
    for key, artifact in accepted.items():
        if module.verify_job(new_job(job_by_id[key], attempt), original_plan) != artifact:
            raise ValueError(f"Resumed accepted receipt differs: {key}")
    if state.get("failed"):
        raise RuntimeError("Prior resumed failure needs inspection")
    stop = False

    def stop_handler(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    gpu_lock_path = RUNTIME / "resource-leases" / HOST / f"gpu-{gpu}.lock"
    gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
    for key in plan["pending_job_ids"]:
        if stop:
            break
        if key in accepted:
            continue
        job = new_job(job_by_id[key], attempt)
        output = Path(job["output"])
        # Recheck original receipts in case another process finished a pending job.
        old_output = Path(job_by_id[key]["output"])
        if (old_output / "extension_eval_receipt.json").exists() and (old_output / "eval_info.json").exists():
            artifact = module.verify_job(job_by_id[key], original_plan)
            originals[key] = artifact
            continue
        original_launch = old_run / "launches" / f"{key}.json"
        if original_launch.exists():
            old_pid = read(original_launch).get("pid")
            while isinstance(old_pid, int) and live_matching_pid(old_pid, old_output):
                write(attempt / "state.json", {"status": "waiting_original_worker", "job": key,
                        "accepted_original": len(originals), "accepted_resumed": accepted,
                        "pending": len(plan["pending_job_ids"]) - len(accepted),
                        "failed": {}, "active": None, "updated_at": now()})
                time.sleep(30)
            if (old_output / "extension_eval_receipt.json").exists() and (old_output / "eval_info.json").exists():
                originals[key] = module.verify_job(job_by_id[key], original_plan)
                continue
        if output.exists():
            if (output / "extension_eval_receipt.json").exists() and (output / "eval_info.json").exists():
                accepted[key] = module.verify_job(job, original_plan)
                continue
            raise RuntimeError(f"Partial resumed output requires inspection: {key}")
        # Wait without owning a lease until both the global lock and memory admit.
        while not stop:
            with gpu_lock_path.open("a+") as gpu_lock:
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    admitted = None
                else:
                    admitted = gpu_row(gpu)
                    if admitted["uuid"] != GPU_UUIDS[gpu]:
                        raise ValueError("GPU UUID changed")
                    if admitted["free_mib"] >= MIN_FREE_MIB:
                        break
            write(attempt / "state.json", {"status": "waiting_gpu", "job": key,
                    "accepted_original": len(originals), "accepted_resumed": accepted,
                    "pending": len(plan["pending_job_ids"]) - len(accepted),
                    "failed": {}, "active": None, "updated_at": now()})
            time.sleep(30)
        if stop:
            break
        # The break above closes the context and releases the lease. Reacquire
        # immediately for the entire child lifetime; a competing claimant wins
        # the race by locking first, in which case we wait again.
        while not stop:
            gpu_lock = gpu_lock_path.open("a+")
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                gpu_lock.close()
                time.sleep(10)
                continue
            admitted = gpu_row(gpu)
            if admitted["free_mib"] >= MIN_FREE_MIB and admitted["uuid"] == GPU_UUIDS[gpu]:
                break
            gpu_lock.close()
            time.sleep(30)
        if stop:
            break
        env = module.environment()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
        log_path = attempt / "logs" / f"{key}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x") as log:
            child = subprocess.Popen(job["command"], cwd=WORK, env=env,
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            write(attempt / "launches" / f"{key}.json", {
                "id": key, "pid": child.pid, "gpu": gpu, "gpu_uuid": GPU_UUIDS[gpu],
                "admission": admitted, "started_at": now(), "command": job["command"],
                "original_plan_sha256": original_sha})
            write(attempt / "state.json", {"status": "running", "job": key,
                    "accepted_original": len(originals), "accepted_resumed": accepted,
                    "pending": len(plan["pending_job_ids"]) - len(accepted) - 1,
                    "failed": {}, "active": {"pid": child.pid, "gpu": gpu}, "updated_at": now()})
            floor = False
            while child.poll() is None:
                time.sleep(15)
                current = gpu_row(gpu)
                if current["uuid"] != GPU_UUIDS[gpu] or current["free_mib"] < FLOOR_MIB:
                    floor = True
                    os.killpg(child.pid, signal.SIGTERM)
                    break
            code = child.wait()
        gpu_lock.close()
        error = None
        artifact = None
        if code == 0 and not floor:
            try:
                artifact = module.verify_job(job, original_plan)
            except Exception as exc:
                error = f"receipt audit: {type(exc).__name__}: {exc}"
        else:
            error = f"child exit {code}; resource_floor={floor}"
        write(attempt / "exits" / f"{key}.json", {"id": key, "returncode": code,
                "accepted": artifact is not None, "artifact": artifact,
                "error": error, "finished_at": now()})
        if error:
            write(attempt / "state.json", {"status": "failed", "job": key,
                    "accepted_original": len(originals), "accepted_resumed": accepted,
                    "pending": len(plan["pending_job_ids"]) - len(accepted) - 1,
                    "failed": {key: error}, "active": None, "updated_at": now()})
            raise RuntimeError(error)
        accepted[key] = artifact
        write(attempt / "state.json", {"status": "running", "job": None,
                "accepted_original": len(originals), "accepted_resumed": accepted,
                "pending": len(plan["pending_job_ids"]) - len(accepted),
                "failed": {}, "active": None, "updated_at": now()})
    complete = len(originals) + len(accepted) == 480
    write(attempt / "state.json", {"status": "complete" if complete else "stopped",
            "accepted_original": len(originals), "accepted_resumed": accepted,
            "pending": 480 - len(originals) - len(accepted), "failed": {},
            "active": None, "updated_at": now()})
    if complete:
        write(attempt / "complete.json", {"schema": "pi05_pro_combined_complete_v1",
                "method": method, "original_plan_sha256": original_sha,
                "accepted_original": len(originals), "accepted_resumed": len(accepted),
                "jobs": 480, "episodes": 4800, "finished_at": now()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--method", choices=("soup", "ties"), required=True)
    parser.add_argument("--gpu", type=int, choices=sorted(GPU_UUIDS), required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.method, args.gpu), indent=2))
    else:
        run(args.method, args.gpu)


if __name__ == "__main__":
    main()
