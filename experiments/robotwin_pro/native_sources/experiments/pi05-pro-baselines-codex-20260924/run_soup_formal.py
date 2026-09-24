#!/usr/bin/env python3
"""Serial formal PRO Soup queue on local GPU 7, with conservative shared-card admission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
VLA = WORK / "vla-merge"
RUNTIME = WORK / "vla-merge-runtime"
SOURCE = WORK / "pi05_lora_finetune_v2_20260826"
ROOT = RUNTIME / "experiments/pi05-pro-soup-codex-20260924"
SELECTION = ROOT / "selection-v1"
RUN = ROOT / "formal-v1"
POLICY = (RUNTIME / "experiments/iclr2027-table1-20260910/libero/model-soups"
          / "repeat-shared/merge/attempt-02-peft-safe-v2/pretrained_model")
EVALUATOR = VLA / "scripts/eval_pi05_policy_with_extension_selection.py"
NATIVE_RUNNER = VLA / "scripts/run_pi05_libero_pro_evaluation.py"
PYTHON = RUNTIME / "envs/iclr2027-libero-pro-py312-v1/bin/python"
PRO = WORK / ".datasets/LIBERO/20260919/pro"
GPU = 7
UUID = "GPU-e027de95-4c80-d8ec-d650-30da9b6eb135"
HOST = "dsw-967394-56ffd4897d-42wft"
METHOD = "model-soups-main10k-codex-v1"
POLICY_SHA = "a92aacc43146dc41663c0057f2999bd90252fb3fb316ef963f165be67e1be01a"
SELECTION_MANIFEST_SHA = "ef24aefb638317a7820409eac535b707b3af506c435d0628deaab388d39128e3"
MIN_FREE_MIB = 40 * 1024
RUNTIME_FLOOR_MIB = 12 * 1024
REPEATS = ("repeat-01", "repeat-02", "repeat-03")
DIMENSIONS = ("object", "swap", "semantic", "task")

sys.path[:0] = [str(VLA / "src"), str(VLA / "scripts")]
import run_pi05_libero_pro_evaluation as native  # noqa: E402
from vla_merge.libero_extension import load_extension_selection  # noqa: E402
from vla_merge.libero_procedural_bank import sha256_file  # noqa: E402


def write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def row() -> dict:
    line = subprocess.check_output([
        "nvidia-smi", "-i", str(GPU),
        "--query-gpu=uuid,memory.free,memory.used", "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    uuid, free, used = [part.strip() for part in line]
    return {"uuid": uuid, "free_mib": int(free), "used_mib": int(used)}


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(GPU), MUJOCO_GL="egl",
               MUJOCO_EGL_DEVICE_ID=str(GPU), TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               TORCHINDUCTOR_COMPILE_THREADS="1", HF_HUB_OFFLINE="1",
               PALIGEMMA_TOKENIZER_PATH=str(SOURCE / "assets/paligemma-3b-pt-224-tokenizer"),
               LIBERO_PRO_REPO=str(PRO / "repo"), LIBERO_PRO_ASSET_DIR=str(PRO / "assets"),
               LIBERO_CONFIG_PATH=str(PRO / "config"))
    env["PYTHONPATH"] = ":".join((str(PRO / "repo"), str(SOURCE / "lerobot/src"),
                                   str(SOURCE / "src"), str(VLA / "scripts"),
                                   str(VLA / "src"), env.get("PYTHONPATH", "")))
    for key in ("PI05_LIBERO_INIT_STATE_OFFSET", "PI05_LIBERO_INIT_STATE_COUNT",
                "PI05_LIBERO_SUITE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        env.pop(key, None)
    return env


def validate_source() -> None:
    if socket.gethostname() != HOST or row()["uuid"] != UUID:
        raise ValueError("Wrong host or physical GPU 7 UUID")
    if sha256_file(SELECTION / "manifest.json") != SELECTION_MANIFEST_SHA:
        raise ValueError("Frozen PRO Soup selection manifest changed")
    if sha256_file(POLICY / "model.safetensors") != POLICY_SHA:
        raise ValueError("Frozen main-table Soup checkpoint changed")


def prepare() -> dict:
    if RUN.exists():
        raise FileExistsError(RUN)
    validate_source()
    jobs = []
    plans = {}
    for repeat in REPEATS:
        selection_path = SELECTION / "selections" / f"{repeat}.json"
        selection = load_extension_selection(selection_path, expected_benchmark="libero_pro", verify_files=True)
        args = argparse.Namespace(policy=POLICY, policy_sha256=POLICY_SHA, method_id=METHOD,
                                  repeat_id=repeat, selection=selection_path, source_suite=None,
                                  dimension=list(DIMENSIONS), output_root=RUN / "results",
                                  gpu=str(GPU), launch=False, resume_existing=False)
        native_plan = native.build_plan(args)
        if len(native_plan["commands"]) != 160 or len(selection.jobs) != 160:
            raise ValueError(f"Missing formal PRO Soup jobs: {repeat}")
        by_id = {job.job_id: job for job in selection.jobs}
        if len(by_id) != 160:
            raise ValueError("Duplicate source job ID")
        for command in native_plan["commands"]:
            source_job = by_id[command["job_id"]]
            jobs.append({
                "id": f"{repeat}-{command['job_id']}", "repeat": repeat,
                "job_id": command["job_id"], "dimension": command["dimension"],
                "suite": command["source_suite"], "task_id": command["task_id"],
                "state_ids": command["state_ids"],
                "state_raw_sha256": list(source_job.state_raw_sha256),
                "bddl_sha256": sha256_file(source_job.bddl_path),
                "init_states_sha256": sha256_file(source_job.init_path),
                "selection_sha256": native_plan["selection_sha256"],
                "eval_seed": native_plan["eval_seed"],
                "command": command["command"], "output": command["output"],
            })
        plans[repeat] = {"selection_sha256": native_plan["selection_sha256"],
                         "eval_seed": native_plan["eval_seed"],
                         "native_plan_sha256": hashlib.sha256(json.dumps(native_plan, sort_keys=True).encode()).hexdigest()}
    if len(jobs) != 480 or len({job["id"] for job in jobs}) != 480:
        raise ValueError("PRO Soup must contain exactly 480 distinct jobs")
    if sum(len(job["state_ids"]) for job in jobs) != 4800:
        raise ValueError("PRO Soup must contain exactly 4800 episodes")
    plan = {"schema": "pi05_pro_soup_formal_shared_v1", "created_at": now(),
            "host": HOST, "gpu": GPU, "gpu_uuid": UUID,
            "resource_mode": "user_authorized_shared_card_with_40gib_admission_12gib_floor",
            "source_manifest_sha256": SELECTION_MANIFEST_SHA, "model_sha256": POLICY_SHA,
            "evaluator_sha256": sha256_file(EVALUATOR),
            "native_runner_sha256": sha256_file(NATIVE_RUNNER),
            "runner_sha256": sha256_file(Path(__file__)),
            "method": METHOD, "jobs_expected": 480, "episodes_expected": 4800,
            "no_retry": True, "no_score_based_scheduling": True,
            "min_free_mib": MIN_FREE_MIB, "runtime_floor_mib": RUNTIME_FLOOR_MIB,
            "repeats": plans, "jobs": jobs}
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN / "plan.json", plan)
    write(RUN / "CLAIM.json", {"owner": "Codex", "host": HOST,
                               "scope": "pi0.5 PRO Model Soups exact main-table 4800 formal episodes",
                               "plan_sha256": sha256_file(RUN / "plan.json"),
                               "gpu_range": [4, 5, 6, 7], "gpu_selected": GPU,
                               "no_training": True, "no_retry": True})
    return {"run": str(RUN), "plan_sha256": sha256_file(RUN / "plan.json"),
            "jobs": 480, "episodes": 4800}


def verify_job(job: dict, plan: dict) -> dict:
    output = Path(job["output"])
    receipt_path = output / "extension_eval_receipt.json"
    info_path = output / "eval_info.json"
    receipt = json.loads(receipt_path.read_text())
    info = json.loads(info_path.read_text())
    expected = {"benchmark": "libero_pro", "job_id": job["job_id"],
                "repeat_id": job["repeat"], "eval_seed": job["eval_seed"],
                "policy_sha256": POLICY_SHA, "policy_checkpoint_sha256": POLICY_SHA,
                "selection_sha256": job["selection_sha256"],
                "dimension": job["dimension"], "source_suite": job["suite"],
                "task_id": job["task_id"], "state_ids": job["state_ids"],
                "state_raw_sha256": job["state_raw_sha256"],
                "bddl_sha256": job["bddl_sha256"],
                "init_states_sha256": job["init_states_sha256"],
                "entrypoint_sha256": plan["evaluator_sha256"],
                "eval_info_sha256": sha256_file(info_path)}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"PRO Soup receipt {job['id']} mismatched {key}")
    rows = info.get("per_task")
    if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("task_id") != job["task_id"]:
        raise ValueError(f"PRO Soup task coverage differs: {job['id']}")
    successes = (rows[0].get("metrics") or {}).get("successes")
    if (not isinstance(successes, list) or len(successes) != 10
            or any(type(value) is not bool for value in successes)):
        raise ValueError(f"PRO Soup lacks ten boolean outcomes: {job['id']}")
    overall = info.get("overall") or {}
    if (overall.get("n_episodes") != 10
            or abs(float(overall.get("pc_success", -1)) - 10 * sum(successes)) > 1e-6):
        raise ValueError(f"PRO Soup overall disagrees with raw outcomes: {job['id']}")
    return {"episodes": 10, "successes": sum(successes),
            "receipt_sha256": sha256_file(receipt_path),
            "eval_info_sha256": sha256_file(info_path)}


def run() -> None:
    plan_path = RUN / "plan.json"
    plan = json.loads(plan_path.read_text())
    if (plan["schema"] != "pi05_pro_soup_formal_shared_v1" or plan["host"] != HOST
            or plan["gpu"] != GPU or plan["gpu_uuid"] != UUID
            or plan["source_manifest_sha256"] != SELECTION_MANIFEST_SHA
            or plan["model_sha256"] != POLICY_SHA or len(plan["jobs"]) != 480
            or plan["evaluator_sha256"] != sha256_file(EVALUATOR)
            or plan["native_runner_sha256"] != sha256_file(NATIVE_RUNNER)
            or plan["runner_sha256"] != sha256_file(Path(__file__))):
        raise ValueError("Frozen formal plan changed")
    if (RUN / "started.json").exists():
        raise FileExistsError("Formal queue has already started; no automatic retry")
    validate_source()
    lock_path = RUN / "gpu-7-shared-lane.lock"
    handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(handle)
        raise RuntimeError("Another local PRO Soup shared lane holds GPU 7")
    admitted = row()
    if admitted["free_mib"] < MIN_FREE_MIB:
        raise RuntimeError("GPU 7 has insufficient memory for shared admission")
    write(RUN / "started.json", {"pid": os.getpid(), "host": HOST, "started_at": now(),
                                  "gpu": GPU, "uuid": UUID, "admission": admitted,
                                  "plan_sha256": sha256_file(plan_path), "no_retry": True})
    accepted: dict[str, dict] = {}
    failed: dict[str, str] = {}
    stopped = False

    def stop_handler(_sig: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    jobs = plan["jobs"]
    for index, job in enumerate(jobs):
        pending = [item["id"] for item in jobs[index:]]
        while not stopped:
            current = row()
            if current["uuid"] != UUID:
                failed[job["id"]] = "GPU 7 UUID changed before admission"
                stopped = True
                break
            if current["free_mib"] >= MIN_FREE_MIB:
                break
            write(RUN / "state.json", {"accepted": accepted, "active": {},
                                        "pending": pending, "failed": failed,
                                        "status": "waiting_memory", "updated_at": now()})
            time.sleep(30)
        if stopped:
            break
        if Path(job["output"]).exists():
            failed[job["id"]] = "Output directory existed before launch"
            break
        log_path = RUN / "logs" / f"{job['id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x") as log:
            child = subprocess.Popen(job["command"], cwd=WORK, env=environment(),
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            write(RUN / "launches" / f"{job['id']}.json", {
                "id": job["id"], "pid": child.pid, "gpu": GPU, "uuid": UUID,
                "admission": current, "command": job["command"],
                "started_at": now(), "plan_sha256": sha256_file(plan_path)})
            write(RUN / "state.json", {"accepted": accepted,
                                        "active": {job["id"]: {"pid": child.pid, "gpu": GPU}},
                                        "pending": pending[1:], "failed": failed,
                                        "status": "running", "updated_at": now()})
            floor = False
            while child.poll() is None:
                time.sleep(30)
                current = row()
                if current["uuid"] != UUID or current["free_mib"] < RUNTIME_FLOOR_MIB:
                    floor = True
                    os.killpg(child.pid, signal.SIGTERM)
                    break
                write(RUN / "state.json", {"accepted": accepted,
                                            "active": {job["id"]: {"pid": child.pid, "gpu": GPU}},
                                            "pending": pending[1:], "failed": failed,
                                            "status": "running", "updated_at": now()})
            code = child.wait()
        artifact = None
        error = None
        if code == 0 and not floor:
            try:
                artifact = verify_job(job, plan)
            except Exception as exc:
                error = f"audit: {type(exc).__name__}: {exc}"
        else:
            error = f"child exit {code}; resource_floor={floor}"
        write(RUN / "exits" / f"{job['id']}.json", {
            "id": job["id"], "returncode": code, "accepted": artifact is not None,
            "artifact": artifact, "error": error, "finished_at": now()})
        if error:
            failed[job["id"]] = error
            break
        accepted[job["id"]] = artifact
        write(RUN / "state.json", {"accepted": accepted, "active": {},
                                    "pending": pending[1:], "failed": failed,
                                    "status": "running", "updated_at": now()})
    complete = len(accepted) == 480 and not failed and not stopped
    remaining = [job["id"] for job in jobs if job["id"] not in accepted and job["id"] not in failed]
    write(RUN / "state.json", {"accepted": accepted, "active": {},
                                "pending": remaining, "failed": failed,
                                "status": "complete" if complete else "incomplete",
                                "updated_at": now()})
    write(RUN / "queue-ended.json", {"complete": complete, "stopped": stopped,
                                      "accepted_jobs": len(accepted), "failed": failed,
                                      "pending_jobs": len(remaining),
                                      "accepted_episodes": 10 * len(accepted),
                                      "expected_jobs": 480, "expected_episodes": 4800,
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
