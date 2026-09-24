#!/usr/bin/env python3
"""Prepare and run the bounded local fresh-expert capture queue.

The queue owns only its child processes on local GPUs 2/3. It launches at most two
collectors per card, admits a launch only when the card reports at least 32 GiB free,
and records the measured snapshot (including a required 12 GiB reserve). A failed or
signal-killed child trips one batch latch: pending jobs are blocked, no job is retried,
and each new attempt must use a new output directory.

This module does not solve or evaluate a model. It creates twelve collection jobs:
three repeat identities (start seeds 291001--3, flow seeds 292001--3, init offsets
0--2) times the four audited 10k dense experts.
"""
from __future__ import annotations

import argparse
import fcntl
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
VLA = HERE.parents[1]
SCRIPTS = VLA / "scripts"
SOURCE = WORK / "pi05_lora_finetune_v2_20260826"
PYTHON = SOURCE / ".venv/bin/python"
TABLE = WORK / "vla-merge-runtime/experiments/iclr2027-table1-20260910"
BANK = TABLE / "expert-dense-bank-peft-v2.json"
DEFAULT_RUN = WORK / "vla-merge-runtime/experiments/tcr-mainline-capture-20260919"
COLLECTOR = HERE / "collect_mainline_table3_execution.py"

LOCAL_GPUS = (2, 3)
MAX_SLOTS_PER_GPU = 2
MIN_FREE_MIB = 36 * 1024
RESERVE_MIB = 12 * 1024
CAP_REQUESTS = 128
FLOW_INDICES = (0, 5, 9)
SUITES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}
REPEATS = (
    {"repeat_id": "repeat-01", "start_seed": 291001, "flow_seed": 292001, "init_offset": 0},
    {"repeat_id": "repeat-02", "start_seed": 291002, "flow_seed": 292002, "init_offset": 1},
    {"repeat_id": "repeat-03", "start_seed": 291003, "flow_seed": 292003, "init_offset": 2},
)

FROZEN_SOURCES = (
    HERE / "mainline_capture.py",
    SCRIPTS / "collect_pi05_table3_execution.py",
    SCRIPTS / "collect_pi05_block_regmeanpp_calibration.py",
    SCRIPTS / "pi05_table3_contract.py",
    SCRIPTS / "eval_pi05_libero_with_init_offset.py",
    SCRIPTS / "run_pi05_table3_queue.py",
    SCRIPTS / "pi05_tcr_e_dense_contract.py",
    SOURCE / "src/eval_with_local_tokenizer.py",
    SOURCE / "src/runtime_overrides.py",
    SCRIPTS / "run_pi05_generation_path_probe.py",
    SCRIPTS / "audit_tcr_request_quantiles_20260919.py",
    WORK / ".datasets/LIBERO/20260919/config-standard/config.yaml",
    VLA / "experiments/claude-balanced-pass2-20260919/batch_safety.py",
    VLA / "experiments/claude-balanced-pass2-20260919/guard_batch_v3.py",
)

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(VLA / "experiments/claude-balanced-pass2-20260919"))
from batch_safety import BatchStop, Child, classify_exit  # noqa: E402
import run_pi05_table3_queue as table3  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def source_hashes() -> dict[str, str]:
    missing = [str(path) for path in FROZEN_SOURCES if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen collector dependency: " + ", ".join(missing))
    return {str(path.resolve()): sha256_file(path) for path in FROZEN_SOURCES}


def load_bank(bank_path: Path = BANK) -> dict[str, Any]:
    bank_path = bank_path.resolve()
    if not bank_path.is_file():
        raise FileNotFoundError(bank_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("kind") != "iclr2027_table1_peft_safe_dense_expert_bank":
        raise ValueError(f"Unexpected expert bank kind: {bank.get('kind')!r}")
    experts = {entry.get("name"): entry for entry in bank.get("experts", [])}
    if set(experts) != set(SUITES):
        raise ValueError(f"Expert names differ: {sorted(experts)}")
    for name, entry in experts.items():
        if entry.get("suite") != SUITES[name]:
            raise ValueError(f"{name}: wrong LIBERO suite")
        dense = entry.get("dense_checkpoint") or {}
        if "/checkpoints/010000/" not in entry.get("source_adapter", {}).get("path", ""):
            raise ValueError(f"{name}: not the registered 10k expert")
        model = Path(dense.get("path", "")).resolve()
        model_file = model / "model.safetensors"
        if not model_file.is_file():
            raise FileNotFoundError(f"{name}: {model_file}")
        declared = dense.get("model_sha256")
        if not declared:
            raise ValueError(f"{name}: dense checkpoint has no declared model hash")
        actual = sha256_file(model_file)
        if actual != declared:
            raise ValueError(f"{name}: model hash mismatch: {actual} != {declared}")
        config = json.loads((model / "config.json").read_text())
        if config.get("num_inference_steps") != 10:
            raise ValueError(f"{name}: native generation must have ten steps")
    return {"path": str(bank_path), "sha256": sha256_file(bank_path), "data": bank, "experts": experts}


def job_output(run: Path, repeat_id: str, expert: str) -> Path:
    return run / repeat_id / "inputs" / expert


def build_jobs(run: Path, bank_info: dict[str, Any]) -> list[dict[str, Any]]:
    bank_sha = bank_info["sha256"]
    jobs: list[dict[str, Any]] = []
    for repeat in REPEATS:
        for expert, suite in SUITES.items():
            dense = bank_info["experts"][expert]["dense_checkpoint"]
            policy = Path(dense["path"]).resolve()
            output = job_output(run, repeat["repeat_id"], expert)
            env = {
                "PI05_BLOCK_REGMEANPP_TASK": expert,
                "PI05_BLOCK_REGMEANPP_TENSOR_OUTPUT": str(output / "raw.safetensors"),
                "PI05_BLOCK_REGMEANPP_MANIFEST_OUTPUT": str(output / "raw.json"),
                "PI05_BLOCK_REGMEANPP_CALIBRATION_POLICY": str(policy),
                "PI05_BLOCK_REGMEANPP_MAX_CALLS": "3840",
                "PI05_BLOCK_REGMEANPP_MAX_CALLS_PER_PROMPT": "1",
                "PI05_BLOCK_REGMEANPP_REQUESTS_PER_EPISODE": str(CAP_REQUESTS),
                "PI05_BLOCK_REGMEANPP_EPISODE_AWARE": "1",
                "PI05_BLOCK_REGMEANPP_FULL_PREFIX": "1",
                "PI05_BLOCK_REGMEANPP_FLOW_INDICES": ",".join(map(str, FLOW_INDICES)),
                "PI05_BLOCK_REGMEANPP_REQUEST_MODE": "initial",
                "PI05_BLOCK_REGMEANPP_START_SEED": str(repeat["start_seed"]),
                "PI05_LIBERO_INIT_STATE_OFFSET": str(repeat["init_offset"]),
                "PI05_LIBERO_INIT_STATE_COUNT": "1",
                "PI05_LIBERO_SUITE": suite,
                "PI05_MAINLINE_REPEAT_ID": repeat["repeat_id"],
                "PI05_MAINLINE_EXPERT_NAME": expert,
                "PI05_MAINLINE_POLICY_SHA256": dense["model_sha256"],
                "PI05_MAINLINE_DENSE_BANK_SHA256": bank_sha,
                "PI05_MAINLINE_START_SEED": str(repeat["start_seed"]),
                "PI05_MAINLINE_FLOW_SEED": str(repeat["flow_seed"]),
                "PI05_MAINLINE_INIT_STATE_OFFSET": str(repeat["init_offset"]),
                "PI05_MAINLINE_MEMORY_FRACTION": "0.30",
            }
            command = [
                str(PYTHON), "-u", str(COLLECTOR),
                f"--output_dir={output / 'rollout'}", "--env.type=libero",
                f"--env.task={suite}", "--env.task_ids=[0,1,2,3,4,5,6,7,8,9]",
                "--env.init_states=true", "--eval.batch_size=1", "--eval.n_episodes=1",
                f"--seed={repeat['start_seed']}", f"--policy.path={policy}",
                "--policy.device=cuda", "--policy.compile_model=false",
                "--policy.gradient_checkpointing=false", "--policy.n_action_steps=10",
            ]
            jobs.append({
                "id": f"collect-{repeat['repeat_id']}-{expert}",
                "kind": "collection", "repeat_id": repeat["repeat_id"],
                "expert_name": expert, "suite": suite,
                "start_seed": repeat["start_seed"], "flow_seed": repeat["flow_seed"],
                "init_state_offset": repeat["init_offset"], "policy": str(policy),
                "policy_sha256": dense["model_sha256"], "dense_bank_sha256": bank_sha,
                "output": str(output), "command": command, "environment": env,
                "min_free_mib": MIN_FREE_MIB, "reserve_mib": RESERVE_MIB,
            })
    return jobs


def prepare_plan(run: Path, bank_path: Path = BANK, max_runtime_hours: float = 13.0) -> dict[str, Any]:
    run = run.resolve()
    if run.exists():
        raise FileExistsError(f"Fresh attempt path already exists: {run}")
    if max_runtime_hours <= 0:
        raise ValueError("max_runtime_hours must be positive")
    bank = load_bank(bank_path)
    plan = {
        "schema": "tcr_mainline_local_collection_v1",
        "status": "prepared_not_launched",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "run": str(run),
        "allowed_gpus": list(LOCAL_GPUS), "max_slots_per_gpu": MAX_SLOTS_PER_GPU,
        "min_free_mib": MIN_FREE_MIB, "reserve_mib": RESERVE_MIB,
        "no_kill_other_processes": True, "no_automatic_restart": True,
        "collection_only": True,
        "protected_paths_not_accessed": [str(WORK / "vla-merge-runtime/experiments/tcr-confirmation-20260917")],
        "protocol": {
            "experts": "four audited 10k dense single-task experts",
            "repeats": [dict(item) for item in REPEATS], "tasks_per_suite": 10,
            "full_request_cap": CAP_REQUESTS,
            "selection": "true observed request quantiles 0,.25,.5,.75,1",
            "flow_indices": list(FLOW_INDICES), "native_action_steps": 10,
            "native_executed_actions": 10, "failures_retained": True,
        },
        "expert_bank": bank, "source_hashes": source_hashes(),
        "adapter_sha256": sha256_file(COLLECTOR),
        "deadline_unix": min(time.time() + max_runtime_hours * 3600.0,
                             datetime(2026, 9, 20, 1, 31, tzinfo=timezone.utc).timestamp()),
    }
    plan["jobs"] = build_jobs(run, bank)
    if len(plan["jobs"]) != 12:
        raise AssertionError(f"Expected 12 collection jobs, got {len(plan['jobs'])}")
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "plan.json", plan)
    return plan


def _nvidia_rows(gpu: int) -> list[dict[str, Any]]:
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip()
    rows = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            continue
        rows.append({
            "index": int(fields[0]), "uuid": fields[1], "total_mib": int(float(fields[2])),
            "used_mib": int(float(fields[3])), "free_mib": int(float(fields[4])),
            "utilization_gpu_percent": int(float(fields[5])),
        })
    return rows


def _compute_rows(gpu: int, owned_pids: set[int]) -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid, memory = int(fields[0]), int(float(fields[1]))
        except ValueError:
            continue
        try:
            command = Path(f"/proc/{pid}/comm").read_text().strip()
        except (OSError, UnicodeError):
            command = "<unavailable>"
        rows.append({"pid": pid, "used_mib": memory, "command": command,
                     "owner": "owned" if pid in owned_pids else "unknown"})
    return rows


def resource_snapshot(gpus: tuple[int, ...], owned_pids: set[int]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "available": True, "gpus": [],
    }
    for gpu in gpus:
        try:
            rows = _nvidia_rows(gpu)
            if len(rows) != 1:
                raise RuntimeError(f"nvidia-smi returned {len(rows)} rows for GPU {gpu}")
            row = rows[0]
            row["compute_processes"] = _compute_rows(gpu, owned_pids)
            snapshot["gpus"].append(row)
        except Exception as error:
            snapshot["available"] = False
            snapshot["gpus"].append({"index": gpu, "available": False, "error": str(error)})
    return snapshot


def record_resource_snapshot(run: Path, gpus: tuple[int, ...], owned_pids: set[int], stage: str) -> Path:
    value = resource_snapshot(gpus, owned_pids)
    value["stage"] = stage
    path = run / "resources" / f"{utc_stamp()}-{time.time_ns()}.json"
    write_json(path, value)
    return path


def admit_gpu(gpu: int) -> tuple[bool, dict[str, Any]]:
    try:
        rows = _nvidia_rows(gpu)
    except Exception as error:
        return False, {"available": False, "error": str(error)}
    if len(rows) != 1:
        return False, {"available": False, "error": f"expected one row, got {len(rows)}"}
    row = rows[0]
    return row["free_mib"] >= MIN_FREE_MIB, {
        "available": True, "gpu": gpu, "uuid": row["uuid"],
        "free_mib": row["free_mib"], "required_free_mib": MIN_FREE_MIB,
        "reserve_mib": RESERVE_MIB, "utilization_gpu_percent": row["utilization_gpu_percent"],
    }


def expected_request_indices(request_count: int) -> list[int]:
    if request_count < 5:
        raise ValueError("Need at least five observed requests")
    if request_count >= CAP_REQUESTS:
        raise ValueError("Request count reached cap; cannot certify full trajectory")
    return [(request_count - 1) * k // 4 for k in range(5)]


def verify_manifest(manifest_path: Path, job: dict[str, Any]) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    tensor_path = manifest_path.with_suffix(".safetensors")
    if not manifest_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"Missing replay artifacts for {job['id']}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "mainline_capture_version": 1, "source_kind": "expert_execution",
        "table3_request_selection": "across", "sample_count": 150,
        "calibration_start_seed": job["start_seed"],
        "calibration_init_state_offset": job["init_state_offset"],
        "generation_noise_seed": job["flow_seed"], "repeat_id": job["repeat_id"],
        "expert_name": job["expert_name"], "calibration_policy_sha256": job["policy_sha256"],
        "dense_expert_bank_sha256": job["dense_bank_sha256"],
    }
    problems = [f"{key}={manifest.get(key)!r}, expected {value!r}"
                for key, value in expected.items() if manifest.get(key) != value]
    if manifest.get("failures_retained") is not True:
        problems.append("failures_retained is not true")
    if manifest.get("flow_indices") != list(FLOW_INDICES):
        problems.append(f"flow_indices={manifest.get('flow_indices')!r}")
    if manifest.get("calibration_policy") and Path(manifest["calibration_policy"]).resolve() != Path(job["policy"]).resolve():
        problems.append("calibration policy path mismatch")
    samples = manifest.get("samples") or []
    if len(samples) != 150 or manifest.get("prompt_count") != 10:
        problems.append("expected 10 tasks x 5 requests x 3 flows")
    grouped: dict[tuple[Any, int], list[int]] = {}
    prompt_indices: dict[Any, set[int]] = {}
    for sample in samples:
        prompt = sample.get("prompt_signature")
        grouped.setdefault((prompt, sample.get("selected_request_slot")), []).append(sample.get("flow_index"))
        prompt_indices.setdefault(prompt, set()).add(sample.get("task_episode_index"))
        if sample.get("init_state_id") != job["init_state_offset"]:
            problems.append("actual sample init state differs from registered repeat")
        if sample.get("simulator_seed") != job["start_seed"]:
            problems.append("actual sample simulator seed differs from registered repeat")
    if len(prompt_indices) != 10 or len(grouped) != 50:
        problems.append("wrong prompt/request group count")
    if any(sorted(values) != list(FLOW_INDICES) for values in grouped.values()):
        problems.append("incomplete flow triple")
    counts = manifest.get("prompt_seen_call_counts") or {}
    selected = manifest.get("selected_requests") or {}
    if len(counts) != 10 or len(selected) != 10:
        problems.append("missing per-task full request provenance")
    else:
        for prompt, seen_calls in counts.items():
            if int(seen_calls) % 10:
                problems.append(f"prompt {prompt} call count is not divisible by native 10 steps")
                continue
            request_count = int(seen_calls) // 10
            try:
                wanted = expected_request_indices(request_count)
            except ValueError as error:
                problems.append(f"prompt {prompt}: {error}")
                continue
            actual = list(selected.get(str(prompt), selected.get(prompt, [])))
            if actual != wanted:
                problems.append(f"prompt {prompt}: selected {actual}, expected {wanted}")
    if problems:
        raise ValueError(f"{job['id']} manifest audit failed: " + "; ".join(problems))
    from audit_tcr_request_quantiles_20260919 import check_quantiles
    quantiles = check_quantiles(manifest)
    if not quantiles["matches_registered_full_execution_quantiles"]:
        raise ValueError(f"{job['id']}: actual sample quantiles failed: {quantiles['issues']}")
    return {
        "job_id": job["id"], "sample_count": len(samples),
        "manifest_sha256": sha256_file(manifest_path), "tensor_sha256": sha256_file(tensor_path),
        "request_provenance_audited": True, "full_trajectory_cap_not_reached": True,
        "collection_only": True, "no_success_rate": True,
    }


def environment(gpu: int) -> dict[str, str]:
    base = table3.environment(gpu)
    base.update({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2",
                 "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                 "TORCHINDUCTOR_COMPILE_THREADS": "1",
                 "LIBERO_CONFIG_PATH": str(WORK / ".datasets/LIBERO/20260919/config-standard"),
                 "ITERATION_PHYSICAL_GPU": str(gpu), "HF_HUB_OFFLINE": "1"})
    for name in ("LIBERO_PRO_REPO", "LIBERO_PRO_ASSET_DIR", "MUJOCO_EGL_DEVICE_ID",
                 "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        base.pop(name, None)
    return base


def ensure_fresh_outputs(plan: dict[str, Any]) -> None:
    run = Path(plan["run"])
    for job in plan["jobs"]:
        output = Path(job["output"])
        if output.exists():
            raise FileExistsError(f"Refusing live-path reuse for {job['id']}: {output} already exists")
    for marker in ("started.json", "queue-ended.json", "batch-stop-report.json"):
        if (run / marker).exists():
            raise FileExistsError(f"Refusing to restart an existing attempt: {run / marker}")


def run_queue(
    plan: dict[str, Any], *, gpus: tuple[int, ...] = LOCAL_GPUS,
    poll_seconds: float = 60.0,
    command_factory: Callable[[dict[str, Any]], list[str]] | None = None,
    resource_probe: Callable[[int], tuple[bool, dict[str, Any]]] = admit_gpu,
    max_slots_per_gpu: int = MAX_SLOTS_PER_GPU,
) -> dict[str, Any]:
    """Run a plan with real Child processes; injectable command/probe only for tests."""
    if tuple(gpus) != LOCAL_GPUS:
        raise ValueError(f"Local capture may use only GPUs {LOCAL_GPUS}")
    if not 1 <= max_slots_per_gpu <= MAX_SLOTS_PER_GPU:
        raise ValueError("max_slots_per_gpu must be1 or2")
    run = Path(plan["run"])
    ensure_fresh_outputs(plan)
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "started.json", {"pid": os.getpid(), "host": socket.gethostname(), "time": time.time()})
    stop = BatchStop(report_path=run / "batch-stop-report.json")
    stop.install_signal_handlers()
    jobs = list(plan["jobs"])
    pending = deque(jobs)
    state_lock = threading.Lock()
    card_locks = {gpu: threading.Lock() for gpu in gpus}
    wake = threading.Event()
    states = {job["id"]: {"status": "pending"} for job in jobs}
    active: dict[int, Child] = {}
    active_lock = threading.Lock()
    started = time.time()
    last_resource = 0.0
    resource_paths: list[str] = []
    report_lock = threading.Lock()
    report_done = threading.Event()
    command_factory = command_factory or (lambda job: list(job["command"]))

    def snapshot(stage: str) -> None:
        nonlocal last_resource
        with active_lock:
            owned = {child.pid for child in active.values()}
        with report_lock:
            path = record_resource_snapshot(run, gpus, owned, stage)
            resource_paths.append(str(path))
            last_resource = time.time()

    snapshot("queue-start")

    def report_periodically():
        while not report_done.wait(20 * 60):
            snapshot("periodic-active")

    reporter = threading.Thread(target=report_periodically, name="capture-resource-reporter")
    reporter.start()

    def worker(gpu: int, slot: int) -> None:
        nonlocal last_resource
        while True:
            if stop.stopped:
                return
            if time.time() >= float(plan["deadline_unix"]):
                stop.trip("bounded queue deadline reached")
                return
            with card_locks[gpu]:
                if stop.stopped:
                    return
                allowed, check = resource_probe(gpu)
                job = None
                child = None
                if allowed:
                    with state_lock:
                        job = pending.popleft() if pending else None
                        if job is not None:
                            states[job["id"]].update(status="launching", gpu=gpu, slot=slot, resource_check=check)
                    if job is not None:
                        output = Path(job["output"])
                        try:
                            output.mkdir(parents=True, exist_ok=False)
                            log_path = run / "logs" / f"{job['id']}.log"
                            child = Child(command_factory(job), {**environment(gpu), **job.get("environment", {})}, log_path, cwd=VLA)
                            if not stop.register(child):
                                with state_lock:
                                    states[job["id"]].update(status="cancelled", reason=stop.reason)
                                return
                            with active_lock:
                                active[child.pid] = child
                            with state_lock:
                                states[job["id"]].update(status="running", pid=child.pid, started_at=time.time(), log=str(log_path))
                            write_json(output / "launch.json", {"job": job["id"], "gpu": gpu, "slot": slot,
                                                                  "host": socket.gethostname(),
                                                                  "plan_sha256": sha256_file(run / "plan.json") if (run / "plan.json").exists() else None,
                                                                  "resource_check": check, "child": child.identity(),
                                                                  "started_unix": time.time()})
                            snapshot(f"launch-{job['id']}")
                        except Exception as error:
                            with state_lock:
                                states[job["id"]].update(status="failed", error=str(error))
                            stop.trip(f"{job['id']} launch failed: {error}")
                            return
            if job is None:
                if not pending:
                    return
                if time.time() - last_resource >= 20 * 60:
                    snapshot("periodic-wait")
                wake.wait(poll_seconds)
                wake.clear()
                continue
            code = child.wait()
            write_json(Path(job["output"]) / "exit.json", {
                "return_code": code, "host": socket.gethostname(),
                "child": child.identity(), "finished_unix": time.time(),
                "wall_seconds": time.time() - child.started_unix})
            with active_lock:
                active.pop(child.pid, None)
            outcome = classify_exit(code)
            if outcome["outcome"] != "success":
                with state_lock:
                    states[job["id"]].update(status="failed", return_code=code, outcome=outcome, finished_at=time.time())
                stop.trip(f"{job['id']} exited: {outcome['outcome']}")
                return
            try:
                receipt = verify_manifest(Path(job["output"]) / "across/replay.json", job)
            except Exception as error:
                with state_lock:
                    states[job["id"]].update(status="failed", error=str(error), finished_at=time.time())
                stop.trip(f"{job['id']} audit failed: {error}")
                return
            with state_lock:
                states[job["id"]].update(status="complete", return_code=code, finished_at=time.time(), receipt=receipt)
            write_json(Path(job["output"]) / "verified.json", receipt)
            snapshot(f"complete-{job['id']}")
            if time.time() - last_resource >= 20 * 60:
                snapshot("periodic")

    threads = [threading.Thread(target=worker, args=(gpu, slot), name=f"capture-gpu{gpu}-{slot}")
               for gpu in gpus for slot in range(max_slots_per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    report_done.set()
    reporter.join()
    with state_lock:
        for job in jobs:
            if states[job["id"]]["status"] in {"pending", "launching", "running"}:
                states[job["id"]].update(status="cancelled", reason=stop.reason or "queue ended")
    status = "stopped" if stop.stopped else ("complete" if all(item["status"] == "complete" for item in states.values()) else "incomplete")
    result = {"schema": "tcr_mainline_local_collection_result_v1", "status": status,
              "stopped": stop.stopped, "stop_reason": stop.reason, "restarted": False,
              "collection_only": True, "no_success_rate_evaluation": True, "states": states,
              "resource_snapshots": resource_paths, "elapsed_seconds": time.time() - started}
    write_json(run / "queue-ended.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--max-runtime-hours", type=float, default=13.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.prepare == args.launch:
        raise SystemExit("Choose exactly one of --prepare or --launch")
    if args.prepare:
        plan = prepare_plan(args.run, args.bank, args.max_runtime_hours)
        print(json.dumps({"prepared": True, "run": plan["run"], "jobs": len(plan["jobs"])}, indent=2))
        return
    plan_path = args.run.resolve() / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing prepared plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "tcr_mainline_local_collection_v1":
        raise ValueError("Wrong or unsupported collection plan schema")
    if plan.get("hostname") != socket.gethostname():
        raise ValueError("Prepared plan belongs to another host")
    if plan.get("source_hashes") != source_hashes() or plan.get("adapter_sha256") != sha256_file(COLLECTOR):
        raise ValueError("Pinned collector dependency changed after preparation")
    with (args.run / "controller.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_queue(plan, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
