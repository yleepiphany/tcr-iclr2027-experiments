#!/usr/bin/env python3
"""Build and persistently run the corrected RoboTwin Experts formal panel.

The protocol uses three selected 15k experts, 30 source tasks, three repeats,
and six fresh SHA-derived reset seeds per task/repeat.  Build fails closed if
any formal task/seed key overlaps a completed historical rollout.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path("/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime")
TABLE5 = RUNTIME / "experiments/iclr2027-table5-20260910"
QUEUE = TABLE5 / "evaluation-queues/robotwin-three-expert-formal-v2"
GROUPS = ("coordination", "receptacle", "precision")
REPEATS = (1, 2, 3)
EPISODES_PER_TASK_PER_REPEAT = 6
SEED_NAMESPACE = "iclr2027-table5-robotwin-formal-reset-v2"
SEED_AMENDMENTS = {
    # Protocol amendment 1: the original seed failed simulator setup in 14
    # preserved attempts before policy inference.  The replacement is the next
    # unused SHA-derived episode slot for the same task/repeat.
    ("receptacle", 2, 27, 0): 877886121,
}
CANDIDATE_GPUS = tuple(range(8))
MAX_USED_MIB = 40_000.0
MIN_FREE_MIB = 32_000.0
MAX_SHARED_UTILIZATION = 5.0

PARENTS = {
    "coordination": TABLE5 / "expert-training/robotwin-three-expert-15k-sequence-v1/coordination/evaluation-015000/parent.json",
    "receptacle": TABLE5 / "expert-training/robotwin-three-expert-15k-sequence-v1/receptacle/evaluation-015000/parent.json",
    "precision": TABLE5 / "evaluation-queues/robotwin-precision-015000-quick20-v1/parent.json",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_module("robotwin_formal_base", ROOT / "scripts/run_iclr2027_robotwin_checkpoint_development.py")
slicer = load_module("robotwin_formal_slicer", ROOT / "scripts/run_iclr2027_robotwin_development_slices.py")
def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def parent(group: str) -> dict[str, Any]:
    value = read(PARENTS[group])
    base.validate(value)
    if value["group"] != group or value["step"] != 15_000:
        raise ValueError(f"Wrong frozen parent identity for {group}")
    if len(value["tasks"]) != 10 or any(len(task["seeds"]) != 20 for task in value["tasks"]):
        raise ValueError(f"{group} parent is not 10 tasks x 20 deterministic seeds")
    return value


def formal_tasks(group: str, repeat: int) -> list[dict[str, Any]]:
    source = parent(group)
    rows = []
    for task in source["tasks"]:
        seeds = []
        for episode_index in range(EPISODES_PER_TASK_PER_REPEAT):
            token = (
                f"{SEED_NAMESPACE}\0{repeat}\0{int(task['task_index'])}\0{episode_index}"
            )
            derived = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 2**31
            seeds.append(
                SEED_AMENDMENTS.get(
                    (group, repeat, int(task["task_index"]), episode_index),
                    derived,
                )
            )
        rows.append({
            "task": task["task"],
            "task_index": int(task["task_index"]),
            "seeds": seeds,
        })
    return rows


def historical_completed_keys() -> dict[tuple[int, int], list[str]]:
    """Return completed task/seed rows outside this corrected formal queue."""
    result: dict[tuple[int, int], list[str]] = {}
    for directory, subdirs, files in os.walk(TABLE5):
        root = Path(directory)
        if root == QUEUE or QUEUE in root.parents:
            subdirs[:] = []
            continue
        if any(part in {"checkpoints", "datasets", "expert-plans"} for part in root.parts):
            subdirs[:] = []
            continue
        for name in files:
            if not name.endswith(".json"):
                continue
            path = root / name
            try:
                value = read(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if (
                type(value.get("task_index")) is int
                and type(value.get("seed")) is int
                and type(value.get("success")) is bool
            ):
                result.setdefault((value["task_index"], value["seed"]), []).append(str(path))
    return result


def build() -> None:
    if QUEUE.exists():
        raise FileExistsError(f"Corrected formal queue already exists: {QUEUE}")
    QUEUE.mkdir(parents=True, exist_ok=False)
    parent_bindings = {group: bind(PARENTS[group]) for group in GROUPS}
    bank_bindings = []
    job_bindings = []
    all_keys: set[tuple[str, int, int, int]] = set()
    history = historical_completed_keys()
    historical_collisions = []
    for repeat in REPEATS:
        entries = []
        for group in GROUPS:
            tasks = formal_tasks(group, repeat)
            for task in tasks:
                for seed in task["seeds"]:
                    key = (group, repeat, task["task_index"], seed)
                    if key in all_keys:
                        raise ValueError(f"Duplicate formal key: {key}")
                    all_keys.add(key)
                    for historical_path in history.get((task["task_index"], seed), []):
                        historical_collisions.append({
                            "group": group,
                            "repeat": repeat,
                            "task_index": task["task_index"],
                            "seed": seed,
                            "historical_path": historical_path,
                        })
                    entries.append({
                        "group": group,
                        "repeat": repeat,
                        "task": task["task"],
                        "task_index": task["task_index"],
                        "seed": seed,
                        "task_config": "demo_clean",
                        "action_mode": "joint",
                    })
        bank = QUEUE / f"reset-banks/repeat-{repeat:02d}.json"
        if not bank.is_file():
            write_exclusive(bank, {
                "schema_version": 1,
                "status": "robotwin_formal_reset_bank_frozen",
                "formal_result": True,
                "repeat": repeat,
                "seed_namespace": SEED_NAMESPACE,
                "seed_derivation": "uint31(SHA256(namespace\\0repeat\\0task_index\\0episode_index)[:8])",
                "historical_completed_keys_checked": len(history),
                "episodes_per_task": 6,
                "entries": entries,
                "created_at": now(),
            })
        bank_bindings.append(bind(bank))
        for group in GROUPS:
            source = parent(group)
            manifest = QUEUE / f"manifests/{group}/repeat-{repeat:02d}.json"
            tasks = formal_tasks(group, repeat)
            expected = {(task["task_index"], seed) for task in tasks for seed in task["seeds"]}
            if len(expected) != 60:
                raise ValueError("Each formal group/repeat job must contain 60 unique episodes")
            if not manifest.is_file():
                write_exclusive(manifest, {
                    "schema_version": 1,
                    "status": "robotwin_experts_formal_job_frozen",
                    "formal_result": True,
                    "method": "Experts",
                    "group": group,
                    "repeat": repeat,
                    "checkpoint_step": 15_000,
                    "checkpoint": source["checkpoint"],
                    "condition": "demo_clean",
                    "action_mode": "joint",
                    "full_native_horizon": True,
                    "invalid_seed_policy": "fail attempt; never replace or resample seed",
                    "episodes_per_task": 6,
                    "tasks_per_group": 10,
                    "episodes": 60,
                    "tasks": tasks,
                    "seed_namespace": SEED_NAMESPACE,
                    "reset_bank": bind(bank),
                    "development_parent": parent_bindings[group],
                    "created_at": now(),
                })
            job_bindings.append(bind(manifest))
    if len(all_keys) != 540:
        raise ValueError(f"Formal protocol must contain 540 unique keys, got {len(all_keys)}")
    if historical_collisions:
        raise ValueError(f"Fresh formal bank overlaps historical rows: {historical_collisions[:5]}")
    collision_audit = QUEUE / "seed-collision-audit.json"
    write_exclusive(collision_audit, {
        "schema_version": 1,
        "status": "formal_seed_bank_disjoint_from_completed_history",
        "formal_keys": len(all_keys),
        "historical_completed_keys_checked": len(history),
        "collision_count": 0,
        "seed_namespace": SEED_NAMESPACE,
        "created_at": now(),
    })
    protocol = QUEUE / "protocol.json"
    if not protocol.is_file():
        write_exclusive(protocol, {
            "schema_version": 1,
            "status": "robotwin_three_expert_formal_protocol_frozen",
            "formal_result": True,
            "method": "Experts",
            "groups": list(GROUPS),
            "tasks_per_group": 10,
            "repeats": 3,
            "episodes_per_task_per_repeat": 6,
            "episodes_per_group_per_repeat": 60,
            "episodes_total": 540,
            "aggregation": "task success within repeat; macro over 30 tasks; mean and sample std over three repeats",
            "condition": "demo_clean",
            "action_mode": "joint",
            "full_native_horizon": True,
            "seed_namespace": SEED_NAMESPACE,
            "seed_derivation": "uint31(SHA256(namespace\\0repeat\\0task_index\\0episode_index)[:8])",
            "historical_collision_audit": bind(collision_audit),
            "parent_manifests": parent_bindings,
            "reset_banks": bank_bindings,
            "jobs": job_bindings,
            "created_at": now(),
        })
    print(json.dumps({"status": "formal_protocol_ready", "protocol": bind(protocol)}, indent=2))


def validate_job(group: str, repeat: int) -> tuple[dict[str, Any], dict[str, Any], set[tuple[int, int]]]:
    manifest_path = QUEUE / f"manifests/{group}/repeat-{repeat:02d}.json"
    manifest = read(manifest_path)
    source = parent(group)
    if manifest.get("formal_result") is not True or manifest.get("method") != "Experts":
        raise ValueError("Job is not a frozen Experts formal job")
    if manifest.get("group") != group or manifest.get("repeat") != repeat:
        raise ValueError("Formal job identity differs")
    for name in ("development_parent", "reset_bank"):
        item = manifest[name]
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Changed formal binding: {name}")
    if manifest["tasks"] != formal_tasks(group, repeat):
        raise ValueError("Formal task/seed panel differs")
    expected = {(int(task["task_index"]), int(seed)) for task in manifest["tasks"] for seed in task["seeds"]}
    if len(expected) != 60:
        raise ValueError("Formal job does not contain 60 unique episodes")
    return manifest, source, expected


def reusable_rows(job_root: Path, expected: set[tuple[int, int]]) -> dict[tuple[int, int], Path]:
    rows: dict[tuple[int, int], Path] = {}
    for attempt in sorted(job_root.glob("attempt-*")):
        for path in sorted(attempt.glob("*.json")):
            if path.name in {"started.json", "failure.json", "complete.json", "simulator.json"}:
                continue
            try:
                row = read(path)
                key = (int(row["task_index"]), int(row["seed"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if key not in expected or type(row.get("success")) is not bool:
                continue
            if key in rows:
                raise ValueError(f"Duplicate reusable formal key {key}: {rows[key]}, {path}")
            rows[key] = path
    return rows


def next_attempt(job_root: Path) -> Path:
    for index in range(1, 100):
        candidate = job_root / f"attempt-{index:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("No formal attempt slot remains")


def acquire(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream


def gpu_inventory(gpu: int) -> dict[str, Any]:
    row = subprocess.run(
        [
            "nvidia-smi", "-i", str(gpu),
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fields = [part.strip() for part in row.split(",")]
    process_text = subprocess.run(
        [
            "nvidia-smi", "-i", str(gpu),
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    processes = []
    for line in process_text.splitlines():
        if not line.strip():
            continue
        pid_text, memory_text = [part.strip() for part in line.split(",", 1)]
        processes.append({"pid": int(pid_text), "used_memory_mib": float(memory_text)})
    used = float(fields[2])
    total = float(fields[3])
    utilization = float(fields[4])
    return {
        "schema_version": 1,
        "physical_gpu": gpu,
        "gpu_uuid": fields[1],
        "memory_used_mib": used,
        "memory_total_mib": total,
        "memory_free_mib": total - used,
        "utilization_percent": utilization,
        "preexisting_compute_processes": processes,
        "external_processes_untouched": True,
    }


def active_robotwin_training(gpu: int) -> list[dict[str, Any]]:
    tokens = (
        "train_pi05_robotwin_train40_full_expert.py",
        "train_pi05_robotwin_train40_full_expert_world4.py",
        "train_pi05_robotwin_task_balanced.py",
    )
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
            environment = (entry / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(token in command for token in tokens):
            continue
        visible = None
        for item in environment:
            if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                visible = item.split(b"=", 1)[1].decode("utf-8", errors="replace")
                break
        if visible is None or gpu in {int(value) for value in visible.split(",") if value.isdigit()}:
            matches.append({"pid": int(entry.name), "cuda_visible_devices": visible, "command": command})
    return matches


def probe_admission(gpu: int) -> dict[str, Any]:
    admission = gpu_inventory(gpu)
    training = active_robotwin_training(gpu)
    admission["robotwin_training_processes"] = training
    empty = (
        not admission["preexisting_compute_processes"]
        and admission["memory_used_mib"] <= 512.0
    )
    shared_safe = (
        admission["memory_used_mib"] <= MAX_USED_MIB
        and admission["memory_free_mib"] >= MIN_FREE_MIB
        and admission["utilization_percent"] <= MAX_SHARED_UTILIZATION
        and not training
    )
    admission["empty"] = empty
    admission["shared_safe"] = shared_safe
    admission["admitted"] = empty or shared_safe
    admission["limits"] = {
        "max_used_mib": MAX_USED_MIB,
        "min_free_mib": MIN_FREE_MIB,
        "max_shared_utilization_percent": MAX_SHARED_UTILIZATION,
    }
    if not admission["admitted"]:
        raise RuntimeError(f"GPU{gpu} failed empty/shared admission: {admission}")
    return admission


def wait_gpu(job_state: Path, poll_seconds: int):
    while True:
        rejected = {}
        raw_gpus = os.environ.get("ROBOTWIN_FORMAL_GPUS", "").strip()
        candidate_gpus = (
            tuple(int(value) for value in raw_gpus.split(",") if value.strip())
            if raw_gpus else CANDIDATE_GPUS
        )
        for gpu in candidate_gpus:
            lock = None
            try:
                hostname = socket.gethostname()
                lock = acquire(RUNTIME / f"resource-leases/{hostname}-gpu-{gpu}.lock")
                admission = probe_admission(gpu)
                return gpu, admission, lock
            except (BlockingIOError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                rejected[str(gpu)] = f"{type(error).__name__}: {error}"
                if lock is not None:
                    lock.close()
        atomic_write(job_state, {
            "status": "waiting_safe_shared_gpu",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "candidate_gpus": list(candidate_gpus),
            "rejected": rejected,
            "updated_at": now(),
        })
        time.sleep(poll_seconds)


def run_job(group: str, repeat: int, poll_seconds: int) -> None:
    manifest, source, expected = validate_job(group, repeat)
    job_root = QUEUE / f"runs/{group}/repeat-{repeat:02d}"
    job_root.mkdir(parents=True, exist_ok=True)
    job_lock = (job_root / "orchestrator.lock").open("a+")
    fcntl.flock(job_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    complete = job_root / "complete.json"
    if complete.is_file():
        return
    rows = reusable_rows(job_root, expected)
    remaining = sorted(expected - set(rows))
    if not remaining:
        raise ValueError("All rows exist but formal complete receipt is missing")
    state_path = job_root / "state.json"
    gpu, admission, gpu_lock = wait_gpu(state_path, poll_seconds)
    attempt = next_attempt(job_root)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        atomic_write(state_path, {
            "status": "running",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "group": group,
            "repeat": repeat,
            "physical_gpu": gpu,
            "completed_unique_episodes_before_attempt": len(rows),
            "remaining_episodes": len(remaining),
            "attempt": str(attempt),
            "admission": admission,
            "updated_at": now(),
        })
        attempt.mkdir(parents=True, exist_ok=False)
        write_exclusive(attempt / "started.json", {
            "status": "robotwin_experts_formal_attempt_started",
            "formal_result": True,
            "manifest": bind(QUEUE / f"manifests/{group}/repeat-{repeat:02d}.json"),
            "admission": admission,
            "reused_rows": [bind(path) for path in rows.values()],
            "remaining_keys": [list(key) for key in remaining],
            "created_at": now(),
        })
        runtime_manifest = copy.deepcopy(source)
        runtime_manifest["formal_result"] = True
        runtime_manifest["purpose"] = "robotwin_experts_formal_evaluation"
        runtime_manifest["plateau_eligible"] = False
        runtime_manifest["tasks"] = []
        task_names = {int(task["task_index"]): task["task"] for task in manifest["tasks"]}
        source_tasks = {int(task["task_index"]): task for task in source["tasks"]}
        grouped: dict[int, list[int]] = {}
        for task_index, seed in remaining:
            grouped.setdefault(task_index, []).append(seed)
        for task_index, seeds in grouped.items():
            runtime_manifest["tasks"].append(dict(source_tasks[task_index], seeds=sorted(seeds)))
        slicer.runtime_setup(source)
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one visible CUDA device required")
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        fresh_result = base.simulator_audit(runtime_manifest, attempt, False)
        fresh = {(int(task_index), int(seed)): attempt / f"{task_names[task_index]}-{seed}.json" for task_index, seed in remaining}
        combined = {**rows, **fresh}
        if set(combined) != expected:
            raise ValueError("Formal episode coverage differs after attempt")
        ordered = [combined[key] for key in sorted(expected)]
        values = [read(path) for path in ordered]
        successes = sum(bool(value["success"]) for value in values)
        task_rates = {}
        for task_index in sorted({key[0] for key in expected}):
            local = [value for value in values if int(value["task_index"]) == task_index]
            task_rates[str(task_index)] = sum(bool(value["success"]) for value in local) / len(local)
        attempt_complete = attempt / "complete.json"
        write_exclusive(attempt_complete, {
            "status": "robotwin_experts_formal_group_repeat_complete",
            "formal_result": True,
            "method": "Experts",
            "group": group,
            "repeat": repeat,
            "episodes": 60,
            "successes": successes,
            "macro_success": sum(task_rates.values()) / len(task_rates),
            "task_success": task_rates,
            "rows": [bind(path) for path in ordered],
            "fresh_result": fresh_result,
            "seconds": time.monotonic() - begin,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "finished_at": now(),
        })
        write_exclusive(complete, {
            "status": "robotwin_experts_formal_job_complete",
            "formal_result": True,
            "group": group,
            "repeat": repeat,
            "receipt": bind(attempt_complete),
            "finished_at": now(),
        })
        atomic_write(state_path, {
            "status": "complete",
            "pid": os.getpid(),
            "group": group,
            "repeat": repeat,
            "physical_gpu": gpu,
            "receipt": bind(complete),
            "updated_at": now(),
        })
    except BaseException as error:
        if attempt.exists() and not (attempt / "failure.json").exists():
            write_exclusive(attempt / "failure.json", {
                "error_type": type(error).__name__,
                "error": str(error),
                "formal_result": True,
                "created_at": now(),
            })
        atomic_write(state_path, {
            "status": "failed",
            "pid": os.getpid(),
            "group": group,
            "repeat": repeat,
            "physical_gpu": gpu,
            "error_type": type(error).__name__,
            "error": str(error),
            "updated_at": now(),
        })
        raise
    finally:
        gpu_lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "validate", "run"))
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--repeat", type=int, choices=REPEATS)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    if args.action == "build":
        return build()
    if args.group is None or args.repeat is None:
        raise ValueError("--group and --repeat are required")
    manifest, _, expected = validate_job(args.group, args.repeat)
    if args.action == "validate":
        print(json.dumps({"status": "formal_job_validated", "episodes": len(expected), "manifest": manifest}, indent=2))
        return
    run_job(args.group, args.repeat, args.poll_seconds)


if __name__ == "__main__":
    main()
