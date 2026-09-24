#!/usr/bin/env python3
"""Run frozen RoboTwin Model Soups or TIES on the Experts formal-v2 reset bank."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path("/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime")
TABLE5 = RUNTIME / "experiments/iclr2027-table5-20260910"
EXPERT_QUEUE = TABLE5 / "evaluation-queues/robotwin-three-expert-formal-v2"
METHOD = os.environ.get("ROBOTWIN_FORMAL_MERGE_METHOD", "model_soups")
if METHOD not in {"model_soups", "ties"}:
    raise ValueError(f"unsupported formal merge method: {METHOD}")
METHOD_LABEL = {"model_soups": "Model Soups", "ties": "TIES"}[METHOD]
QUEUE = TABLE5 / f"evaluation-queues/robotwin-three-expert-table5-formal-methods-v1/evaluations/{METHOD}"
POLICY = TABLE5 / f"evaluation-queues/robotwin-three-expert-table5-formal-methods-v1/materialized/{METHOD}/pretrained_model"
GROUPS = ("coordination", "receptacle", "precision")
REPEATS = (1, 2, 3)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experts = load_module("robotwin_experts_formal", ROOT / "scripts/watch_robotwin_experts_formal_v2.py")


def method_manifest(group: str, repeat: int) -> Path:
    return QUEUE / f"manifests/{group}/repeat-{repeat:02d}.json"


def build() -> None:
    if METHOD == "model_soups":
        required = {
            "fusion_manifest": POLICY / "fusion_manifest.json",
            "adapter": POLICY / "adapter_model.safetensors",
            "adapter_config": POLICY / "adapter_config.json",
        }
    else:
        required = {
            "parameter_baseline_manifest": POLICY / "parameter_baseline_manifest.json",
            "dense_model": POLICY / "model.safetensors",
        }
    for path in (*required.values(), POLICY / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    bindings = {name: experts.bind(path) for name, path in required.items()}
    for group in GROUPS:
        for repeat in REPEATS:
            source, _, expected = experts.validate_job(group, repeat)
            target = method_manifest(group, repeat)
            target.parent.mkdir(parents=True, exist_ok=True)
            value = {
                "schema_version": 1,
                "status": f"robotwin_{METHOD}_formal_job_frozen",
                "formal_result": True,
                "method": METHOD_LABEL,
                "group": group,
                "repeat": repeat,
                "episodes": len(expected),
                "tasks": source["tasks"],
                "reset_bank": source["reset_bank"],
                "policy": bindings,
                "experts_protocol": experts.bind(EXPERT_QUEUE / "protocol.json"),
                "created_at": experts.now(),
            }
            if target.is_file():
                existing = experts.read(target)
                existing.pop("created_at", None)
                comparable = dict(value)
                comparable.pop("created_at", None)
                if existing != comparable:
                    raise ValueError(f"existing method manifest differs: {target}")
            else:
                experts.write_exclusive(target, value)
    protocol = QUEUE / "protocol.json"
    document = {
        "schema_version": 1,
        "status": f"robotwin_{METHOD}_formal_protocol_frozen",
        "method": METHOD_LABEL,
        "jobs": [experts.bind(method_manifest(group, repeat)) for group in GROUPS for repeat in REPEATS],
        "policy": bindings,
        "experts_protocol": experts.bind(EXPERT_QUEUE / "protocol.json"),
        "created_at": experts.now(),
    }
    if not protocol.is_file():
        experts.write_exclusive(protocol, document)


def validate_job(group: str, repeat: int) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    manifest = experts.read(method_manifest(group, repeat))
    if manifest.get("method") != METHOD_LABEL or manifest.get("formal_result") is not True:
        raise ValueError(f"not a frozen {METHOD_LABEL} job")
    expert_manifest, source, expected = experts.validate_job(group, repeat)
    if manifest["tasks"] != expert_manifest["tasks"] or len(expected) != 60:
        raise ValueError(f"{METHOD_LABEL} task/reset panel differs from Experts")
    for binding in manifest["policy"].values():
        if experts.bind(Path(binding["path"])) != binding:
            raise ValueError(f"{METHOD_LABEL} policy binding changed")
    runtime = copy.deepcopy(source)
    runtime["checkpoint"] = str(POLICY.parent)
    runtime["formal_result"] = True
    runtime["purpose"] = f"robotwin_{METHOD}_formal_evaluation"
    runtime["plateau_eligible"] = False
    return runtime, expected


def run_job(group: str, repeat: int, poll_seconds: int) -> None:
    runtime, expected = validate_job(group, repeat)
    job_root = QUEUE / f"runs/{group}/repeat-{repeat:02d}"
    job_root.mkdir(parents=True, exist_ok=True)
    lock = experts.acquire(job_root / "orchestrator.lock")
    complete = job_root / "complete.json"
    if complete.is_file():
        return
    rows = experts.reusable_rows(job_root, expected)
    remaining = sorted(expected - set(rows))
    if not remaining:
        raise ValueError("all rows exist without complete receipt")
    state = job_root / "state.json"
    gpu, admission, gpu_lock = experts.wait_gpu(state, poll_seconds)
    attempt = experts.next_attempt(job_root)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        experts.atomic_write(state, {"status": "running", "method": METHOD_LABEL, "group": group, "repeat": repeat, "gpu": gpu, "remaining": len(remaining), "updated_at": experts.now()})
        attempt.mkdir(parents=True, exist_ok=False)
        experts.write_exclusive(attempt / "started.json", {"status": f"robotwin_{METHOD}_formal_attempt_started", "manifest": experts.bind(method_manifest(group, repeat)), "admission": admission, "remaining_keys": [list(key) for key in remaining], "created_at": experts.now()})
        task_names = {int(task["task_index"]): task["task"] for task in runtime["tasks"]}
        source_tasks = {int(task["task_index"]): task for task in runtime["tasks"]}
        grouped: dict[int, list[int]] = {}
        for task_index, seed in remaining:
            grouped.setdefault(task_index, []).append(seed)
        runtime["tasks"] = [dict(source_tasks[index], seeds=sorted(seeds)) for index, seeds in grouped.items()]
        experts.slicer.runtime_setup(runtime)
        import torch
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        fresh_result = experts.base.simulator_audit(runtime, attempt, False)
        fresh = {(index, seed): attempt / f"{task_names[index]}-{seed}.json" for index, seed in remaining}
        combined = {**rows, **fresh}
        if set(combined) != expected:
            raise ValueError("formal episode coverage differs")
        ordered = [combined[key] for key in sorted(expected)]
        values = [experts.read(path) for path in ordered]
        successes = sum(bool(value["success"]) for value in values)
        task_rates = {str(index): sum(bool(value["success"]) for value in values if int(value["task_index"]) == index) / 6 for index in sorted({key[0] for key in expected})}
        attempt_complete = attempt / "complete.json"
        experts.write_exclusive(attempt_complete, {"status": f"robotwin_{METHOD}_formal_group_repeat_complete", "formal_result": True, "method": METHOD_LABEL, "group": group, "repeat": repeat, "episodes": 60, "successes": successes, "macro_success": sum(task_rates.values()) / len(task_rates), "task_success": task_rates, "rows": [experts.bind(path) for path in ordered], "fresh_result": fresh_result, "seconds": time.monotonic() - begin, "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2, "finished_at": experts.now()})
        experts.write_exclusive(complete, {"status": f"robotwin_{METHOD}_formal_job_complete", "formal_result": True, "method": METHOD_LABEL, "group": group, "repeat": repeat, "receipt": experts.bind(attempt_complete), "finished_at": experts.now()})
        experts.atomic_write(state, {"status": "complete", "method": METHOD_LABEL, "group": group, "repeat": repeat, "receipt": experts.bind(complete), "updated_at": experts.now()})
    except BaseException as error:
        if attempt.exists() and not (attempt / "failure.json").exists():
            experts.write_exclusive(attempt / "failure.json", {"error_type": type(error).__name__, "error": str(error), "formal_result": True, "created_at": experts.now()})
        experts.atomic_write(state, {"status": "failed", "method": METHOD_LABEL, "group": group, "repeat": repeat, "error_type": type(error).__name__, "error": str(error), "updated_at": experts.now()})
        raise
    finally:
        gpu_lock.close()
        lock.close()


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
    if args.action == "validate":
        runtime, expected = validate_job(args.group, args.repeat)
        print(json.dumps({"status": "valid", "episodes": len(expected), "checkpoint": runtime["checkpoint"]}))
        return
    run_job(args.group, args.repeat, args.poll_seconds)


if __name__ == "__main__":
    main()
