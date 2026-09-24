#!/usr/bin/env python3
"""Frozen RoboTwin native-execution A/B capture contract for M=3 TCR."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
VLA = HERE.parents[1]
WORK = VLA.parent
RUNTIME = WORK / "vla-merge-runtime"
TABLE5 = RUNTIME / "experiments/iclr2027-table5-20260910"
FORMAL = TABLE5 / "evaluation-queues/robotwin-three-expert-formal-v2"
CAPTURE_ROOT = RUNTIME / "experiments/claude-robotwin-tcr-20260923"
GROUPS = ("coordination", "receptacle", "precision")
REPEATS = (1, 2, 3)
POOLS = ("A", "B")
FLOW_INDICES = (0, 5, 9)
REQUESTS_PER_TASK = 5
SEED_NAMESPACE = "iclr2027-table5-robotwin-tcr-calibration-v1"
PARENTS = {
    "coordination": TABLE5 / "expert-training/robotwin-three-expert-15k-sequence-v1/coordination/evaluation-015000/parent.json",
    "receptacle": TABLE5 / "expert-training/robotwin-three-expert-15k-sequence-v1/receptacle/evaluation-015000/parent.json",
    "precision": TABLE5 / "evaluation-queues/robotwin-precision-015000-quick20-v1/parent.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def quantile_request_indices(request_count: int) -> list[int]:
    """Five deterministic full-trajectory request positions, including endpoints."""
    if type(request_count) is not int or request_count < REQUESTS_PER_TASK:
        raise ValueError("A complete task trajectory must expose at least five native requests")
    values = [(request_count - 1) * numerator // 4 for numerator in range(5)]
    if len(set(values)) != REQUESTS_PER_TASK:
        raise AssertionError(f"Quantile request indices are not distinct: {values}")
    return values


def select_request_quantiles(records: list[Any], metadata: list[dict[str, Any]]) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    if len(records) != len(metadata):
        raise ValueError("Record/metadata lengths differ")
    by_request: dict[int, list[int]] = {}
    for index, row in enumerate(metadata):
        request = row.get("request_index")
        flow = row.get("flow_index")
        if type(request) is not int or type(flow) is not int:
            raise ValueError("Capture metadata lacks integer request/flow identity")
        by_request.setdefault(request, []).append(index)
    observed = sorted(by_request)
    if observed != list(range(len(observed))):
        raise ValueError("Native request indices are not a contiguous full trajectory")
    chosen = quantile_request_indices(len(observed))
    slots: list[int] = []
    for request in chosen:
        indices = sorted(by_request[request], key=lambda value: metadata[value]["flow_index"])
        if [metadata[value]["flow_index"] for value in indices] != list(FLOW_INDICES):
            raise ValueError(f"Request {request} does not contain exact flow 0/5/9")
        slots.extend(indices)
    return (
        [records[index] for index in slots],
        [{**metadata[index], "selected_request_slot": ordinal // len(FLOW_INDICES)}
         for ordinal, index in enumerate(slots)],
        {"available_request_count": len(observed), "selected_request_indices": chosen},
    )


def uint31(*parts: object) -> int:
    token = "\0".join(map(str, parts))
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 2**31


def occupied_task_seeds() -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for bank_path in sorted((FORMAL / "reset-banks").glob("repeat-*.json")):
        for row in read(bank_path).get("entries", []):
            occupied.add((int(row["task_index"]), int(row["seed"])))
    # Historical rows are used only to prevent reset reuse.  No reward or success
    # value participates in seed selection.
    for path in TABLE5.rglob("*.json"):
        if CAPTURE_ROOT in path.parents:
            continue
        try:
            row = read(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if type(row.get("task_index")) is int and type(row.get("seed")) is int:
            occupied.add((row["task_index"], row["seed"]))
    return occupied


def derive_unique_seed(task_index: int, repeat: int, pool: str, occupied: set[tuple[int, int]]) -> tuple[int, int]:
    for counter in range(10_000):
        seed = uint31(SEED_NAMESPACE, "reset", repeat, pool, task_index, counter)
        if (task_index, seed) not in occupied:
            occupied.add((task_index, seed))
            return seed, counter
    raise RuntimeError("Could not derive an unused deterministic calibration seed")


def build_protocol(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    formal_protocol = read(FORMAL / "protocol.json")
    if formal_protocol.get("episodes_total") != 540 or formal_protocol.get("status") != "robotwin_three_expert_formal_protocol_frozen_amendment1":
        raise ValueError("RoboTwin formal-v2 protocol identity differs")
    occupied = occupied_task_seeds()
    occupied_before = len(occupied)
    jobs = []
    parents = {group: bind(path) for group, path in PARENTS.items()}
    for repeat in REPEATS:
        for pool in POOLS:
            flow_seed = uint31(SEED_NAMESPACE, "flow", repeat, pool)
            for group in GROUPS:
                parent = read(PARENTS[group])
                if parent.get("group") != group or len(parent.get("tasks", [])) != 10:
                    raise ValueError(f"Unexpected frozen parent for {group}")
                tasks = []
                for task in parent["tasks"]:
                    task_index = int(task["task_index"])
                    seed, counter = derive_unique_seed(task_index, repeat, pool, occupied)
                    tasks.append({"task": task["task"], "task_index": task_index,
                                  "seed": seed, "derivation_counter": counter})
                jobs.append({
                    "id": f"r{repeat:02d}-{pool}-{group}", "repeat": repeat, "pool": pool,
                    "group": group, "parent": parents[group], "checkpoint": parent["checkpoint"],
                    "flow_seed": flow_seed, "tasks": tasks, "episodes": 10,
                    "requests": 50, "flow_rows": 150,
                    "output": str(output / "jobs" / f"r{repeat:02d}-{pool}-{group}"),
                })
    if len(jobs) != 18 or sum(job["flow_rows"] for job in jobs) != 2700:
        raise AssertionError("RoboTwin A/B capture matrix coverage differs")
    keys = [(row["task_index"], row["seed"]) for job in jobs for row in job["tasks"]]
    if len(keys) != 180 or len(set(keys)) != 180:
        raise AssertionError("RoboTwin calibration reset keys are not unique")
    protocol = {
        "schema": "robotwin_tcr_native_ab_capture_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "three-expert-native-execution-calibration-only",
        "formal_result": False, "success_filtering": False,
        "seed_namespace": SEED_NAMESPACE,
        "seed_selection_uses_only_identity_collision": True,
        "historical_and_formal_keys_checked": occupied_before,
        "formal_protocol": bind(FORMAL / "protocol.json"), "parents": parents,
        "groups": list(GROUPS), "repeats": list(REPEATS), "pools": list(POOLS),
        "tasks_per_group_pool_repeat": 10, "requests_per_task": REQUESTS_PER_TASK,
        "request_selection": "full-trajectory quantiles 0,25,50,75,100 percent",
        "flow_indices": list(FLOW_INDICES), "episodes": 180, "requests": 900,
        "flow_rows": 2700, "jobs": jobs,
        "ab_disjoint_from_formal_and_historical_by_task_seed": True,
        "no_score_based_scheduling_or_stopping": True,
    }
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(
        sha256_file(output / "protocol.json") + "  protocol.json\n", encoding="utf-8"
    )
    return protocol

