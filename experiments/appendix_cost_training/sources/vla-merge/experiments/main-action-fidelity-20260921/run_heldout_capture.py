#!/usr/bin/env python3
"""Prepare and dynamically dispatch the frozen C146 held-out request bank."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
BASE_PATH = WORK / "vla-merge/experiments/tcr-mainline-local-20260919/mainline_capture.py"
FLOCK_DIR = WORK / "vla-merge/experiments/claude-firstpass-cause-20260920"
COLLECTOR = HERE / "collect_heldout_requests.py"
DEFAULT_RUN = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/heldout-attempt-01"
SELECTION_ROOT = WORK / "vla-merge-runtime/experiments/iclr2027-table1-20260910/reset-banks/libero-procedural-clean-v1/selections"
CANDIDATE_GPUS = tuple(range(8))
MAX_ACTIVE = 4
MIN_FREE_MIB = 48 * 1024
RESERVE_MIB = 12 * 1024
EPISODES = (
    {"repeat_id": "heldout-D1", "start_seed": 331001, "flow_seed": 332001, "init_offset": 40},
    {"repeat_id": "heldout-D2", "start_seed": 331002, "flow_seed": 332002, "init_offset": 41},
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("main_fidelity_base_capture", BASE_PATH)
sys.path.insert(0, str(FLOCK_DIR))
import card_flock  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def heldout_indices() -> set[int]:
    result = set()
    for repeat in range(1, 4):
        payload = json.loads((SELECTION_ROOT / f"repeat-{repeat:02}.json").read_text())
        result.update(int(index) for values in payload["tasks"].values() for index in values)
    if result != set(range(30)):
        raise ValueError(f"Formal state identity changed: {sorted(result)}")
    return result


def existing_offsets_and_collisions() -> tuple[set[int], list[dict[str, Any]]]:
    offsets, collisions = set(), []
    wanted = {(e["start_seed"], e["flow_seed"], e["init_offset"]) for e in EPISODES}
    for path in (WORK / "vla-merge-runtime/experiments").rglob("replay.json"):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        offset = value.get("calibration_init_state_offset", value.get("init_state_offset"))
        if offset is not None:
            offsets.add(int(offset))
        identity = (value.get("calibration_start_seed", value.get("start_seed")),
                    value.get("generation_noise_seed", value.get("flow_seed")), offset)
        if identity in wanted:
            collisions.append({"path": str(path.resolve()), "identity": list(identity)})
    return offsets, collisions


def configure_base() -> None:
    base.LOCAL_GPUS = CANDIDATE_GPUS
    base.MAX_SLOTS_PER_GPU = 1
    base.MIN_FREE_MIB = MIN_FREE_MIB
    base.RESERVE_MIB = RESERVE_MIB
    base.REPEATS = EPISODES
    base.FLOW_INDICES = (0,)
    base.COLLECTOR = COLLECTOR


def build_jobs(run: Path, bank: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = base.build_jobs(run, bank)
    for job in jobs:
        job["id"] = job["id"].replace("collect-", "heldout-")
        job["kind"] = "heldout_request_capture"
        job["environment"].update({
            "PI05_BLOCK_REGMEANPP_FLOW_INDICES": "0",
            "PI05_MAIN_FIDELITY_MEMORY_FRACTION": "0.30",
            "PI05_MAIN_FIDELITY_EPISODE_ID": job["repeat_id"],
            "PI05_MAIN_FIDELITY_EXPERT_NAME": job["expert_name"],
            "PI05_MAIN_FIDELITY_POLICY_SHA256": job["policy_sha256"],
            "PI05_MAIN_FIDELITY_DENSE_BANK_SHA256": job["dense_bank_sha256"],
            "PI05_MAIN_FIDELITY_START_SEED": str(job["start_seed"]),
            "PI05_MAIN_FIDELITY_FLOW_SEED": str(job["flow_seed"]),
            "PI05_MAIN_FIDELITY_INIT_STATE_OFFSET": str(job["init_state_offset"]),
        })
        job["command"][2] = str(COLLECTOR)
    return jobs


def verify_manifest(path: Path, job: dict[str, Any]) -> dict[str, Any]:
    tensor = path.with_suffix(".safetensors")
    if not path.is_file() or not tensor.is_file():
        raise ValueError(f"Missing held-out artifact for {job['id']}")
    value = json.loads(path.read_text())
    expected = {
        "main_fidelity_capture_version": 1,
        "source_kind": "heldout_expert_execution",
        "heldout_only": True, "forbidden_as_calibration": True,
        "sample_count": 50, "prompt_count": 10, "flow_indices": [0],
        "episode_id": job["repeat_id"], "expert_name": job["expert_name"],
        "capture_start_seed": job["start_seed"],
        "generation_noise_seed": job["flow_seed"],
        "capture_init_state_offset": job["init_state_offset"],
        "calibration_policy_sha256": job["policy_sha256"],
        "dense_expert_bank_sha256": job["dense_bank_sha256"],
        "failures_retained": True,
    }
    problems = [f"{key}={value.get(key)!r} expected {wanted!r}"
                for key, wanted in expected.items() if value.get(key) != wanted]
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in value.get("samples", []):
        grouped.setdefault(row.get("prompt_signature"), []).append(row)
        if row.get("flow_index") != 0 or row.get("init_state_id") != job["init_state_offset"]:
            problems.append("sample flow/state identity mismatch")
    if len(grouped) != 10:
        problems.append("expected ten task prompts")
    for rows in grouped.values():
        if sorted(r.get("selected_request_slot") for r in rows) != list(range(5)):
            problems.append("task lacks five fixed request slots")
        if len({r.get("request_index") for r in rows}) != 5:
            problems.append("task request indices are not unique")
        if len({r.get("initial_observation_sha256") for r in rows}) != 1:
            problems.append("one episode does not share an initial observation")
    if problems:
        raise ValueError(f"{job['id']} held-out audit failed: {'; '.join(problems)}")
    return {"job_id": job["id"], "samples": 50, "tasks": 10,
            "manifest_sha256": sha256_file(path), "tensor_sha256": sha256_file(tensor),
            "heldout_only": True, "forbidden_as_calibration": True}


def prepare(run: Path, max_runtime_hours: float) -> dict[str, Any]:
    run = run.resolve()
    if run.exists():
        raise FileExistsError(run)
    formal = heldout_indices()
    existing, collisions = existing_offsets_and_collisions()
    new_offsets = {e["init_offset"] for e in EPISODES}
    if formal & new_offsets or existing & new_offsets or collisions:
        raise ValueError({"formal_overlap": sorted(formal & new_offsets),
                          "calibration_overlap": sorted(existing & new_offsets),
                          "identity_collisions": collisions})
    configure_base()
    bank = base.load_bank()
    jobs = build_jobs(run, bank)
    sources = dict(base.source_hashes())
    for path in (Path(__file__).resolve(), COLLECTOR.resolve(), (FLOCK_DIR / "card_flock.py").resolve()):
        sources[str(path)] = sha256_file(path)
    plan = {
        "schema": "main_action_fidelity_heldout_capture_v1",
        "status": "prepared_not_launched", "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "run": str(run),
        "candidate_gpus": list(CANDIDATE_GPUS), "max_active": MAX_ACTIVE,
        "max_workers_per_gpu": 1, "min_free_mib": MIN_FREE_MIB,
        "runtime_floor_mib": RESERVE_MIB, "no_retry": True,
        "dynamic_selection": "highest current free memory among admissible UUID-locked cards",
        "episodes": list(EPISODES), "formal_state_indices": sorted(formal),
        "known_calibration_state_indices": sorted(existing),
        "heldout_state_indices": sorted(new_offsets), "all_known_sources_disjoint": True,
        "requests": {"tasks": 40, "episodes_per_task": 2, "requests_per_episode": 5,
                     "total": 400, "flow_index": 0, "success_filter": False},
        "usage": "heldout action-fidelity only; forbidden for calibration, tuning, or model selection",
        "expert_bank": bank, "source_hashes": sources, "jobs": jobs,
        "deadline_unix": time.time() + max_runtime_hours * 3600,
    }
    run.mkdir(parents=True)
    write_json(run / "plan.json", plan)
    write_json(run / "identity-audit.json", {
        "formal_state_indices": sorted(formal), "known_calibration_state_indices": sorted(existing),
        "heldout_state_indices": sorted(new_offsets), "all_disjoint": True,
        "identity_collisions": collisions, "usage": plan["usage"],
    })
    return plan


def gpu_rows():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, free, used, util = [item.strip() for item in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "free_mib": int(float(free)),
                     "used_mib": int(float(used)), "utilization_gpu_percent": int(float(util))})
    return sorted(rows, key=lambda row: (-row["free_mib"], row["index"]))


def launch(run: Path, poll_seconds: float):
    run = run.resolve()
    plan = json.loads((run / "plan.json").read_text())
    if plan.get("schema") != "main_action_fidelity_heldout_capture_v1" or plan.get("hostname") != socket.gethostname():
        raise ValueError("Wrong plan identity/host")
    if {path: sha256_file(Path(path)) for path in plan["source_hashes"]} != plan["source_hashes"]:
        raise ValueError("Pinned held-out source changed")
    configure_base()
    base.verify_manifest = lambda path, job: verify_manifest(
        Path(job["output"]) / "heldout/requests.json", job)
    uuids, leases, selected = card_flock.gpu_uuids(), [], []
    snapshot = gpu_rows()
    try:
        for row in snapshot:
            gpu = row["index"]
            if row["free_mib"] < MIN_FREE_MIB or uuids.get(gpu) != row["uuid"]:
                continue
            lease = card_flock.take_card(gpu, row["uuid"], "main-fidelity-heldout", "capture")
            if lease is None:
                continue
            leases.append(lease); selected.append(gpu)
            if len(selected) == MAX_ACTIVE:
                break
        write_json(run / "dynamic-card-selection.json", {
            "created_at": datetime.now(timezone.utc).isoformat(), "snapshot": snapshot,
            "selected_gpus": selected, "required_free_mib": MIN_FREE_MIB,
            "max_active": MAX_ACTIVE, "one_worker_per_card": True,
        })
        if not selected:
            raise RuntimeError("No admissible lockable GPU is available")
        base.LOCAL_GPUS = tuple(selected)
        result = base.run_queue(plan, gpus=tuple(selected), poll_seconds=poll_seconds,
                                max_slots_per_gpu=1)
        result["dynamically_selected_gpus"] = selected
        write_json(run / "queue-ended.json", result)
        return result
    finally:
        for lease in reversed(leases):
            lease.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--max-runtime-hours", type=float, default=6.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.prepare == args.launch:
        raise SystemExit("Choose exactly one of --prepare/--launch")
    result = prepare(args.run, args.max_runtime_hours) if args.prepare else launch(args.run, args.poll_seconds)
    print(json.dumps(result if args.launch else {"prepared": result["run"], "jobs": len(result["jobs"])}, indent=2))


if __name__ == "__main__":
    main()
