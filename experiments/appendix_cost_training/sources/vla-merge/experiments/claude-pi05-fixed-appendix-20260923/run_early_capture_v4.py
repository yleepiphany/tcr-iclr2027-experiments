#!/usr/bin/env python3
"""Collect the paired early-request A/B cache for one fixed appendix control.

The scientific identities are fixed before launch.  Resource placement is not:
at launch the dispatcher sorts all eight authorized cards by current free memory,
takes UUID-keyed flock leases on up to four admissible cards, and runs one worker
per selected card.  It never waits for a preferred GPU while another admissible
card is available.
"""
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
BASE_PATH = WORK / "vla-merge/experiments/tcr-mainline-local-20260919/early_capture_queue_v2.py"
FLOCK_DIR = WORK / "vla-merge/experiments/claude-firstpass-cause-20260920"
DEFAULT_RUN = WORK / "vla-merge-runtime/experiments/claude-pi05-fixed-appendix-20260923/early-capture-attempt-04"
SELECTION_ROOT = WORK / "vla-merge-runtime/experiments/iclr2027-table1-20260910/reset-banks/libero-procedural-clean-v1/selections"
PRIOR_CAPTURE = WORK / "vla-merge-runtime/experiments/claude-abc-20260920/attempt-01"

CANDIDATE_GPUS = tuple(range(8))
MAX_ACTIVE = 2
MAX_FREE_MIB_FOR_EARLY = 71_000
MIN_FREE_MIB = 48 * 1024
RESERVE_MIB = 12 * 1024
POOLS = (
    {"repeat_id": "A_early", "start_seed": 321001, "flow_seed": 322001, "init_offset": 30},
    {"repeat_id": "B_early", "start_seed": 321002, "flow_seed": 322002, "init_offset": 31},
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(BASE_PATH.parent))
base = _load("appendix_early_capture_backend", BASE_PATH)
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


def formal_state_indices() -> tuple[set[int], dict[str, Any]]:
    all_indices: set[int] = set()
    records: dict[str, Any] = {}
    for repeat in range(1, 4):
        path = SELECTION_ROOT / f"repeat-{repeat:02}.json"
        payload = json.loads(path.read_text())
        indices = {int(index) for values in payload["tasks"].values() for index in values}
        expected = set(range((repeat - 1) * 10, repeat * 10))
        if indices != expected:
            raise ValueError(f"Unexpected formal state set in {path}: {sorted(indices)}")
        all_indices.update(indices)
        records[str(path.resolve())] = {
            "sha256": sha256_file(path), "state_indices": sorted(indices),
            "eval_seed": payload["eval_seed"], "repeat_id": payload["repeat_id"],
        }
    return all_indices, records


def existing_identity_collisions() -> list[dict[str, Any]]:
    wanted = {(p["start_seed"], p["flow_seed"], p["init_offset"]) for p in POOLS}
    collisions: list[dict[str, Any]] = []
    root = WORK / "vla-merge-runtime/experiments"
    for path in root.rglob("replay.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        identity = (
            payload.get("calibration_start_seed", payload.get("start_seed")),
            payload.get("generation_noise_seed", payload.get("flow_seed")),
            payload.get("calibration_init_state_offset", payload.get("init_state_offset")),
        )
        if identity in wanted:
            collisions.append({"path": str(path.resolve()), "identity": list(identity)})
    return collisions


def paired_across_evidence() -> list[dict[str, Any]]:
    audit_path = WORK / "coordination/2026-09-23/appendix-abc-pools-content-audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("accepted") is not True or audit.get("jobs") != 12:
        raise ValueError("Current A/B/C source content audit is not accepted")
    approved = [row for row in audit["accepted_jobs"] if row["pool"] in ("A_new", "B_new")]
    if len(approved) != 8:
        raise ValueError("Exact eight A/B across sources are required")
    old = WORK / "vla-merge-runtime/experiments/claude-cache-sequence-budget-20260921/pools-attempt-01"
    evidence = []
    for row in approved:
        root = old / row["pool"] / "inputs" / row["suite"] / "across"
        manifest, tensor = root / "replay.json", root / "replay.safetensors"
        if sha256_file(manifest) != row["manifest_sha256"] or \
                sha256_file(tensor) != row["tensor_sha256"]:
            raise ValueError(f"Audited paired across source changed: {root}")
        payload = json.loads(manifest.read_text())
        entry = next(item for item in POOLS if item["start_seed"] == payload.get("calibration_start_seed"))
        if (payload.get("repeat_id") != row["pool"] or
                payload.get("generation_noise_seed") != entry["flow_seed"] or
                payload.get("calibration_init_state_offset") != entry["init_offset"]):
            raise ValueError(f"Paired across source reset/noise differs: {root}")
        evidence.append({"pool": row["pool"], "suite": row["suite"],
                         "manifest": str(manifest), "manifest_sha256": row["manifest_sha256"],
                         "tensor_sha256": row["tensor_sha256"]})
    collisions = existing_identity_collisions()
    expected = {str(Path(row["manifest"]).resolve()) for row in evidence}
    failed_run = WORK / "vla-merge-runtime/experiments/claude-pi05-fixed-appendix-20260923/early-capture-attempt-03"
    failed_plan = json.loads((failed_run / "plan.json").read_text())
    failed_terminal = json.loads((failed_run / "queue-ended.json").read_text())
    failed_job = next(item for item in failed_plan["jobs"]
                      if item["id"] == "collect-A_early-spatial")
    failed_manifest = failed_run / "A_early/inputs/spatial/early/replay.json"
    failed_exit = json.loads((failed_run / "A_early/inputs/spatial/exit.json").read_text())
    if (failed_terminal.get("status") != "stopped"
            or failed_terminal.get("states", {}).get("collect-A_early-spatial", {}).get("error")
            != "No module named 'audit_early_requests'"
            or failed_exit.get("return_code") != 0):
        raise ValueError("The extra early source is not the registered failed-audit capture")
    recovered = base.verify_manifest(failed_manifest, failed_job)
    if recovered["sample_count"] != 150 or recovered["request_provenance_audited"] is not True:
        raise ValueError("Failed-audit capture has invalid raw replay")
    expected.add(str(failed_manifest.resolve()))
    observed = {str(Path(row["path"]).resolve()) for row in collisions}
    if observed != expected or len(collisions) != 9:
        raise ValueError(f"Unexpected source-identity collision: {collisions}")
    return evidence


def prior_capture_evidence() -> dict[str, Any]:
    queue = PRIOR_CAPTURE / "queue-ended.json"
    plan = PRIOR_CAPTURE / "plan.json"
    q = json.loads(queue.read_text())
    p = json.loads(plan.read_text())
    verified = sorted(PRIOR_CAPTURE.rglob("verified.json"))
    if q.get("status") != "complete" or q.get("stopped") is not False or len(verified) != 4:
        raise ValueError("Prior exact-collector evidence is not a healthy four-suite terminal run")
    receipts = [json.loads(path.read_text()) for path in verified]
    if any(r.get("sample_count") != 150 or r.get("request_provenance_audited") is not True for r in receipts):
        raise ValueError("Prior exact-collector receipts do not pass the expected gate")
    return {
        "role": "stronger-than-small-smoke evidence for the identical collector and expert bank",
        "run": str(PRIOR_CAPTURE.resolve()),
        "queue_sha256": sha256_file(queue),
        "plan_sha256": sha256_file(plan),
        "terminal_status": q["status"],
        "stopped": q["stopped"],
        "four_verified_suite_captures": True,
        "samples_per_suite": 150,
        "elapsed_seconds": q.get("elapsed_seconds"),
        "admission_free_mib": p.get("min_free_mib"),
        "runtime_reserve_mib": p.get("reserve_mib"),
    }


def configure_base() -> None:
    base.LOCAL_GPUS = CANDIDATE_GPUS
    base.MAX_SLOTS_PER_GPU = 1
    base.MIN_FREE_MIB = MIN_FREE_MIB
    base.RESERVE_MIB = RESERVE_MIB
    base.REPEATS = POOLS


def prepare(run: Path, max_runtime_hours: float) -> dict[str, Any]:
    run = run.resolve()
    if run.exists():
        raise FileExistsError(run)
    protected, selections = formal_state_indices()
    pool_offsets = {p["init_offset"] for p in POOLS}
    if protected & pool_offsets:
        raise ValueError("Calibration pool overlaps the formal held-out state indices")
    paired = paired_across_evidence()
    collisions = [{"path": str((WORK / "vla-merge-runtime/experiments/claude-pi05-fixed-appendix-20260923/early-capture-attempt-03/A_early/inputs/spatial/early/replay.json").resolve()),
                   "reason": "attempt-03 GPU child exited zero, but queue audit import failed; raw replay is independently validated and retained"}]
    prior = prior_capture_evidence()
    configure_base()
    bank = base.load_bank()
    jobs = base.build_jobs(run, bank)
    if len(jobs) != 8:
        raise AssertionError(f"Expected 8 jobs, got {len(jobs)}")
    sources = dict(base.source_hashes())
    sources[str(Path(__file__).resolve())] = sha256_file(Path(__file__).resolve())
    sources[str(BASE_PATH.resolve())] = sha256_file(BASE_PATH)
    sources[str(base.COLLECTOR.resolve())] = sha256_file(base.COLLECTOR)
    sources[str((FLOCK_DIR / "card_flock.py").resolve())] = sha256_file(FLOCK_DIR / "card_flock.py")
    plan = {
        "schema": "pi05_appendix_early_capture_v1",
        "status": "prepared_not_launched",
        "technical_recovery_of": {
            "attempt": "early-capture-attempt-03",
            "reason": "first job wrote complete replay, then backend audit could not import audit_early_requests",
            "old_terminal": str((WORK / "vla-merge-runtime/experiments/claude-pi05-fixed-appendix-20260923/early-capture-attempt-03/queue-ended.json").resolve()),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "run": str(run),
        "candidate_gpus": list(CANDIDATE_GPUS),
        "dynamic_selection": "descending current free memory among admissible UUID-locked cards",
        "max_active": MAX_ACTIVE,
        "max_workers_per_gpu": 1,
        "min_free_mib": MIN_FREE_MIB,
        "max_free_mib_for_early": MAX_FREE_MIB_FOR_EARLY,
        "reserve_mib": RESERVE_MIB,
        "no_automatic_retry": True,
        "collection_only": True,
        "pools": [dict(item) for item in POOLS],
        "heldout_selections": selections,
        "heldout_state_indices": sorted(protected),
        "calibration_state_indices": sorted(pool_offsets),
        "heldout_disjoint": True,
        "identity_collisions": collisions,
        "paired_across_sources": paired,
        "prior_smoke_evidence": prior,
        "expert_bank": bank,
        "source_hashes": sources,
        "jobs": jobs,
        "deadline_unix": time.time() + max_runtime_hours * 3600,
    }
    run.mkdir(parents=True)
    write_json(run / "plan.json", plan)
    write_json(run / "identity-audit.json", {
        "heldout_disjoint": True, "heldout_state_indices": sorted(protected),
        "calibration_state_indices": sorted(pool_offsets), "pools": list(POOLS),
        "exact_identity_collisions": collisions, "paired_across_sources": paired,
        "heldout_selections": selections,
    })
    write_json(run / "prior-smoke-evidence.json", prior)
    return plan


def gpu_rows() -> list[dict[str, Any]]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, free, used, util = [value.strip() for value in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "free_mib": int(float(free)),
                     "used_mib": int(float(used)), "utilization_gpu_percent": int(float(util))})
    return sorted(rows, key=lambda row: (-row["free_mib"], row["index"]))


def launch(run: Path, poll_seconds: float) -> dict[str, Any]:
    run = run.resolve()
    plan_path = run / "plan.json"
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "pi05_appendix_early_capture_v1":
        raise ValueError("Wrong plan schema")
    if plan.get("hostname") != socket.gethostname():
        raise ValueError("Plan belongs to another host")
    observed = {path: sha256_file(Path(path)) for path in plan["source_hashes"]}
    if observed != plan["source_hashes"]:
        raise ValueError("Pinned source changed after preparation")
    configure_base()
    uuids = card_flock.gpu_uuids()
    leases = []
    selected = []
    snapshot = gpu_rows()
    try:
        for row in snapshot:
            gpu = row["index"]
            if (gpu not in CANDIDATE_GPUS or row["free_mib"] < MIN_FREE_MIB
                    or row["free_mib"] >= MAX_FREE_MIB_FOR_EARLY):
                continue
            if uuids.get(gpu) != row["uuid"]:
                raise ValueError(f"GPU {gpu} UUID changed during selection")
            lease = card_flock.take_card(gpu, row["uuid"], "appendix-early-capture", "capture")
            if lease is None:
                continue
            legacy = card_flock._try_lock(
                WORK / "vla-merge-runtime/resource-leases" / socket.gethostname() / f"gpu-{gpu}.lock",
                "appendix-early-capture", {"gpu": gpu, "stage": "capture"})
            if legacy is None:
                lease.release()
                continue
            leases.append((lease, legacy))
            selected.append(gpu)
            if len(selected) == MAX_ACTIVE:
                break
        write_json(run / "dynamic-card-selection.json", {
            "created_at": datetime.now(timezone.utc).isoformat(), "snapshot": snapshot,
            "selected_gpus": selected, "required_free_mib": MIN_FREE_MIB,
            "max_active": MAX_ACTIVE, "one_worker_per_card": True,
            "dual_uuid_legacy_lease": True,
            "reserved_for_71000mib_builds": "cards with >=71000 MiB free are excluded",
        })
        if not selected:
            raise RuntimeError("No admissible and lockable GPU is currently available")
        base.LOCAL_GPUS = tuple(selected)
        result = base.run_queue(plan, gpus=tuple(selected), poll_seconds=poll_seconds,
                                max_slots_per_gpu=1)
        result["dynamically_selected_gpus"] = selected
        write_json(run / "queue-ended.json", result)
        return result
    finally:
        for card, legacy in reversed(leases):
            legacy.release()
            card.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--max-runtime-hours", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.prepare == args.launch:
        raise SystemExit("Choose exactly one of --prepare or --launch")
    result = prepare(args.run, args.max_runtime_hours) if args.prepare else launch(args.run, args.poll_seconds)
    print(json.dumps(result if args.launch else {"prepared": result["run"], "jobs": len(result["jobs"])}, indent=2))


if __name__ == "__main__":
    main()
