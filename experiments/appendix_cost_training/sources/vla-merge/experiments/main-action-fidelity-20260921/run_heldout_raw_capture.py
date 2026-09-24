#!/usr/bin/env python3
"""Dynamic queue for complete raw C146 held-out inputs (attempt 02)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path.insert(0, str(HERE))
import run_heldout_capture as prior

COLLECTOR = HERE / "collect_heldout_raw_requests.py"
DEFAULT_RUN = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/heldout-raw-attempt-02"
SUPERSEDED = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/heldout-attempt-01"


def configure():
    prior.configure_base()
    prior.base.COLLECTOR = COLLECTOR


def jobs(run, bank):
    result = prior.build_jobs(run, bank)
    for job in result:
        job["id"] = job["id"].replace("heldout-", "heldout-raw-", 1)
        job["kind"] = "heldout_raw_native_input_capture"
        job["command"][2] = str(COLLECTOR)
        output = Path(job["output"]) / "raw-heldout"
        job["environment"].update({
            "PI05_BLOCK_REGMEANPP_TENSOR_OUTPUT": str(output / "requests.safetensors"),
            "PI05_BLOCK_REGMEANPP_MANIFEST_OUTPUT": str(output / "requests.json"),
            "FEATCAL_EXEC_NOISE_SEED": str(job["flow_seed"]),
            "FEATCAL_EXEC_MEMORY_FRACTION": "0.35",
        })
    return result


def verify(path, job):
    manifest = Path(job["output"]) / "raw-heldout/requests.json"
    tensor = manifest.with_suffix(".safetensors")
    if not manifest.is_file() or not tensor.is_file():
        raise ValueError(f"Missing raw held-out outputs for {job['id']}")
    value = json.loads(manifest.read_text())
    expected = {
        "method": "main_fidelity_native_execution_input_capture",
        "source_kind": "heldout_expert_execution", "heldout_only": True,
        "forbidden_as_calibration": True, "sample_count": 50,
        "episode_id": job["repeat_id"], "expert": job["expert_name"],
        "source_policy_sha256": job["policy_sha256"],
        "dense_expert_bank_sha256": job["dense_bank_sha256"],
        "noise_seed": job["flow_seed"], "capture_seed": job["start_seed"],
        "init_state_offset": job["init_state_offset"], "flow_indices": [0],
        "demonstration_actions_used": False, "success_filtering": False,
        "failures_retained": True,
    }
    problems = [f"{key}={value.get(key)!r} expected {wanted!r}"
                for key, wanted in expected.items() if value.get(key) != wanted]
    samples = value.get("samples", [])
    required = {"tokens", "masks", "x_t", "time", "native_velocity"}
    required |= {f"image_{i}" for i in range(3)} | {f"image_mask_{i}" for i in range(3)}
    from safetensors import safe_open
    with safe_open(str(tensor), framework="pt") as handle:
        keys = set(handle.keys())
    grouped = {}
    for row in samples:
        grouped.setdefault(row.get("prompt_signature"), []).append(row)
        prefix = f"sample_{row.get('index', -1):03d}."
        if not {prefix + key for key in required} <= keys:
            problems.append(f"sample {row.get('index')} lacks complete native inputs")
        if row.get("flow_index") != 0 or row.get("init_state_id") != job["init_state_offset"]:
            problems.append("sample flow/state mismatch")
    if len(samples) != 50 or len(grouped) != 10:
        problems.append("expected 10 tasks x 5 requests")
    for rows in grouped.values():
        if sorted(r.get("request_slot") for r in rows) != list(range(5)):
            problems.append("request slot coverage differs")
        if len({r.get("request_index") for r in rows}) != 5:
            problems.append("request quantiles are not unique")
        if len({r.get("initial_observation_sha256") for r in rows}) != 1:
            problems.append("episode initial observation is not stable")
    if prior.sha256_file(tensor) != value.get("tensor_sha256"):
        problems.append("tensor content hash mismatch")
    if problems:
        raise ValueError(f"{job['id']} raw audit failed: {'; '.join(problems)}")
    return {"job_id": job["id"], "requests": 50, "tasks": 10,
            "manifest_sha256": prior.sha256_file(manifest),
            "tensor_sha256": prior.sha256_file(tensor), "complete_native_inputs": True,
            "heldout_only": True, "forbidden_as_calibration": True}


def prepare(run: Path, hours: float):
    run = run.resolve()
    if run.exists():
        raise FileExistsError(run)
    old = json.loads((SUPERSEDED / "queue-ended.json").read_text())
    if old.get("status") != "complete" or old.get("stopped") is not False:
        raise ValueError("Prefix-only attempt has no healthy terminal evidence")
    configure()
    bank = prior.base.load_bank()
    all_jobs = jobs(run, bank)
    sources = dict(prior.base.source_hashes())
    for path in (Path(__file__).resolve(), COLLECTOR.resolve(), prior.Path(prior.__file__).resolve(),
                 (prior.FLOCK_DIR / "card_flock.py").resolve()):
        sources[str(path)] = prior.sha256_file(path)
    plan = {
        "schema": "main_action_fidelity_raw_heldout_capture_v1",
        "status": "prepared_not_launched", "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "run": str(run),
        "candidate_gpus": list(prior.CANDIDATE_GPUS), "max_active": prior.MAX_ACTIVE,
        "max_workers_per_gpu": 1, "min_free_mib": prior.MIN_FREE_MIB,
        "runtime_floor_mib": prior.RESERVE_MIB, "no_retry": True,
        "dynamic_selection": "highest current free memory among admissible UUID-locked cards",
        "episodes": list(prior.EPISODES), "requests": {"tasks": 40,
            "episodes_per_task": 2, "requests_per_episode": 5, "total": 400,
            "flow_index": 0, "success_filter": False},
        "usage": "heldout action-fidelity only; forbidden for calibration, tuning, or model selection",
        "supersedes_prefix_only_attempt": str(SUPERSEDED.resolve()),
        "superseded_reason": "prefix-only artifacts are model-specific and cannot support full cross-model native forward",
        "same_frozen_episode_and_request_contract": True,
        "expert_bank": bank, "source_hashes": sources, "jobs": all_jobs,
        "deadline_unix": time.time() + hours * 3600,
    }
    run.mkdir(parents=True)
    prior.write_json(run / "plan.json", plan)
    prior.write_json(SUPERSEDED / "SUPERSEDED-FOR-MAIN-FIDELITY.json", {
        "superseded_by": str(run), "reason": plan["superseded_reason"],
        "raw_scores_read": False, "artifacts_preserved": True,
    })
    return plan


def launch(run: Path, poll_seconds: float):
    run = run.resolve()
    plan = json.loads((run / "plan.json").read_text())
    if plan.get("schema") != "main_action_fidelity_raw_heldout_capture_v1" or plan.get("hostname") != socket.gethostname():
        raise ValueError("Wrong raw-heldout plan identity/host")
    if {path: prior.sha256_file(Path(path)) for path in plan["source_hashes"]} != plan["source_hashes"]:
        raise ValueError("Pinned raw-heldout source changed")
    configure()
    prior.base.verify_manifest = verify
    uuids, leases, selected = prior.card_flock.gpu_uuids(), [], []
    snapshot = prior.gpu_rows()
    try:
        for row in snapshot:
            gpu = row["index"]
            if row["free_mib"] < prior.MIN_FREE_MIB or uuids.get(gpu) != row["uuid"]:
                continue
            lease = prior.card_flock.take_card(gpu, row["uuid"], "main-fidelity-heldout-raw", "capture")
            if lease is None:
                continue
            leases.append(lease); selected.append(gpu)
            if len(selected) == prior.MAX_ACTIVE:
                break
        prior.write_json(run / "dynamic-card-selection.json", {
            "created_at": datetime.now(timezone.utc).isoformat(), "snapshot": snapshot,
            "selected_gpus": selected, "required_free_mib": prior.MIN_FREE_MIB,
            "max_active": prior.MAX_ACTIVE, "one_worker_per_card": True,
        })
        if not selected:
            raise RuntimeError("No admissible lockable GPU is available")
        prior.base.LOCAL_GPUS = tuple(selected)
        result = prior.base.run_queue(plan, gpus=tuple(selected), poll_seconds=poll_seconds,
                                      max_slots_per_gpu=1)
        result["dynamically_selected_gpus"] = selected
        prior.write_json(run / "queue-ended.json", result)
        return result
    finally:
        for lease in reversed(leases): lease.release()


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


if __name__ == "__main__": main()
