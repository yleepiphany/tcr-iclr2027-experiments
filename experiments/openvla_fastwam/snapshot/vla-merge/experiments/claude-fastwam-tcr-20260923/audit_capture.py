#!/usr/bin/env python3
"""Strict CPU audit of one Fast-WAM closed-loop calibration episode."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from capture_closed_loop import observation_sha256, request_quantiles


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_tensor(value, *, shape=None):
    if not torch.is_tensor(value) or (shape is not None and tuple(value.shape) != shape):
        raise ValueError(f"Expected native tensor with shape {shape}")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError("Nonfinite tensor in native capture")


def audit(plan_path: Path, episode_dir: Path, job_id: str) -> dict:
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "fastwam_four_expert_ab_calibration_bank_v1" or \
            len(plan.get("jobs", [])) != 80:
        raise ValueError("Wrong frozen calibration bank")
    jobs = [row for row in plan["jobs"] if row["id"] == job_id]
    if len(jobs) != 1:
        raise ValueError("Calibration job is missing or duplicated")
    job = jobs[0]
    receipt = json.loads((episode_dir / "episode.json").read_text())
    identity = receipt.get("identity", {})
    if receipt.get("schema") != "fastwam_closed_loop_episode_v1" or \
            receipt.get("complete") is not True or type(receipt.get("success")) is not bool:
        raise ValueError("Episode receipt is incomplete")
    for key, expected in {
        "bank_job_id": job_id, "suite": job["suite"], "task_id": job["task_id"],
        "expert": job["expert"], "seed": job["seed"],
        "checkpoint": job["checkpoint"]["path"],
        "checkpoint_size": job["checkpoint"]["size"],
        "stats_path": job["stats_path"], "stats_sha256": job["stats_sha256"],
        "calibration_round": job["round"],
    }.items():
        if identity.get(key) != expected:
            raise ValueError(f"Episode identity mismatch: {key}")
    if Path(identity["bank_plan"]).resolve() != plan_path.resolve():
        raise ValueError("Episode used a different bank plan")
    checkpoint = Path(job["checkpoint"]["path"])
    stat = checkpoint.stat()
    if stat.st_size != job["checkpoint"]["size"] or \
            stat.st_mtime_ns != job["checkpoint"]["mtime_ns"]:
        raise ValueError("Source checkpoint changed")
    if sha(Path(job["stats_path"])) != job["stats_sha256"]:
        raise ValueError("Normalizer changed")
    demo = Path(job["demo_file"])
    stat = demo.stat()
    if stat.st_size != job["demo_size"] or stat.st_mtime_ns != job["demo_mtime_ns"]:
        raise ValueError("Demonstration reset source changed")
    with h5py.File(demo) as handle:
        reset = np.ascontiguousarray(handle["data"][f"demo_{job['demo_index']}"].attrs["init_state"])
    reset_hash = hashlib.sha256(reset.tobytes()).hexdigest()
    if (reset_hash != job["reset_sha256"] or
            reset_hash != receipt["initial_state_sha256"] or
            reset_hash != identity.get("reset", {}).get("state_sha256") or
            list(reset.shape) != receipt["initial_state_shape"] or
            reset.dtype.str != receipt["initial_state_dtype"]):
        raise ValueError("Initial simulator state differs from frozen reset")
    for item in plan["source_code"].values():
        if sha(Path(item["path"])) != item["sha256"]:
            raise ValueError("Native execution source changed")
    count = receipt.get("request_count")
    if type(count) is not int or count < 1 or \
            receipt.get("selected_request_indices") != request_quantiles(count) or \
            receipt.get("selection_rule") != "first_mid_last_floor_position_only":
        raise ValueError("Request selection is incomplete or outcome-dependent")
    files = sorted(episode_dir.glob("request-*.pt"))
    if [path.name for path in files] != [f"request-{i:06d}.pt" for i in range(count)]:
        raise ValueError("Native request files have gaps or extras")
    hashes = []
    for index, path in enumerate(files):
        row = torch.load(path, map_location="cpu", weights_only=True)
        if row.get("request_index") != index or \
                observation_sha256(row["raw_observation"]) != row["raw_observation_sha256"]:
            raise ValueError(f"Raw closed-loop observation mismatch at request {index}")
        action = row["executed_action_chunk"]
        check_tensor(action, shape=(32, 7))
        if hashlib.sha256(action.numpy().tobytes()).hexdigest() != row["executed_action_sha256"]:
            raise ValueError(f"Executed action chunk mismatch at request {index}")
        request = row["native_request"]
        if request.get("seed") != job["seed"] or request.get("num_inference_steps") != 10 or \
                request.get("action_horizon") != 32 or not isinstance(request.get("prompt"), str):
            raise ValueError(f"Native request schedule/seed differs at {index}")
        check_tensor(request.get("input_image"), shape=(1, 3, 224, 448))
        check_tensor(request.get("proprio"), shape=(1, 8))
        trace = row["trace"]
        if trace.get("schema") != "fastwam_native_action_trace_v1" or \
                trace.get("mode") != "action_only_with_visual_prefill" or \
                trace.get("expected_denoising_calls") != 10 or \
                trace.get("selected_call_indices") != [0, 5, 9] or \
                len(trace.get("video_cache_shapes", [])) != 30 or \
                [call.get("call_index") for call in trace.get("calls", [])] != [0, 5, 9]:
            raise ValueError(f"Native action trace contract differs at {index}")
        check_tensor(trace.get("native_action"), shape=(32, 7))
        for call in trace["calls"]:
            check_tensor(call.get("prediction"), shape=(1, 32, 7))
            check_tensor(call["inputs"].get("latents_action"), shape=(1, 32, 7))
            check_tensor(call["inputs"].get("timestep_action"), shape=(1,))
        hashes.append({"index": index, "sha256": sha(path)})
    return {"accepted": True, "job_id": job_id, "requests": count,
            "selected_request_indices": receipt["selected_request_indices"],
            "episode_receipt_sha256": sha(episode_dir / "episode.json"),
            "request_files": hashes,
            "formal_evaluation": False, "success_used_for_selection": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.plan, args.episode, args.job_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"accepted": result["accepted"], "job_id": result["job_id"],
                      "requests": result["requests"],
                      "selected": result["selected_request_indices"]}))
