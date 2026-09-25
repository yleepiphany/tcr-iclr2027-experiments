#!/usr/bin/env python3
"""Bounded native-path diagnostic. No checkpoint writes or success-rate claims."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
SOURCE = WORK / "pi05_lora_finetune_v2_20260826"
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(SOURCE / "src"), str(SOURCE / "lerobot/src")]
PILOT = WORK / "vla-merge-runtime/experiments/featcal-execution-pilot-20260916"
FULL = WORK / "vla-merge-runtime/experiments/table3-ablation-20260916/checkpoints/full"
RUN = WORK / "vla-merge-runtime/experiments/generation-path-probe-20260916"
SUITES = {0: ("spatial", "goal"), 1: ("object", "long")}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_requests(rows, smoke=False):
    selected = []
    # Choose reservoir slots, not best-performing tasks/episodes.
    for task in range(1 if smoke else 10):
        for request in range(1 if smoke else 2):
            matches = [r for r in rows if r["task_slot"] == task and r["task_ordinal"] == 3 * request]
            if len(matches) != 1 or matches[0]["flow_index"] != 0:
                raise ValueError("Missing unique initial-noise record")
            selected.append(matches[0])
    return selected


def rest_seconds(elapsed, duty):
    if not 0 < duty <= .25:
        raise ValueError("Duty fraction must be at most 0.25")
    return max(0., elapsed) * (1 / duty - 1)


def memory_free(gpu):
    return float(subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], text=True).strip())


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("GPU visibility mismatch")
    out = args.run / (f"smoke-gpu{args.gpu}" if args.smoke else f"gpu{args.gpu}")
    out.mkdir(parents=True, exist_ok=False)
    wait_start = time.monotonic()
    while memory_free(args.gpu) < 32 * 1024:
        if time.monotonic() - wait_start > 7200:
            raise RuntimeError("No memory window within two hours")
        print("WAIT_MEMORY", args.gpu, flush=True)
        time.sleep(20)
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file, load_file
    from scripts.run_pi05_featcal_execution_pilot import load_policy
    from vla_merge.featcal_execution import validate_raw_manifest

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.225, 0)
    # Same native mixed precision as existing capture, including its TF32 setting.
    source = read(PILOT / "build-v2/plan.json")
    checkpoint_receipt = read(FULL.parents[1] / "status.json")["jobs"]["solve-full"]
    if checkpoint_receipt["status"] != "complete":
        raise ValueError("Full checkpoint is incomplete")
    expected = checkpoint_receipt["receipt"]["model_sha256"]
    if sha(FULL / "model.safetensors") != expected:
        raise ValueError("Full checkpoint changed")
    samples = {}
    sources = {}
    for suite in SUITES[args.gpu]:
        entry = source["raw"][suite]
        if sha(entry["manifest"]) != entry["manifest_sha256"] or sha(entry["tensor_file"]) != entry["tensor_sha256"]:
            raise ValueError("Raw source identity changed")
        meta = read(entry["manifest"])
        rows = choose_requests(validate_raw_manifest(meta), args.smoke)
        expert = source["models"]["experts"][suite]
        if sha(Path(expert["path"]) / "model.safetensors") != expert["model_sha256"]:
            raise ValueError("Expert changed")
        sources[suite] = {"raw": entry, "expert": expert, "samples": rows}
        with safe_open(entry["tensor_file"], framework="pt") as f:
            keys = list(f.keys())
            for row in rows:
                prefix = f"sample_{row['index']:03d}."
                key = f"{suite}_{row['index']:03d}"
                samples[key] = (suite, row, {k[len(prefix):]: f.get_tensor(k) for k in keys if k.startswith(prefix)})
    write(out / "plan.json", {
        "model": str(FULL), "model_sha256": expected, "rows_used_to_build_model": 1331600,
        "script_sha256": sha(__file__), "sources": sources, "smoke": args.smoke,
        "allocator_cap_gib": 18, "duty_fraction": args.duty, "cpu_threads": 2,
        "wall_limit_seconds_after_preflight": 3600, "native_steps": 10,
        "claims": "Exploratory fixed-request path diagnostic, no optimization or closed-loop score.",
        "independence": "Existing execution cache; no claim of independent episodes or untouched held-out data.",
        "tf32_override": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
    })
    start = time.monotonic()
    progress = out / "progress.jsonl"

    def event(phase, key, **extra):
        with progress.open("a") as stream:
            stream.write(json.dumps({"phase": phase, "key": key, "elapsed": time.monotonic()-start, **extra}) + "\n")
        print(phase, key, flush=True)

    def guard():
        if time.monotonic() - start > 3600:
            raise RuntimeError("One-hour diagnostic wall budget exhausted")
        if memory_free(args.gpu) < 16 * 1024:
            raise RuntimeError("Shared GPU reserve fell below 16 GiB; stop without retry")

    def pause(begin):
        torch.cuda.synchronize()
        time.sleep(rest_seconds(time.monotonic()-begin, args.duty))

    def unload(policy):
        policy.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def path(policy, batch, queries=None):
        model = policy.model
        original = model.denoise_step
        captured = {"x": [], "v": [], "time": [], "query_v": []}

        def hook(*a, **kw):
            if a:
                raise ValueError("Unexpected positional native call")
            index = len(captured["x"])
            captured["x"].append(kw["x_t"].detach().cpu().clone())
            captured["time"].append(kw["timestep"].detach().cpu().clone())
            v = original(**kw)
            captured["v"].append(v.detach().cpu().clone())
            if queries is not None:
                if not torch.equal(captured["time"][-1], queries["time"][index]):
                    raise ValueError("Schedules differ")
                q = dict(kw, x_t=queries["x"][index].to("cuda"))
                captured["query_v"].append(original(**q).detach().cpu().clone())
            return v

        model.denoise_step = hook
        try:
            actions = model.sample_actions(
                images=[batch[f"image_{i}"] for i in range(3)],
                img_masks=[batch[f"image_mask_{i}"] for i in range(3)],
                tokens=batch["tokens"], masks=batch["masks"],
                states=batch.get("states"), state_masks=batch.get("state_masks"),
                noise=batch["x_t"].clone(), num_steps=10)
        finally:
            model.denoise_step = original
        if len(captured["x"]) != 10:
            raise ValueError("Incomplete native path")
        result = {k: torch.stack(v) for k, v in captured.items() if v}
        result["action"] = actions.detach().cpu().clone()
        if any(not torch.isfinite(v).all() for v in result.values()):
            raise ValueError("Nonfinite path")
        return result

    policy = load_policy(str(FULL))
    for key, (_, _, cpu_batch) in samples.items():
        guard(); begin = time.monotonic()
        batch = {k: v.to("cuda") for k, v in cpu_batch.items()}
        result = path(policy, batch)
        if key == next(iter(samples)):
            repeated = path(policy, batch)
            if any(not torch.equal(result[k], repeated[k]) for k in result):
                raise ValueError("Native replay is not deterministic")
        save_file(result, out / f"{key}.merged.safetensors")
        event("merged_path", key); del batch; pause(begin)
    unload(policy); del policy

    for suite in SUITES[args.gpu]:
        policy = load_policy(source["models"]["experts"][suite]["path"])
        for key, (name, _, cpu_batch) in samples.items():
            if name != suite:
                continue
            guard(); begin = time.monotonic()
            batch = {k: v.to("cuda") for k, v in cpu_batch.items()}
            merged = load_file(out / f"{key}.merged.safetensors")
            result = path(policy, batch, merged)
            if not torch.equal(result["v"][0], cpu_batch["native_velocity"]):
                raise ValueError("Native teacher no longer reproduces captured velocity")
            save_file(result, out / f"{key}.expert.safetensors")
            event("expert_path_and_teacher_on_merged", key); del batch; pause(begin)
        unload(policy); del policy

    policy = load_policy(str(FULL))
    results = []
    for key, (suite, meta, cpu_batch) in samples.items():
        guard(); begin = time.monotonic()
        batch = {k: v.to("cuda") for k, v in cpu_batch.items()}
        expert = load_file(out / f"{key}.expert.safetensors")
        merged = load_file(out / f"{key}.merged.safetensors")
        result = path(policy, batch, expert)
        if not torch.equal(result["action"], merged["action"]):
            raise ValueError("Querying teacher states changed native generation")
        save_file({"v_on_expert": result["query_v"]}, out / f"{key}.cross.safetensors")

        def stage_mse(a, b):
            return (a[..., :10, :7].float()-b[..., :10, :7].float()).square().flatten(1).mean(1).tolist()

        row = {"key": key, "suite": suite, "request": meta,
               "velocity_mse_on_expert_path": stage_mse(result["query_v"], expert["v"]),
               "velocity_mse_on_merged_path": stage_mse(merged["v"], expert["query_v"]),
               "latent_mse": stage_mse(merged["x"], expert["x"]),
               "action_mse": float((merged["action"][:, :10, :7].float()-expert["action"][:, :10, :7].float()).square().mean())}
        results.append(row)
        event("metrics", key, metrics=row); del batch; pause(begin)
    unload(policy); del policy
    write(out / "summary.json", {"status": "complete", "gpu": args.gpu, "results": results,
          "requests": len(results), "elapsed_seconds": time.monotonic()-start,
          "peak_allocator_gib": torch.cuda.max_memory_allocated()/1024**3,
          "no_checkpoint_modified": True, "no_success_rate_claim": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, choices=[0, 1], required=True)
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--duty", type=float, default=.2)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    rest_seconds(1., args.duty)
    try:
        run(args)
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise
