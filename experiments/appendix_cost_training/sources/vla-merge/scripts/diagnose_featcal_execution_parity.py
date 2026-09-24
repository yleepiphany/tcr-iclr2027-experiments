#!/usr/bin/env python3
"""Separate batching, precision and full-joint/native replay differences.

Read-only for existing captures/checkpoints. Emits independent diagnostic files;
never relaxes an acceptance threshold or changes a calibration objective.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_pi05_featcal_execution_pilot as pilot
from scripts.run_featcal_execution_pilot_lane import wait_memory
import torch
from safetensors.torch import save_file
from vla_merge.featcal_execution import explicit_velocity_forward


@torch.inference_mode()
def native_velocity_forward(policy, batch):
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d
    model = policy.model
    prefix, padding, attention = model.embed_prefix(
        [batch[f"image_{i}"] for i in range(3)],
        [batch[f"image_mask_{i}"] for i in range(3)],
        batch["tokens"], batch["masks"], batch.get("states"), batch.get("state_masks"))
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=prepare_attention_masks_4d(make_att_2d_masks(padding, attention)),
        position_ids=torch.cumsum(padding, dim=1)-1,
        inputs_embeds=[prefix, None], past_key_values=None, use_cache=True)
    return model.denoise_step(prefix_pad_masks=padding, past_key_values=cache,
                              x_t=batch["x_t"], timestep=batch["time"])


def metrics(actual, expected):
    result = {}
    for label, a, b in [("all_50x32", actual, expected),
                         ("action_50x7", actual[:, :, :7], expected[:, :, :7]),
                         ("executed_10x7", actual[:, :10, :7], expected[:, :10, :7])]:
        a, b = a.float(), b.float()
        d = a-b
        result[label] = {"relative_l2": float(d.norm()/b.norm().clamp_min(1e-8)),
                         "max_abs": float(d.abs().max()), "rmse": float(d.square().mean().sqrt()),
                         "equal": bool(torch.equal(a, b))}
    return result


@torch.inference_mode()
def run(args):
    wait_memory(args.gpu)
    torch.cuda.set_per_process_memory_fraction(.35, 0)
    torch.set_num_threads(4)
    out = pilot.PILOT / "diagnostics" / f"parity-{args.expert}-task{args.task:02d}-v1"
    out.mkdir(parents=True, exist_ok=False)
    plan = pilot.baseline.load_json(pilot.BUILD / "plan.json")
    raw = pilot.RawInputs(plan)
    started = time.monotonic()
    model = pilot.load_policy(plan["models"]["experts"][args.expert]["path"])
    outputs, results = {}, []
    try:
        for precision in ["original_mixed_bf16", "fp32"]:
            if precision == "fp32":
                model.float()
                gc.collect()
                torch.cuda.empty_cache()
            for flow in pilot.FLOWS:
                batch, rows = raw.batch(args.expert, args.task, flow)
                saved = batch["native_velocity"].cpu()
                native5 = native_velocity_forward(model, batch).cpu()
                joint5 = explicit_velocity_forward(model, batch).cpu()
                natives, joints = [], []
                for i in range(5):
                    single = {k: v[i:i+1] for k, v in batch.items()}
                    natives.append(native_velocity_forward(model, single).cpu())
                    joints.append(explicit_velocity_forward(model, single).cpu())
                native1, joint1 = torch.cat(natives), torch.cat(joints)
                tensors = {"saved": saved, "native_b1": native1, "native_b5": native5,
                           "joint_b1": joint1, "joint_b5": joint5}
                for k, v in tensors.items():
                    if not torch.isfinite(v).all():
                        raise ValueError("Nonfinite diagnostic output")
                    outputs[f"{precision}.flow{flow}.{k}"] = v.contiguous()
                comparisons = {
                    "native_b1_vs_saved": metrics(native1, saved),
                    "native_b5_vs_native_b1": metrics(native5, native1),
                    "joint_b1_vs_native_b1": metrics(joint1, native1),
                    "joint_b5_vs_native_b5": metrics(joint5, native5),
                    "joint_b5_vs_joint_b1": metrics(joint5, joint1),
                    "joint_b1_vs_saved": metrics(joint1, saved),
                }
                row = {"precision": precision, "flow": flow, "comparisons": comparisons}
                results.append(row)
                pilot.baseline.append_jsonl(out / "progress.jsonl", row)
                print(json.dumps(row), flush=True)
                del batch, single
        save_file(outputs, out / "outputs.safetensors")
        pilot.baseline.write_exclusive_json(out / "summary.json", {
            "status": "complete", "expert": args.expert, "task_slot": args.task,
            "source_plan_sha256": pilot.sha256_file(pilot.BUILD / "plan.json"),
            "diagnostic_source_sha256": pilot.sha256_file(Path(__file__)),
            "results": results, "elapsed_seconds": time.monotonic()-started,
            "allocator_peak_bytes": torch.cuda.max_memory_allocated(),
            "does_not_establish_success_rate": True})
    finally:
        raw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", required=True, choices=["spatial", "object"])
    parser.add_argument("--task", required=True, type=int)
    parser.add_argument("--gpu", required=True, type=int, choices=[6, 7])
    args = parser.parse_args()
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("Physical GPU must match CUDA visibility")
    run(args)
