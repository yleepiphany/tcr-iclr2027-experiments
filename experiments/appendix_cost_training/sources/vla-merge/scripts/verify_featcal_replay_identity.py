#!/usr/bin/env python3
"""Check same-graph identity and strict-FP32 native/full-joint equivalence."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import diagnose_featcal_execution_parity as diag
from scripts import run_pi05_featcal_execution_pilot as pilot
from scripts.run_featcal_execution_pilot_lane import wait_memory
import torch


@torch.inference_mode()
def baseline_oracle(policy, batch):
    """Use the *existing* baseline model.forward, substituting only external x_t.

There are no demo actions for an execution record. Inject the recorded latent
at the one training-input construction boundary; do not change any layer code.
"""
    from lerobot.policies.pi05 import modeling_pi05
    captured = []
    handle = policy.model.action_out_proj.register_forward_hook(
        lambda _module, _args, out: captured.append(out.detach().clone()))
    try:
        with patch.object(modeling_pi05, "_build_flow_matching_inputs",
                          return_value=(batch["x_t"], batch["time"])):
            policy.model.forward(
                images=[batch[f"image_{i}"] for i in range(3)],
                img_masks=[batch[f"image_mask_{i}"] for i in range(3)],
                tokens=batch["tokens"], masks=batch["masks"],
                actions=torch.zeros_like(batch["x_t"]), noise=torch.zeros_like(batch["x_t"]),
                time=batch["time"], states=batch.get("states"), state_masks=batch.get("state_masks"))
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Baseline oracle must call action head exactly once")
    return captured[0]


@torch.inference_mode()
def run(args):
    wait_memory(args.gpu)
    torch.cuda.set_per_process_memory_fraction(.35, 0)
    torch.set_num_threads(4)
    if args.precision == "strict-fp32":
        if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") != "0":
            raise ValueError("Disable environment override BEFORE importing torch")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    out = pilot.PILOT / "diagnostics" / f"identity-{args.expert}-{args.precision}-v1"
    out.mkdir(parents=True, exist_ok=False)
    plan = pilot.baseline.load_json(pilot.BUILD / "plan.json")
    raw = pilot.RawInputs(plan)
    model = pilot.load_policy(plan["models"]["experts"][args.expert]["path"])
    if args.precision == "strict-fp32":
        model.float()
    backend = {"tf32_override": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
               "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
               "float32_matmul_precision": torch.get_float32_matmul_precision()}
    results = []
    try:
        for flow in pilot.FLOWS:
            batch, rows = raw.batch(args.expert, args.task, flow)
            joint = diag.explicit_velocity_forward(model, batch)
            oracle = baseline_oracle(model, batch)
            native = diag.native_velocity_forward(model, batch)
            row = {"flow": flow, "baseline_oracle_bitwise_equal": torch.equal(joint, oracle),
                   "joint_vs_native_b5": diag.metrics(joint, native)}
            if not row["baseline_oracle_bitwise_equal"]:
                raise ValueError("New adapter differs from original full-joint graph")
            if args.precision == "strict-fp32":
                row["strict_fp32_close"] = torch.allclose(joint, native, atol=1e-4, rtol=1e-4)
                if not row["strict_fp32_close"]:
                    raise ValueError(f"Strict FP32 paths differ: {row}")
            else:
                native1 = torch.cat([diag.native_velocity_forward(model, {k:v[i:i+1] for k,v in batch.items()}) for i in range(5)])
                row["native_b1_saved_bitwise_equal"] = torch.equal(native1, batch["native_velocity"])
                if not row["native_b1_saved_bitwise_equal"]:
                    raise ValueError("Raw capture not reproduced exactly")
            results.append(row)
            pilot.baseline.append_jsonl(out / "progress.jsonl", row)
            print(json.dumps(row), flush=True)
        pilot.baseline.write_exclusive_json(out / "summary.json", {
            "status": "passed", "expert": args.expert, "task": args.task,
            "precision": args.precision, "backend": backend, "results": results,
            "code_sha256": pilot.sha256_file(Path(__file__))})
    finally:
        raw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", choices=["spatial", "object"], required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--gpu", type=int, choices=[6,7], required=True)
    parser.add_argument("--precision", choices=["baseline", "strict-fp32"], required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("Physical GPU visibility mismatch")
    run(args)
