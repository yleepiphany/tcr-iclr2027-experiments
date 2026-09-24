#!/usr/bin/env python3
"""Freeze the actual Fast-WAM action-only linear scope and execution order."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "vla-merge-runtime"
SOUP = RUNTIME / "experiments/openvla-fastwam-diagnostics-20260923/fastwam-soup/manifest.json"


def classify(module: str) -> str:
    if module == "proprio_encoder":
        return "00-proprio"
    if module.startswith("mot.mixtures.video.blocks."):
        index = int(module.split(".")[4])
        if index not in range(30):
            raise ValueError("Video block outside native 30-layer prefill")
        return f"02-video-block-{index:02d}"
    if module.startswith("mot.mixtures.action.blocks."):
        index = int(module.split(".")[4])
        if index not in range(30):
            raise ValueError("Action block outside native 30-layer denoising")
        return f"04-action-block-{index:02d}"
    if module.startswith("mot.mixtures.video."):
        if module == "mot.mixtures.video.head.head":
            return "inactive-video-generation-head"
        if module.split(".")[3] in {"text_embedding", "time_embedding", "time_projection"}:
            return "01-video-pre"
    if module.startswith("mot.mixtures.action."):
        if module == "mot.mixtures.action.head":
            return "05-action-head"
        if module.split(".")[3] in {"action_encoder", "text_embedding",
                                    "time_embedding", "time_projection"}:
            return "03-action-pre"
    raise ValueError(f"Unclassified native linear: {module}")


def plan() -> dict:
    source = json.loads(SOUP.read_text())
    if source.get("complete") is not True or source.get("is_tcr") is not False:
        raise ValueError("Required fixed Soup prior is absent")
    checkpoint = Path(source["sources"]["spatial"]["path"])
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    groups = {}
    all_linears = {}
    for component in ("mot", "proprio_encoder"):
        for name, value in weights[component].items():
            if (name.endswith(".weight") or
                    (component == "proprio_encoder" and name == "weight")) and value.ndim == 2:
                module = component + ("." + name[:-7] if name != "weight" else "")
                group = classify(module)
                bias_name = name[:-7] + ".bias" if name != "weight" else "bias"
                bias = weights[component].get(bias_name)
                if bias is not None and (bias.ndim != 1 or bias.shape[0] != value.shape[0]):
                    raise ValueError(f"Malformed linear bias: {module}")
                descriptor = {"module": module, "weight_shape": list(value.shape),
                              "weight_dtype": str(value.dtype), "has_bias": bias is not None,
                              "native_calls_per_selected_request": 3 if group.startswith("03-") or
                                  group.startswith("04-") or group.startswith("05-") else 1}
                all_linears[module] = descriptor
                groups.setdefault(group, []).append(descriptor)
    inactive = groups.pop("inactive-video-generation-head")
    ordered = OrderedDict((key, groups[key]) for key in sorted(groups))
    if len(ordered) != 64 or sum(map(len, ordered.values())) != 613 or len(inactive) != 1:
        raise ValueError("Fast-WAM action-only linear scope differs from expected 64/613+1")
    if [len(ordered[key]) for key in ordered if "-block-" in key] != [10] * 60:
        raise ValueError("Native MoT block linear scope changed")
    if [len(ordered[key]) for key in ("00-proprio", "01-video-pre", "03-action-pre",
                                      "05-action-head")] != [1, 5, 6, 1]:
        raise ValueError("Native input/output linear scope changed")
    total_tensors = len(weights["mot"]) + len(weights["proprio_encoder"])
    return {
        "schema": "fastwam_action_only_tcr_block_plan_v1",
        "source_soup_manifest": str(SOUP),
        "source_soup_sha256": source["checkpoint"]["sha256"],
        "expert_checkpoint_sha256": {name: row["sha256"] for name, row in source["sources"].items()},
        "execution_mode": "native_visual_prefill_then_10_step_action_denoise_no_future_video",
        "block_order": list(ordered), "blocks": ordered,
        "active_linear_modules": 613, "blocks_per_pass": 64,
        "inactive_video_generation_linear": inactive[0]["module"],
        "all_checkpoint_tensors": total_tensors,
        "nonlinear_and_inactive_tensor_rule": "preserve fixed uniform-Soup prior unless in an active solved Linear; pass B starts from pass A",
        "selected_native_denoising_calls": [0, 5, 9],
        "row_cap_per_selected_request_per_linear": 6,
        "row_selection": "seeded position sampling; one expert-trajectory request is the unit; retain every completed episode",
        "pass_A": {"source": "A", "expert_masses": "inverse_relative_prior_error",
                   "ridge_multiplier": 0.05, "max_correction_ratio": 3.0},
        "pass_B": {"source": "B", "expert_masses": "uniform",
                   "ridge_multiplier": 0.05, "max_correction_ratio": 3.0},
        "no_success_evaluation_in_build": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = plan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"blocks": result["blocks_per_pass"],
                      "active_linears": result["active_linear_modules"],
                      "inactive_video_head": result["inactive_video_generation_linear"]}))
