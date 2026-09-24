#!/usr/bin/env python3
"""Materialize a full PI0.5 checkpoint with dense original-style RegMean."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import mean, median

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial", type=Path, required=True)
    parser.add_argument("--object", dest="object_adapter", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--spatial-activations", type=Path, required=True)
    parser.add_argument("--object-activations", type=Path, required=True)
    parser.add_argument("--spatial-manifest", type=Path, required=True)
    parser.add_argument("--object-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge-ratio", type=float, default=0.05)
    parser.add_argument(
        "--ridge-scale",
        choices=("feature_energy", "kernel_diagonal"),
        default="feature_energy",
        help=(
            "Scale ridge by the primal feature-energy mean (original-style) or "
            "by the sampled dual-kernel diagonal mean (sampling-aware)."
        ),
    )
    parser.add_argument("--max-correction-ratio", type=float, default=3.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def task_loss(x: torch.Tensor, weight: torch.Tensor, target: torch.Tensor) -> float:
    residual = x @ (weight - target).T
    return float((residual * residual).mean())


def solve_weight(
    x_spatial: torch.Tensor,
    x_object: torch.Tensor,
    weight_spatial: torch.Tensor,
    weight_object: torch.Tensor,
    prior: torch.Tensor,
    ridge_ratio: float,
    ridge_scale: str,
    max_correction_ratio: float,
) -> tuple[torch.Tensor, dict]:
    device = prior.device
    x_spatial = x_spatial.to(device=device, dtype=torch.float32)
    x_object = x_object.to(device=device, dtype=torch.float32)
    weight_spatial = weight_spatial.to(device=device, dtype=torch.float32)
    weight_object = weight_object.to(device=device, dtype=torch.float32)
    prior = prior.to(dtype=torch.float32)
    if x_spatial.shape[1] != prior.shape[1] or x_object.shape[1] != prior.shape[1]:
        raise ValueError(
            f"Activation/weight width mismatch: {x_spatial.shape}, {x_object.shape}, {prior.shape}"
        )

    spatial_scale = math.sqrt(2.0 * x_spatial.shape[0])
    object_scale = math.sqrt(2.0 * x_object.shape[0])
    x = torch.cat((x_spatial / spatial_scale, x_object / object_scale), dim=0)
    residual_target = torch.cat(
        (
            (x_spatial @ (weight_spatial - prior).T) / spatial_scale,
            (x_object @ (weight_object - prior).T) / object_scale,
        ),
        dim=0,
    )
    kernel = x @ x.T
    feature_energy = max(float((x * x).sum() / x.shape[1]), 1e-12)
    kernel_diagonal = max(float(torch.diagonal(kernel).mean()), 1e-12)
    if ridge_scale == "feature_energy":
        ridge_reference = feature_energy
    elif ridge_scale == "kernel_diagonal":
        ridge_reference = kernel_diagonal
    else:
        raise ValueError(f"Unsupported ridge scale: {ridge_scale}")
    ridge = ridge_ratio * ridge_reference
    kernel.diagonal().add_(ridge)
    cholesky, info = torch.linalg.cholesky_ex(kernel)
    if int(info.max()) != 0:
        raise RuntimeError(f"Dense RegMean kernel is not positive definite; info={int(info.max())}")
    alpha = torch.cholesky_solve(residual_target, cholesky)
    correction = alpha.T @ x

    task_delta_reference = 0.5 * (
        torch.linalg.vector_norm(weight_spatial - prior)
        + torch.linalg.vector_norm(weight_object - prior)
    )
    correction_norm = torch.linalg.vector_norm(correction)
    trust_scale = 1.0
    limit = max_correction_ratio * max(float(task_delta_reference), 1e-12)
    if float(correction_norm) > limit:
        trust_scale = limit / float(correction_norm)
        correction = correction * trust_scale
    merged = prior + correction

    mean_loss = 0.5 * (
        task_loss(x_spatial, prior, weight_spatial)
        + task_loss(x_object, prior, weight_object)
    )
    merged_loss = 0.5 * (
        task_loss(x_spatial, merged, weight_spatial)
        + task_loss(x_object, merged, weight_object)
    )
    eigenvalues = torch.linalg.eigvalsh(kernel)
    metrics = {
        "spatial_rows": int(x_spatial.shape[0]),
        "object_rows": int(x_object.shape[0]),
        "input_width": int(prior.shape[1]),
        "output_width": int(prior.shape[0]),
        "ridge": ridge,
        "ridge_scale": ridge_scale,
        "ridge_reference": ridge_reference,
        "feature_energy": feature_energy,
        "kernel_diagonal_mean": kernel_diagonal,
        "kernel_condition": float(eigenvalues[-1] / eigenvalues[0]),
        "trust_scale": trust_scale,
        "correction_norm": float(torch.linalg.vector_norm(correction)),
        "prior_loss": mean_loss,
        "dense_regmean_loss": merged_loss,
        "improvement_vs_prior": (mean_loss - merged_loss) / max(mean_loss, 1e-12),
    }
    return merged.detach().cpu(), metrics


def main() -> None:
    args = parse_args()
    if args.ridge_ratio <= 0:
        raise ValueError("ridge-ratio must be positive")
    if args.max_correction_ratio <= 0:
        raise ValueError("max-correction-ratio must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    spatial_root = args.spatial.expanduser().resolve()
    object_root = args.object_adapter.expanduser().resolve()
    base_root = args.base_model.expanduser().resolve()
    spatial_activations = args.spatial_activations.expanduser().resolve()
    object_activations = args.object_activations.expanduser().resolve()
    spatial_manifest_path = args.spatial_manifest.expanduser().resolve()
    object_manifest_path = args.object_manifest.expanduser().resolve()
    output = args.output.expanduser().absolute()
    for path in (
        spatial_root,
        object_root,
        base_root,
        spatial_activations,
        object_activations,
        spatial_manifest_path,
        object_manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")

    spatial_config = load_json(spatial_root / "adapter_config.json")
    object_config = load_json(object_root / "adapter_config.json")
    if spatial_config != object_config:
        raise ValueError("Source adapter configs differ")
    source_rank = int(spatial_config["r"])
    source_scale = float(spatial_config["lora_alpha"]) / source_rank
    spatial_manifest = load_json(spatial_manifest_path)
    object_manifest = load_json(object_manifest_path)
    if spatial_manifest["task"] != "spatial" or object_manifest["task"] != "object":
        raise ValueError("Activation manifests must be spatial and object")
    shared_input_keys = ("spatial_adapter", "object_adapter", "base_model")
    if any(
        spatial_manifest["inputs"].get(key) != object_manifest["inputs"].get(key)
        for key in shared_input_keys
    ):
        raise ValueError("Activation manifests do not share the same expert/base triplet")
    if set(spatial_manifest["modules"]) != set(object_manifest["modules"]):
        raise ValueError("Activation module sets differ")

    spatial_weights_path = spatial_root / "adapter_model.safetensors"
    object_weights_path = object_root / "adapter_model.safetensors"
    base_weights_path = base_root / "model.safetensors"

    output.mkdir(parents=True)
    support_files = (
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    for name in support_files:
        first = spatial_root / name
        second = object_root / name
        if first.read_bytes() != second.read_bytes():
            raise ValueError(f"Expert support file differs: {name}")
        shutil.copy2(first, output / name)
    shutil.copytree(spatial_root / "tokenizer", output / "tokenizer")
    policy_config = load_json(spatial_root / "config.json")
    policy_config["use_peft"] = False
    policy_config["pretrained_path"] = str(output)
    (output / "config.json").write_text(
        json.dumps(policy_config, indent=4, sort_keys=False) + "\n", encoding="utf-8"
    )

    metrics: dict[str, dict] = {}
    modified_keys: set[str] = set()
    with safe_open(base_weights_path, framework="pt", device="cpu") as base:
        output_tensors = {key: base.get_tensor(key) for key in base.keys()}

    with safe_open(spatial_weights_path, framework="pt", device="cpu") as spatial, safe_open(
        object_weights_path, framework="pt", device="cpu"
    ) as object_adapter, safe_open(
        spatial_activations, framework="pt", device="cpu"
    ) as spatial_inputs, safe_open(
        object_activations, framework="pt", device="cpu"
    ) as object_inputs:
        adapter_keys = list(spatial.keys())
        if adapter_keys != list(object_adapter.keys()):
            raise ValueError("Adapter tensor keys differ")
        activation_modules = set(spatial_inputs.keys())
        if activation_modules != set(object_inputs.keys()):
            raise ValueError("Activation tensor module sets differ")

        prefix = "base_model.model."
        for key in adapter_keys:
            if ".lora_A." not in key:
                continue
            pair_key = key.replace(".lora_A.", ".lora_B.")
            module_name = key.split(".lora_A.", 1)[0]
            if module_name not in activation_modules:
                raise ValueError(f"Missing activations for {module_name}")
            if not module_name.startswith(prefix):
                raise ValueError(f"Cannot map adapter module to base: {module_name}")
            base_module = module_name[len(prefix) :]
            weight_key = f"{base_module}.weight"
            base_weight = output_tensors[weight_key]
            a_spatial = spatial.get_tensor(key)
            b_spatial = spatial.get_tensor(pair_key)
            a_object = object_adapter.get_tensor(key)
            b_object = object_adapter.get_tensor(pair_key)
            weight_spatial = base_weight.float() + source_scale * (
                b_spatial.float() @ a_spatial.float()
            )
            weight_object = base_weight.float() + source_scale * (
                b_object.float() @ a_object.float()
            )
            prior = 0.5 * (weight_spatial + weight_object)
            merged, module_metrics = solve_weight(
                spatial_inputs.get_tensor(module_name),
                object_inputs.get_tensor(module_name),
                weight_spatial,
                weight_object,
                prior.to(device),
                args.ridge_ratio,
                args.ridge_scale,
                args.max_correction_ratio,
            )
            if not torch.isfinite(merged).all():
                raise ValueError(f"Non-finite dense solution for {module_name}")
            output_tensors[weight_key] = merged.to(dtype=base_weight.dtype).contiguous()
            modified_keys.add(weight_key)
            metrics[module_name] = {"kind": "lora_materialized_dense", **module_metrics}

        dense_modules = sorted(
            {
                key.rsplit(".", 1)[0]
                for key in adapter_keys
                if ".lora_" not in key and key.endswith((".weight", ".bias"))
            }
        )
        for module_name in dense_modules:
            if module_name not in activation_modules:
                raise ValueError(f"Missing activations for {module_name}")
            if not module_name.startswith(prefix):
                raise ValueError(f"Cannot map dense adapter module to base: {module_name}")
            base_module = module_name[len(prefix) :]
            weight_key = f"{base_module}.weight"
            bias_key = f"{base_module}.bias"
            adapter_weight_key = f"{module_name}.weight"
            adapter_bias_key = f"{module_name}.bias"
            base_weight = output_tensors[weight_key]
            base_bias = output_tensors[bias_key]
            spatial_weight = spatial.get_tensor(adapter_weight_key)
            spatial_bias = spatial.get_tensor(adapter_bias_key)
            object_weight = object_adapter.get_tensor(adapter_weight_key)
            object_bias = object_adapter.get_tensor(adapter_bias_key)
            spatial_augmented = torch.cat(
                (spatial_weight.float(), spatial_bias.float().unsqueeze(1)), dim=1
            )
            object_augmented = torch.cat(
                (object_weight.float(), object_bias.float().unsqueeze(1)), dim=1
            )
            prior = 0.5 * (spatial_augmented + object_augmented)
            x_spatial = spatial_inputs.get_tensor(module_name).float()
            x_object = object_inputs.get_tensor(module_name).float()
            x_spatial = torch.cat((x_spatial, torch.ones((x_spatial.shape[0], 1))), dim=1)
            x_object = torch.cat((x_object, torch.ones((x_object.shape[0], 1))), dim=1)
            merged, module_metrics = solve_weight(
                x_spatial,
                x_object,
                spatial_augmented,
                object_augmented,
                prior.to(device),
                args.ridge_ratio,
                args.ridge_scale,
                args.max_correction_ratio,
            )
            output_tensors[weight_key] = merged[:, :-1].to(dtype=base_weight.dtype).contiguous()
            output_tensors[bias_key] = merged[:, -1].to(dtype=base_bias.dtype).contiguous()
            modified_keys.update((weight_key, bias_key))
            metrics[module_name] = {"kind": "modules_to_save_dense", **module_metrics}

    expected_modified = 414 + 2 * 4
    if len(modified_keys) != expected_modified:
        raise ValueError(f"Expected {expected_modified} modified tensors, found {len(modified_keys)}")
    output_weights = output / "model.safetensors"
    save_file(output_tensors, output_weights, metadata={"format": "pt"})

    improvements = [entry["improvement_vs_prior"] for entry in metrics.values()]
    trust_scales = [entry["trust_scale"] for entry in metrics.values()]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "pi05_dense_original_style_regmean_dual_ridge_to_mean",
        "training": False,
        "parameterization": "full_linear_weight_space",
        "task_balance": "equal after per-task row normalization",
        "ridge_ratio": args.ridge_ratio,
        "ridge_scale": args.ridge_scale,
        "ridge_prior": "dense_mean_0p5_0p5",
        "max_correction_ratio": args.max_correction_ratio,
        "module_count": len(metrics),
        "modified_tensor_count": len(modified_keys),
        "improvement_summary": {
            "min": min(improvements),
            "median": median(improvements),
            "mean": mean(improvements),
            "max": max(improvements),
        },
        "trust_limited_module_count": sum(value < 1.0 for value in trust_scales),
        "inputs": {
            "spatial_adapter": str(spatial_root),
            "object_adapter": str(object_root),
            "base_model": str(base_root),
            "spatial_activations": str(spatial_activations),
            "object_activations": str(object_activations),
            "spatial_adapter_sha256": sha256(spatial_weights_path),
            "object_adapter_sha256": sha256(object_weights_path),
        },
        "output": str(output),
        "model_sha256": sha256(output_weights),
        "modules": metrics,
    }
    (output / "dense_regmean_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "method",
                    "module_count",
                    "modified_tensor_count",
                    "improvement_summary",
                    "trust_limited_module_count",
                    "output",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
