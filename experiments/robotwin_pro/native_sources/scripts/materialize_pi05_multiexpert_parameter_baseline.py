#!/usr/bin/env python3
"""Materialize full-scope M-way parameter baselines for PI0.5.

The global fusion domain is the union of all LoRA-materialized Linear weights
and directly saved interface tensors. TIES trimming/sign election is performed
once over that complete flat domain, rather than independently per module.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from materialize_pi05_multiexpert_wudi import (  # noqa: E402
    ExpertAdapter,
    copy_policy_sidecars,
    expert_delta_for_key,
    load_expert_adapter,
    load_json,
    parse_expert_specs,
    sha256,
)
from vla_merge.streaming_parameter_fusion import (  # noqa: E402
    dare_ties_merge_flat_chunked,
    ties_merge_flat_chunked,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--expert", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-bank-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-expert-bank-manifest-sha256",
        required=True,
        help="Frozen SHA256 of --expert-bank-manifest; mismatches fail closed.",
    )
    parser.add_argument(
        "--method",
        choices=("uniform_soup", "weighted_soup", "task_arithmetic", "ties", "dare_ties"),
        required=True,
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Expert coefficient for weighted_soup. Supply every expert exactly once; "
            "coefficients must be nonnegative and sum to one."
        ),
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ties-keep-fraction", type=float, default=0.2)
    parser.add_argument("--dare-drop-probability", type=float, default=0.9)
    parser.add_argument("--dare-seed", type=int, default=20260906)
    parser.add_argument("--disjoint", choices=("mean", "sum"), default="mean")
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def expert_metadata(expert: ExpertAdapter) -> dict[str, Any]:
    return {
        "name": expert.name,
        "path": str(expert.root),
        "rank": expert.rank,
        "alpha": expert.alpha,
        "peft_scale": expert.scale,
        "adapter_sha256": expert.adapter_sha256,
    }


def uniform_soup_weights(expert_names: list[str]) -> dict[str, float]:
    """Return the paper's fixed equal weighting, preserving expert order."""
    if not expert_names or len(expert_names) != len(set(expert_names)):
        raise ValueError("Uniform soup requires distinct, non-empty expert names")
    coefficient = 1.0 / len(expert_names)
    return {name: coefficient for name in expert_names}


def parse_named_weights(specs: list[str], expert_names: list[str]) -> dict[str, float]:
    """Parse one simplex coefficient for every expert, independent of CLI order."""
    parsed: dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid weight binding: {spec}")
        name, raw_value = spec.split("=", 1)
        if not name or name in parsed:
            raise ValueError(f"Invalid or duplicate weight name: {name}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid weight value for {name}: {raw_value}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Weight for {name} must be finite and nonnegative")
        parsed[name] = value
    expected = set(expert_names)
    if set(parsed) != expected:
        missing = sorted(expected - set(parsed))
        extra = sorted(set(parsed) - expected)
        raise ValueError(f"Weight names differ from experts; missing={missing}, extra={extra}")
    total = sum(parsed.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Expert weights must sum to one, observed {total:.17g}")
    return {name: parsed[name] for name in expert_names}


def validate_expert_bank_manifest(
    path: Path,
    expected_sha256: str,
    *,
    base_root: Path,
    base_model_sha256: str,
    experts: list[ExpertAdapter],
    adapted_keys: list[str],
) -> dict[str, Any]:
    """Bind a build to the independently audited frozen Table 1 expert bank."""
    path = path.expanduser().resolve()
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError("Expected expert-bank manifest SHA256 must be lowercase hex")
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_sha256 = sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Expert-bank manifest SHA256 differs: "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    manifest = load_json(path)
    if manifest.get("kind") != "iclr2027_table1_frozen_expert_bank":
        raise ValueError("Expert-bank manifest kind differs")
    validation = manifest.get("validation", {})
    if (
        validation.get("status") != "valid"
        or validation.get("failed") != 0
        or validation.get("errors")
    ):
        raise ValueError("Expert-bank manifest is not independently valid")
    bank = manifest.get("expert_bank", {})
    if bank.get("frozen") is not True or bank.get("shared_training_step") != 10000:
        raise ValueError("Expert bank is not frozen at the shared 10k checkpoint")
    base = manifest.get("base", {})
    if Path(base.get("canonical_path", "")).resolve() != base_root:
        raise ValueError("Expert-bank base path differs")
    if base.get("model", {}).get("actual_sha256") != base_model_sha256:
        raise ValueError("Expert-bank base model SHA256 differs")

    recorded_experts = bank.get("experts")
    if not isinstance(recorded_experts, list) or len(recorded_experts) != len(experts):
        raise ValueError("Expert-bank expert count differs")
    for recorded, expert in zip(recorded_experts, experts, strict=True):
        if recorded.get("name") != expert.name:
            raise ValueError("Expert-bank expert order/name differs")
        if Path(recorded.get("path", "")).resolve() != expert.root:
            raise ValueError(f"Expert-bank path differs for {expert.name}")
        if recorded.get("training_step") != 10000:
            raise ValueError(f"Expert-bank training step differs for {expert.name}")
        if recorded.get("adapter", {}).get("actual_sha256") != expert.adapter_sha256:
            raise ValueError(f"Expert-bank adapter SHA256 differs for {expert.name}")

    domain = manifest.get("adaptation_domain", {})
    expected_domain = domain.get("expected", {})
    if expected_domain != {
        "logical_tensor_count": 422,
        "lora_pair_count": 414,
        "direct_tensor_count": 8,
        "serialized_adapter_tensor_count": 836,
    }:
        raise ValueError("Expert-bank adaptation-domain contract differs")
    recorded_keys = [row.get("base_key") for row in domain.get("logical_tensors", [])]
    if len(recorded_keys) != 422 or set(recorded_keys) != set(adapted_keys):
        raise ValueError("Expert-bank logical adapted tensor keys differ")
    return {
        "path": str(path),
        "sha256": observed_sha256,
        "kind": manifest["kind"],
        "validation_status": validation["status"],
        "validation_checks_passed": validation.get("passed"),
        "shared_training_step": bank["shared_training_step"],
        "expert_order": [expert.name for expert in experts],
        "adapter_sha256": {
            expert.name: expert.adapter_sha256 for expert in experts
        },
        "base_model_sha256": base_model_sha256,
        "logical_adapted_tensor_count": len(recorded_keys),
        "lora_pair_count": expected_domain["lora_pair_count"],
        "direct_tensor_count": expected_domain["direct_tensor_count"],
    }


def materialized_delta(
    expert: ExpertAdapter,
    key: str,
    base_value: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    del device  # PEFT CPU merge semantics are fixed and device-independent here.
    return expert_delta_for_key(expert, key, base_value)


def main() -> None:
    args = parse_args()
    if len(args.expert) < 2:
        raise ValueError("At least two experts are required")
    if not math.isfinite(args.alpha):
        raise ValueError("alpha must be finite")
    if args.method in {"uniform_soup", "weighted_soup"} and args.alpha != 1.0:
        raise ValueError(
            "--alpha is not part of a soup recipe; leave it at 1.0"
        )
    if not 0 < args.ties_keep_fraction <= 1:
        raise ValueError("ties keep fraction must lie in (0,1]")
    if not 0 <= args.dare_drop_probability < 1 or args.chunk_size <= 0:
        raise ValueError("invalid DARE probability or chunk size")
    base_root = args.base_model.expanduser().resolve()
    output = args.output.expanduser().absolute()
    base_weights = base_root / "model.safetensors"
    if not base_weights.is_file():
        raise FileNotFoundError(base_weights)
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    experts = [load_expert_adapter(name, path) for name, path in parse_expert_specs(args.expert)]
    expert_names = [expert.name for expert in experts]
    if args.method == "uniform_soup":
        if args.weight:
            raise ValueError("--weight is not valid for uniform_soup")
        expert_weights = uniform_soup_weights(expert_names)
    elif args.method == "weighted_soup":
        expert_weights = parse_named_weights(args.weight, expert_names)
    else:
        if args.weight:
            raise ValueError("--weight is only valid for weighted_soup")
        expert_weights = None
    first_lora = set(experts[0].lora_a)
    first_saved = set(experts[0].saved_tensors)
    for expert in experts[1:]:
        if set(expert.lora_a) != first_lora or set(expert.saved_tensors) != first_saved:
            raise ValueError(f"{expert.name}: adapted tensor schema differs")
    adapted_keys = sorted(first_lora | first_saved)
    segments = []
    offset = 0
    with safe_open(base_weights, framework="pt", device="cpu") as base:
        base_keys = set(base.keys())
        missing = set(adapted_keys) - base_keys
        if missing:
            raise ValueError(f"Adapted keys missing from base: {sorted(missing)[:5]}")
        for key in adapted_keys:
            shape = tuple(base.get_slice(key).get_shape())
            count = math.prod(shape)
            segments.append({"key": key, "shape": list(shape), "start": offset, "end": offset + count})
            offset += count
    base_model_sha256 = sha256(base_weights)
    expert_bank_receipt = validate_expert_bank_manifest(
        args.expert_bank_manifest,
        args.expected_expert_bank_manifest_sha256,
        base_root=base_root,
        base_model_sha256=base_model_sha256,
        experts=experts,
        adapted_keys=adapted_keys,
    )
    plan = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "training": False,
        "gradient_or_backward": False,
        "base_model": str(base_root),
        "base_model_sha256": base_model_sha256,
        "expert_bank_receipt": expert_bank_receipt,
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "experts": [expert_metadata(expert) for expert in experts],
        "domain": {
            "scope": "global union of LoRA-materialized weights and directly saved interface tensors",
            "tensor_count": len(adapted_keys),
            "coordinate_count": offset,
            "one_fp32_task_vector_gib": offset * 4 / 2**30,
            "four_fp32_task_vectors_gib": offset * 4 * len(experts) / 2**30,
            "segments": segments,
        },
        "settings": {
            "alpha": args.alpha,
            "alpha_semantics": (
                "not_applicable_to_soup; guarded at exactly 1.0"
                if args.method in {"uniform_soup", "weighted_soup"}
                else "task_vector_scaling"
            ),
            "expert_weights": expert_weights,
            "ties_keep_fraction": args.ties_keep_fraction,
            "dare_drop_probability": args.dare_drop_probability,
            "dare_seed": args.dare_seed,
            "disjoint": args.disjoint,
            "chunk_size": args.chunk_size,
            "dare_effective_chunk_size": max(args.chunk_size, 8 * 1024 * 1024),
            "task_vector_dtype": "float32",
            "fusion_accumulation_dtype": "float64",
            "device_for_lora_materialization": str(device),
            "expert_weight_semantics": (
                "peft_cpu_safe_merge_delta_cast_to_base_dtype_then_base_dtype_add"
            ),
            "task_vector_semantics": (
                "peft_safe_merged_expert_weight_minus_base"
            ),
        },
        "output": str(output),
    }
    print(json.dumps({**plan, "domain": {k: v for k, v in plan["domain"].items() if k != "segments"}}, indent=2), flush=True)
    if args.plan_only:
        return

    vectors = [torch.empty(offset, dtype=torch.float32) for _ in experts]
    with safe_open(base_weights, framework="pt", device="cpu") as base:
        for index, segment in enumerate(segments):
            key = segment["key"]
            base_value = base.get_tensor(key)
            start, end = segment["start"], segment["end"]
            for vector, expert in zip(vectors, experts, strict=True):
                vector[start:end].copy_(materialized_delta(expert, key, base_value, device).reshape(-1))
            if (index + 1) % 25 == 0 or index + 1 == len(segments):
                print(json.dumps({"materialized_tensors": index + 1, "total": len(segments)}), flush=True)
    if any(not torch.isfinite(vector).all() for vector in vectors):
        raise ValueError("Non-finite task vector")

    if args.method in {"uniform_soup", "weighted_soup"}:
        assert expert_weights is not None
        merged_delta = torch.empty(offset, dtype=torch.float64)
        for start in range(0, offset, args.chunk_size):
            end = min(start + args.chunk_size, offset)
            combined = torch.zeros(end - start, dtype=torch.float64)
            for expert, vector in zip(experts, vectors, strict=True):
                combined.add_(
                    vector[start:end].double(), alpha=expert_weights[expert.name]
                )
            merged_delta[start:end] = combined
        fusion_metadata = {
            "operator": "base_plus_simplex_weighted_task_vectors",
            "expert_weights": expert_weights,
            "soup_recipe": "uniform" if args.method == "uniform_soup" else "explicit_weighted",
            "weight_sum": sum(expert_weights.values()),
        }
    elif args.method == "task_arithmetic":
        merged_delta = torch.empty(offset, dtype=torch.float64)
        for start in range(0, offset, args.chunk_size):
            end = min(start + args.chunk_size, offset)
            combined = torch.zeros(end - start, dtype=torch.float64)
            for vector in vectors:
                combined.add_(vector[start:end].double())
            merged_delta[start:end] = combined.mul_(args.alpha)
        fusion_metadata = {"alpha": args.alpha, "operator": "base_plus_alpha_sum_task_vectors"}
    elif args.method == "ties":
        merged_delta, fusion_metadata = ties_merge_flat_chunked(
            vectors, keep_fraction=args.ties_keep_fraction, alpha=args.alpha,
            disjoint=args.disjoint, chunk_size=args.chunk_size,
        )
    else:
        merged_delta, fusion_metadata = dare_ties_merge_flat_chunked(
            vectors, drop_probability=args.dare_drop_probability, seed=args.dare_seed,
            alpha=args.alpha, disjoint=args.disjoint,
            chunk_size=max(args.chunk_size, 8 * 1024 * 1024),
        )
    del vectors
    gc.collect()

    with safe_open(base_weights, framework="pt", device="cpu") as base:
        dense_tensors = {key: base.get_tensor(key).contiguous() for key in base.keys()}
    for segment in segments:
        key = segment["key"]
        start, end = segment["start"], segment["end"]
        base_value = dense_tensors[key]
        delta = merged_delta[start:end].reshape(segment["shape"])
        dense_tensors[key] = (base_value.double() + delta).to(base_value.dtype).contiguous()
    if any(not torch.isfinite(value).all() for value in dense_tensors.values()):
        raise ValueError("Non-finite output tensor")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.parent.name}-parameter-baseline-", dir=output.parent))
    copy_policy_sidecars(experts[0].root, temporary)
    config_source = experts[0].root / "config.json"
    policy_config = load_json(config_source if config_source.is_file() else base_root / "config.json")
    policy_config["use_peft"] = False
    policy_config["pretrained_path"] = str(output)
    (temporary / "config.json").write_text(json.dumps(policy_config, indent=4) + "\n")
    output_weights = temporary / "model.safetensors"
    save_file(dense_tensors, output_weights, metadata={"format": "pt"})
    manifest = {
        **plan,
        "fusion_metadata": fusion_metadata,
        "model_sha256": sha256(output_weights),
        "modified_tensor_count": len(adapted_keys),
    }
    (temporary / "parameter_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    temporary.rename(output)
    print(json.dumps({"output": str(output), "model_sha256": manifest["model_sha256"],
                      "method": args.method, "coordinates": offset}, indent=2), flush=True)


if __name__ == "__main__":
    main()
