#!/usr/bin/env python3
"""Materialize strict sequential Block RegMean++ for PI0.5.

In full-prefix mode, all 27 Vision blocks, all 18 Language blocks, and all 18
Action Expert blocks are solved in full dense weight space.  Every solved block
is immediately replayed so the next block sees the activation produced by the
already-merged prefix.  The action front-end and output head are likewise
solved in computation-graph order.  No gradient, optimizer, or training step
is used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import mean, median
import tempfile
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

import eval_with_local_tokenizer  # noqa: F401  (runtime compatibility overrides)
from materialize_pi05_dense_regmean import solve_weight


ADAPTER_PREFIX = "base_model.model."
ACTION_LAYER_FRAGMENT = "paligemma_with_expert.gemma_expert.model.layers."
VISION_LAYER_FRAGMENT = (
    "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers."
)
LANGUAGE_LAYER_FRAGMENT = "paligemma_with_expert.paligemma.model.language_model.layers."
ACTION_LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
FRONTEND_MODULES = ("action_in_proj", "time_mlp_in", "time_mlp_out")
OUTPUT_MODULE = "action_out_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial", type=Path, required=True)
    parser.add_argument("--object", dest="object_adapter", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument(
        "--prior-model",
        type=Path,
        help=(
            "Optional materialized dense checkpoint used as the local trust prior. "
            "When omitted, the original 0.5/0.5 expert mean is used."
        ),
    )
    parser.add_argument("--spatial-calibration", type=Path, required=True)
    parser.add_argument("--object-calibration", type=Path, required=True)
    parser.add_argument("--spatial-manifest", type=Path, required=True)
    parser.add_argument("--object-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge-ratio", type=float, default=0.05)
    parser.add_argument(
        "--ridge-scale", choices=("feature_energy", "kernel_diagonal"), default="feature_energy"
    )
    parser.add_argument("--max-correction-ratio", type=float, default=3.0)
    parser.add_argument("--max-rows-per-sample", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-prior-calibration-mismatch",
        action="store_true",
        help=(
            "Allow anchored solving around --prior-model even when the replay "
            "calibration was collected from a different policy. This is intended "
            "for activation-source ablations and is recorded in the manifest."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "median": median(values), "mean": mean(values), "max": max(values)}


def sample_rows(value: torch.Tensor, limit: int) -> torch.Tensor:
    value = value.reshape(-1, value.shape[-1])
    if value.shape[0] > limit:
        indices = torch.linspace(0, value.shape[0] - 1, limit, device=value.device).round().long()
        value = value.index_select(0, indices)
    return value.detach()


@dataclass
class VisionReplayState:
    hidden: torch.Tensor
    attention_mask: torch.Tensor | None


@dataclass
class ReplayState:
    action_input: torch.Tensor
    time_input: torch.Tensor
    hidden_reference: torch.Tensor
    cond_reference: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    prefix_keys: list[torch.Tensor]
    prefix_values: list[torch.Tensor]
    hidden: torch.Tensor | None = None
    cond: torch.Tensor | None = None
    vision: list[VisionReplayState] | None = None
    language_hidden_reference: torch.Tensor | None = None
    language_attention_mask: torch.Tensor | None = None
    language_position_ids: torch.Tensor | None = None
    language_hidden: torch.Tensor | None = None
    flow_index: int | None = None
    request_index: int | None = None
    prompt_signature: int | None = None


@dataclass
class LinearSpec:
    adapter_name: str
    base_name: str
    weight_key: str
    module: torch.nn.Module
    spatial_weight: torch.Tensor
    object_weight: torch.Tensor
    prior: torch.Tensor


def load_replay_states(
    tensor_path: Path,
    manifest_path: Path,
    expected_task: str,
    device: torch.device,
    max_states: int = 0,
    required_flow_index: int | None = None,
) -> tuple[list[ReplayState], dict[str, Any]]:
    manifest = load_json(manifest_path)
    if manifest.get("task") != expected_task:
        raise ValueError(f"Expected {expected_task} calibration, got {manifest.get('task')}")
    count = int(manifest["sample_count"])
    if max_states < 0:
        raise ValueError("max_states must be nonnegative")
    source_indices = list(range(count))
    sample_metadata = manifest.get("samples")
    if required_flow_index is not None:
        if not isinstance(sample_metadata, list) or len(sample_metadata) != count:
            raise ValueError("Flow-index filtering requires aligned manifest sample metadata")
        source_indices = [
            index
            for index, sample in enumerate(sample_metadata)
            if int(sample.get("flow_index", -1)) == required_flow_index
        ]
        if not source_indices:
            raise ValueError(f"No replay states found for flow_index={required_flow_index}")
    if max_states and len(source_indices) > max_states:
        source_count = len(source_indices)
        selected_offsets = (
            torch.linspace(0, source_count - 1, max_states)
            .round()
            .to(dtype=torch.int64)
            .tolist()
        )
        source_indices = [source_indices[offset] for offset in selected_offsets]
    states: list[ReplayState] = []
    with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
        for index in source_indices:
            prefix = f"sample_{index:03d}."
            get = lambda name: tensors.get_tensor(prefix + name).to(device)  # noqa: E731
            full_prefix = manifest.get("method") == (
                "pi05_full_vision_language_action_block_regmeanpp_replay_calibration"
            )
            vision: list[VisionReplayState] | None = None
            language_hidden_reference = None
            language_attention_mask = None
            language_position_ids = None
            if full_prefix:
                vision_count = int(tensors.get_tensor(prefix + "vision_count"))
                vision = []
                keys = set(tensors.keys())
                for vision_index in range(vision_count):
                    vision_prefix = prefix + f"vision_{vision_index:02d}."
                    attention_key = vision_prefix + "attention_mask"
                    vision.append(
                        VisionReplayState(
                            hidden=tensors.get_tensor(vision_prefix + "hidden").to(device),
                            attention_mask=(
                                tensors.get_tensor(attention_key).to(device)
                                if attention_key in keys
                                else None
                            ),
                        )
                    )
                language_hidden_reference = get("language_hidden_mean")
                language_attention_mask = get("language_attention_mask")
                language_position_ids = get("language_position_ids")
            states.append(
                ReplayState(
                    action_input=get("action_input"),
                    time_input=get("time_input"),
                    hidden_reference=get("hidden_mean"),
                    cond_reference=get("cond_mean"),
                    attention_mask=get("attention_mask"),
                    position_ids=get("position_ids"),
                    prefix_keys=[get(f"prefix_key_{layer:02d}") for layer in range(18)],
                    prefix_values=[get(f"prefix_value_{layer:02d}") for layer in range(18)],
                    vision=vision,
                    language_hidden_reference=language_hidden_reference,
                    language_attention_mask=language_attention_mask,
                    language_position_ids=language_position_ids,
                    flow_index=(
                        int(sample_metadata[index]["flow_index"])
                        if isinstance(sample_metadata, list)
                        and sample_metadata[index].get("flow_index") is not None
                        else None
                    ),
                    request_index=(
                        int(sample_metadata[index]["request_index"])
                        if isinstance(sample_metadata, list)
                        and sample_metadata[index].get("request_index") is not None
                        else None
                    ),
                    prompt_signature=(
                        int(sample_metadata[index]["prompt_signature"])
                        if isinstance(sample_metadata, list)
                        and sample_metadata[index].get("prompt_signature") is not None
                        else None
                    ),
                )
            )
    return states, manifest


def resolve_module(modules: dict[str, torch.nn.Module], suffix: str) -> torch.nn.Module:
    if suffix in modules:
        return modules[suffix]
    matches = [module for name, module in modules.items() if name.endswith(suffix)]
    if len(matches) != 1:
        names = [name for name in modules if name.endswith(suffix)]
        raise KeyError(f"Could not uniquely resolve {suffix!r}: {names}")
    return matches[0]


def linear_forward(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    weight = module.weight
    if value.dtype != weight.dtype:
        value = value.to(weight.dtype)
    return F.linear(value, weight, module.bias)


def manual_action_block(
    layer: torch.nn.Module,
    rotary_emb: torch.nn.Module,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    prefix_key: torch.Tensor | None,
    prefix_value: torch.Tensor | None,
    cond: torch.Tensor | None,
) -> torch.Tensor:
    """Run one PiGemma decoder layer without mutating a DynamicCache."""
    from transformers.models.gemma import modeling_gemma
    from lerobot.policies.pi_gemma import _gated_residual, layernorm_forward

    residual = hidden
    normalized, gate = layernorm_forward(layer.input_layernorm, hidden, cond)
    input_shape = normalized.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(normalized).view(hidden_shape).transpose(1, 2)
    key = layer.self_attn.k_proj(normalized).view(hidden_shape).transpose(1, 2)
    value = layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2)
    cos, sin = rotary_emb(normalized, position_ids)
    query, key = modeling_gemma.apply_rotary_pos_emb(query, key, cos, sin)
    if prefix_key is not None or prefix_value is not None:
        if prefix_key is None or prefix_value is None:
            raise ValueError("prefix_key and prefix_value must either both be set or both be None")
        key = torch.cat((prefix_key.to(dtype=key.dtype), key), dim=-2)
        value = torch.cat((prefix_value.to(dtype=value.dtype), value), dim=-2)
    attn_output, _ = modeling_gemma.eager_attention_forward(
        layer.self_attn,
        query,
        key,
        value,
        attention_mask,
        scaling=layer.self_attn.scaling,
        dropout=0.0,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = layer.self_attn.o_proj(attn_output)
    hidden = _gated_residual(residual, attn_output, gate)
    residual = hidden
    normalized, gate = layernorm_forward(layer.post_attention_layernorm, hidden, cond)
    if normalized.dtype != layer.mlp.up_proj.weight.dtype:
        normalized = normalized.to(layer.mlp.up_proj.weight.dtype)
    hidden = layer.mlp(normalized)
    return _gated_residual(residual, hidden, gate)


def decoder_prefix_kv(
    layer: torch.nn.Module,
    rotary_emb: torch.nn.Module,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    cond: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact post-RoPE K/V tensors cached by a Gemma prefix layer."""
    from transformers.models.gemma import modeling_gemma
    from lerobot.policies.pi_gemma import layernorm_forward

    normalized, _ = layernorm_forward(layer.input_layernorm, hidden, cond)
    input_shape = normalized.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(normalized).view(hidden_shape).transpose(1, 2)
    key = layer.self_attn.k_proj(normalized).view(hidden_shape).transpose(1, 2)
    value = layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2)
    cos, sin = rotary_emb(normalized, position_ids)
    _, key = modeling_gemma.apply_rotary_pos_emb(query, key, cos, sin)
    return key, value


def augmented(weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.cat((weight.float(), bias.float().unsqueeze(1)), dim=1)


def with_ones(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    return torch.cat((value, torch.ones((*value.shape[:-1], 1), device=value.device)), dim=-1)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.ridge_ratio <= 0 or args.max_correction_ratio <= 0:
        raise ValueError("ridge-ratio and max-correction-ratio must be positive")
    if args.max_rows_per_sample <= 0:
        raise ValueError("max-rows-per-sample must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    roots = {
        "spatial": args.spatial.expanduser().resolve(),
        "object": args.object_adapter.expanduser().resolve(),
        "base": args.base_model.expanduser().resolve(),
    }
    prior_root = args.prior_model.expanduser().resolve() if args.prior_model else None
    paths = (
        *roots.values(),
        *((prior_root,) if prior_root is not None else ()),
        args.spatial_calibration.expanduser().resolve(),
        args.object_calibration.expanduser().resolve(),
        args.spatial_manifest.expanduser().resolve(),
        args.object_manifest.expanduser().resolve(),
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")

    spatial_config = load_json(roots["spatial"] / "adapter_config.json")
    object_config = load_json(roots["object"] / "adapter_config.json")
    if spatial_config != object_config:
        raise ValueError("Source adapter configs differ")
    rank = int(spatial_config["r"])
    scale = float(spatial_config["lora_alpha"]) / rank

    spatial_states, spatial_manifest = load_replay_states(
        args.spatial_calibration.resolve(), args.spatial_manifest.resolve(), "spatial", device
    )
    object_states, object_manifest = load_replay_states(
        args.object_calibration.resolve(), args.object_manifest.resolve(), "object", device
    )
    if spatial_manifest["calibration_policy"] != object_manifest["calibration_policy"]:
        raise ValueError("Spatial/Object calibration did not use the same merged reference")
    if prior_root is not None and not args.allow_prior_calibration_mismatch:
        calibration_policy = Path(spatial_manifest["calibration_policy"]).resolve()
        if calibration_policy != prior_root:
            raise ValueError(
                "Anchored solve requires the prior model to equal the rollout calibration "
                f"policy: prior={prior_root}, calibration={calibration_policy}"
            )
    full_prefix_method = "pi05_full_vision_language_action_block_regmeanpp_replay_calibration"
    full_prefix = spatial_manifest.get("method") == full_prefix_method
    if (object_manifest.get("method") == full_prefix_method) != full_prefix:
        raise ValueError("Spatial/Object calibration disagree on full-prefix replay mode")

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    print(f"Loading common base runtime model on {device}", flush=True)
    policy = PI05Policy.from_pretrained(roots["base"], device=str(device))
    policy.eval()
    modules = dict(policy.named_modules())
    core = policy.model
    expert_model = core.paligemma_with_expert.gemma_expert.model
    # PI05Pytorch.denoise_step explicitly selects eager attention before every
    # suffix pass.  Replay must use the same implementation; SDPA rejects the
    # float32 additive mask captured from the evaluator with bf16 queries.
    expert_model.config._attn_implementation = "eager"  # noqa: SLF001
    language_model = core.paligemma_with_expert.paligemma.model.language_model
    language_model.config._attn_implementation = "eager"  # noqa: SLF001
    if len(expert_model.layers) != 18:
        raise ValueError(f"Expected 18 Action Expert blocks, found {len(expert_model.layers)}")

    base_weights_path = roots["base"] / "model.safetensors"
    prior_weights_path = prior_root / "model.safetensors" if prior_root is not None else None
    spatial_weights_path = roots["spatial"] / "adapter_model.safetensors"
    object_weights_path = roots["object"] / "adapter_model.safetensors"
    initial_weights_path = prior_weights_path or base_weights_path
    with safe_open(base_weights_path, framework="pt", device="cpu") as base, safe_open(
        initial_weights_path, framework="pt", device="cpu"
    ) as initial:
        if set(base.keys()) != set(initial.keys()):
            raise ValueError("Base and prior checkpoint tensor keys differ")
        output_tensors = {key: initial.get_tensor(key) for key in initial.keys()}

    metrics: dict[str, dict[str, Any]] = {}
    modified_keys: set[str] = set()
    vision_specs: dict[int, list[LinearSpec]] = {index: [] for index in range(27)}
    language_specs: dict[int, list[LinearSpec]] = {index: [] for index in range(18)}
    action_specs: dict[int, list[LinearSpec]] = {index: [] for index in range(18)}
    dense_sources: dict[
        str,
        tuple[
            str,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ] = {}

    with safe_open(spatial_weights_path, framework="pt", device="cpu") as spatial, safe_open(
        object_weights_path, framework="pt", device="cpu"
    ) as object_adapter, safe_open(base_weights_path, framework="pt", device="cpu") as base:
        keys = list(spatial.keys())
        if keys != list(object_adapter.keys()):
            raise ValueError("Source adapter tensor keys differ")

        # Initialize every task-modified module from either the previous merged
        # checkpoint (anchored fixed-point mode) or the original 0.5/0.5 mean.
        # Strict block solutions below then replace modules in graph order.
        for key in keys:
            if ".lora_A." not in key:
                continue
            pair_key = key.replace(".lora_A.", ".lora_B.")
            adapter_name = key.split(".lora_A.", 1)[0]
            if not adapter_name.startswith(ADAPTER_PREFIX):
                raise ValueError(f"Cannot map adapter module {adapter_name}")
            base_name = adapter_name[len(ADAPTER_PREFIX) :]
            weight_key = f"{base_name}.weight"
            base_weight = base.get_tensor(weight_key)
            spatial_weight = base_weight.float() + scale * (
                spatial.get_tensor(pair_key).float() @ spatial.get_tensor(key).float()
            )
            object_weight = base_weight.float() + scale * (
                object_adapter.get_tensor(pair_key).float() @ object_adapter.get_tensor(key).float()
            )
            prior = (
                output_tensors[weight_key].float()
                if prior_weights_path is not None
                else 0.5 * (spatial_weight + object_weight)
            )
            output_tensors[weight_key] = prior.to(base_weight.dtype).contiguous()
            modified_keys.add(weight_key)
            module = resolve_module(modules, base_name)
            module.weight.data.copy_(prior.to(device=device, dtype=module.weight.dtype))
            spec = LinearSpec(
                adapter_name=adapter_name,
                base_name=base_name,
                weight_key=weight_key,
                module=module,
                spatial_weight=spatial_weight,
                object_weight=object_weight,
                prior=prior,
            )
            for fragment, destination in (
                (VISION_LAYER_FRAGMENT, vision_specs),
                (LANGUAGE_LAYER_FRAGMENT, language_specs),
                (ACTION_LAYER_FRAGMENT, action_specs),
            ):
                if fragment in base_name:
                    remainder = base_name.split(fragment, 1)[1]
                    block_index = int(remainder.split(".", 1)[0])
                    destination[block_index].append(spec)
                    break
            else:
                raise ValueError(f"LoRA target is outside known block families: {base_name}")

        dense_names = sorted(
            {
                key.rsplit(".", 1)[0]
                for key in keys
                if ".lora_" not in key and key.endswith((".weight", ".bias"))
            }
        )
        for adapter_name in dense_names:
            base_name = adapter_name[len(ADAPTER_PREFIX) :]
            weight_key = f"{base_name}.weight"
            bias_key = f"{base_name}.bias"
            sw = spatial.get_tensor(f"{adapter_name}.weight").float()
            sb = spatial.get_tensor(f"{adapter_name}.bias").float()
            ow = object_adapter.get_tensor(f"{adapter_name}.weight").float()
            ob = object_adapter.get_tensor(f"{adapter_name}.bias").float()
            mean_weight = 0.5 * (sw + ow)
            mean_bias = 0.5 * (sb + ob)
            prior_weight = (
                output_tensors[weight_key].float()
                if prior_weights_path is not None
                else mean_weight
            )
            prior_bias = (
                output_tensors[bias_key].float()
                if prior_weights_path is not None
                else mean_bias
            )
            output_tensors[weight_key] = prior_weight.to(
                output_tensors[weight_key].dtype
            ).contiguous()
            output_tensors[bias_key] = prior_bias.to(output_tensors[bias_key].dtype).contiguous()
            modified_keys.update((weight_key, bias_key))
            module = resolve_module(modules, base_name)
            module.weight.data.copy_(prior_weight.to(device=device, dtype=module.weight.dtype))
            module.bias.data.copy_(prior_bias.to(device=device, dtype=module.bias.dtype))
            dense_sources[base_name.rsplit(".", 1)[-1]] = (
                adapter_name,
                sw,
                sb,
                ow,
                ob,
                prior_weight,
                prior_bias,
            )

    for block_index, specs in vision_specs.items():
        if len(specs) != 6:
            raise ValueError(f"Vision block {block_index} has {len(specs)} target Linear modules, expected 6")
    for block_index, specs in language_specs.items():
        if len(specs) != 7:
            raise ValueError(f"Language block {block_index} has {len(specs)} target Linear modules, expected 7")
    for block_index, specs in action_specs.items():
        if len(specs) != 7:
            raise ValueError(f"Action block {block_index} has {len(specs)} target Linear modules, expected 7")
        specs.sort(key=lambda spec: ACTION_LINEAR_SUFFIXES.index(spec.base_name.split(f"layers.{block_index}.", 1)[1]))
    if set(dense_sources) != {*FRONTEND_MODULES, OUTPUT_MODULE}:
        raise ValueError(f"Unexpected dense action modules: {sorted(dense_sources)}")

    def solve_dense_module(
        short_name: str, x_spatial: torch.Tensor, x_object: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        _adapter_name, sw, sb, ow, ob, prior_weight, prior_bias = dense_sources[short_name]
        spatial_aug = augmented(sw, sb)
        object_aug = augmented(ow, ob)
        prior = augmented(prior_weight, prior_bias)
        return solve_weight(
            with_ones(x_spatial),
            with_ones(x_object),
            spatial_aug,
            object_aug,
            prior.to(device),
            args.ridge_ratio,
            args.ridge_scale,
            args.max_correction_ratio,
        )

    def assign_source(specs: list[LinearSpec], which: str) -> None:
        for spec in specs:
            weight = spec.spatial_weight if which == "spatial" else spec.object_weight
            spec.module.weight.data.copy_(weight.to(device=device, dtype=spec.module.weight.dtype))

    def capture_inputs(
        specs: list[LinearSpec], which: str, forwards: list[Any]
    ) -> dict[str, torch.Tensor]:
        assign_source(specs, which)
        captures: dict[str, list[torch.Tensor]] = {spec.base_name: [] for spec in specs}
        handles = []
        for spec in specs:
            def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], name: str = spec.base_name) -> None:
                captures[name].append(sample_rows(inputs[0], args.max_rows_per_sample))

            handles.append(spec.module.register_forward_pre_hook(hook))
        try:
            for forward in forwards:
                forward()
        finally:
            for handle in handles:
                handle.remove()
        return {name: torch.cat(values, dim=0) for name, values in captures.items()}

    vision_reference_error: float | None = None
    prefix_kv_reference_error: float | None = None
    if full_prefix:
        all_states = spatial_states + object_states
        if any(
            state.vision is None
            or state.language_hidden_reference is None
            or state.language_attention_mask is None
            or state.language_position_ids is None
            for state in all_states
        ):
            raise ValueError("Full-prefix calibration is missing vision/language replay tensors")

        vision_transformer = (
            core.paligemma_with_expert.paligemma.model.vision_tower.vision_model
        )
        vision_layers = vision_transformer.encoder.layers
        projector = core.paligemma_with_expert.paligemma.model.multi_modal_projector
        if len(vision_layers) != 27:
            raise ValueError(f"Expected 27 Vision blocks, found {len(vision_layers)}")

        # Validate that the stored block-0 inputs reproduce the visual segment
        # of the rollout-reference language prefix before changing Vision.
        vision_errors: list[float] = []
        for state in all_states:
            assert state.vision is not None and state.language_hidden_reference is not None
            projected: list[torch.Tensor] = []
            for vision_state in state.vision:
                value = vision_state.hidden
                for layer in vision_layers:
                    value = layer(value, vision_state.attention_mask)
                value = vision_transformer.post_layernorm(value)
                projected.append(projector(value))
            visual = torch.cat(projected, dim=1)
            reference = state.language_hidden_reference[:, : visual.shape[1]]
            vision_errors.append(float((visual.float() - reference.float()).abs().max()))
        vision_reference_error = max(vision_errors)
        print(
            f"Reference replay Vision-to-language max error: {vision_reference_error:.6g}",
            flush=True,
        )

        # Independently verify that reference language replay rebuilds the exact
        # prefix K/V cache used by the paired action denoising state.
        kv_errors: list[float] = []
        for state in all_states:
            assert state.language_hidden_reference is not None
            assert state.language_attention_mask is not None
            assert state.language_position_ids is not None
            value = state.language_hidden_reference
            for block_index, layer in enumerate(language_model.layers):
                key, cached_value = decoder_prefix_kv(
                    layer, language_model.rotary_emb, value, state.language_position_ids, None
                )
                kv_errors.extend(
                    (
                        float((key.float() - state.prefix_keys[block_index].float()).abs().max()),
                        float(
                            (cached_value.float() - state.prefix_values[block_index].float())
                            .abs()
                            .max()
                        ),
                    )
                )
                value = manual_action_block(
                    layer,
                    language_model.rotary_emb,
                    value,
                    state.language_attention_mask,
                    state.language_position_ids,
                    None,
                    None,
                    None,
                )
        prefix_kv_reference_error = max(kv_errors)
        print(
            f"Reference replay Language-KV max error: {prefix_kv_reference_error:.6g}",
            flush=True,
        )

        # Strict sequential Vision RegMean++: each next block consumes the
        # output of the already merged Vision prefix.
        def vision_items(states: list[ReplayState]) -> list[VisionReplayState]:
            return [item for state in states for item in (state.vision or [])]

        for block_index in range(27):
            specs = vision_specs[block_index]
            layer = vision_layers[block_index]
            spatial_items = vision_items(spatial_states)
            object_items = vision_items(object_states)
            spatial_inputs = capture_inputs(
                specs,
                "spatial",
                [
                    (lambda item=item: layer(item.hidden, item.attention_mask))
                    for item in spatial_items
                ],
            )
            object_inputs = capture_inputs(
                specs,
                "object",
                [
                    (lambda item=item: layer(item.hidden, item.attention_mask))
                    for item in object_items
                ],
            )
            for spec in specs:
                merged, module_metrics = solve_weight(
                    spatial_inputs[spec.base_name],
                    object_inputs[spec.base_name],
                    spec.spatial_weight,
                    spec.object_weight,
                    spec.prior.to(device),
                    args.ridge_ratio,
                    args.ridge_scale,
                    args.max_correction_ratio,
                )
                spec.module.weight.data.copy_(merged.to(device=device, dtype=spec.module.weight.dtype))
                output_tensors[spec.weight_key] = merged.to(
                    output_tensors[spec.weight_key].dtype
                ).contiguous()
                metrics[spec.base_name] = {
                    "stage": "vision_block_sequential",
                    "block_index": block_index,
                    "kind": "lora_materialized_dense",
                    **module_metrics,
                }
            for item in spatial_items + object_items:
                item.hidden = layer(item.hidden, item.attention_mask)
            del spatial_inputs, object_inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"Merged Vision block {block_index:02d}/26", flush=True)

        # Rebuild the actual language block-0 input with merged visual tokens;
        # language token embeddings remain the paired reference tail.
        language_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        for state in all_states:
            assert state.vision is not None and state.language_hidden_reference is not None
            projected = []
            for vision_state in state.vision:
                value = vision_transformer.post_layernorm(vision_state.hidden)
                projected.append(projector(value))
            visual = torch.cat(projected, dim=1)
            tail = state.language_hidden_reference[:, visual.shape[1] :]
            state.language_hidden = torch.cat((visual.to(tail.dtype), tail), dim=1).to(language_dtype)

        # Strict sequential Language RegMean++; after each merged block, cache
        # its merged K/V for the paired Action Expert replay.
        for block_index in range(18):
            specs = language_specs[block_index]
            layer = language_model.layers[block_index]

            def language_forwards(states: list[ReplayState]) -> list[Any]:
                forwards = []
                for state in states:
                    assert state.language_hidden is not None
                    assert state.language_attention_mask is not None
                    assert state.language_position_ids is not None
                    forwards.append(
                        lambda state=state: manual_action_block(
                            layer,
                            language_model.rotary_emb,
                            state.language_hidden,
                            state.language_attention_mask,
                            state.language_position_ids,
                            None,
                            None,
                            None,
                        )
                    )
                return forwards

            spatial_inputs = capture_inputs(specs, "spatial", language_forwards(spatial_states))
            object_inputs = capture_inputs(specs, "object", language_forwards(object_states))
            for spec in specs:
                merged, module_metrics = solve_weight(
                    spatial_inputs[spec.base_name],
                    object_inputs[spec.base_name],
                    spec.spatial_weight,
                    spec.object_weight,
                    spec.prior.to(device),
                    args.ridge_ratio,
                    args.ridge_scale,
                    args.max_correction_ratio,
                )
                spec.module.weight.data.copy_(merged.to(device=device, dtype=spec.module.weight.dtype))
                output_tensors[spec.weight_key] = merged.to(
                    output_tensors[spec.weight_key].dtype
                ).contiguous()
                metrics[spec.base_name] = {
                    "stage": "language_block_sequential",
                    "block_index": block_index,
                    "kind": "lora_materialized_dense",
                    **module_metrics,
                }
            for state in all_states:
                assert state.language_hidden is not None
                assert state.language_attention_mask is not None
                assert state.language_position_ids is not None
                key, value = decoder_prefix_kv(
                    layer,
                    language_model.rotary_emb,
                    state.language_hidden,
                    state.language_position_ids,
                    None,
                )
                state.prefix_keys[block_index] = key
                state.prefix_values[block_index] = value
                state.language_hidden = manual_action_block(
                    layer,
                    language_model.rotary_emb,
                    state.language_hidden,
                    state.language_attention_mask,
                    state.language_position_ids,
                    None,
                    None,
                    None,
                )
            del spatial_inputs, object_inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"Merged Language block {block_index:02d}/17", flush=True)

    # Check that the serialized records reconstruct the rollout-reference
    # policy before changing any front-end module.
    reference_errors: list[float] = []
    for state in spatial_states + object_states:
        action = linear_forward(core.action_in_proj, state.action_input)
        time = F.silu(linear_forward(core.time_mlp_in, state.time_input))
        cond = F.silu(linear_forward(core.time_mlp_out, time))
        reference_errors.extend(
            (
                float((action.float() - state.hidden_reference.float()).abs().max()),
                float((cond.float() - state.cond_reference.float()).abs().max()),
            )
        )
    print(f"Reference replay front-end max error: {max(reference_errors):.6g}", flush=True)

    # Guard the hand-written single-block replay against the exact transformers
    # layer implementation before using it for all 18 sequential steps.
    from transformers.cache_utils import DynamicCache

    validation_state = spatial_states[0]
    validation_cache = DynamicCache(
        tuple(
            (keys.clone(), values.clone(), None)
            for keys, values in zip(
                validation_state.prefix_keys, validation_state.prefix_values, strict=True
            )
        )
    )
    validation_position_embeddings = expert_model.rotary_emb(
        validation_state.hidden_reference, validation_state.position_ids
    )
    validation_cache_position = torch.arange(
        validation_state.prefix_keys[0].shape[-2],
        validation_state.prefix_keys[0].shape[-2] + validation_state.hidden_reference.shape[-2],
        device=device,
    )
    native_block_output = expert_model.layers[0](
        validation_state.hidden_reference,
        attention_mask=validation_state.attention_mask,
        position_ids=validation_state.position_ids,
        past_key_values=validation_cache,
        use_cache=False,
        cache_position=validation_cache_position,
        position_embeddings=validation_position_embeddings,
        adarms_cond=validation_state.cond_reference,
    )
    manual_block_output = manual_action_block(
        expert_model.layers[0],
        expert_model.rotary_emb,
        validation_state.hidden_reference,
        validation_state.attention_mask,
        validation_state.position_ids,
        validation_state.prefix_keys[0],
        validation_state.prefix_values[0],
        validation_state.cond_reference,
    )
    manual_replay_error = float(
        (native_block_output.float() - manual_block_output.float()).abs().max()
    )
    if manual_replay_error > 0.02:
        raise RuntimeError(
            f"Manual Action block replay disagrees with native layer: {manual_replay_error}"
        )
    print(f"Manual/native Action block max error: {manual_replay_error:.6g}", flush=True)

    # action_in_proj and time_mlp_in are independent front-end branches.
    action_solution, action_metrics = solve_dense_module(
        "action_in_proj",
        torch.cat([sample_rows(s.action_input, args.max_rows_per_sample) for s in spatial_states]),
        torch.cat([sample_rows(s.action_input, args.max_rows_per_sample) for s in object_states]),
    )
    time_in_solution, time_in_metrics = solve_dense_module(
        "time_mlp_in",
        torch.cat([sample_rows(s.time_input, args.max_rows_per_sample) for s in spatial_states]),
        torch.cat([sample_rows(s.time_input, args.max_rows_per_sample) for s in object_states]),
    )
    for short_name, solution, module_metrics in (
        ("action_in_proj", action_solution, action_metrics),
        ("time_mlp_in", time_in_solution, time_in_metrics),
    ):
        module = getattr(core, short_name)
        module.weight.data.copy_(solution[:, :-1].to(device=device, dtype=module.weight.dtype))
        module.bias.data.copy_(solution[:, -1].to(device=device, dtype=module.bias.dtype))
        base_name = f"model.{short_name}"
        output_tensors[f"{base_name}.weight"] = solution[:, :-1].to(output_tensors[f"{base_name}.weight"].dtype)
        output_tensors[f"{base_name}.bias"] = solution[:, -1].to(output_tensors[f"{base_name}.bias"].dtype)
        metrics[base_name] = {"stage": "front_end", "kind": "dense_augmented", **module_metrics}

    # time_mlp_out is calibrated on the activation produced by the already
    # merged time_mlp_in, which is the module-level RegMean++ dependency.
    spatial_time_hidden = torch.cat(
        [sample_rows(F.silu(linear_forward(core.time_mlp_in, s.time_input)), args.max_rows_per_sample) for s in spatial_states]
    )
    object_time_hidden = torch.cat(
        [sample_rows(F.silu(linear_forward(core.time_mlp_in, s.time_input)), args.max_rows_per_sample) for s in object_states]
    )
    time_out_solution, time_out_metrics = solve_dense_module(
        "time_mlp_out", spatial_time_hidden, object_time_hidden
    )
    core.time_mlp_out.weight.data.copy_(
        time_out_solution[:, :-1].to(device=device, dtype=core.time_mlp_out.weight.dtype)
    )
    core.time_mlp_out.bias.data.copy_(
        time_out_solution[:, -1].to(device=device, dtype=core.time_mlp_out.bias.dtype)
    )
    output_tensors["model.time_mlp_out.weight"] = time_out_solution[:, :-1].to(
        output_tensors["model.time_mlp_out.weight"].dtype
    )
    output_tensors["model.time_mlp_out.bias"] = time_out_solution[:, -1].to(
        output_tensors["model.time_mlp_out.bias"].dtype
    )
    metrics["model.time_mlp_out"] = {
        "stage": "front_end_sequential",
        "kind": "dense_augmented",
        **time_out_metrics,
    }

    first_weight_dtype = expert_model.layers[0].self_attn.q_proj.weight.dtype
    for state in spatial_states + object_states:
        state.hidden = linear_forward(core.action_in_proj, state.action_input).to(first_weight_dtype)
        time = F.silu(linear_forward(core.time_mlp_in, state.time_input))
        state.cond = F.silu(linear_forward(core.time_mlp_out, time))

    def run_and_capture(
        block_index: int, states: list[ReplayState], specs: list[LinearSpec], which: str
    ) -> dict[str, torch.Tensor]:
        assign_source(specs, which)
        captures: dict[str, list[torch.Tensor]] = {spec.base_name: [] for spec in specs}
        handles = []
        for spec in specs:
            def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], name: str = spec.base_name) -> None:
                captures[name].append(sample_rows(inputs[0], args.max_rows_per_sample))

            handles.append(spec.module.register_forward_pre_hook(hook))
        try:
            for state in states:
                assert state.hidden is not None and state.cond is not None
                manual_action_block(
                    expert_model.layers[block_index],
                    expert_model.rotary_emb,
                    state.hidden,
                    state.attention_mask,
                    state.position_ids,
                    state.prefix_keys[block_index],
                    state.prefix_values[block_index],
                    state.cond,
                )
        finally:
            for handle in handles:
                handle.remove()
        return {name: torch.cat(values, dim=0) for name, values in captures.items()}

    for block_index in range(18):
        specs = action_specs[block_index]
        spatial_inputs = run_and_capture(block_index, spatial_states, specs, "spatial")
        object_inputs = run_and_capture(block_index, object_states, specs, "object")
        for spec in specs:
            merged, module_metrics = solve_weight(
                spatial_inputs[spec.base_name],
                object_inputs[spec.base_name],
                spec.spatial_weight,
                spec.object_weight,
                spec.prior.to(device),
                args.ridge_ratio,
                args.ridge_scale,
                args.max_correction_ratio,
            )
            if not torch.isfinite(merged).all():
                raise ValueError(f"Non-finite solution for {spec.base_name}")
            spec.module.weight.data.copy_(merged.to(device=device, dtype=spec.module.weight.dtype))
            output_tensors[spec.weight_key] = merged.to(output_tensors[spec.weight_key].dtype).contiguous()
            metrics[spec.base_name] = {
                "stage": "action_block_sequential",
                "block_index": block_index,
                "kind": "lora_materialized_dense",
                **module_metrics,
            }

        # This propagation is the defining RegMean++ step: block l+1 will see
        # the output of merged blocks 0..l, not either expert's private prefix.
        for state in spatial_states + object_states:
            assert state.hidden is not None and state.cond is not None
            state.hidden = manual_action_block(
                expert_model.layers[block_index],
                expert_model.rotary_emb,
                state.hidden,
                state.attention_mask,
                state.position_ids,
                state.prefix_keys[block_index],
                state.prefix_values[block_index],
                state.cond,
            )
        del spatial_inputs, object_inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        block_improvements = [
            metrics[spec.base_name]["improvement_vs_prior"] for spec in specs
        ]
        print(
            f"Merged Action block {block_index:02d}/17: "
            f"mean offline improvement={mean(block_improvements):.2%}",
            flush=True,
        )

    def final_hidden(state: ReplayState) -> torch.Tensor:
        assert state.hidden is not None and state.cond is not None
        value, _ = expert_model.norm(state.hidden, state.cond)
        return value[:, -core.config.chunk_size :].float()

    spatial_final = torch.cat(
        [sample_rows(final_hidden(state), args.max_rows_per_sample) for state in spatial_states]
    )
    object_final = torch.cat(
        [sample_rows(final_hidden(state), args.max_rows_per_sample) for state in object_states]
    )
    output_solution, output_metrics = solve_dense_module(
        OUTPUT_MODULE, spatial_final, object_final
    )
    output_tensors["model.action_out_proj.weight"] = output_solution[:, :-1].to(
        output_tensors["model.action_out_proj.weight"].dtype
    )
    output_tensors["model.action_out_proj.bias"] = output_solution[:, -1].to(
        output_tensors["model.action_out_proj.bias"].dtype
    )
    metrics["model.action_out_proj"] = {
        "stage": "output_head_sequential",
        "kind": "dense_augmented",
        **output_metrics,
    }

    expected_modified = 414 + 2 * 4
    if len(modified_keys) != expected_modified:
        raise ValueError(f"Expected {expected_modified} modified tensors, found {len(modified_keys)}")
    if any(not torch.isfinite(tensor).all() for tensor in output_tensors.values()):
        raise ValueError("Output checkpoint contains NaN or Inf")
    # Slicing the augmented [W | b] solutions produces strided views.  A
    # safetensors checkpoint requires every serialized tensor to be contiguous.
    output_tensors = {key: tensor.contiguous() for key, tensor in output_tensors.items()}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.parent.name}-block-regmeanpp-", dir=output.parent))
    support_files = (
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    for name in support_files:
        first = roots["spatial"] / name
        second = roots["object"] / name
        if first.read_bytes() != second.read_bytes():
            raise ValueError(f"Expert support file differs: {name}")
        shutil.copy2(first, temporary / name)
    shutil.copytree(roots["spatial"] / "tokenizer", temporary / "tokenizer")
    config = load_json(roots["spatial"] / "config.json")
    config["use_peft"] = False
    config["pretrained_path"] = str(output)
    (temporary / "config.json").write_text(
        json.dumps(config, indent=4, sort_keys=False) + "\n", encoding="utf-8"
    )
    output_weights = temporary / "model.safetensors"
    save_file(output_tensors, output_weights, metadata={"format": "pt"})

    improvements = [entry["improvement_vs_prior"] for entry in metrics.values()]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": (
            "pi05_strict_sequential_vision27_language18_action18_block_regmeanpp"
            if full_prefix
            else "pi05_strict_sequential_action18_block_regmeanpp"
        ),
        "training": False,
        "gradient_or_backward": False,
        "parameterization": "full_linear_weight_space",
        "local_prior": (
            "rollout_reference_dense_checkpoint"
            if prior_root is not None
            else "expert_mean_0p5_0p5"
        ),
        "vision_language_merge": (
            "27_vision_and_18_language_blocks_sequential_dense_regmean"
            if full_prefix
            else "dense_mean_0p5_0p5"
        ),
        "action_frontend_merge": "sequential_dense_regmean",
        "action_expert_merge": "18_blocks_sequential_merged_prefix_dense_regmean",
        "action_output_merge": "merged_final_hidden_dense_regmean",
        "ridge_ratio": args.ridge_ratio,
        "ridge_scale": args.ridge_scale,
        "max_correction_ratio": args.max_correction_ratio,
        "max_rows_per_sample": args.max_rows_per_sample,
        "module_count": len(metrics),
        "modified_tensor_count": len(modified_keys),
        "replay_reference_max_error": max(reference_errors),
        "manual_native_block_max_error": manual_replay_error,
        "vision_reference_max_error": vision_reference_error,
        "prefix_kv_reference_max_error": prefix_kv_reference_error,
        "improvement_summary": summarize(improvements),
        "trust_limited_module_count": sum(entry["trust_scale"] < 1.0 for entry in metrics.values()),
        "inputs": {
            "spatial_adapter": str(roots["spatial"]),
            "object_adapter": str(roots["object"]),
            "base_model": str(roots["base"]),
            "prior_model": str(prior_root) if prior_root is not None else None,
            "calibration_policy": spatial_manifest["calibration_policy"],
            "allow_prior_calibration_mismatch": args.allow_prior_calibration_mismatch,
            "spatial_calibration": str(args.spatial_calibration.resolve()),
            "object_calibration": str(args.object_calibration.resolve()),
            "spatial_sample_count": len(spatial_states),
            "object_sample_count": len(object_states),
            "spatial_adapter_sha256": sha256(spatial_weights_path),
            "object_adapter_sha256": sha256(object_weights_path),
        },
        "output": str(output),
        "model_sha256": sha256(output_weights),
        "modules": metrics,
    }
    (temporary / "block_regmeanpp_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.rename(output)
    print(
        json.dumps(
            {
                "method": manifest["method"],
                "module_count": manifest["module_count"],
                "modified_tensor_count": manifest["modified_tensor_count"],
                "replay_reference_max_error": manifest["replay_reference_max_error"],
                "improvement_summary": manifest["improvement_summary"],
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
