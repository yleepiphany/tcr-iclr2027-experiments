#!/usr/bin/env python3
"""Isolated full-scope solver for the fixed early-request A→B control.

Numerical code is copied from the audited TCR-E full-418 solver.  The added contract
accepts only the demo replacement, pins the matching Full repeat's 418 numeric ridges,
uses the legacy cap-10 allocation in pass A, and the shared cap-16 allocation in pass B.
It also accepts the independently registered fresh matched Full/last-call/
expert-prefix batch; that batch uses new common-budget inputs and is never
relabeled as the historical headline Full.

Each LoRA expert is first treated as a dense full-weight expert:

    W_i = exact frozen dense checkpoint tensor, converted to float32 for solve

Then every target Linear is solved in full dense weight space from replay
calibration activations.  In full-prefix mode, Vision, Language and Action
blocks are merged in graph order and replayed after each block, matching the
RegMean++ "merged prefix produces next-layer activations" rule.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
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

import eval_with_local_tokenizer  # noqa: F401
from pi05_replay_prefix import advance_by_prefix
from materialize_pi05_block_regmeanpp import (
    ACTION_LAYER_FRAGMENT,
    ACTION_LINEAR_SUFFIXES,
    ADAPTER_PREFIX,
    FRONTEND_MODULES,
    LANGUAGE_LAYER_FRAGMENT,
    OUTPUT_MODULE,
    VISION_LAYER_FRAGMENT,
    ReplayState,
    augmented,
    decoder_prefix_kv,
    linear_forward,
    load_json,
    load_replay_states,
    manual_action_block,
    resolve_module,
    sample_rows,
    with_ones,
)


def parse_binding(raw: str, kind: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"{kind} must be NAME=PATH, got {raw!r}")
    name, path = raw.split("=", 1)
    name = name.strip().lower()
    if not name or not path:
        raise ValueError(f"Invalid {kind}: {raw!r}")
    return name, Path(path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--dense-expert-bank", type=Path, required=True)
    parser.add_argument("--calibration", action="append", required=True, metavar="NAME=FILE")
    parser.add_argument("--manifest", action="append", required=True, metavar="NAME=FILE")
    parser.add_argument(
        "--calibration-source-weight",
        action="append",
        default=[],
        metavar="NAME=FLOAT",
        help=(
            "Optional positive mass for each repeated calibration source, in the same "
            "per-expert order as --calibration/--manifest. Omit to weight every source "
            "by 1.0. This controls expert-success versus deployed-policy occupancy with "
            "one global continuous trust parameter rather than suite-specific weights."
        ),
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--prior-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ablation-config", type=Path, help="Opt-in Table-3 single-repeat contract; never relabel as the legacy Table-1 build")
    parser.add_argument("--ridge-ratio", type=float, default=0.05)
    parser.add_argument(
        "--ridge-scale", choices=("feature_energy", "kernel_diagonal"), default="feature_energy"
    )
    parser.add_argument("--max-correction-ratio", type=float, default=3.0)
    parser.add_argument("--max-rows-per-sample", type=int, default=16)
    parser.add_argument(
        "--expert-loss-normalization",
        choices=("none", "prior", "hardness"),
        default="none",
        help=(
            "Set automatic objective weighting: none is uniform, prior uses inverse prior "
            "loss to optimize relative degradation, and hardness emphasizes groups whose "
            "prior is currently farthest from their expert target."
        ),
    )
    parser.add_argument(
        "--expert-loss-normalization-power",
        type=float,
        default=1.0,
        help=(
            "Exponent gamma for prior normalization: objective weight_i is proportional "
            "to prior_loss_i**(-gamma). Values below one temper noisy module-level weights."
        ),
    )
    parser.add_argument(
        "--expert-aggregation",
        choices=("mean", "minimax_relative_regret", "fixed_weight_plan"),
        default="mean",
        help=(
            "Aggregate normalized expert function losses by their weighted mean, or use "
            "an automatic exponentiated-dual outer loop that minimizes the worst loss "
            "relative to the local prior. fixed_weight_plan consumes graph-coherent "
            "per-module expert weights from --expert-weight-plan. These modes use no "
            "reward or suite labels."
        ),
    )
    parser.add_argument(
        "--expert-weight-plan",
        type=Path,
        help=(
            "JSON plan containing modules.<module>.expert_objective_weights. Required "
            "for fixed_weight_plan and intended for graph-coherent minimax dual weights."
        ),
    )
    parser.add_argument(
        "--minimax-iterations",
        type=int,
        default=4,
        help="Number of closed-form primal solves in the minimax expert-weight outer loop.",
    )
    parser.add_argument(
        "--minimax-temperature",
        type=float,
        default=1.0,
        help="Exponentiated-dual step size for automatic worst-expert reweighting.",
    )
    parser.add_argument(
        "--require-objective-improvement",
        action="store_true",
        help=(
            "Reject a module correction and retain its local prior unless the solved "
            "weight improves the same weighted calibration objective used by the solver."
        ),
    )
    parser.add_argument(
        "--objective-grouping",
        choices=("expert", "expert_prompt"),
        default="expert",
        help=(
            "Balance calibration either per expert or per expert-by-natural-language-prompt. "
            "Prompt grouping uses collector prompt signatures and remains a static merge."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--replay-prefix", choices=("merged", "expert"), default="merged",
        help=(
            "merged: strict Block RegMean++ boundary replay (existing behavior). "
            "expert: propagate each candidate's own Vision/Language/action/time "
            "prefix for a matched-data dense RegMean control. Full-prefix replay "
            "and independent Linear solves are required in expert mode."
        ),
    )
    parser.add_argument(
        "--allow-prior-calibration-mismatch",
        action="store_true",
        help="Allow --prior-model to differ from the rollout calibration policy.",
    )
    parser.add_argument(
        "--allow-mixed-calibration-policies",
        action="store_true",
        help=(
            "Allow each expert replay file to be collected from a different "
            "policy. This is intended for per-expert successful-trajectory "
            "calibration in M-way merging."
        ),
    )
    parser.add_argument(
        "--freeze-prefix-from-prior",
        action="store_true",
        help=(
            "Keep the prior checkpoint's Vision and Language blocks fixed while replaying "
            "their outputs into trajectory-calibrated Action Expert refinement. Requires "
            "--prior-model and full-prefix replay calibration."
        ),
    )
    parser.add_argument(
        "--freeze-action-interface-from-prior",
        action="store_true",
        help=(
            "Keep action_in_proj, time_mlp_in/out and action_out_proj at the prior. "
            "This isolates Action-Expert block refinement and requires --prior-model."
        ),
    )
    parser.add_argument(
        "--action-block-solver",
        choices=(
            "independent_linear",
            "joint_subspace",
            "joint_path_subspace",
            "joint_flow_terminal_subspace",
        ),
        default="independent_linear",
        help=(
            "Use the existing per-Linear dense RegMean++ solve, or jointly solve each "
            "Action block from its final residual output in an M-expert task-delta subspace."
        ),
    )
    parser.add_argument(
        "--joint-subspace-grouping",
        choices=("family", "module"),
        default="family",
        help=(
            "For joint_subspace, share one coefficient per expert across Q/K/V/O and "
            "another across gate/up/down (family), or use one per expert and Linear (module)."
        ),
    )
    parser.add_argument("--joint-finite-difference", type=float, default=0.1)
    parser.add_argument("--joint-ridge-ratio", type=float, default=0.05)
    parser.add_argument("--joint-max-states-per-expert", type=int, default=40)
    parser.add_argument("--joint-max-correction-ratio", type=float, default=1.0)
    parser.add_argument(
        "--joint-path-layer-groups",
        type=int,
        default=3,
        help=(
            "Number of contiguous early/middle/late Action-layer groups for "
            "joint_path_subspace. Each group has one Attention and one MLP direction per expert."
        ),
    )
    parser.add_argument(
        "--max-states-per-calibration-source",
        type=int,
        default=0,
        help=(
            "Deterministically subsample each replay source before all prefix/block replay. "
            "Zero keeps every state; positive values are useful for implementation smoke tests."
        ),
    )
    return parser.parse_args()


def assignments(values: list[str], kind: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for raw in values:
        name, path = parse_binding(raw, kind)
        if name in parsed:
            raise ValueError(f"Duplicate {kind} name: {name}")
        parsed[name] = path
    return parsed


def multi_assignments(values: list[str], kind: str) -> dict[str, list[Path]]:
    """Parse repeatable NAME=PATH bindings while retaining per-name source order."""
    parsed: dict[str, list[Path]] = {}
    for raw in values:
        name, path = parse_binding(raw, kind)
        parsed.setdefault(name, []).append(path)
    return parsed


def multi_float_assignments(values: list[str], kind: str) -> dict[str, list[float]]:
    """Parse repeatable NAME=FLOAT bindings while retaining per-name source order."""
    parsed: dict[str, list[float]] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid {kind}: {raw!r}")
        name, raw_value = raw.split("=", 1)
        name = name.strip().lower()
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"Invalid {kind}: {raw!r}") from error
        if not name or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {kind}: {raw!r}")
        parsed.setdefault(name, []).append(value)
    return parsed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "median": median(values), "mean": mean(values), "max": max(values)}


def task_loss(x: torch.Tensor, weight: torch.Tensor, target: torch.Tensor) -> float:
    residual = x @ (weight - target).T
    return float((residual * residual).mean())


def concatenate_grouped(
    values: list[torch.Tensor], labels: list[str]
) -> dict[str, torch.Tensor]:
    if len(values) != len(labels):
        raise ValueError(f"Grouped value/label mismatch: {len(values)} != {len(labels)}")
    grouped: dict[str, list[torch.Tensor]] = {}
    for value, label in zip(values, labels, strict=True):
        grouped.setdefault(label, []).append(value)
    if not grouped:
        raise ValueError("Cannot construct an empty objective group")
    return {label: torch.cat(rows, dim=0) for label, rows in grouped.items()}


def solve_weight_multi(
    inputs: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
    prior: torch.Tensor,
    ridge_ratio: float,
    ridge_scale: str,
    max_correction_ratio: float,
    expert_loss_normalization: str = "none",
    expert_loss_normalization_power: float = 1.0,
    require_objective_improvement: bool = False,
    expert_aggregation: str = "mean",
    minimax_iterations: int = 4,
    minimax_temperature: float = 1.0,
    objective_weight_override: dict[str, float] | None = None,
    fixed_ridge: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    names = list(inputs)
    if set(names) != set(weights):
        raise ValueError("Input/weight expert names differ")
    device = prior.device
    prior = prior.to(dtype=torch.float32)
    prepared_inputs: dict[str, torch.Tensor] = {}
    prepared_weights: dict[str, torch.Tensor] = {}
    row_counts: dict[str, int] = {}
    for name in names:
        x_i = inputs[name].to(device=device, dtype=torch.float32)
        w_i = weights[name].to(device=device, dtype=torch.float32)
        if x_i.shape[1] != prior.shape[1] or w_i.shape != prior.shape:
            raise ValueError(f"{name}: activation/weight mismatch {tuple(x_i.shape)} {tuple(w_i.shape)}")
        prepared_inputs[name] = x_i
        prepared_weights[name] = w_i
        row_counts[name] = int(x_i.shape[0])

    prior_loss_by_expert = {
        name: task_loss(prepared_inputs[name], prior, prepared_weights[name]) for name in names
    }
    if expert_loss_normalization == "none":
        base_objective_weights = {name: 1.0 / len(names) for name in names}
        normalization_floor = None
    elif expert_loss_normalization == "prior":
        mean_prior_loss = sum(prior_loss_by_expert.values()) / len(names)
        normalization_floor = max(mean_prior_loss * 1e-6, 1e-12)
        inverse_losses = {
            name: max(prior_loss_by_expert[name], normalization_floor)
            ** (-expert_loss_normalization_power)
            for name in names
        }
        inverse_sum = sum(inverse_losses.values())
        base_objective_weights = {
            name: inverse_losses[name] / inverse_sum for name in names
        }
    elif expert_loss_normalization == "hardness":
        mean_prior_loss = sum(prior_loss_by_expert.values()) / len(names)
        normalization_floor = max(mean_prior_loss * 1e-6, 1e-12)
        hardness = {
            name: max(prior_loss_by_expert[name], normalization_floor)
            ** expert_loss_normalization_power
            for name in names
        }
        hardness_sum = sum(hardness.values())
        base_objective_weights = {name: hardness[name] / hardness_sum for name in names}
    else:
        raise ValueError(f"Unsupported expert loss normalization: {expert_loss_normalization}")

    task_delta_reference = sum(
        float(torch.linalg.vector_norm(weights[name].to(device=device, dtype=torch.float32) - prior))
        for name in names
    ) / len(names)

    def normalized(raw: dict[str, float]) -> dict[str, float]:
        total = sum(raw.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"Invalid expert objective masses: {raw}")
        return {name: raw[name] / total for name in names}

    def solve_once(objective_weights: dict[str, float]) -> dict[str, Any]:
        scaled_x: list[torch.Tensor] = []
        scaled_targets: list[torch.Tensor] = []
        for name in names:
            x_i = prepared_inputs[name]
            w_i = prepared_weights[name]
            scale = math.sqrt(x_i.shape[0] / objective_weights[name])
            scaled_x.append(x_i / scale)
            scaled_targets.append((x_i @ (w_i - prior).T) / scale)
        x = torch.cat(scaled_x, dim=0)
        residual_target = torch.cat(scaled_targets, dim=0)
        kernel = x @ x.T
        feature_energy = max(float((x * x).sum() / x.shape[1]), 1e-12)
        kernel_diagonal = max(float(torch.diagonal(kernel).mean()), 1e-12)
        ridge_reference = (
            feature_energy if ridge_scale == "feature_energy" else kernel_diagonal
        )
        ridge = ridge_ratio * ridge_reference if fixed_ridge is None else fixed_ridge
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError('Fixed ridge must be positive and finite')
        kernel.diagonal().add_(ridge)
        cholesky, info = torch.linalg.cholesky_ex(kernel)
        if int(info.max()) != 0:
            raise RuntimeError(
                f"RegMean++ kernel is not positive definite; info={int(info.max())}"
            )
        alpha = torch.cholesky_solve(residual_target, cholesky)
        correction = alpha.T @ x
        raw_correction_norm = float(torch.linalg.vector_norm(correction))
        trust_scale = 1.0
        limit = max_correction_ratio * max(task_delta_reference, 1e-12)
        if raw_correction_norm > limit:
            trust_scale = limit / raw_correction_norm
            correction = correction * trust_scale
        merged = prior + correction
        losses = {
            name: task_loss(prepared_inputs[name], merged, prepared_weights[name])
            for name in names
        }
        relative_losses = {
            name: losses[name] / max(prior_loss_by_expert[name], 1e-12)
            for name in names
        }
        return {
            "merged": merged,
            "correction": correction,
            "losses": losses,
            "relative_losses": relative_losses,
            "robust_objective": max(relative_losses.values()),
            "objective_weights": dict(objective_weights),
            "feature_energy": feature_energy,
            "kernel_diagonal": kernel_diagonal,
            "ridge_reference": ridge_reference,
            "ridge": ridge,
            "kernel": kernel,
            "cholesky": cholesky,
            "trust_scale": trust_scale,
        }

    minimax_history: list[dict[str, Any]] = []
    if expert_aggregation == "mean":
        if objective_weight_override is not None:
            raise ValueError("mean aggregation does not accept an objective-weight override")
        selected = solve_once(base_objective_weights)
        selected_iteration: int | None = None
        selected_prior = False
    elif expert_aggregation == "minimax_relative_regret":
        dual_weights = {name: 1.0 / len(names) for name in names}
        candidates: list[tuple[float, int, dict[str, float]]] = []
        for iteration in range(minimax_iterations):
            effective_weights = normalized(
                {
                    name: base_objective_weights[name] * dual_weights[name]
                    for name in names
                }
            )
            trial = solve_once(effective_weights)
            candidates.append((trial["robust_objective"], iteration, effective_weights))
            minimax_history.append(
                {
                    "iteration": iteration,
                    "effective_objective_weights": effective_weights,
                    "relative_losses": trial["relative_losses"],
                    "worst_relative_loss": trial["robust_objective"],
                }
            )
            mean_relative_loss = max(
                sum(trial["relative_losses"].values()) / len(names), 1e-12
            )
            dual_weights = normalized(
                {
                    name: dual_weights[name]
                    * math.exp(
                        max(
                            -20.0,
                            min(
                                20.0,
                                minimax_temperature
                                * (
                                    trial["relative_losses"][name]
                                    / mean_relative_loss
                                    - 1.0
                                ),
                            ),
                        )
                    )
                    for name in names
                }
            )
        best_robust, selected_iteration, selected_weights = min(
            candidates, key=lambda item: item[0]
        )
        selected_prior = best_robust > 1.0
        selected = solve_once(selected_weights)
        if selected_prior:
            selected["merged"] = prior
            selected["correction"] = torch.zeros_like(prior)
            selected["losses"] = dict(prior_loss_by_expert)
            selected["relative_losses"] = {name: 1.0 for name in names}
            selected["robust_objective"] = 1.0
            selected["trust_scale"] = 1.0
    elif expert_aggregation == "fixed_weight_plan":
        if objective_weight_override is None:
            raise ValueError("fixed_weight_plan requires an objective-weight override")
        if set(objective_weight_override) != set(names):
            raise ValueError(
                "Planned expert weights differ from objective groups: "
                f"{sorted(objective_weight_override)} != {sorted(names)}"
            )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in objective_weight_override.values()
        ):
            raise ValueError(f"Invalid planned expert weights: {objective_weight_override}")
        selected = solve_once(
            normalized(
                {name: float(objective_weight_override[name]) for name in names}
            )
        )
        selected_iteration = None
        selected_prior = False
    else:
        raise ValueError(f"Unsupported expert aggregation: {expert_aggregation}")

    objective_weights = selected["objective_weights"]
    merged = selected["merged"]
    correction = selected["correction"]
    merged_loss_by_expert = selected["losses"]
    feature_energy = selected["feature_energy"]
    kernel_diagonal = selected["kernel_diagonal"]
    ridge_reference = selected["ridge_reference"]
    ridge = selected["ridge"]
    kernel = selected["kernel"]
    cholesky = selected["cholesky"]
    trust_scale = selected["trust_scale"]
    prior_loss = sum(prior_loss_by_expert.values()) / len(names)
    merged_loss = sum(merged_loss_by_expert.values()) / len(names)
    objective_prior_loss = sum(
        objective_weights[name] * prior_loss_by_expert[name] for name in names
    )
    objective_merged_loss = sum(
        objective_weights[name] * merged_loss_by_expert[name] for name in names
    )
    objective_tolerance = max(abs(objective_prior_loss) * 1e-6, 1e-12)
    rejected_nonimproving = selected_prior or (
        require_objective_improvement
        and objective_merged_loss > objective_prior_loss + objective_tolerance
    )
    if rejected_nonimproving:
        merged = prior
        correction = torch.zeros_like(correction)
        correction_norm = 0.0
        merged_loss_by_expert = dict(prior_loss_by_expert)
        merged_loss = prior_loss
        objective_merged_loss = objective_prior_loss
    # Exact eigendecomposition is diagnostic-only and cuSolver's syevd path
    # rejects sufficiently large dual kernels on some CUDA builds.  The
    # Cholesky solve above has already succeeded, so avoid turning a valid
    # merge into a failure merely to report a condition number.  For large
    # systems, use the squared Cholesky diagonal ratio as a finite,
    # inexpensive conditioning proxy.
    if kernel.shape[0] <= 8192:
        eigenvalues = torch.linalg.eigvalsh(kernel)
        kernel_condition = float(eigenvalues[-1] / eigenvalues[0])
        kernel_condition_method = "exact_eigvalsh"
    else:
        cholesky_diagonal = torch.diagonal(cholesky)
        kernel_condition = float(
            (cholesky_diagonal.max() / cholesky_diagonal.min()).square()
        )
        kernel_condition_method = "cholesky_diagonal_proxy"
    return merged.detach().cpu(), {
        "expert_count": len(names),
        "rows_by_expert": row_counts,
        "expert_loss_normalization": expert_loss_normalization,
        "expert_loss_normalization_power": expert_loss_normalization_power,
        "expert_aggregation": expert_aggregation,
        "base_expert_objective_weights": base_objective_weights,
        "objective_weight_override": objective_weight_override,
        "minimax_iterations": minimax_iterations,
        "minimax_temperature": minimax_temperature,
        "minimax_history": minimax_history,
        "minimax_selected_iteration": selected_iteration,
        "minimax_selected_prior": selected_prior,
        "worst_relative_loss": max(
            merged_loss_by_expert[name] / max(prior_loss_by_expert[name], 1e-12)
            for name in names
        ),
        "require_objective_improvement": require_objective_improvement,
        "rejected_nonimproving": rejected_nonimproving,
        "expert_objective_weights": objective_weights,
        "expert_normalization_floor": normalization_floor,
        "input_width": int(prior.shape[1]),
        "output_width": int(prior.shape[0]),
        "ridge": ridge,
        "ridge_scale": ridge_scale,
        "ridge_reference": ridge_reference,
        "feature_energy": feature_energy,
        "kernel_diagonal_mean": kernel_diagonal,
        "kernel_rows": int(kernel.shape[0]),
        "kernel_condition": kernel_condition,
        "kernel_condition_method": kernel_condition_method,
        "trust_scale": trust_scale,
        "correction_norm": float(torch.linalg.vector_norm(correction)),
        "prior_loss": prior_loss,
        "dense_regmean_loss": merged_loss,
        "objective_prior_loss": objective_prior_loss,
        "objective_dense_regmean_loss": objective_merged_loss,
        "objective_acceptance_tolerance": objective_tolerance,
        "objective_improvement_vs_prior": (
            (objective_prior_loss - objective_merged_loss) / max(objective_prior_loss, 1e-12)
        ),
        "prior_loss_by_expert": prior_loss_by_expert,
        "dense_regmean_loss_by_expert": merged_loss_by_expert,
        "relative_improvement_by_expert": {
            name: (prior_loss_by_expert[name] - merged_loss_by_expert[name])
            / max(prior_loss_by_expert[name], 1e-12)
            for name in names
        },
        "improvement_vs_prior": (prior_loss - merged_loss) / max(prior_loss, 1e-12),
    }


@dataclass
class LinearSpec:
    adapter_name: str
    base_name: str
    weight_key: str
    module: torch.nn.Module
    weights: dict[str, torch.Tensor]
    prior: torch.Tensor


def copy_support_files(experts: dict[str, Path], output: Path) -> None:
    names = list(experts)
    first = experts[names[0]]
    support_files = (
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    for filename in support_files:
        reference = first / filename
        for name in names[1:]:
            if reference.read_bytes() != (experts[name] / filename).read_bytes():
                raise ValueError(f"Support file {filename} differs for {name}")
        shutil.copy2(reference, output / filename)
    shutil.copytree(first / "tokenizer", output / "tokenizer")
    config = load_json(first / "config.json")
    config["use_peft"] = False
    config["pretrained_path"] = str(output)
    (output / "config.json").write_text(
        json.dumps(config, indent=4, sort_keys=False) + "\n", encoding="utf-8"
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    from pi05_tcr_e_dense_contract import validate_recipe, validate_dense_bank, validate_trace, validate_realized_rows
    ablation = load_json(args.ablation_config) if args.ablation_config else None
    if ablation is None or ablation.get('schema') != 'pi05_fixed_appendix_early_requests_v1':
        raise ValueError('This isolated solver accepts only the fixed early-request control')
    full_ridges = {}
    if ablation is None:
        validate_recipe(args)
    else:
        from pi05_table3_contract import validate_config, validate_trace, validate_rows, row_indices
        import three_level_contract as three_level
        import matched_main_contract as matched_main
        import pi05_subset_contract as subset
        import appendix_early_contract_v8 as appendix
        is_three_level = ablation.get('schema') == three_level.SCHEMA
        is_matched_main = ablation.get('schema') == matched_main.SCHEMA
        is_subset = ablation.get('schema') == subset.SCHEMA
        is_appendix = ablation.get('schema') == appendix.SCHEMA
        if is_three_level:
            repeat, pass_index = three_level.validate_config(ablation)
            recipe = three_level.PASS[pass_index]
        elif is_matched_main:
            repeat, pass_index, _ = matched_main.validate_config(ablation)
            recipe = matched_main.PASS[pass_index]
        elif is_subset:
            subset_names, pass_index = subset.validate_config(ablation)
            recipe = {"row_cap": subset.CAP,
                      "normalization": "prior" if pass_index == 1 else "none"}
        elif is_appendix:
            pass_index = appendix.validate_config(ablation)
            recipe = {"row_cap": appendix.CAP[pass_index],
                      "normalization": appendix.NORMALIZATION[pass_index]}
        else:
            validate_config(ablation)
            pass_index = None
            recipe = {"row_cap": 16, "normalization": None}
        expected = dict(ridge_ratio=.05, ridge_scale='feature_energy', max_correction_ratio=3.0,
                        max_rows_per_sample=recipe["row_cap"], expert_loss_normalization_power=1.0,
                        expert_aggregation='mean', objective_grouping='expert',
                        action_block_solver='independent_linear', require_objective_improvement=False,
                        freeze_prefix_from_prior=False, freeze_action_interface_from_prior=False,
                        max_states_per_calibration_source=0, expert_weight_plan=None)
        if is_three_level or is_matched_main or is_subset or is_appendix:
            expected['replay_prefix'] = ('expert' if is_matched_main and ablation['variant'] == 'expert_prefix'
                                         else 'merged')
            expected['expert_loss_normalization'] = recipe['normalization']
        else:
            expected['replay_prefix'] = 'expert' if ablation['variant'] == 'expert_prefix' else 'merged'
            expected['expert_loss_normalization'] = 'none' if ablation['variant'] == 'uniform' else 'prior'
        for key, value in expected.items():
            if getattr(args, key) != value:
                raise ValueError(f'Ablation recipe drift: {key}')
        if not args.prior_model:
            raise ValueError('Ablation requires an explicit frozen prior')
        if is_three_level or is_matched_main or is_subset or is_appendix:
            if Path(args.prior_model).resolve() != Path(ablation['start_point']).resolve():
                raise ValueError('Prior model is not the registered pass start point')
            observed_start = sha256(Path(ablation['start_point']) / 'model.safetensors')
            if observed_start != ablation['start_point_sha256']:
                raise ValueError('Registered pass start point hash differs on disk')
            ridge_source = (ablation.get('ridge_source_manifest') if is_three_level or is_subset or is_appendix
                            else ablation.get('full_manifest'))
            if ridge_source:
                ridge_manifest = load_json(Path(ridge_source))
                if len(ridge_manifest.get('modules', {})) != 418:
                    raise ValueError('Full numerical ridge source does not cover 418 modules')
                full_ridges = {key: value['ridge'] for key, value in ridge_manifest['modules'].items()}
                if not all(math.isfinite(value) and value > 0 for value in full_ridges.values()):
                    raise ValueError('Full numerical ridge source contains an invalid ridge')
        elif ablation['variant'] != 'full':
            full = load_json(Path(ablation['full_manifest']))
            if full.get('ablation', {}).get('variant') != 'full':
                raise ValueError('Ridge reference is not the new matched Full')
            full_ridges = {key:value['ridge'] for key,value in full['modules'].items()}
        if not is_three_level and ablation['variant'] == 'action_only':
            args.freeze_prefix_from_prior = True
    if args.replay_prefix == "expert" and (
        args.freeze_prefix_from_prior
        or args.freeze_action_interface_from_prior
        or args.action_block_solver != "independent_linear"
        or args.expert_aggregation != "mean"
    ):
        raise ValueError(
            "Expert-prefix control requires unfrozen full scope, independent_linear "
            "and mean aggregation; joint/refinement variants are separate experiments."
        )
    if args.ridge_ratio <= 0 or args.max_correction_ratio <= 0:
        raise ValueError("ridge-ratio and max-correction-ratio must be positive")
    if (
        args.joint_finite_difference <= 0
        or args.joint_ridge_ratio < 0
        or args.joint_max_states_per_expert <= 0
        or args.joint_max_correction_ratio <= 0
        or not 1 <= args.joint_path_layer_groups <= 18
    ):
        raise ValueError("Joint-subspace finite difference/state/trust settings are invalid")
    if args.expert_loss_normalization_power <= 0:
        raise ValueError("expert-loss-normalization-power must be positive")
    if args.minimax_iterations <= 0 or args.minimax_temperature <= 0:
        raise ValueError("minimax iterations and temperature must be positive")
    if (
        args.expert_aggregation == "minimax_relative_regret"
        and args.objective_grouping != "expert"
    ):
        raise ValueError(
            "minimax_relative_regret currently requires --objective-grouping=expert"
        )
    if args.expert_aggregation == "fixed_weight_plan":
        if args.objective_grouping != "expert":
            raise ValueError("fixed_weight_plan currently requires --objective-grouping=expert")
        if args.expert_weight_plan is None:
            raise ValueError("fixed_weight_plan requires --expert-weight-plan")
        if args.action_block_solver != "independent_linear":
            raise ValueError("fixed_weight_plan currently requires independent_linear blocks")
    elif args.expert_weight_plan is not None:
        raise ValueError("--expert-weight-plan requires --expert-aggregation=fixed_weight_plan")
    if args.max_rows_per_sample <= 0:
        raise ValueError("max-rows-per-sample must be positive")
    if args.max_states_per_calibration_source < 0:
        raise ValueError("max-states-per-calibration-source must be nonnegative")
    experts = assignments(args.expert, "expert")
    calibrations = multi_assignments(args.calibration, "calibration")
    manifests = multi_assignments(args.manifest, "manifest")
    calibration_source_weights = multi_float_assignments(
        args.calibration_source_weight, "calibration source weight"
    )
    if len(experts) < 2:
        raise ValueError("At least two experts are required")
    if set(experts) != set(calibrations) or set(experts) != set(manifests):
        raise ValueError("Expert, calibration and manifest names must match")
    for name in experts:
        if len(calibrations[name]) != len(manifests[name]):
            raise ValueError(
                f"{name}: calibration/manifest source count mismatch: "
                f"{len(calibrations[name])} != {len(manifests[name])}"
            )
        if name not in calibration_source_weights:
            calibration_source_weights[name] = [1.0] * len(calibrations[name])
        elif len(calibration_source_weights[name]) != len(calibrations[name]):
            raise ValueError(
                f"{name}: calibration/source-weight count mismatch: "
                f"{len(calibrations[name])} != {len(calibration_source_weights[name])}"
            )
    unknown_weight_names = set(calibration_source_weights) - set(experts)
    if unknown_weight_names:
        raise ValueError(
            f"Calibration source weights reference unknown experts: {sorted(unknown_weight_names)}"
        )
    calibration_paths = [path for paths in calibrations.values() for path in paths]
    manifest_paths = [path for paths in manifests.values() for path in paths]
    for path in (*experts.values(), *calibration_paths, *manifest_paths):
        if not path.exists():
            raise FileNotFoundError(path)

    base_root = args.base_model.expanduser().resolve()
    prior_root = args.prior_model.expanduser().resolve() if args.prior_model else None
    output = args.output.expanduser().absolute()
    if not (base_root / "model.safetensors").is_file():
        raise FileNotFoundError(base_root / "model.safetensors")
    if prior_root is not None and not (prior_root / "model.safetensors").is_file():
        raise FileNotFoundError(prior_root / "model.safetensors")
    if args.freeze_prefix_from_prior and prior_root is None:
        raise ValueError("--freeze-prefix-from-prior requires --prior-model")
    if args.freeze_action_interface_from_prior and prior_root is None:
        raise ValueError("--freeze-action-interface-from-prior requires --prior-model")
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    names = list(experts)
    if ablation is not None and ablation.get('schema') == subset.SCHEMA:
        if tuple(names) != subset_names:
            raise ValueError('CLI experts differ from frozen subset membership/order')
        dense_bank = subset.validate_subset_bank(
            args.dense_expert_bank, experts, verify_weights=True)
    else:
        dense_bank = validate_dense_bank(args.dense_expert_bank, experts, verify_weights=True)
    expert_weight_plan_path: Path | None = None
    expert_weight_plan_method: str | None = None
    expert_weight_plan_modules: dict[str, Any] = {}
    if args.expert_weight_plan is not None:
        expert_weight_plan_path = args.expert_weight_plan.expanduser().resolve()
        if not expert_weight_plan_path.is_file():
            raise FileNotFoundError(expert_weight_plan_path)
        expert_weight_plan = load_json(expert_weight_plan_path)
        expert_weight_plan_method = str(
            expert_weight_plan.get("method", "unspecified_fixed_weight_plan")
        )
        planned_experts = expert_weight_plan.get("experts")
        if planned_experts is not None and set(planned_experts) != set(names):
            raise ValueError(
                f"Weight-plan experts differ: {sorted(planned_experts)} != {sorted(names)}"
            )
        expert_weight_plan_modules = expert_weight_plan.get("modules", {})
        if not isinstance(expert_weight_plan_modules, dict):
            raise ValueError("Weight plan must contain an object-valued modules field")

    used_weight_plan_modules: set[str] = set()

    def planned_objective_weights(module_name: str) -> dict[str, float] | None:
        if not expert_weight_plan_modules:
            return None
        entry = expert_weight_plan_modules.get(module_name)
        if not isinstance(entry, dict):
            raise ValueError(f"Weight plan is missing module: {module_name}")
        raw = entry.get("expert_objective_weights")
        if not isinstance(raw, dict):
            raise ValueError(f"Weight plan module lacks expert_objective_weights: {module_name}")
        if set(raw) != set(names):
            raise ValueError(
                f"Weight plan expert mismatch for {module_name}: "
                f"{sorted(raw)} != {sorted(names)}"
            )
        used_weight_plan_modules.add(module_name)
        return {name: float(raw[name]) for name in names}

    first_config = load_json(experts[names[0]] / "adapter_config.json")
    for name in names[1:]:
        if load_json(experts[name] / "adapter_config.json") != first_config:
            raise ValueError(f"Adapter config differs for {name}")
    rank = int(first_config["r"])
    scale = float(first_config["lora_alpha"]) / rank

    states_by_name: dict[str, list[ReplayState]] = {}
    state_weights_by_name: dict[str, list[float]] = {}
    manifests_by_name: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        states_by_name[name] = []
        state_weights_by_name[name] = []
        manifests_by_name[name] = []
        for calibration_path, manifest_path, source_weight in zip(
            calibrations[name],
            manifests[name],
            calibration_source_weights[name],
            strict=True,
        ):
            states, manifest = load_replay_states(
                calibration_path,
                manifest_path,
                name,
                device,
                args.max_states_per_calibration_source,
                0 if args.action_block_solver == "joint_flow_terminal_subspace" else None,
            )
            if ablation is not None and ablation.get('schema') == three_level.SCHEMA:
                three_level.validate_demo_trace(
                    manifest, dense_bank[name]["path"], name,
                    int(ablation['repeat']), int(ablation['pass_index']))
            elif ablation is not None and ablation.get('schema') == matched_main.SCHEMA:
                matched_main.validate_trace(
                    manifest, dense_bank[name]["path"], name,
                    int(ablation['repeat']), int(ablation['pass_index']))
            elif ablation is not None and ablation.get('schema') == subset.SCHEMA:
                subset.validate_trace(
                    manifest, dense_bank[name]["path"], name,
                    int(ablation['pass_index']))
            elif ablation is not None and ablation.get('schema') == appendix.SCHEMA:
                appendix.validate_trace(
                    manifest, Path(dense_bank[name]["path"]), name,
                    int(ablation['pass_index']))
            else:
                validate_trace(manifest, dense_bank[name]["path"], name)
            states_by_name[name].extend(states)
            state_weights_by_name[name].extend([source_weight] * len(states))
            manifests_by_name[name].append(manifest)
    objective_labels_by_name: dict[str, list[str]] = {}
    for name in names:
        states = states_by_name[name]
        if args.objective_grouping == "expert":
            objective_labels_by_name[name] = [name] * len(states)
            continue
        labels: list[str] = []
        expected_prompt_counts: list[int] = []
        for manifest in manifests_by_name[name]:
            sample_metadata = manifest.get("samples")
            expected_samples = int(manifest.get("sample_count", -1))
            if not isinstance(sample_metadata, list) or len(sample_metadata) != expected_samples:
                raise ValueError(
                    f"{name}: expert_prompt grouping requires one manifest sample per replay state"
                )
            signatures = [sample.get("prompt_signature") for sample in sample_metadata]
            if any(signature is None for signature in signatures):
                raise ValueError(
                    f"{name}: expert_prompt grouping requires prompt_signature metadata"
                )
            labels.extend(f"{name}/prompt={signature}" for signature in signatures)
            expected_prompt_counts.append(int(manifest.get("prompt_count", 0)))
        if len(labels) != len(states):
            raise ValueError(
                f"{name}: aggregated objective labels/states mismatch: "
                f"{len(labels)} != {len(states)}"
            )
        objective_labels_by_name[name] = labels
        actual_prompts = len(set(objective_labels_by_name[name]))
        nonzero_expected = {count for count in expected_prompt_counts if count}
        if len(nonzero_expected) > 1 or (
            nonzero_expected and actual_prompts != next(iter(nonzero_expected))
        ):
            raise ValueError(
                f"{name}: source prompt counts={expected_prompt_counts}, "
                f"aggregated prompt groups={actual_prompts}"
            )
    calibration_policy_sources_by_name = {
        name: [manifest["calibration_policy"] for manifest in source_manifests]
        for name, source_manifests in manifests_by_name.items()
    }
    calibration_policies = {
        policy
        for policies in calibration_policy_sources_by_name.values()
        for policy in policies
    }
    if len(calibration_policies) != 1 and not args.allow_mixed_calibration_policies:
        raise ValueError(
            "Calibrations used different reference policies; pass "
            "--allow-mixed-calibration-policies for per-expert calibration: "
            f"{calibration_policy_sources_by_name}"
        )
    if (
        prior_root is not None
        and not args.allow_prior_calibration_mismatch
        and len(calibration_policies) == 1
    ):
        calibration_policy = Path(next(iter(calibration_policies))).resolve()
        if calibration_policy != prior_root:
            raise ValueError(f"prior={prior_root} differs from calibration={calibration_policy}")
    full_prefix_method = "pi05_full_vision_language_action_block_regmeanpp_replay_calibration"
    full_prefix_values = {
        manifest.get("method") == full_prefix_method
        for source_manifests in manifests_by_name.values()
        for manifest in source_manifests
    }
    if len(full_prefix_values) != 1:
        raise ValueError("Calibrations disagree on full-prefix replay mode")
    full_prefix = full_prefix_values.pop()
    if args.replay_prefix == "expert" and not full_prefix:
        raise ValueError("Expert-prefix control requires full Vision/Language replay inputs")
    if args.freeze_prefix_from_prior and not full_prefix:
        raise ValueError("--freeze-prefix-from-prior requires full-prefix replay calibration")

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    print(f"Loading common base runtime model on {device}", flush=True)
    policy = PI05Policy.from_pretrained(base_root, device=str(device))
    policy.eval()
    if ablation is not None:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    modules = dict(policy.named_modules())
    core = policy.model
    expert_model = core.paligemma_with_expert.gemma_expert.model
    expert_model.config._attn_implementation = "eager"  # noqa: SLF001
    language_model = core.paligemma_with_expert.paligemma.model.language_model
    language_model.config._attn_implementation = "eager"  # noqa: SLF001

    base_weights_path = base_root / "model.safetensors"
    initial_weights_path = (prior_root / "model.safetensors") if prior_root is not None else base_weights_path
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
    dense_sources: dict[str, tuple[str, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]] = {}

    adapter_paths = {name: experts[name] / "adapter_model.safetensors" for name in names}
    with ExitStack() as stack:
        handles = {
            name: stack.enter_context(safe_open(adapter_paths[name], framework="pt", device="cpu"))
            for name in names
        }
        dense_handles = {
            name: stack.enter_context(safe_open(Path(dense_bank[name]["path"]) / "model.safetensors", framework="pt", device="cpu"))
            for name in names
        }
        base = stack.enter_context(safe_open(base_weights_path, framework="pt", device="cpu"))
        keys = list(handles[names[0]].keys())
        for name in names[1:]:
            if keys != list(handles[name].keys()):
                raise ValueError(f"Adapter tensor keys differ for {name}")

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
            expert_weights = {
                name: dense_handles[name].get_tensor(weight_key).float()
                for name in names
            }
            prior = (
                output_tensors[weight_key].float()
                if prior_root is not None
                else sum(expert_weights.values()) / len(expert_weights)
            )
            output_tensors[weight_key] = prior.to(base_weight.dtype).contiguous()
            modified_keys.add(weight_key)
            module = resolve_module(modules, base_name)
            module.weight.data.copy_(prior.to(device=device, dtype=module.weight.dtype))
            spec = LinearSpec(adapter_name, base_name, weight_key, module, expert_weights, prior)
            for fragment, destination in (
                (VISION_LAYER_FRAGMENT, vision_specs),
                (LANGUAGE_LAYER_FRAGMENT, language_specs),
                (ACTION_LAYER_FRAGMENT, action_specs),
            ):
                if fragment in base_name:
                    block_index = int(base_name.split(fragment, 1)[1].split(".", 1)[0])
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
            expert_weights = {name: dense_handles[name].get_tensor(weight_key).float() for name in names}
            expert_biases = {name: dense_handles[name].get_tensor(bias_key).float() for name in names}
            prior_weight = (
                output_tensors[weight_key].float()
                if prior_root is not None
                else sum(expert_weights.values()) / len(expert_weights)
            )
            prior_bias = (
                output_tensors[bias_key].float()
                if prior_root is not None
                else sum(expert_biases.values()) / len(expert_biases)
            )
            output_tensors[weight_key] = prior_weight.to(output_tensors[weight_key].dtype).contiguous()
            output_tensors[bias_key] = prior_bias.to(output_tensors[bias_key].dtype).contiguous()
            modified_keys.update((weight_key, bias_key))
            module = resolve_module(modules, base_name)
            module.weight.data.copy_(prior_weight.to(device=device, dtype=module.weight.dtype))
            module.bias.data.copy_(prior_bias.to(device=device, dtype=module.bias.dtype))
            dense_sources[base_name.rsplit(".", 1)[-1]] = (
                adapter_name,
                expert_weights,
                expert_biases,
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

    def assign_source(specs: list[LinearSpec], name: str) -> None:
        for spec in specs:
            spec.module.weight.data.copy_(spec.weights[name].to(device=device, dtype=spec.module.weight.dtype))

    def restore_merged(specs: list[LinearSpec]) -> None:
        for spec in specs:
            spec.module.weight.data.copy_(
                output_tensors[spec.weight_key].to(device=device, dtype=spec.module.weight.dtype)
            )

    def interface_forward(short_name: str, value: torch.Tensor, name: str) -> torch.Tensor:
        module = getattr(core, short_name)
        if args.replay_prefix == "merged":
            return linear_forward(module, value)
        _, expert_weights, expert_biases, _, _ = dense_sources[short_name]
        # Match the native interface dtype, not a higher-precision teacher.
        return F.linear(
            value.to(dtype=module.weight.dtype),
            expert_weights[name].to(device=device, dtype=module.weight.dtype),
            expert_biases[name].to(device=device, dtype=module.bias.dtype),
        )

    def capture_inputs(
        specs: list[LinearSpec],
        name: str,
        forwards: list[Any],
        labels: list[str],
        sample_weights: list[float],
    ) -> dict[str, dict[str, torch.Tensor]]:
        if len(forwards) != len(labels) or len(forwards) != len(sample_weights):
            raise ValueError(
                f"{name}: forward/objective-label/source-weight count mismatch: "
                f"{len(forwards)} != {len(labels)} != {len(sample_weights)}"
            )
        assign_source(specs, name)
        captures: dict[str, dict[str, list[torch.Tensor]]] = {
            spec.base_name: {} for spec in specs
        }
        current_label: str | None = None
        current_sqrt_weight = 1.0
        current_forward_index = 0
        handles = []
        for spec in specs:
            def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], module_name: str = spec.base_name) -> None:
                if current_label is None:
                    raise RuntimeError("Linear hook fired without an objective-group label")
                captures[module_name].setdefault(current_label, []).append(
                    selected_rows(inputs[0], name, current_forward_index, module_name) * current_sqrt_weight
                )

            handles.append(spec.module.register_forward_pre_hook(hook))
        try:
            for current_forward_index, (forward, label, sample_weight) in enumerate(zip(
                forwards, labels, sample_weights, strict=True
            )):
                current_label = label
                current_sqrt_weight = math.sqrt(sample_weight)
                forward()
        finally:
            for handle in handles:
                handle.remove()
        return {
            module_name: {
                label: torch.cat(values, dim=0) for label, values in grouped.items()
            }
            for module_name, grouped in captures.items()
        }

    def solve_spec(
        spec: LinearSpec,
        inputs_by_expert: dict[str, dict[str, dict[str, torch.Tensor]]],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        grouped_inputs: dict[str, torch.Tensor] = {}
        grouped_weights: dict[str, torch.Tensor] = {}
        for name in names:
            for label, value in inputs_by_expert[name][spec.base_name].items():
                if label in grouped_inputs:
                    raise ValueError(f"Duplicate objective-group label: {label}")
                grouped_inputs[label] = value
                grouped_weights[label] = spec.weights[name]
        return solve_weight_multi(
            grouped_inputs,
            grouped_weights,
            spec.prior.to(device),
            args.ridge_ratio,
            args.ridge_scale,
            args.max_correction_ratio,
            args.expert_loss_normalization,
            args.expert_loss_normalization_power,
            args.require_objective_improvement,
            args.expert_aggregation,
            args.minimax_iterations,
            args.minimax_temperature,
            planned_objective_weights(spec.base_name),
            full_ridges.get(spec.base_name),
        )

    def solve_dense_module(
        short_name: str,
        x_by_expert: dict[str, dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        _adapter_name, expert_weights, expert_biases, prior_weight, prior_bias = dense_sources[short_name]
        grouped_inputs: dict[str, torch.Tensor] = {}
        grouped_weights: dict[str, torch.Tensor] = {}
        for name in names:
            for label, value in x_by_expert[name].items():
                if label in grouped_inputs:
                    raise ValueError(f"Duplicate objective-group label: {label}")
                grouped_inputs[label] = with_ones(value)
                grouped_weights[label] = augmented(expert_weights[name], expert_biases[name])
        return solve_weight_multi(
            grouped_inputs,
            grouped_weights,
            augmented(prior_weight, prior_bias).to(device),
            args.ridge_ratio,
            args.ridge_scale,
            args.max_correction_ratio,
            args.expert_loss_normalization,
            args.expert_loss_normalization_power,
            args.require_objective_improvement,
            args.expert_aggregation,
            args.minimax_iterations,
            args.minimax_temperature,
            planned_objective_weights(f"model.{short_name}"),
            full_ridges.get(f"model.{short_name}"),
        )

    all_states = [state for states in states_by_name.values() for state in states]

    def selected_rows(value, name, index, module_name=''):
        if ablation is None:
            return sample_rows(value, args.max_rows_per_sample)
        if (ablation.get('schema') == three_level.SCHEMA and ablation['pass_index'] == 1
                or ablation.get('schema') == appendix.SCHEMA and ablation['pass_index'] == 1):
            # The fixed A pass applies cap 10 independently to each of the
            # three saved flow states, realizing 4,441,200 rows.
            return sample_rows(value, args.max_rows_per_sample)
        is_vision = '.vision_tower.' in module_name
        camera = index % 3 if is_vision else 0
        state_index = index // 3 if is_vision else index
        sample = [s for m in manifests_by_name[name] for s in m['samples']][state_index]
        flat = value.reshape(-1, value.shape[-1]).float()
        ids = row_indices(len(flat), args.max_rows_per_sample,
                          sample['flow_index'], sample['selected_request_slot'],
                          cameras=3 if is_vision else 1, camera=camera,
                          last_only=ablation.get('variant') == 'last')
        return flat[ids]

    def grouped_state_rows(name: str, value_fn: Any) -> dict[str, torch.Tensor]:
        values = [
            selected_rows(value_fn(state), name, index) * math.sqrt(sample_weight)
            for index, (state, sample_weight) in enumerate(zip(
                states_by_name[name], state_weights_by_name[name], strict=True
            ))
        ]
        return concatenate_grouped(values, objective_labels_by_name[name])

    vision_reference_error: float | None = None
    prefix_kv_reference_error: float | None = None
    expert_prefix_vision_error: float | None = None
    expert_prefix_kv_error: float | None = None
    expert_prefix_kv_errors: list[float] = []

    if full_prefix:
        if any(
            state.vision is None
            or state.language_hidden_reference is None
            or state.language_attention_mask is None
            or state.language_position_ids is None
            for state in all_states
        ):
            raise ValueError("Full-prefix calibration is missing vision/language replay tensors")
        vision_transformer = core.paligemma_with_expert.paligemma.model.vision_tower.vision_model
        vision_layers = vision_transformer.encoder.layers
        projector = core.paligemma_with_expert.paligemma.model.multi_modal_projector

        vision_errors: list[float] = []
        for state in all_states:
            assert state.vision is not None and state.language_hidden_reference is not None
            projected = []
            for vision_state in state.vision:
                value = vision_state.hidden
                for layer in vision_layers:
                    value = layer(value, vision_state.attention_mask)
                projected.append(projector(vision_transformer.post_layernorm(value)))
            visual = torch.cat(projected, dim=1)
            vision_errors.append(float((visual.float() - state.language_hidden_reference[:, : visual.shape[1]].float()).abs().max()))
        vision_reference_error = max(vision_errors)
        print(
            "Stored-reference Vision mismatch before sequential replay "
            f"(calibration-policy dependent): {vision_reference_error:.6g}",
            flush=True,
        )

        kv_errors: list[float] = []
        for state in all_states:
            assert state.language_hidden_reference is not None
            assert state.language_attention_mask is not None
            assert state.language_position_ids is not None
            value = state.language_hidden_reference
            for block_index, layer in enumerate(language_model.layers):
                key, cached_value = decoder_prefix_kv(layer, language_model.rotary_emb, value, state.language_position_ids, None)
                kv_errors.extend(
                    (
                        float((key.float() - state.prefix_keys[block_index].float()).abs().max()),
                        float((cached_value.float() - state.prefix_values[block_index].float()).abs().max()),
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
            "Stored-reference Language-KV mismatch before sequential replay "
            f"(calibration-policy dependent): {prefix_kv_reference_error:.6g}",
            flush=True,
        )

        def vision_items(states: list[ReplayState]) -> list[Any]:
            return [item for state in states for item in (state.vision or [])]

        def vision_objective_labels(name: str) -> list[str]:
            return [
                label
                for state, label in zip(
                    states_by_name[name], objective_labels_by_name[name], strict=True
                )
                for _item in (state.vision or [])
            ]

        def vision_source_weights(name: str) -> list[float]:
            return [
                sample_weight
                for state, sample_weight in zip(
                    states_by_name[name], state_weights_by_name[name], strict=True
                )
                for _item in (state.vision or [])
            ]

        for block_index in range(27):
            specs = vision_specs[block_index]
            layer = vision_layers[block_index]
            if args.freeze_prefix_from_prior or (
                    ablation is not None and ablation.get('variant') == 'conditioning_action'):
                for spec in specs:
                    metrics[spec.base_name] = {
                        "stage": "vision_block_frozen_prior",
                        "block_index": block_index,
                        "kind": "frozen_prior",
                        "improvement_vs_prior": 0.0,
                        "trust_scale": 1.0,
                        "rejected_nonimproving": False,
                    }
            else:
                inputs_by_expert = {
                    name: capture_inputs(
                        specs,
                        name,
                        [
                            (lambda item=item: layer(item.hidden, item.attention_mask))
                            for item in vision_items(states_by_name[name])
                        ],
                        vision_objective_labels(name),
                        vision_source_weights(name),
                    )
                    for name in names
                }
                for spec in specs:
                    merged, module_metrics = solve_spec(spec, inputs_by_expert)
                    spec.module.weight.data.copy_(
                        merged.to(device=device, dtype=spec.module.weight.dtype)
                    )
                    output_tensors[spec.weight_key] = merged.to(
                        output_tensors[spec.weight_key].dtype
                    ).contiguous()
                    metrics[spec.base_name] = {
                        "stage": "vision_block_sequential",
                        "block_index": block_index,
                        "kind": "multi_lora_materialized_dense",
                        **module_metrics,
                    }
                del inputs_by_expert
            def advance_vision(state: ReplayState) -> None:
                for item in state.vision or []:
                    item.hidden = layer(item.hidden, item.attention_mask)

            advance_by_prefix(
                states_by_name, args.replay_prefix,
                lambda name: assign_source(specs, name),
                lambda: restore_merged(specs), advance_vision,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            verb = "Replayed frozen" if args.freeze_prefix_from_prior else "Merged"
            print(f"{verb} Vision block {block_index:02d}/26", flush=True)

        language_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        if args.replay_prefix == "expert":
            errors = []
            for state in all_states:
                assert state.vision is not None and state.language_hidden_reference is not None
                projected = [projector(vision_transformer.post_layernorm(v.hidden)) for v in state.vision]
                visual = torch.cat(projected, dim=1)
                errors.append(
                    float(
                        (
                            visual.float()
                            - state.language_hidden_reference[:, : visual.shape[1]].float()
                        )
                        .abs()
                        .max()
                    )
                )
            expert_prefix_vision_error = max(errors)
            print(
                "Expert-prefix Vision-to-language stored-reference max error: "
                f"{expert_prefix_vision_error:.6g}",
                flush=True,
            )
        for state in all_states:
            assert state.vision is not None and state.language_hidden_reference is not None
            projected = [projector(vision_transformer.post_layernorm(v.hidden)) for v in state.vision]
            visual = torch.cat(projected, dim=1)
            tail = state.language_hidden_reference[:, visual.shape[1] :]
            state.language_hidden = torch.cat((visual.to(tail.dtype), tail), dim=1).to(language_dtype)

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

            if args.freeze_prefix_from_prior:
                for spec in specs:
                    metrics[spec.base_name] = {
                        "stage": "language_block_frozen_prior",
                        "block_index": block_index,
                        "kind": "frozen_prior",
                        "improvement_vs_prior": 0.0,
                        "trust_scale": 1.0,
                        "rejected_nonimproving": False,
                    }
            else:
                inputs_by_expert = {
                    name: capture_inputs(
                        specs,
                        name,
                        language_forwards(states_by_name[name]),
                        objective_labels_by_name[name],
                        state_weights_by_name[name],
                    )
                    for name in names
                }
                for spec in specs:
                    merged, module_metrics = solve_spec(spec, inputs_by_expert)
                    spec.module.weight.data.copy_(
                        merged.to(device=device, dtype=spec.module.weight.dtype)
                    )
                    output_tensors[spec.weight_key] = merged.to(
                        output_tensors[spec.weight_key].dtype
                    ).contiguous()
                    metrics[spec.base_name] = {
                        "stage": "language_block_sequential",
                        "block_index": block_index,
                        "kind": "multi_lora_materialized_dense",
                        **module_metrics,
                    }
                del inputs_by_expert
            def advance_language(state: ReplayState) -> None:
                assert state.language_hidden is not None
                assert state.language_attention_mask is not None
                assert state.language_position_ids is not None
                key, value = decoder_prefix_kv(layer, language_model.rotary_emb, state.language_hidden, state.language_position_ids, None)
                if args.replay_prefix == "expert":
                    expert_prefix_kv_errors.extend(
                        (
                            float((key.float() - state.prefix_keys[block_index].float()).abs().max()),
                            float((value.float() - state.prefix_values[block_index].float()).abs().max()),
                        )
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
            advance_by_prefix(
                states_by_name, args.replay_prefix,
                lambda name: assign_source(specs, name),
                lambda: restore_merged(specs), advance_language,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            verb = "Replayed frozen" if args.freeze_prefix_from_prior else "Merged"
            print(f"{verb} Language block {block_index:02d}/17", flush=True)
        if args.replay_prefix == "expert":
            expert_prefix_kv_error = max(expert_prefix_kv_errors)
            print(
                "Expert-prefix Language-KV stored-reference max error: "
                f"{expert_prefix_kv_error:.6g}",
                flush=True,
            )

    reference_errors: list[float] = []
    for state in all_states:
        action = linear_forward(core.action_in_proj, state.action_input)
        time = F.silu(linear_forward(core.time_mlp_in, state.time_input))
        cond = F.silu(linear_forward(core.time_mlp_out, time))
        reference_errors.extend(
            (
                float((action.float() - state.hidden_reference.float()).abs().max()),
                float((cond.float() - state.cond_reference.float()).abs().max()),
            )
        )
    print(
        "Stored-reference action front-end mismatch before interface solve "
        f"(calibration-policy dependent): {max(reference_errors):.6g}",
        flush=True,
    )
    expert_prefix_frontend_error: float | None = None
    if args.replay_prefix == "expert":
        errors = []
        for name in names:
            for state in states_by_name[name]:
                action = interface_forward("action_in_proj", state.action_input, name)
                time = F.silu(interface_forward("time_mlp_in", state.time_input, name))
                cond = F.silu(interface_forward("time_mlp_out", time, name))
                errors.extend(
                    (
                        float((action.float() - state.hidden_reference.float()).abs().max()),
                        float((cond.float() - state.cond_reference.float()).abs().max()),
                    )
                )
        expert_prefix_frontend_error = max(errors)
        print(
            "Expert-prefix action front-end stored-reference max error: "
            f"{expert_prefix_frontend_error:.6g}",
            flush=True,
        )

    from transformers.cache_utils import DynamicCache

    validation_state = all_states[0]
    validation_cache = DynamicCache(
        tuple(
            (keys.clone(), values.clone(), None)
            for keys, values in zip(validation_state.prefix_keys, validation_state.prefix_values, strict=True)
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
    manual_replay_error = float((native_block_output.float() - manual_block_output.float()).abs().max())
    if manual_replay_error > 0.02:
        raise RuntimeError(f"Manual Action block replay disagrees with native layer: {manual_replay_error}")
    print(f"Manual/native Action block max error: {manual_replay_error:.6g}", flush=True)

    if args.freeze_action_interface_from_prior:
        for short_name in FRONTEND_MODULES:
            metrics[f"model.{short_name}"] = {
                "stage": "action_interface_frozen_prior",
                "kind": "frozen_prior",
                "improvement_vs_prior": 0.0,
                "trust_scale": 1.0,
                "rejected_nonimproving": False,
            }
    else:
        action_solution, action_metrics = solve_dense_module(
            "action_in_proj",
            {name: grouped_state_rows(name, lambda state: state.action_input) for name in names},
        )
        time_in_solution, time_in_metrics = solve_dense_module(
            "time_mlp_in",
            {name: grouped_state_rows(name, lambda state: state.time_input) for name in names},
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
            metrics[base_name] = {"stage": "front_end", "kind": "multi_dense_augmented", **module_metrics}

        time_hidden_by_expert = {
            name: grouped_state_rows(
                name,
                lambda state, name=name: F.silu(interface_forward("time_mlp_in", state.time_input, name)),
            )
            for name in names
        }
        time_out_solution, time_out_metrics = solve_dense_module("time_mlp_out", time_hidden_by_expert)
        core.time_mlp_out.weight.data.copy_(time_out_solution[:, :-1].to(device=device, dtype=core.time_mlp_out.weight.dtype))
        core.time_mlp_out.bias.data.copy_(time_out_solution[:, -1].to(device=device, dtype=core.time_mlp_out.bias.dtype))
        output_tensors["model.time_mlp_out.weight"] = time_out_solution[:, :-1].to(output_tensors["model.time_mlp_out.weight"].dtype)
        output_tensors["model.time_mlp_out.bias"] = time_out_solution[:, -1].to(output_tensors["model.time_mlp_out.bias"].dtype)
        metrics["model.time_mlp_out"] = {"stage": "front_end_sequential", "kind": "multi_dense_augmented", **time_out_metrics}

    first_weight_dtype = expert_model.layers[0].self_attn.q_proj.weight.dtype
    for name in names:
        for state in states_by_name[name]:
            state.hidden = interface_forward("action_in_proj", state.action_input, name).to(first_weight_dtype)
            time = F.silu(interface_forward("time_mlp_in", state.time_input, name))
            state.cond = F.silu(interface_forward("time_mlp_out", time, name))

    def run_and_capture(
        block_index: int,
        states: list[ReplayState],
        specs: list[LinearSpec],
        name: str,
    ) -> dict[str, dict[str, torch.Tensor]]:
        forwards = []
        for state in states:
            assert state.hidden is not None and state.cond is not None
            forwards.append(
                lambda state=state: manual_action_block(
                    expert_model.layers[block_index],
                    expert_model.rotary_emb,
                    state.hidden,
                    state.attention_mask,
                    state.position_ids,
                    state.prefix_keys[block_index],
                    state.prefix_values[block_index],
                    state.cond,
                )
            )
        return capture_inputs(
            specs,
            name,
            forwards,
            objective_labels_by_name[name],
            state_weights_by_name[name],
        )

    def joint_state_indices(name: str) -> list[int]:
        count = len(states_by_name[name])
        limit = min(count, args.joint_max_states_per_expert)
        if limit == count:
            return list(range(count))
        return (
            torch.linspace(0, count - 1, limit)
            .round()
            .to(dtype=torch.int64)
            .tolist()
        )

    def joint_block_solve(
        block_index: int,
        specs: list[LinearSpec],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Fit one Action block from its final residual output without gradients.

        The local parameterization is the M-expert task-delta span around the
        deployed prior.  A forward finite-difference Jacobian couples all
        selected Linear modules, after which a small ridge system is solved and
        validated with an exact nonlinear line search on the same block output.
        """
        if args.objective_grouping != "expert":
            raise ValueError("joint_subspace currently requires --objective-grouping=expert")

        layer = expert_model.layers[block_index]
        selected_indices = {name: joint_state_indices(name) for name in names}

        if args.joint_subspace_grouping == "family":
            grouped_specs = (
                ("attention", specs[:4]),
                ("mlp", specs[4:]),
            )
        else:
            grouped_specs = tuple(
                (spec.base_name.rsplit(".", 2)[-2] + "." + spec.base_name.rsplit(".", 1)[-1], [spec])
                for spec in specs
            )
        directions = [
            (family, expert_name, family_specs)
            for family, family_specs in grouped_specs
            for expert_name in names
        ]

        def assign_prior() -> None:
            for spec in specs:
                spec.module.weight.data.copy_(
                    spec.prior.to(device=device, dtype=spec.module.weight.dtype)
                )

        def assign_expert(expert_name: str) -> None:
            for spec in specs:
                spec.module.weight.data.copy_(
                    spec.weights[expert_name].to(device=device, dtype=spec.module.weight.dtype)
                )

        def assign_coefficients(coefficients: torch.Tensor) -> None:
            merged = {spec.base_name: spec.prior.to(device=device, dtype=torch.float32).clone() for spec in specs}
            for coefficient, (_family, expert_name, family_specs) in zip(
                coefficients, directions, strict=True
            ):
                for spec in family_specs:
                    merged[spec.base_name].add_(
                        spec.weights[expert_name].to(device=device, dtype=torch.float32)
                        - spec.prior.to(device=device, dtype=torch.float32),
                        alpha=float(coefficient),
                    )
            for spec in specs:
                spec.module.weight.data.copy_(
                    merged[spec.base_name].to(dtype=spec.module.weight.dtype)
                )

        def output_rows(data_name: str) -> torch.Tensor:
            rows: list[torch.Tensor] = []
            for state_index in selected_indices[data_name]:
                state = states_by_name[data_name][state_index]
                assert state.hidden is not None and state.cond is not None
                value = manual_action_block(
                    layer,
                    expert_model.rotary_emb,
                    state.hidden,
                    state.attention_mask,
                    state.position_ids,
                    state.prefix_keys[block_index],
                    state.prefix_values[block_index],
                    state.cond,
                )
                rows.append(
                    sample_rows(value.float(), args.max_rows_per_sample)
                    * math.sqrt(state_weights_by_name[data_name][state_index])
                )
            return torch.cat(rows, dim=0)

        assign_prior()
        prior_outputs = {name: output_rows(name) for name in names}
        target_outputs: dict[str, torch.Tensor] = {}
        for name in names:
            assign_expert(name)
            target_outputs[name] = output_rows(name)
        assign_prior()

        prior_loss_by_expert = {
            name: float((prior_outputs[name] - target_outputs[name]).square().mean())
            for name in names
        }
        if args.expert_loss_normalization == "none":
            objective_weights = {name: 1.0 / len(names) for name in names}
            normalization_floor = None
        else:
            mean_prior_loss = sum(prior_loss_by_expert.values()) / len(names)
            normalization_floor = max(mean_prior_loss * 1e-6, 1e-12)
            sign = -1.0 if args.expert_loss_normalization == "prior" else 1.0
            masses = {
                name: max(prior_loss_by_expert[name], normalization_floor)
                ** (sign * args.expert_loss_normalization_power)
                for name in names
            }
            total_mass = sum(masses.values())
            objective_weights = {name: masses[name] / total_mass for name in names}

        derivatives: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        epsilon = args.joint_finite_difference
        for direction_index in range(len(directions)):
            coefficients = torch.zeros(len(directions), device=device, dtype=torch.float32)
            coefficients[direction_index] = epsilon
            assign_coefficients(coefficients)
            for name in names:
                derivatives[name].append((output_rows(name) - prior_outputs[name]) / epsilon)
        assign_prior()

        dimension = len(directions)
        hessian = torch.zeros((dimension, dimension), device=device, dtype=torch.float64)
        rhs = torch.zeros(dimension, device=device, dtype=torch.float64)
        for name in names:
            jacobian = torch.stack(
                [value.reshape(-1) for value in derivatives[name]], dim=1
            ).to(dtype=torch.float64)
            target = (target_outputs[name] - prior_outputs[name]).reshape(-1).to(dtype=torch.float64)
            mass = objective_weights[name] / max(target.numel(), 1)
            hessian.addmm_(jacobian.T, jacobian, beta=1.0, alpha=mass)
            rhs.addmv_(jacobian.T, target, beta=1.0, alpha=mass)

        ridge_reference = max(float(torch.trace(hessian) / max(dimension, 1)), 1e-12)
        ridge = args.joint_ridge_ratio * ridge_reference
        regularized = hessian + ridge * torch.eye(dimension, device=device, dtype=torch.float64)
        coefficients = torch.linalg.solve(regularized, rhs).to(dtype=torch.float32)

        correction_norm_squared = 0.0
        expert_delta_norms_squared = {name: 0.0 for name in names}
        for spec in specs:
            correction = torch.zeros_like(spec.prior, device=device, dtype=torch.float32)
            for coefficient, (_family, expert_name, family_specs) in zip(
                coefficients, directions, strict=True
            ):
                if any(spec is member for member in family_specs):
                    delta = (
                        spec.weights[expert_name].to(device=device, dtype=torch.float32)
                        - spec.prior.to(device=device, dtype=torch.float32)
                    )
                    correction.add_(delta, alpha=float(coefficient))
            correction_norm_squared += float(correction.square().sum())
            for name in names:
                delta = (
                    spec.weights[name].to(device=device, dtype=torch.float32)
                    - spec.prior.to(device=device, dtype=torch.float32)
                )
                expert_delta_norms_squared[name] += float(delta.square().sum())
        correction_norm = math.sqrt(correction_norm_squared)
        delta_reference = sum(math.sqrt(value) for value in expert_delta_norms_squared.values()) / len(names)
        trust_limit = args.joint_max_correction_ratio * max(delta_reference, 1e-12)
        trust_scale = min(1.0, trust_limit / max(correction_norm, 1e-12))
        coefficients.mul_(trust_scale)

        def exact_losses(scale: float) -> tuple[float, dict[str, float]]:
            assign_coefficients(coefficients * scale)
            losses = {
                name: float((output_rows(name) - target_outputs[name]).square().mean())
                for name in names
            }
            objective = sum(objective_weights[name] * losses[name] for name in names)
            return objective, losses

        line_search = {}
        for scale_value in (0.0, 0.25, 0.5, 1.0):
            line_search[scale_value] = exact_losses(scale_value)
        selected_scale = min(line_search, key=lambda value: line_search[value][0])
        selected_objective, selected_losses = line_search[selected_scale]
        prior_objective = line_search[0.0][0]
        coefficients.mul_(selected_scale)
        assign_coefficients(coefficients)

        solved_weights = {
            spec.base_name: spec.module.weight.detach().float().cpu() for spec in specs
        }
        direction_labels = [f"{family}/{expert_name}" for family, expert_name, _ in directions]
        result_metrics = {
            "expert_count": len(names),
            "rows_by_expert": {
                name: int(prior_outputs[name].shape[0]) for name in names
            },
            "selected_states_by_expert": {
                name: len(selected_indices[name]) for name in names
            },
            "joint_subspace_grouping": args.joint_subspace_grouping,
            "joint_subspace_dimension": dimension,
            "joint_direction_labels": direction_labels,
            "joint_coefficients": {
                label: float(value)
                for label, value in zip(direction_labels, coefficients, strict=True)
            },
            "finite_difference": epsilon,
            "ridge": ridge,
            "ridge_reference": ridge_reference,
            "kernel_condition": float(torch.linalg.cond(regularized)),
            "trust_scale": trust_scale,
            "selected_line_search_scale": selected_scale,
            "line_search_objectives": {
                str(scale_value): objective for scale_value, (objective, _losses) in line_search.items()
            },
            "prior_loss": sum(prior_loss_by_expert.values()) / len(names),
            "dense_regmean_loss": sum(selected_losses.values()) / len(names),
            "objective_prior_loss": prior_objective,
            "objective_dense_regmean_loss": selected_objective,
            "prior_loss_by_expert": prior_loss_by_expert,
            "dense_regmean_loss_by_expert": selected_losses,
            "expert_objective_weights": objective_weights,
            "expert_normalization_floor": normalization_floor,
            "improvement_vs_prior": (
                prior_objective - selected_objective
            ) / max(prior_objective, 1e-12),
            "rejected_nonimproving": selected_scale == 0.0,
            "correction_norm": correction_norm * trust_scale * selected_scale,
            "task_delta_reference": delta_reference,
        }
        return solved_weights, result_metrics

    def joint_action_path_solve() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Jointly fit Action-layer groups from velocity or integrated terminal action."""
        if args.objective_grouping != "expert":
            raise ValueError("joint_path_subspace currently requires --objective-grouping=expert")
        all_action_specs = [spec for index in range(18) for spec in action_specs[index]]
        prior_gpu = {
            spec.base_name: spec.prior.to(device=device, dtype=torch.float32)
            for spec in all_action_specs
        }
        delta_gpu = {
            (spec.base_name, name): (
                spec.weights[name].to(device=device, dtype=torch.float32)
                - prior_gpu[spec.base_name]
            )
            for spec in all_action_specs
            for name in names
        }
        selected_indices = {name: joint_state_indices(name) for name in names}
        layer_groups = []
        for group_index in range(args.joint_path_layer_groups):
            start = group_index * 18 // args.joint_path_layer_groups
            end = (group_index + 1) * 18 // args.joint_path_layer_groups
            layer_groups.append((f"layers{start:02d}-{end - 1:02d}", start, end))
        directions: list[tuple[str, str, list[LinearSpec]]] = []
        for layer_label, start, end in layer_groups:
            attention_specs = [spec for index in range(start, end) for spec in action_specs[index][:4]]
            mlp_specs = [spec for index in range(start, end) for spec in action_specs[index][4:]]
            for family, family_specs in (("attention", attention_specs), ("mlp", mlp_specs)):
                for expert_name in names:
                    directions.append((f"{layer_label}/{family}", expert_name, family_specs))

        def assign_prior() -> None:
            for spec in all_action_specs:
                spec.module.weight.data.copy_(
                    prior_gpu[spec.base_name].to(dtype=spec.module.weight.dtype)
                )

        def assign_expert(expert_name: str) -> None:
            for spec in all_action_specs:
                spec.module.weight.data.copy_(
                    (prior_gpu[spec.base_name] + delta_gpu[(spec.base_name, expert_name)]).to(
                        dtype=spec.module.weight.dtype
                    )
                )

        def assign_coefficients(coefficients: torch.Tensor) -> None:
            merged = {
                spec.base_name: prior_gpu[spec.base_name].clone()
                for spec in all_action_specs
            }
            for coefficient, (_group, expert_name, direction_specs) in zip(
                coefficients, directions, strict=True
            ):
                for spec in direction_specs:
                    merged[spec.base_name].add_(
                        delta_gpu[(spec.base_name, expert_name)],
                        alpha=float(coefficient),
                    )
            for spec in all_action_specs:
                spec.module.weight.data.copy_(
                    merged[spec.base_name].to(dtype=spec.module.weight.dtype)
                )

        def output_rows(data_name: str) -> torch.Tensor:
            rows: list[torch.Tensor] = []
            for state_index in selected_indices[data_name]:
                state = states_by_name[data_name][state_index]
                assert state.hidden is not None and state.cond is not None
                def velocity_at(x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                    value, _pad_masks, _att_masks, cond = core.embed_suffix(x_t, timestep)
                    value = value.to(first_weight_dtype)
                    for block_index in range(18):
                        value = manual_action_block(
                            expert_model.layers[block_index],
                            expert_model.rotary_emb,
                            value,
                            state.attention_mask,
                            state.position_ids,
                            state.prefix_keys[block_index],
                            state.prefix_values[block_index],
                            cond,
                        )
                    value, _ = expert_model.norm(value, cond)
                    return linear_forward(
                        core.action_out_proj,
                        value[:, -core.config.chunk_size :],
                    )

                if args.action_block_solver == "joint_flow_terminal_subspace":
                    num_steps = int(core.config.num_inference_steps)
                    dt = -1.0 / num_steps
                    terminal = state.action_input.float()
                    for step in range(num_steps):
                        timestep = torch.full(
                            (terminal.shape[0],),
                            1.0 + step * dt,
                            device=device,
                            dtype=torch.float32,
                        )
                        terminal = terminal + dt * velocity_at(terminal, timestep)
                    objective_output = terminal
                else:
                    value = state.hidden
                    for block_index in range(18):
                        value = manual_action_block(
                            expert_model.layers[block_index],
                            expert_model.rotary_emb,
                            value,
                            state.attention_mask,
                            state.position_ids,
                            state.prefix_keys[block_index],
                            state.prefix_values[block_index],
                            state.cond,
                        )
                    value, _ = expert_model.norm(value, state.cond)
                    objective_output = linear_forward(
                        core.action_out_proj,
                        value[:, -core.config.chunk_size :],
                    )
                rows.append(
                    sample_rows(objective_output.float(), args.max_rows_per_sample)
                    * math.sqrt(state_weights_by_name[data_name][state_index])
                )
            return torch.cat(rows, dim=0)

        assign_prior()
        prior_outputs = {name: output_rows(name) for name in names}
        target_outputs: dict[str, torch.Tensor] = {}
        for name in names:
            assign_expert(name)
            target_outputs[name] = output_rows(name)
        assign_prior()
        prior_loss_by_expert = {
            name: float((prior_outputs[name] - target_outputs[name]).square().mean())
            for name in names
        }
        if args.expert_loss_normalization == "none":
            objective_weights = {name: 1.0 / len(names) for name in names}
            normalization_floor = None
        else:
            mean_prior_loss = sum(prior_loss_by_expert.values()) / len(names)
            normalization_floor = max(mean_prior_loss * 1e-6, 1e-12)
            sign = -1.0 if args.expert_loss_normalization == "prior" else 1.0
            masses = {
                name: max(prior_loss_by_expert[name], normalization_floor)
                ** (sign * args.expert_loss_normalization_power)
                for name in names
            }
            total_mass = sum(masses.values())
            objective_weights = {name: masses[name] / total_mass for name in names}

        derivatives: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        epsilon = args.joint_finite_difference
        for direction_index in range(len(directions)):
            coefficients = torch.zeros(len(directions), device=device, dtype=torch.float32)
            coefficients[direction_index] = epsilon
            assign_coefficients(coefficients)
            for name in names:
                derivatives[name].append((output_rows(name) - prior_outputs[name]) / epsilon)
        assign_prior()

        dimension = len(directions)
        hessian = torch.zeros((dimension, dimension), device=device, dtype=torch.float64)
        rhs = torch.zeros(dimension, device=device, dtype=torch.float64)
        for name in names:
            jacobian = torch.stack(
                [value.reshape(-1) for value in derivatives[name]], dim=1
            ).to(dtype=torch.float64)
            target = (target_outputs[name] - prior_outputs[name]).reshape(-1).to(dtype=torch.float64)
            mass = objective_weights[name] / max(target.numel(), 1)
            hessian.addmm_(jacobian.T, jacobian, beta=1.0, alpha=mass)
            rhs.addmv_(jacobian.T, target, beta=1.0, alpha=mass)
        ridge_reference = max(float(torch.trace(hessian) / max(dimension, 1)), 1e-12)
        ridge = args.joint_ridge_ratio * ridge_reference
        regularized = hessian + ridge * torch.eye(dimension, device=device, dtype=torch.float64)
        coefficients = torch.linalg.solve(regularized, rhs).to(dtype=torch.float32)

        correction_norm_squared = 0.0
        expert_delta_norms_squared = {name: 0.0 for name in names}
        for spec in all_action_specs:
            correction = torch.zeros_like(spec.prior, device=device, dtype=torch.float32)
            for coefficient, (_group, expert_name, direction_specs) in zip(
                coefficients, directions, strict=True
            ):
                if any(spec is member for member in direction_specs):
                    correction.add_(
                        delta_gpu[(spec.base_name, expert_name)],
                        alpha=float(coefficient),
                    )
            correction_norm_squared += float(correction.square().sum())
            for name in names:
                delta = delta_gpu[(spec.base_name, name)]
                expert_delta_norms_squared[name] += float(delta.square().sum())
        correction_norm = math.sqrt(correction_norm_squared)
        delta_reference = sum(math.sqrt(value) for value in expert_delta_norms_squared.values()) / len(names)
        trust_limit = args.joint_max_correction_ratio * max(delta_reference, 1e-12)
        trust_scale = min(1.0, trust_limit / max(correction_norm, 1e-12))
        coefficients.mul_(trust_scale)

        def exact_losses(scale: float) -> tuple[float, dict[str, float]]:
            assign_coefficients(coefficients * scale)
            losses = {
                name: float((output_rows(name) - target_outputs[name]).square().mean())
                for name in names
            }
            return sum(objective_weights[name] * losses[name] for name in names), losses

        line_search = {
            scale_value: exact_losses(scale_value)
            for scale_value in (0.0, 0.25, 0.5, 1.0)
        }
        selected_scale = min(line_search, key=lambda value: line_search[value][0])
        selected_objective, selected_losses = line_search[selected_scale]
        prior_objective = line_search[0.0][0]
        coefficients.mul_(selected_scale)
        assign_coefficients(coefficients)
        solved_weights = {
            spec.base_name: spec.module.weight.detach().float().cpu()
            for spec in all_action_specs
        }
        direction_labels = [f"{group}/{expert_name}" for group, expert_name, _ in directions]
        metrics = {
            "expert_count": len(names),
            "rows_by_expert": {name: int(prior_outputs[name].shape[0]) for name in names},
            "selected_states_by_expert": {
                name: len(selected_indices[name]) for name in names
            },
            "joint_subspace_grouping": (
                "flow_terminal_action_path_layer_family"
                if args.action_block_solver == "joint_flow_terminal_subspace"
                else "action_path_layer_family"
            ),
            "flow_terminal_steps": (
                int(core.config.num_inference_steps)
                if args.action_block_solver == "joint_flow_terminal_subspace"
                else None
            ),
            "joint_path_layer_groups": args.joint_path_layer_groups,
            "joint_subspace_dimension": dimension,
            "joint_direction_labels": direction_labels,
            "joint_coefficients": {
                label: float(value)
                for label, value in zip(direction_labels, coefficients, strict=True)
            },
            "finite_difference": epsilon,
            "ridge": ridge,
            "ridge_reference": ridge_reference,
            "kernel_condition": float(torch.linalg.cond(regularized)),
            "trust_scale": trust_scale,
            "selected_line_search_scale": selected_scale,
            "line_search_objectives": {
                str(scale_value): objective
                for scale_value, (objective, _losses) in line_search.items()
            },
            "prior_loss": sum(prior_loss_by_expert.values()) / len(names),
            "dense_regmean_loss": sum(selected_losses.values()) / len(names),
            "objective_prior_loss": prior_objective,
            "objective_dense_regmean_loss": selected_objective,
            "prior_loss_by_expert": prior_loss_by_expert,
            "dense_regmean_loss_by_expert": selected_losses,
            "expert_objective_weights": objective_weights,
            "expert_normalization_floor": normalization_floor,
            "improvement_vs_prior": (
                prior_objective - selected_objective
            ) / max(prior_objective, 1e-12),
            "rejected_nonimproving": selected_scale == 0.0,
            "correction_norm": correction_norm * trust_scale * selected_scale,
            "task_delta_reference": delta_reference,
        }
        return solved_weights, metrics

    if args.action_block_solver in ("joint_path_subspace", "joint_flow_terminal_subspace"):
        solved_weights, path_metrics = joint_action_path_solve()
        for block_index in range(18):
            for spec in action_specs[block_index]:
                merged = solved_weights[spec.base_name]
                output_tensors[spec.weight_key] = merged.to(
                    output_tensors[spec.weight_key].dtype
                ).contiguous()
                metrics[spec.base_name] = {
                    "stage": (
                        "flow_terminal_action_path_joint_subspace"
                        if args.action_block_solver == "joint_flow_terminal_subspace"
                        else "action_path_joint_subspace"
                    ),
                    "block_index": block_index,
                    "kind": (
                        "mway_joint_integrated_terminal_action_subspace"
                        if args.action_block_solver == "joint_flow_terminal_subspace"
                        else "mway_joint_final_flow_velocity_subspace"
                    ),
                    **path_metrics,
                }
        print(
            "Merged complete Action path objective: "
            f"offline improvement={path_metrics['improvement_vs_prior']:.2%}",
            flush=True,
        )
        for state in all_states:
            assert state.hidden is not None and state.cond is not None
            for block_index in range(18):
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
    else:
      for block_index in range(18):
        specs = action_specs[block_index]
        if args.action_block_solver == "joint_subspace":
            solved_weights, block_metrics = joint_block_solve(block_index, specs)
            for spec in specs:
                merged = solved_weights[spec.base_name]
                if not torch.isfinite(merged).all():
                    raise ValueError(f"Non-finite solution for {spec.base_name}")
                spec.module.weight.data.copy_(
                    merged.to(device=device, dtype=spec.module.weight.dtype)
                )
                output_tensors[spec.weight_key] = merged.to(
                    output_tensors[spec.weight_key].dtype
                ).contiguous()
                metrics[spec.base_name] = {
                    "stage": "action_block_joint_subspace_sequential",
                    "block_index": block_index,
                    "kind": "mway_joint_block_residual_subspace",
                    **block_metrics,
                }
        else:
            inputs_by_expert = {
                name: run_and_capture(block_index, states_by_name[name], specs, name) for name in names
            }
            for spec in specs:
                merged, module_metrics = solve_spec(spec, inputs_by_expert)
                if not torch.isfinite(merged).all():
                    raise ValueError(f"Non-finite solution for {spec.base_name}")
                spec.module.weight.data.copy_(merged.to(device=device, dtype=spec.module.weight.dtype))
                output_tensors[spec.weight_key] = merged.to(output_tensors[spec.weight_key].dtype).contiguous()
                metrics[spec.base_name] = {
                    "stage": "action_block_sequential",
                    "block_index": block_index,
                    "kind": "multi_lora_materialized_dense",
                    **module_metrics,
                }
            del inputs_by_expert
        def advance_action(state: ReplayState) -> None:
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
        advance_by_prefix(
            states_by_name, args.replay_prefix,
            lambda name: assign_source(specs, name),
            lambda: restore_merged(specs), advance_action,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        block_improvements = [metrics[spec.base_name]["improvement_vs_prior"] for spec in specs]
        print(
            f"Merged Action block {block_index:02d}/17: "
            f"mean offline improvement={mean(block_improvements):.2%}",
            flush=True,
        )

    def final_hidden(state: ReplayState) -> torch.Tensor:
        assert state.hidden is not None and state.cond is not None
        value, _ = expert_model.norm(state.hidden, state.cond)
        return value[:, -core.config.chunk_size :].float()

    if args.freeze_action_interface_from_prior:
        metrics["model.action_out_proj"] = {
            "stage": "action_interface_frozen_prior",
            "kind": "frozen_prior",
            "improvement_vs_prior": 0.0,
            "trust_scale": 1.0,
            "rejected_nonimproving": False,
        }
    else:
        output_solution, output_metrics = solve_dense_module(
            OUTPUT_MODULE,
            {name: grouped_state_rows(name, final_hidden) for name in names},
        )
        output_tensors["model.action_out_proj.weight"] = output_solution[:, :-1].to(output_tensors["model.action_out_proj.weight"].dtype)
        output_tensors["model.action_out_proj.bias"] = output_solution[:, -1].to(output_tensors["model.action_out_proj.bias"].dtype)
        metrics["model.action_out_proj"] = {"stage": "output_head_sequential", "kind": "multi_dense_augmented", **output_metrics}

    expected_modified = 414 + 2 * 4
    if ablation is None:
        realized_row_total = validate_realized_rows(metrics)
    elif ablation.get('schema') == three_level.SCHEMA:
        realized_row_total = three_level.validate_rows(
            metrics, names, int(ablation['pass_index']))
    elif ablation.get('schema') == matched_main.SCHEMA:
        realized_row_total = matched_main.validate_rows(metrics, names)
    elif ablation.get('schema') == subset.SCHEMA:
        realized_row_total = subset.validate_rows(metrics, tuple(names))
    elif ablation.get('schema') == appendix.SCHEMA:
        realized_row_total = appendix.validate_rows(metrics, names, int(ablation['pass_index']))
    else:
        realized_row_total = validate_rows(metrics, names, ablation['variant'])
    if len(modified_keys) != expected_modified:
        raise ValueError(f"Expected {expected_modified} modified tensors, found {len(modified_keys)}")
    if expert_weight_plan_modules:
        unused_plan_modules = set(expert_weight_plan_modules) - used_weight_plan_modules
        if unused_plan_modules:
            raise ValueError(
                "Weight plan contains modules that were not solved: "
                f"{sorted(unused_plan_modules)}"
            )
    if any(not torch.isfinite(tensor).all() for tensor in output_tensors.values()):
        raise ValueError("Output checkpoint contains NaN or Inf")
    output_tensors = {key: tensor.contiguous() for key, tensor in output_tensors.items()}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.parent.name}-multi-block-regmeanpp-", dir=output.parent))
    copy_support_files(experts, temporary)
    output_weights = temporary / "model.safetensors"
    save_file(output_tensors, output_weights, metadata={"format": "pt"})

    improvements = [entry["improvement_vs_prior"] for entry in metrics.values()]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": (
            "pi05_multiexpert_dense_regmean_expert_prefix"
            if args.replay_prefix == "expert"
            else "pi05_mway_fixed_weight_plan_regmeanpp"
            if args.expert_aggregation == "fixed_weight_plan"
            else "pi05_mway_causal_flow_terminal_joint_subspace"
            if args.action_block_solver == "joint_flow_terminal_subspace"
            else "pi05_mway_causal_action_path_joint_subspace"
            if args.action_block_solver == "joint_path_subspace"
            else "pi05_mway_causal_block_joint_subspace"
            if args.action_block_solver == "joint_subspace"
            else (
                "pi05_multiexpert_strict_sequential_vision27_language18_action18_block_regmeanpp"
                if full_prefix
                else "pi05_multiexpert_strict_sequential_action18_block_regmeanpp"
            )
        ),
        "training": False,
        "table1_method": "TCR-E" if ablation is None else None,
        "ablation": ablation,
        "ablation_config_sha256": sha256(args.ablation_config) if args.ablation_config else None,
        "fixed_ridge_reference": (
            ablation.get('ridge_source_manifest')
            if ablation and ablation.get('schema') in (three_level.SCHEMA, subset.SCHEMA, appendix.SCHEMA)
            else ablation.get('full_manifest')
            if ablation and ablation.get('schema') == matched_main.SCHEMA
            else ablation.get('full_manifest') if ablation else None),
        "subset_experts": (
            list(names) if ablation and ablation.get('schema') == subset.SCHEMA else None),
        "subset_pass": (
            ablation['pass_index'] if ablation and ablation.get('schema') == subset.SCHEMA else None),
        "fixed_appendix_arm": (
            ablation.get('arm') if ablation and ablation.get('schema') == appendix.SCHEMA else None),
        "fixed_appendix_pass": (
            ablation.get('pass_index') if ablation and ablation.get('schema') == appendix.SCHEMA else None),
        "three_level_main_arm": (
            ablation.get('arm') if ablation and ablation.get('schema') == three_level.SCHEMA
            else None),
        "three_level_main_repeat": (
            ablation.get('repeat') if ablation and ablation.get('schema') == three_level.SCHEMA
            else None),
        "three_level_main_pass": (
            ablation.get('pass_index') if ablation and ablation.get('schema') == three_level.SCHEMA
            else None),
        "matched_main_variant": (
            ablation.get('variant') if ablation and ablation.get('schema') == matched_main.SCHEMA
            else None),
        "matched_main_repeat": (
            ablation.get('repeat') if ablation and ablation.get('schema') == matched_main.SCHEMA
            else None),
        "matched_main_pass": (
            ablation.get('pass_index') if ablation and ablation.get('schema') == matched_main.SCHEMA
            else None),
        "dense_expert_bank": str(args.dense_expert_bank),
        "dense_expert_bank_sha256": sha256(args.dense_expert_bank),
        "expert_weight_semantics": "audited_peft_safe_dense_tensors_no_lora_reconstruction",
        "realized_row_total": realized_row_total,
        "gradient_or_backward": False,
        "parameterization": (
            "full_linear_weight_space_fixed_expert_weights"
            if args.expert_aggregation == "fixed_weight_plan"
            else "mway_task_delta_joint_flow_terminal_subspace"
            if args.action_block_solver == "joint_flow_terminal_subspace"
            else "mway_task_delta_joint_action_path_subspace"
            if args.action_block_solver == "joint_path_subspace"
            else "mway_task_delta_joint_action_block_subspace"
            if args.action_block_solver == "joint_subspace"
            else "full_linear_weight_space"
        ),
        "expert_count": len(names),
        "replay_prefix": args.replay_prefix,
        "within_block_capture": "candidate_internal_inputs_before_any_linear_solve",
        "experts": {name: str(experts[name]) for name in names},
        "local_prior": "rollout_reference_dense_checkpoint" if prior_root else "equal_expert_dense_mean",
        "calibration_policy": (
            next(iter(calibration_policies)) if len(calibration_policies) == 1 else None
        ),
        "calibration_policies": calibration_policy_sources_by_name,
        "calibration_source_counts": {
            name: len(calibrations[name]) for name in names
        },
        "calibration_source_weights": calibration_source_weights,
        "ridge_ratio": args.ridge_ratio,
        "ridge_scale": args.ridge_scale,
        "max_correction_ratio": args.max_correction_ratio,
        "max_rows_per_sample": args.max_rows_per_sample,
        "objective_grouping": args.objective_grouping,
        "objective_group_count": len(
            {label for labels in objective_labels_by_name.values() for label in labels}
        ),
        "objective_group_counts_by_expert": {
            name: len(set(objective_labels_by_name[name])) for name in names
        },
        "expert_loss_normalization": args.expert_loss_normalization,
        "expert_loss_normalization_power": args.expert_loss_normalization_power,
        "expert_aggregation": args.expert_aggregation,
        "expert_weight_plan": (
            str(expert_weight_plan_path) if expert_weight_plan_path is not None else None
        ),
        "expert_weight_plan_sha256": (
            sha256(expert_weight_plan_path) if expert_weight_plan_path is not None else None
        ),
        "expert_weight_plan_method": expert_weight_plan_method,
        "minimax_iterations": args.minimax_iterations,
        "minimax_temperature": args.minimax_temperature,
        "freeze_prefix_from_prior": args.freeze_prefix_from_prior,
        "freeze_action_interface_from_prior": args.freeze_action_interface_from_prior,
        "action_block_solver": args.action_block_solver,
        "joint_subspace_grouping": args.joint_subspace_grouping,
        "joint_finite_difference": args.joint_finite_difference,
        "joint_ridge_ratio": args.joint_ridge_ratio,
        "joint_max_states_per_expert": args.joint_max_states_per_expert,
        "joint_max_correction_ratio": args.joint_max_correction_ratio,
        "joint_path_layer_groups": args.joint_path_layer_groups,
        "max_states_per_calibration_source": args.max_states_per_calibration_source,
        "require_objective_improvement": args.require_objective_improvement,
        "module_count": len(metrics),
        "modified_tensor_count": len(modified_keys),
        "replay_reference_max_error": max(reference_errors),
        "manual_native_block_max_error": manual_replay_error,
        "vision_reference_max_error": vision_reference_error,
        "prefix_kv_reference_max_error": prefix_kv_reference_error,
        "reference_diagnostic_semantics": (
            "Legacy reference fields compare the pass prior/current modules with stored "
            "calibration-policy tensors before sequential solving; they are not replay "
            "equivalence tests when calibration policies differ from the prior."
        ),
        "expert_prefix_frontend_reference_max_error": expert_prefix_frontend_error,
        "expert_prefix_vision_reference_max_error": expert_prefix_vision_error,
        "expert_prefix_kv_reference_max_error": expert_prefix_kv_error,
        "improvement_summary": summarize(improvements),
        "trust_limited_module_count": sum(entry["trust_scale"] < 1.0 for entry in metrics.values()),
        "rejected_nonimproving_module_count": sum(
            entry["rejected_nonimproving"] for entry in metrics.values()
        ),
        "inputs": {
            "base_model": str(base_root),
            "prior_model": str(prior_root) if prior_root else None,
            "calibrations": {
                name: [str(path) for path in calibrations[name]] for name in names
            },
            "manifests": {
                name: [str(path) for path in manifests[name]] for name in names
            },
            "sample_counts": {name: len(states_by_name[name]) for name in names},
            "adapter_sha256": {name: sha256(adapter_paths[name]) for name in names},
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
                "improvement_summary": manifest["improvement_summary"],
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
