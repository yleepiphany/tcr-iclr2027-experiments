"""Exact primary-source RegMean++ Linear solver for the ICLR 2027 PI0.5 port."""

from __future__ import annotations

import math
from typing import Any

import torch

ORIGINAL_EXPERT_ORDER = ("spatial", "object", "goal", "long")


def _task_loss(x: torch.Tensor, weight: torch.Tensor, target: torch.Tensor) -> float:
    residual = x @ (weight - target).T
    return float((residual * residual).mean())


def solve_original_regmeanpp_weight(
    inputs: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
    metric_reference: torch.Tensor,
    *,
    offdiag_scale: float,
    residual_tolerance: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve ``sum_i Gtilde_i W_i^T`` with no prior or regularizer.

    The metric reference is the equal expert mean. It is only used to report
    the offline improvement and never enters the solved equation.
    """
    if tuple(inputs) != ORIGINAL_EXPERT_ORDER or tuple(weights) != ORIGINAL_EXPERT_ORDER:
        raise ValueError("Original RegMean++ requires spatial/object/goal/long order")
    if not 0.0 < offdiag_scale <= 1.0:
        raise ValueError("offdiag_scale must be in (0, 1]")
    if not 0.0 < residual_tolerance < 1.0:
        raise ValueError("residual_tolerance must be in (0, 1)")

    device = metric_reference.device
    output_width, input_width = map(int, metric_reference.shape)
    prepared_inputs: dict[str, torch.Tensor] = {}
    prepared_weights: dict[str, torch.Tensor] = {}
    rows_by_expert: dict[str, int] = {}
    for name in ORIGINAL_EXPERT_ORDER:
        x = inputs[name].to(device=device, dtype=torch.float64)
        weight = weights[name].to(device=device, dtype=torch.float64)
        if x.ndim != 2 or tuple(weight.shape) != (output_width, input_width):
            raise ValueError(f"{name}: RegMean++ input/weight shape differs")
        if int(x.shape[1]) != input_width or int(x.shape[0]) <= 0:
            raise ValueError(f"{name}: RegMean++ feature shape differs")
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(weight).all()):
            raise ValueError(f"{name}: RegMean++ input is non-finite")
        prepared_inputs[name] = x
        prepared_weights[name] = weight
        rows_by_expert[name] = int(x.shape[0])
    if len(set(rows_by_expert.values())) != 1:
        raise ValueError(f"Original RegMean++ requires equal rows: {rows_by_expert}")

    total_rows = sum(rows_by_expert.values())
    mean_weight_t = torch.stack(
        [prepared_weights[name] for name in ORIGINAL_EXPERT_ORDER]
    ).mean(dim=0).T.contiguous()
    strategy = "exact_active_support_primal_general_solve"
    if input_width <= total_rows:
        matrix = torch.zeros(
            (input_width, input_width), dtype=torch.float64, device=device
        )
        rhs = torch.zeros(
            (input_width, output_width), dtype=torch.float64, device=device
        )
        for name in ORIGINAL_EXPERT_ORDER:
            x = prepared_inputs[name]
            weight_t = prepared_weights[name].T
            gram = x.T @ x
            diagonal = torch.diagonal(gram).clone()
            gram.mul_(offdiag_scale)
            torch.diagonal(gram).add_((1.0 - offdiag_scale) * diagonal)
            matrix.add_(gram)
            rhs.add_(gram @ weight_t)
        support_diagonal = torch.diagonal(matrix)
        if not bool(torch.isfinite(support_diagonal).all()) or bool(
            torch.any(support_diagonal < 0.0)
        ):
            raise RuntimeError("Original RegMean++ support diagonal is invalid")
        active = support_diagonal > 0.0
        active_count = int(active.sum())
        solved_t = mean_weight_t.clone()
        if active_count:
            try:
                solved_t[active] = torch.linalg.solve(
                    matrix[active][:, active], rhs[active]
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Original RegMean++ active-support system is singular; "
                    "ridge/pinv is forbidden"
                ) from exc
        residual = matrix @ solved_t - rhs
        rowspace_shape = None
    else:
        strategy = "exact_active_support_diagonal_plus_woodbury_rowspace"
        diagonal_by_expert = {
            name: x.square().sum(dim=0) for name, x in prepared_inputs.items()
        }
        diagonal = (1.0 - offdiag_scale) * sum(diagonal_by_expert.values())
        if not bool(torch.isfinite(diagonal).all()) or bool(torch.any(diagonal < 0.0)):
            raise RuntimeError(
                "Original RegMean++ support diagonal is invalid; no fallback is allowed"
            )
        active = diagonal > 0.0
        active_count = int(active.sum())
        active_diagonal = diagonal[active]
        root_alpha = math.sqrt(offdiag_scale)
        z_parts = [
            root_alpha * prepared_inputs[name][:, active]
            for name in ORIGINAL_EXPERT_ORDER
        ]
        z = torch.cat(z_parts, dim=0)
        scaled = z * torch.rsqrt(active_diagonal).unsqueeze(0)
        kernel = scaled @ scaled.T
        torch.diagonal(kernel).add_(1.0)
        cholesky, info = torch.linalg.cholesky_ex(kernel)
        if bool(torch.any(info != 0)):
            raise RuntimeError(
                "Original RegMean++ exact row-space kernel failed Cholesky"
            )
        rhs = torch.zeros((active_count, output_width), dtype=torch.float64, device=device)
        for name, z_expert in zip(ORIGINAL_EXPERT_ORDER, z_parts, strict=True):
            weight_t = prepared_weights[name].T[active]
            rhs.add_(z_expert.T @ (z_expert @ weight_t))
            rhs.add_(
                (1.0 - offdiag_scale)
                * diagonal_by_expert[name][active].unsqueeze(1)
                * weight_t
            )
        base = rhs / active_diagonal.unsqueeze(1)
        dual = torch.cholesky_solve(z @ base, cholesky)
        solved_active = base - (z.T @ dual) / active_diagonal.unsqueeze(1)
        solved_t = mean_weight_t.clone()
        solved_t[active] = solved_active
        residual = (
            active_diagonal.unsqueeze(1) * solved_active
            + z.T @ (z @ solved_active)
            - rhs
        )
        rowspace_shape = [int(z.shape[0]), int(z.shape[0])]

    relative_residual = float(torch.linalg.vector_norm(residual)) / max(
        float(torch.linalg.vector_norm(rhs)), torch.finfo(torch.float64).tiny
    )
    if not math.isfinite(relative_residual) or relative_residual > residual_tolerance:
        raise RuntimeError(f"Original RegMean++ residual differs: {relative_residual}")
    merged = solved_t.T.to(dtype=torch.float32)
    if not bool(torch.isfinite(merged).all()):
        raise RuntimeError("Original RegMean++ solution is non-finite")

    reference = metric_reference.to(device=device, dtype=torch.float32)
    merged_losses = {
        name: _task_loss(prepared_inputs[name].float(), merged, prepared_weights[name].float())
        for name in ORIGINAL_EXPERT_ORDER
    }
    reference_losses = {
        name: _task_loss(prepared_inputs[name].float(), reference, prepared_weights[name].float())
        for name in ORIGINAL_EXPERT_ORDER
    }
    merged_loss = sum(merged_losses.values()) / len(ORIGINAL_EXPERT_ORDER)
    reference_loss = sum(reference_losses.values()) / len(ORIGINAL_EXPERT_ORDER)
    return merged.detach().cpu(), {
        "method": "original_regmean_pp",
        "strategy": strategy,
        "offdiag_scale": offdiag_scale,
        "solve_dtype": "torch.float64",
        "equal_expert_weighting": True,
        "raw_xtx_no_normalization": True,
        "prior_ridge": False,
        "pseudo_inverse": False,
        "relative_weighting": False,
        "correction_cap": False,
        "rejection_gate": False,
        "rows_by_expert": rows_by_expert,
        "input_width": input_width,
        "output_width": output_width,
        "relative_residual": relative_residual,
        "rowspace_kernel_shape": rowspace_shape,
        "active_support_columns": active_count,
        "inactive_zero_support_columns": input_width - active_count,
        "inactive_tie_break": "equal_expert_weight_mean",
        "expert_objective_weights": {
            name: 1.0 / len(ORIGINAL_EXPERT_ORDER) for name in ORIGINAL_EXPERT_ORDER
        },
        "prior_loss": reference_loss,
        "dense_regmean_loss": merged_loss,
        "prior_loss_by_expert": reference_losses,
        "dense_regmean_loss_by_expert": merged_losses,
        "improvement_vs_prior": (reference_loss - merged_loss)
        / max(reference_loss, 1e-12),
        "trust_scale": 1.0,
        "rejected_nonimproving": False,
    }
