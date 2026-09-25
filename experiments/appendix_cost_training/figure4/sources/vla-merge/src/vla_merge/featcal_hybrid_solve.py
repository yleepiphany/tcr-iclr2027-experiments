"""Exact memory-bounded FeatCal solve selection for the formal PI0.5 lane.

The existing Woodbury implementation is ideal for the 16k language MLP where
retained rows are fewer than input features.  It is not memory safe for the
vision layers: their frozen contract retains 18k rows while the input width is
at most 4304.  This module adds the mathematically identical primal SPD solve
and selects the smaller exact system without changing any FeatCal weighting.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from vla_merge.featcal_forward_order import (
    FeatCalForwardOrderError,
    FeatCalRowFactors,
    featcal_rowspace_linear_weight,
)


def estimate_exact_solve_workspace(
    *,
    input_width: int,
    output_width: int,
    task_rows: Sequence[int],
    element_bytes: int,
    output_chunk_size: int,
) -> dict[str, int]:
    if input_width < 1 or output_width < 1 or not task_rows:
        raise ValueError("Solve dimensions must be positive")
    if element_bytes not in (4, 8) or output_chunk_size < 1:
        raise ValueError("Unsupported solve workspace settings")
    total_rows = sum(int(value) for value in task_rows)
    maximum_rows = max(int(value) for value in task_rows)
    chunk = min(int(output_chunk_size), output_width)
    # Matches the allocation bound enforced by featcal_rowspace_linear_weight.
    rowspace_elements = (
        2 * total_rows * input_width
        + 2 * total_rows * output_width
        + 2 * total_rows * total_rows
        + output_width * input_width
        + max(
            3 * maximum_rows * input_width
            + maximum_rows * maximum_rows
            + maximum_rows * output_width,
            2 * input_width * chunk + 3 * total_rows * chunk,
        )
    )
    # Primal path keeps one D x D SPD system and Cholesky workspace, one D x O
    # rhs/solution pair, and only one task's weighted factors at a time.
    primal_elements = (
        4 * input_width * input_width
        + 2 * input_width * output_width
        + 3 * maximum_rows * input_width
        + maximum_rows * maximum_rows
        + maximum_rows * output_width
    )
    out_of_core_rowspace_elements = (
        3 * total_rows * total_rows
        + total_rows * 128
        + 4 * total_rows * chunk
        + 3 * input_width * chunk
    )
    return {
        "rowspace_bytes": element_bytes * rowspace_elements,
        "out_of_core_rowspace_bytes": element_bytes
        * out_of_core_rowspace_elements,
        "primal_bytes": element_bytes * primal_elements,
        "total_rows": total_rows,
        "maximum_task_rows": maximum_rows,
    }


def _write_memmap_rows(path: Path, rows: list[Tensor], width: int) -> np.memmap:
    total = sum(int(row.shape[0]) for row in rows)
    value = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float64, shape=(total, width)
    )
    offset = 0
    for row in rows:
        stop = offset + int(row.shape[0])
        value[offset:stop] = row.detach().cpu().double().numpy()
        offset = stop
    value.flush()
    return value


def _out_of_core_rowspace_linear_weight(
    expert_weights: Sequence[Tensor],
    task_factors: Sequence[FeatCalRowFactors],
    *,
    soup_weight: Tensor,
    base_weight: Tensor,
    ridge_lambda: float,
    anchor_blend_rho: float,
    covariance_eps: float,
    solve_dtype: torch.dtype,
    solve_device: torch.device | str,
    output_chunk_size: int,
    max_workspace_bytes: int,
    workspace: dict[str, int],
    temporary_root: Path | None,
) -> tuple[Tensor, dict[str, Any]]:
    if solve_dtype != torch.float64:
        raise FeatCalForwardOrderError(
            "Formal out-of-core row-space solve requires float64"
        )
    if workspace["out_of_core_rowspace_bytes"] > max_workspace_bytes:
        raise MemoryError(
            "FeatCal out-of-core row-space estimated workspace exceeds cap: "
            f"{workspace['out_of_core_rowspace_bytes']} > {max_workspace_bytes}"
        )
    device = torch.device(solve_device)
    ridge = float(ridge_lambda)
    rho = float(anchor_blend_rho)
    eps = float(covariance_eps)
    mu = ridge + eps
    output_width, input_width = (int(value) for value in soup_weight.shape)
    chunk = min(int(output_chunk_size), output_width)
    task_info: list[dict[str, Any]] = []
    z_rows: list[Tensor] = []
    q_rows: list[Tensor] = []
    for task_index, (expert_weight, factors) in enumerate(
        zip(expert_weights, task_factors, strict=True)
    ):
        xs = factors.student.double()
        xt = factors.target.double()
        weights = factors.row_weights.double()
        if xs.ndim != 2 or xt.shape != xs.shape or xs.shape[1] != input_width:
            raise FeatCalForwardOrderError(
                f"Task {task_index} out-of-core factor dimensions differ"
            )
        weight_sum = weights.sum()
        sqrt_weights = weights.sqrt().unsqueeze(-1)
        weighted_xs = sqrt_weights * xs
        weighted_xt = sqrt_weights * xt
        # R_task is at most 1500 for the language path selected here.
        row_gram = weighted_xs @ weighted_xs.transpose(0, 1)
        covariance_norm = (row_gram.norm(p="fro") / weight_sum).clamp_min(eps)
        factor_scale = torch.rsqrt(weight_sum * covariance_norm)
        z_task = (factor_scale * weighted_xs).contiguous()
        q_task = torch.empty(
            (xs.shape[0], output_width), dtype=torch.float64, device="cpu"
        )
        for start in range(0, output_width, chunk):
            stop = min(start + chunk, output_width)
            q_task[:, start:stop] = (
                factor_scale
                * weighted_xt
                @ expert_weight[start:stop].double().transpose(0, 1)
            )
        z_rows.append(z_task)
        q_rows.append(q_task)
        task_info.append(
            {
                "task": task_index,
                "rows": int(xs.shape[0]),
                "weight_sum": float(weight_sum),
                "covariance_frobenius_norm": float(covariance_norm),
                "normalization_factor": float(factor_scale),
                "forward_identity": factors.forward_identity,
            }
        )
        del xs, xt, weights, sqrt_weights, weighted_xs, weighted_xt, row_gram

    temp_parent = None if temporary_root is None else str(temporary_root)
    with tempfile.TemporaryDirectory(prefix=".featcal-formal-solve-", dir=temp_parent) as tmp:
        tmp_path = Path(tmp)
        z = _write_memmap_rows(tmp_path / "z.npy", z_rows, input_width)
        q = _write_memmap_rows(tmp_path / "q.npy", q_rows, output_width)
        del z_rows, q_rows

        def z_tile(start: int, stop: int) -> Tensor:
            return torch.from_numpy(np.array(z[:, start:stop], copy=True)).to(
                device=device, dtype=torch.float64
            )

        rows = workspace["total_rows"]
        feature_tile = 128
        kernel = torch.zeros((rows, rows), dtype=torch.float64, device=device)
        for start in range(0, input_width, feature_tile):
            stop = min(start + feature_tile, input_width)
            tile = z_tile(start, stop)
            kernel.addmm_(tile, tile.transpose(0, 1))
            del tile
        kernel.diagonal().add_(mu)
        cholesky, info = torch.linalg.cholesky_ex(kernel)
        if torch.any(info != 0):
            raise FeatCalForwardOrderError(
                f"Out-of-core row-space Cholesky failed: {info.tolist()}"
            )
        del kernel, info
        output = torch.empty_like(soup_weight, device="cpu")
        rhs_norm_squared = 0.0
        residual_norm_squared = 0.0
        max_abs_residual = 0.0
        for output_start in range(0, output_width, chunk):
            output_stop = min(output_start + chunk, output_width)
            width = output_stop - output_start
            q_chunk = torch.from_numpy(
                np.array(q[:, output_start:output_stop], copy=True)
            ).to(device=device, dtype=torch.float64)
            anchor_chunk = (
                rho
                * soup_weight[output_start:output_stop].to(
                    device=device, dtype=torch.float64
                )
                + (1.0 - rho)
                * base_weight[output_start:output_stop].to(
                    device=device, dtype=torch.float64
                )
            )
            rhs = torch.empty((input_width, width), dtype=torch.float64, device=device)
            for feature_start in range(0, input_width, feature_tile):
                feature_stop = min(feature_start + feature_tile, input_width)
                tile = z_tile(feature_start, feature_stop)
                rhs[feature_start:feature_stop].copy_(
                    tile.transpose(0, 1) @ q_chunk
                    + ridge * anchor_chunk[:, feature_start:feature_stop].transpose(0, 1)
                )
                del tile
            projected_rhs = torch.zeros((rows, width), dtype=torch.float64, device=device)
            for feature_start in range(0, input_width, feature_tile):
                feature_stop = min(feature_start + feature_tile, input_width)
                tile = z_tile(feature_start, feature_stop)
                projected_rhs.add_(tile @ rhs[feature_start:feature_stop])
                del tile
            correction = torch.cholesky_solve(projected_rhs, cholesky)
            solution = torch.empty_like(rhs)
            for feature_start in range(0, input_width, feature_tile):
                feature_stop = min(feature_start + feature_tile, input_width)
                tile = z_tile(feature_start, feature_stop)
                solution[feature_start:feature_stop].copy_(
                    (
                        rhs[feature_start:feature_stop]
                        - tile.transpose(0, 1) @ correction
                    )
                    / mu
                )
                del tile
            z_solution = torch.zeros((rows, width), dtype=torch.float64, device=device)
            for feature_start in range(0, input_width, feature_tile):
                feature_stop = min(feature_start + feature_tile, input_width)
                tile = z_tile(feature_start, feature_stop)
                z_solution.add_(tile @ solution[feature_start:feature_stop])
                del tile
            residual = mu * solution - rhs
            for feature_start in range(0, input_width, feature_tile):
                feature_stop = min(feature_start + feature_tile, input_width)
                tile = z_tile(feature_start, feature_stop)
                residual[feature_start:feature_stop].add_(
                    tile.transpose(0, 1) @ z_solution
                )
                del tile
            rhs_norm_squared += float(rhs.square().sum())
            residual_norm_squared += float(residual.square().sum())
            max_abs_residual = max(max_abs_residual, float(residual.abs().max()))
            output[output_start:output_stop].copy_(
                solution.transpose(0, 1).to(device="cpu", dtype=soup_weight.dtype)
            )
            del q_chunk, anchor_chunk, rhs, projected_rhs, correction, solution
            del z_solution, residual
        relative_residual = math.sqrt(residual_norm_squared) / max(
            math.sqrt(rhs_norm_squared), torch.finfo(torch.float64).eps
        )
        temporary_factor_bytes = (
            (tmp_path / "z.npy").stat().st_size + (tmp_path / "q.npy").stat().st_size
        )
        del cholesky, z, q
    if not torch.isfinite(output).all() or relative_residual > 1e-8:
        raise FeatCalForwardOrderError(
            f"Out-of-core row-space residual differs: {relative_residual}"
        )
    metadata = {
        "strategy": "exact_featcal_woodbury_rowspace_out_of_core",
        "input_width": input_width,
        "output_width": output_width,
        "task_count": len(task_factors),
        "total_rows": workspace["total_rows"],
        "rowspace_kernel_shape": [workspace["total_rows"], workspace["total_rows"]],
        "dense_input_gram_allocated": False,
        "ridge_lambda": ridge,
        "covariance_eps": eps,
        "mu": mu,
        "anchor_blend_rho": rho,
        "solve_dtype": str(solve_dtype),
        "solve_device": str(device),
        "output_chunk_size": chunk,
        "estimated_workspace_bytes": workspace["out_of_core_rowspace_bytes"],
        "workspace_cap_bytes": max_workspace_bytes,
        "temporary_factor_bytes": temporary_factor_bytes,
        "max_abs_primal_residual": max_abs_residual,
        "relative_primal_residual": relative_residual,
        "tasks": task_info,
    }
    return output, metadata


def _primal_linear_weight(
    expert_weights: Sequence[Tensor],
    task_factors: Sequence[FeatCalRowFactors],
    *,
    soup_weight: Tensor,
    base_weight: Tensor,
    ridge_lambda: float,
    anchor_blend_rho: float,
    covariance_eps: float,
    solve_dtype: torch.dtype,
    solve_device: torch.device | str,
    max_workspace_bytes: int,
    workspace: dict[str, int],
) -> tuple[Tensor, dict[str, Any]]:
    device = torch.device(solve_device)
    ridge = float(ridge_lambda)
    rho = float(anchor_blend_rho)
    eps = float(covariance_eps)
    mu = ridge + eps
    output_width, input_width = (int(value) for value in soup_weight.shape)
    if workspace["primal_bytes"] > max_workspace_bytes:
        raise MemoryError(
            "FeatCal primal estimated workspace exceeds cap: "
            f"{workspace['primal_bytes']} > {max_workspace_bytes}"
        )
    solve_matrix = torch.eye(input_width, dtype=solve_dtype, device=device)
    solve_matrix.mul_(mu)
    anchor = (
        rho * soup_weight.to(device=device, dtype=solve_dtype)
        + (1.0 - rho) * base_weight.to(device=device, dtype=solve_dtype)
    )
    rhs = ridge * anchor.transpose(0, 1).contiguous()
    task_info: list[dict[str, Any]] = []
    for task_index, (expert_weight, factors) in enumerate(
        zip(expert_weights, task_factors, strict=True)
    ):
        xs = factors.student.to(device=device, dtype=solve_dtype)
        xt = factors.target.to(device=device, dtype=solve_dtype)
        weights = factors.row_weights.to(device=device, dtype=solve_dtype)
        if (
            xs.ndim != 2
            or xt.shape != xs.shape
            or xs.shape[1] != input_width
            or weights.shape != (xs.shape[0],)
            or expert_weight.shape != soup_weight.shape
        ):
            raise FeatCalForwardOrderError(
                f"Task {task_index} primal factor dimensions differ"
            )
        if not all(torch.isfinite(value).all() for value in (xs, xt, weights)):
            raise FeatCalForwardOrderError("Primal factors are non-finite")
        weight_sum = weights.sum()
        if float(weight_sum) <= 0:
            raise FeatCalForwardOrderError("Primal row-weight sum is not positive")
        sqrt_weights = weights.sqrt().unsqueeze(-1)
        weighted_xs = sqrt_weights * xs
        weighted_xt = sqrt_weights * xt
        row_gram = weighted_xs @ weighted_xs.transpose(0, 1)
        covariance_norm = (row_gram.norm(p="fro") / weight_sum).clamp_min(eps)
        normalization = 1.0 / (weight_sum * covariance_norm)
        solve_matrix.addmm_(
            weighted_xs.transpose(0, 1), weighted_xs, alpha=float(normalization)
        )
        projected_target = weighted_xt @ expert_weight.to(
            device=device, dtype=solve_dtype
        ).transpose(0, 1)
        rhs.addmm_(
            weighted_xs.transpose(0, 1),
            projected_target,
            alpha=float(normalization),
        )
        task_info.append(
            {
                "task": task_index,
                "rows": int(xs.shape[0]),
                "weight_sum": float(weight_sum),
                "covariance_frobenius_norm": float(covariance_norm),
                "normalization_factor_squared": float(normalization),
                "forward_identity": factors.forward_identity,
            }
        )
        del xs, xt, weights, sqrt_weights, weighted_xs, weighted_xt, row_gram
        del projected_target

    cholesky, info = torch.linalg.cholesky_ex(solve_matrix)
    if torch.any(info != 0):
        raise FeatCalForwardOrderError(f"Primal Cholesky failed: {info.tolist()}")
    solved = torch.cholesky_solve(rhs, cholesky)
    residual = solve_matrix @ solved - rhs
    relative_residual = float(residual.norm() / rhs.norm().clamp_min(torch.finfo(solve_dtype).eps))
    result = solved.transpose(0, 1).to(device="cpu", dtype=soup_weight.dtype).contiguous()
    if not torch.isfinite(result).all() or not math.isfinite(relative_residual):
        raise FeatCalForwardOrderError("Primal FeatCal solution is non-finite")
    metadata = {
        "strategy": "exact_featcal_primal_spd",
        "input_width": input_width,
        "output_width": output_width,
        "task_count": len(task_factors),
        "total_rows": workspace["total_rows"],
        "primal_matrix_shape": [input_width, input_width],
        "dense_input_gram_allocated": True,
        "rowspace_kernel_allocated": False,
        "ridge_lambda": ridge,
        "covariance_eps": eps,
        "mu": mu,
        "anchor_blend_rho": rho,
        "solve_dtype": str(solve_dtype),
        "solve_device": str(device),
        "estimated_workspace_bytes": workspace["primal_bytes"],
        "workspace_cap_bytes": max_workspace_bytes,
        "max_abs_primal_residual": float(residual.abs().max()),
        "relative_primal_residual": relative_residual,
        "tasks": task_info,
    }
    return result, metadata


def featcal_hybrid_linear_weight(
    expert_weights: Sequence[Tensor],
    task_factors: Sequence[FeatCalRowFactors],
    *,
    soup_weight: Tensor,
    base_weight: Tensor,
    ridge_lambda: float,
    anchor_blend_rho: float,
    covariance_eps: float,
    solve_dtype: torch.dtype = torch.float64,
    solve_device: torch.device | str = "cpu",
    output_chunk_size: int = 64,
    max_workspace_bytes: int = 2 * 1024**3,
    force_strategy: str | None = None,
    temporary_root: Path | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Solve the exact FeatCal system using its smaller primal/dual form."""

    if solve_dtype not in (torch.float32, torch.float64):
        raise FeatCalForwardOrderError("solve_dtype must be float32 or float64")
    if not expert_weights or len(expert_weights) != len(task_factors):
        raise FeatCalForwardOrderError("Expert weights and task factors must align")
    if soup_weight.ndim != 2 or base_weight.shape != soup_weight.shape:
        raise FeatCalForwardOrderError("Soup/base weights must be equal-shape matrices")
    if force_strategy not in (None, "rowspace", "rowspace_out_of_core", "primal"):
        raise ValueError(
            "force_strategy must be rowspace, rowspace_out_of_core, primal, or None"
        )
    element_bytes = torch.empty((), dtype=solve_dtype).element_size()
    workspace = estimate_exact_solve_workspace(
        input_width=int(soup_weight.shape[1]),
        output_width=int(soup_weight.shape[0]),
        task_rows=[int(factors.student.shape[0]) for factors in task_factors],
        element_bytes=element_bytes,
        output_chunk_size=output_chunk_size,
    )
    candidates = {
        "rowspace": workspace["rowspace_bytes"],
        "rowspace_out_of_core": workspace["out_of_core_rowspace_bytes"],
        "primal": workspace["primal_bytes"],
    }
    # Prefer the simpler in-memory dual implementation when it fits and is no
    # larger than primal.  Otherwise compare primal with the tiled exact dual.
    if force_strategy is not None:
        strategy = force_strategy
    elif candidates["rowspace"] <= max_workspace_bytes and candidates["rowspace"] <= candidates["primal"]:
        strategy = "rowspace"
    else:
        strategy = min(("rowspace_out_of_core", "primal"), key=candidates.get)
    if candidates[strategy] > max_workspace_bytes:
        raise MemoryError(
            "Neither selected exact FeatCal solve fits the workspace cap: "
            f"strategy={strategy}, bytes={candidates[strategy]}, cap={max_workspace_bytes}"
        )
    if strategy == "rowspace":
        result, metadata = featcal_rowspace_linear_weight(
            expert_weights,
            task_factors,
            soup_weight=soup_weight,
            base_weight=base_weight,
            ridge_lambda=ridge_lambda,
            anchor_blend_rho=anchor_blend_rho,
            covariance_eps=covariance_eps,
            solve_dtype=solve_dtype,
            solve_device=solve_device,
            output_chunk_size=output_chunk_size,
            max_workspace_bytes=max_workspace_bytes,
        )
        metadata = {**metadata, "hybrid_selected_strategy": "rowspace"}
    elif strategy == "rowspace_out_of_core":
        result, metadata = _out_of_core_rowspace_linear_weight(
            expert_weights,
            task_factors,
            soup_weight=soup_weight,
            base_weight=base_weight,
            ridge_lambda=ridge_lambda,
            anchor_blend_rho=anchor_blend_rho,
            covariance_eps=covariance_eps,
            solve_dtype=solve_dtype,
            solve_device=solve_device,
            output_chunk_size=output_chunk_size,
            max_workspace_bytes=max_workspace_bytes,
            workspace=workspace,
            temporary_root=temporary_root,
        )
        metadata = {**metadata, "hybrid_selected_strategy": "rowspace_out_of_core"}
    else:
        result, metadata = _primal_linear_weight(
            expert_weights,
            task_factors,
            soup_weight=soup_weight,
            base_weight=base_weight,
            ridge_lambda=ridge_lambda,
            anchor_blend_rho=anchor_blend_rho,
            covariance_eps=covariance_eps,
            solve_dtype=solve_dtype,
            solve_device=solve_device,
            max_workspace_bytes=max_workspace_bytes,
            workspace=workspace,
        )
        metadata = {**metadata, "hybrid_selected_strategy": "primal"}
    metadata["hybrid_workspace_estimates"] = workspace
    return result, metadata


__all__ = ["estimate_exact_solve_workspace", "featcal_hybrid_linear_weight"]
