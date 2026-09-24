"""Covariance-scale-balanced, alpha-interpolated ridge solve (2026-09-19 study).

Two independent knobs, crossed 2x2:

    teacher alpha       Y_i = [(1-alpha) X_i + alpha X_i^expert] W_i^T
    covariance balance  the per-expert coefficient on each task's residual

The design matrix is always the student's own ``X_i``; only the target and the
coefficients move.  The solved system is

    min_W  sum_i coefficient_i || X_i W^T - Y_i ||_F^2  +  lambda || W - P ||_F^2

with

    off :  coefficient_i = (1/N) / n_i
    on  :  coefficient_i = s * b_i / n_i

BALANCE DEFINITION (registered; not to be tuned after seeing scores).  Per module, with
``H_i = X_i^T X_i / n_i``:

    c_i  = max(||H_i||_F, 1e-12)
    b_i  = (1/c_i) / sum_j (1/c_j)
    E_U  = sum_i (1/N) tr(H_i)
    E_B  = sum_i b_i tr(H_i)
    s    = E_U / E_B          (s = 1 when E_B is zero; recorded explicitly)

``c_i`` is computed through the row-Gram identity ``||X^T X||_F == ||X X^T||_F``, so the
largest matrix formed is n x n (800 x 800 here) and never D x D (up to 16384 x 16384).

WHY THE TRACE MATCH.  Re-weighting experts changes the overall scale of the loss, and a
fixed numeric ridge would then mean something different on each arm, so a pure scale
change could masquerade as a balancing benefit.  ``s`` rescales the balanced coefficients
so the student's total Gram trace ``sum_i coefficient_i n_i tr(H_i)`` matches the uniform
arm's on the same current features.  Both arms therefore carry the same registered ridge.

ATTRIBUTION.  Normalising by a feature-covariance Frobenius norm is **FeatCal's**
(`vla_merge/featcal_forward_order.py`: ``row_gram.norm(p="fro") / weight_sum``).  This is a
FeatCal-inspired, trace-matched variant transplanted into the TCR solve.  It is **not a new
normalisation principle**, and it does not carry over FeatCal's epsilon or its anchor
blending.

All four arms take this same code path with the same float operations; nothing branches to
a different numeric route, so a rounding difference can never be mistaken for a factor
effect.
"""
from __future__ import annotations

import math

import torch

REGISTERED_ARMS = {
    #          teacher alpha, covariance balance
    "s0":  (0.0, False),
    "s03": (0.3, False),
    "n0":  (0.0, True),
    "n03": (0.3, True),
    # Pre-registered replication of the single contrast n0 - s0 on an independent
    # calibration draw. Same alpha (0) and same balance settings as s0/n0; only the
    # calibration cache differs, so these are not new design cells.
    "s0r1": (0.0, False),
    "n0r1": (0.0, True),
}
# Arms whose evidence comes from the independent replication draw.
REPLICATION_ARMS = ("s0r1", "n0r1")
COVARIANCE_FLOOR = 1e-12


def interpolate_inputs(design: torch.Tensor, teacher: torch.Tensor,
                       alpha: float) -> torch.Tensor:
    """``(1 - alpha) X + alpha X_expert`` with exact endpoints."""
    if alpha == 0.0:
        return design
    if alpha == 1.0:
        return teacher
    return (1.0 - alpha) * design + alpha * teacher


def module_statistics(design: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Per-expert ``n_i``, ``tr(H_i)`` and ``c_i = ||H_i||_F``.

    Reductions accumulate in float64.  These are a handful of scalars per module so the
    cost is negligible, while float32 accumulation over 800 x 16384 entries would make
    ``c_i`` noticeably noisy - and that noise would feed straight into the coefficients.
    """
    statistics = {}
    for name, x in design.items():
        rows = int(x.shape[0])
        if rows == 0:
            raise ValueError(f"{name}: empty design matrix")
        if not torch.isfinite(x).all():
            raise ValueError(f"{name}: design matrix contains non-finite values")
        x64 = x.to(dtype=torch.float64)
        trace = float((x64 * x64).sum()) / rows                      # tr(H) = ||X||_F^2/n
        row_gram = x64 @ x64.T                                       # n x n, never D x D
        covariance_norm = float(torch.linalg.matrix_norm(row_gram)) / rows
        statistics[name] = {"rows": rows, "trace": trace,
                            "covariance_norm": covariance_norm}
    return statistics


def objective_masses(design: dict[str, torch.Tensor], balance: bool):
    """Per-expert objective mass and the diagnostics that justify it.

    Returns ``(masses, diagnostics)`` with ``coefficient_i = masses[i] / n_i``.  The
    unbalanced arm returns exactly ``1/N``, so ``balance=False`` is bit-for-bit the
    uniform solve rather than an approximation of it.
    """
    names = list(design)
    uniform = 1.0 / len(names)
    statistics = module_statistics(design)
    if not balance:
        return ({name: uniform for name in names},
                {"balance": False, "uniform_mass": uniform, "per_expert": statistics})

    inverse = {}
    for name, entry in statistics.items():
        floored = max(entry["covariance_norm"], COVARIANCE_FLOOR)
        entry["covariance_norm_floored"] = floored
        inverse[name] = 1.0 / floored
    inverse_sum = sum(inverse.values())
    if not math.isfinite(inverse_sum) or inverse_sum <= 0:
        raise ValueError(f"Invalid covariance normalisation: {inverse}")
    raw = {name: inverse[name] / inverse_sum for name in names}

    energy_uniform = sum(uniform * statistics[name]["trace"] for name in names)
    energy_balanced = sum(raw[name] * statistics[name]["trace"] for name in names)
    degenerate = not (energy_balanced > 0 and math.isfinite(energy_balanced))
    # A module whose features are identically zero carries no energy to match; keep the
    # scale at 1 and record it, rather than dividing by zero.
    scale = 1.0 if degenerate else energy_uniform / energy_balanced
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid trace-matching scale {scale}")
    masses = {name: scale * raw[name] for name in names}
    diagnostics = {"balance": True, "per_expert": statistics, "b_raw": raw,
                   "trace_scale_s": scale, "energy_uniform": energy_uniform,
                   "energy_balanced": energy_balanced,
                   "degenerate_zero_energy": degenerate, "masses": masses}
    return masses, diagnostics


def prepare(inputs, targets, weights, prior, device):
    names = list(inputs)
    if set(names) != set(weights):
        raise ValueError("Input/weight expert names differ")
    if targets is None:
        targets = inputs
    if set(names) != set(targets):
        raise ValueError("Input/teacher expert names differ")
    prior = prior.to(device=device, dtype=torch.float32)
    design, teacher, weight = {}, {}, {}
    for name in names:
        x_i = inputs[name].to(device=device, dtype=torch.float32)
        t_i = targets[name].to(device=device, dtype=torch.float32)
        w_i = weights[name].to(device=device, dtype=torch.float32)
        if x_i.shape[1] != prior.shape[1] or w_i.shape != prior.shape:
            raise ValueError(
                f"{name}: activation/weight mismatch {tuple(x_i.shape)} {tuple(w_i.shape)}")
        if t_i.shape != x_i.shape:
            raise ValueError(
                f"{name}: teacher rows {tuple(t_i.shape)} do not align with the design "
                f"matrix {tuple(x_i.shape)}; row correspondence must be exact")
        design[name], teacher[name], weight[name] = x_i, t_i, w_i
    return names, design, teacher, weight, prior


def solve_covariance(inputs, weights, prior, ridge_ratio, ridge_scale,
                     max_correction_ratio, targets=None, alpha=0.0, balance=False,
                     fixed_ridge=None, device=None):
    """One module of the 2x2 study.  ``alpha`` and ``balance`` are the only knobs."""
    device = device or prior.device
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"teacher alpha must lie in [0, 1], got {alpha}")
    if targets is None and alpha != 0.0:
        raise ValueError("A non-zero alpha requires a separate expert-prefix teacher")
    names, design, teacher, weight, prior = prepare(inputs, targets, weights, prior, device)

    masses, balance_diagnostics = objective_masses(design, balance)
    blended = {name: interpolate_inputs(design[name], teacher[name], alpha)
               for name in names}
    rows = {name: blended[name] @ weight[name].T for name in names}
    prefix_difference = {
        name: float(torch.linalg.matrix_norm(teacher[name] - design[name])
                    / torch.linalg.matrix_norm(design[name]).clamp_min(1e-12))
        for name in names}

    def loss_of(candidate):
        out = {}
        for name in names:
            residual = design[name] @ candidate.T - rows[name]
            out[name] = float((residual * residual).mean())
        return out

    prior_loss = loss_of(prior)
    delta_reference = sum(
        float(torch.linalg.vector_norm(weight[name] - prior)) for name in names) / len(names)

    scaled_x, scaled_targets = [], []
    for name in names:
        x_i = design[name]
        # coefficient_i = masses[i] / n_i, applied once as a row scale.  Applying it here
        # and nowhere else is what keeps the coefficients from being normalised twice.
        scale = math.sqrt(x_i.shape[0] / masses[name])
        scaled_x.append(x_i / scale)
        scaled_targets.append((rows[name] - x_i @ prior.T) / scale)
    x = torch.cat(scaled_x, dim=0)
    residual_target = torch.cat(scaled_targets, dim=0)

    kernel = x @ x.T
    feature_energy = max(float((x * x).sum() / x.shape[1]), 1e-12)
    kernel_diagonal = max(float(torch.diagonal(kernel).mean()), 1e-12)
    reference = feature_energy if ridge_scale == "feature_energy" else kernel_diagonal
    ridge = ridge_ratio * reference if fixed_ridge is None else fixed_ridge
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Ridge must be positive and finite")
    kernel.diagonal().add_(ridge)
    cholesky, info = torch.linalg.cholesky_ex(kernel)
    if int(info.max()) != 0:
        raise RuntimeError(f"Kernel is not positive definite; info={int(info.max())}")
    coefficients = torch.cholesky_solve(residual_target, cholesky)
    correction = coefficients.T @ x

    raw_norm = float(torch.linalg.vector_norm(correction))
    trust_scale = 1.0
    limit = max_correction_ratio * max(delta_reference, 1e-12)
    if raw_norm > limit:
        trust_scale = limit / raw_norm
        correction = correction * trust_scale
    merged = prior + correction

    losses = loss_of(merged)
    # The quantity the trace match is meant to equalise, recorded so the claim can be
    # checked from the receipts rather than taken on trust.
    gram_trace = sum(masses[name] * balance_diagnostics["per_expert"][name]["trace"]
                     for name in names)
    return merged, {
        "kind": "covariance_balanced_split_teacher",
        "teacher_alpha": alpha,
        "covariance_balance": bool(balance),
        "expert_objective_weights": masses,
        "student_gram_trace": gram_trace,
        "balance_diagnostics": balance_diagnostics,
        "prefix_relative_difference_by_expert": prefix_difference,
        "prefix_relative_difference": sum(prefix_difference.values()) / len(names),
        "rows_by_expert": {name: int(design[name].shape[0]) for name in names},
        "prior_loss_by_expert": prior_loss,
        "dense_regmean_loss_by_expert": losses,
        "prior_loss": sum(prior_loss.values()) / len(names),
        "dense_regmean_loss": sum(losses.values()) / len(names),
        "improvement_vs_prior": 1.0 - (sum(losses.values()) / max(sum(prior_loss.values()), 1e-12)),
        "feature_energy": feature_energy,
        "kernel_diagonal": kernel_diagonal,
        "ridge_reference": reference,
        "ridge": ridge,
        "trust_scale": trust_scale,
        "correction_norm": raw_norm,
    }
