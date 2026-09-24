"""Bounded prior-centered regression for wide OFT Linear modules.

For expert i: mass_i / n_i * ||X_i C^T - X_i (W_i-P)^T||_F^2.
Caller supplies the frozen numerical ridge and expert masses. No new weighting
heuristic or row selection is introduced here. Bias, if calibrated, must be an
explicit augmented column in both X and W-P. This is a solver primitive, not a
complete TCR model export.
"""
import math
import torch


def solve(xs, deltas, masses, ridge, *, max_system=4096, branch="auto"):
    if (not xs or len(xs) != len(deltas) or len(xs) != len(masses)
            or branch not in {"auto", "primal", "dual"}):
        raise ValueError("Invalid regression inputs")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("A finite positive numerical ridge is required")
    if any(not math.isfinite(float(a)) or a <= 0 for a in masses) or not math.isclose(sum(masses), 1., abs_tol=1e-10):
        raise ValueError("Expert masses must be finite, positive and sum to one")
    device = xs[0].device
    if deltas[0].ndim != 2:
        raise ValueError("Expert delta must have [out,in] shape")
    out_dim, in_dim = deltas[0].shape
    n = sum(x.shape[0] if x.ndim == 2 else 0 for x in xs)
    use_dual = n < in_dim if branch == "auto" else branch == "dual"
    system = n if use_dual else in_dim
    if system > max_system or system <= 0:
        raise ValueError(f"Refusing unbounded {system}x{system} regression system")
    weighted_x, weighted_y = [], []
    for x, delta, mass in zip(xs, deltas, masses):
        if (x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != in_dim
            or delta.shape != (out_dim, in_dim) or x.device != device or delta.device != device
            or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(delta).all())):
            raise ValueError("Non-finite, empty, wrong-shape or mixed-device expert data")
        x = x.to(torch.float64) * math.sqrt(float(mass) / x.shape[0])
        weighted_x.append(x)
        weighted_y.append(x @ delta.to(torch.float64).T)
    x, y = torch.cat(weighted_x), torch.cat(weighted_y)
    if use_dual:
        gram = x @ x.T
        gram.diagonal().add_(ridge)
        correction = (x.T @ torch.linalg.solve(gram, y)).T
    else:
        gram = x.T @ x
        gram.diagonal().add_(ridge)
        correction = torch.linalg.solve(gram, x.T @ y).T
    if not bool(torch.isfinite(correction).all()):
        raise ValueError("Non-finite regression correction")
    return correction, {"branch": "dual" if use_dual else "primal", "rows": n,
                        "input_dim": in_dim, "output_dim": out_dim, "ridge": ridge,
                        "system_dim": system, "masses": list(masses)}
