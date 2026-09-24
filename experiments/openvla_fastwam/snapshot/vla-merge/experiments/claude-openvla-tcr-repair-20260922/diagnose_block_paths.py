"""Read-only comparison of the inputs seen by a block's native Linears.

This is an independent diagnostic, not a calibration or experiment launcher.
The caller supplies one frozen request and a separately loaded full expert policy.
"""

from __future__ import annotations

import math

import torch

from linear_calibration import expert_block_state


def _capture(policy, request, names):
    modules = dict(policy.linear_modules())
    if set(names) - set(modules):
        raise ValueError("Missing named Linear in replay policy")
    rows = {name: [] for name in names}
    handles = []
    try:
        for name in names:
            def hook(module, args, *, name=name):
                if len(args) != 1 or args[0].shape[-1] != module.in_features:
                    raise ValueError(f"Unexpected Linear input: {name}")
                rows[name].append(args[0].detach().cpu().float().clone())
            handles.append(modules[name].register_forward_pre_hook(hook))
        with torch.no_grad():
            policy.replay(request)
    finally:
        for handle in handles:
            handle.remove()
    if any(not calls for calls in rows.values()):
        raise ValueError("A requested Linear was not reached by native replay")
    return rows


def _metrics(left, right):
    if len(left) != len(right):
        raise ValueError("Linear call counts differ")
    values = []
    for a, b in zip(left, right):
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Linear input shape/nonfinite mismatch")
        x, y = a.double().reshape(-1), b.double().reshape(-1)
        relative_mse = float((x - y).square().mean() / max(float(y.square().mean()), 1e-12))
        denom = max(float(x.norm() * y.norm()), 1e-12)
        cosine = float(torch.dot(x, y) / denom)
        if not math.isfinite(relative_mse) or not math.isfinite(cosine):
            raise ValueError("Nonfinite path comparison")
        values.append({"shape": list(a.shape), "relative_mse": relative_mse, "cosine": cosine})
    return values


def compare_paths(policy, expert_policy, block, expert_block_state_dict, request, names):
    """Compare live merged, current solve-time, and full-expert inputs.

    The solve-time path exactly mirrors ``calibrate_block``: the already merged
    policy prefix stays live while the *whole current block* is expert state.
    The expert path additionally uses the separately loaded expert prefix.
    No policy state is committed or changed by this function.
    """
    if not names or len(names) != len(set(names)):
        raise ValueError("Nonempty unique Linear names required")
    live = _capture(policy, request, names)
    with expert_block_state(block, expert_block_state_dict):
        solve = _capture(policy, request, names)
    expert = _capture(expert_policy, request, names)
    return {name: {
        "calls": len(live[name]),
        "solve_vs_live": _metrics(solve[name], live[name]),
        "expert_vs_solve": _metrics(expert[name], solve[name]),
        "expert_vs_live": _metrics(expert[name], live[name]),
    } for name in names}
