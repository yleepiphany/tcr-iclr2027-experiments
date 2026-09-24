"""One Fast-WAM TCR block under a merged native execution prefix.

This is a build primitive only. The caller must independently audit all A/B
capture jobs, freeze request identities, run both passes in block order,
export a complete native checkpoint, and verify reload before evaluation.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import sys
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "openvla-tcr-20260921"))
from linear_calibration import affine_parameters, row_indices, solve_module  # noqa: E402
from replay_merged_prefix import replay_selected_expert_states  # noqa: E402


def group_scope(group: str) -> tuple[str, tuple[str, ...]]:
    """Return component and complete current-group tensor prefixes."""
    if group == "00-proprio":
        return "proprio_encoder", ("",)
    if group == "01-video-pre":
        return "mot", tuple(f"mixtures.video.{part}." for part in
                            ("patch_embedding", "text_embedding", "time_embedding",
                             "time_projection"))
    if group.startswith("02-video-block-"):
        index = int(group.rsplit("-", 1)[1])
        if index not in range(30):
            raise ValueError("Video block outside native order")
        return "mot", (f"mixtures.video.blocks.{index}.",)
    if group == "03-action-pre":
        return "mot", tuple(f"mixtures.action.{part}." for part in
                            ("action_encoder", "text_embedding", "time_embedding",
                             "time_projection"))
    if group.startswith("04-action-block-"):
        index = int(group.rsplit("-", 1)[1])
        if index not in range(30):
            raise ValueError("Action block outside native order")
        return "mot", (f"mixtures.action.blocks.{index}.",)
    if group == "05-action-head":
        return "mot", ("mixtures.action.head.",)
    raise ValueError(f"Unknown Fast-WAM native action group: {group}")


def selected_state(model: Any, group: str) -> dict[str, torch.Tensor]:
    component, prefixes = group_scope(group)
    state = getattr(model, component).state_dict()
    chosen = {key: value for key, value in state.items()
              if any(key.startswith(prefix) for prefix in prefixes)}
    if not chosen:
        raise ValueError(f"Empty current-group tensor scope: {group}")
    return chosen


@contextmanager
def expert_group_state(model: Any, group: str, expert: dict[str, Any]):
    """Swap only this entire group; restore its prior on every exit path."""
    component, _ = group_scope(group)
    current = selected_state(model, group)
    source = expert[component]
    if not set(current).issubset(source):
        raise ValueError("Expert checkpoint lacks current-group tensors")
    saved = {key: value.detach().cpu().clone() for key, value in current.items()}
    for key, value in current.items():
        other = source[key]
        if other.shape != value.shape or other.dtype != value.dtype or \
                (other.is_floating_point() and not torch.isfinite(other).all()):
            raise ValueError(f"Expert current-group tensor differs: {key}")
    try:
        with torch.no_grad():
            for key, value in current.items():
                value.copy_(source[key].to(device=value.device))
        yield
    finally:
        with torch.no_grad():
            for key, value in current.items():
                value.copy_(saved[key].to(device=value.device))


def resolve_modules(model: Any, names: list[str]) -> dict[str, torch.nn.Linear]:
    """Resolve exact block-plan paths, including aliases omitted by named_modules."""
    if len(names) != len(set(names)):
        raise ValueError("Duplicate current-group Linear scope")
    modules = {}
    for name in names:
        try:
            module = model.get_submodule(name)
        except (AttributeError, IndexError, KeyError) as error:
            raise ValueError(f"Current-group Linear path absent: {name}") from error
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"Current-group path is not Linear: {name}")
        modules[name] = module
    if len({id(module) for module in modules.values()}) != len(modules):
        raise ValueError("Distinct frozen paths alias the same Linear module")
    return modules


def capture_request_features(model: Any, descriptors: list[dict], record: dict,
                             *, seed: int, cap: int,
                             device: torch.device | str) -> tuple[dict, dict]:
    """One request yields all current-group Linear inputs in native order."""
    names = [item["module"] for item in descriptors]
    modules = resolve_modules(model, names)
    if len(names) != len(set(names)) or any(name not in modules for name in names):
        raise ValueError("Current-group Linear scope differs")
    if type(cap) is not int or cap < 3:
        raise ValueError("A positive frozen row cap covering all flow states is required")
    rows = {name: [] for name in names}
    receipts = {name: [] for name in names}
    handles = []
    try:
        for descriptor in descriptors:
            name = descriptor["module"]
            calls = descriptor["native_calls_per_selected_request"]
            module = modules[name]
            if not isinstance(module, torch.nn.Linear) or \
                    list(module.weight.shape) != descriptor["weight_shape"] or \
                    calls not in (1, 3):
                raise ValueError(f"Native Linear module contract changed: {name}")

            def hook(_module, inputs, *, module_name=name, expected=calls):
                ordinal = len(rows[module_name])
                if ordinal >= expected or len(inputs) != 1 or \
                        inputs[0].shape[-1] != _module.in_features:
                    raise ValueError(f"Native call/input contract changed: {module_name}")
                x = inputs[0].detach().reshape(-1, _module.in_features)
                quota = cap // expected + (ordinal < cap % expected)
                indices = row_indices(len(x), quota, seed=seed,
                                      request_id=record["id"], module_name=module_name,
                                      call_index=ordinal)
                picked = x.index_select(0, indices.to(x.device)).cpu().clone()
                if not torch.isfinite(picked).all():
                    raise ValueError("Nonfinite native Linear input")
                rows[module_name].append(picked)
                receipts[module_name].append({"call": ordinal, "available": len(x),
                                              "selected": indices.tolist()})

            handles.append(module.register_forward_pre_hook(hook))
        replay_selected_expert_states(
            model, record["native_request"], record["trace"], device=device)
    finally:
        for handle in handles:
            handle.remove()
    features = {}
    for descriptor in descriptors:
        name = descriptor["module"]
        if len(rows[name]) != descriptor["native_calls_per_selected_request"]:
            raise ValueError(f"Incomplete native Linear call coverage: {name}")
        x = torch.cat(rows[name], dim=0)
        if modules[name].bias is not None:
            x = torch.cat((x, torch.ones((len(x), 1), dtype=x.dtype)), dim=1)
        features[name] = x
    return features, receipts


def calibrate_group(model: Any, group: str, descriptors: list[dict],
                    experts: dict[str, dict], requests: dict[str, list[dict]],
                    *, mass_rule: str, seed: int, cap: int = 6,
                    ridge_multiplier: float = 0.05,
                    max_correction_ratio: float = 3.0,
                    device: torch.device | str = "cuda") -> dict:
    """Capture all expert interfaces before solving any Linear in one group."""
    if set(experts) != {"spatial", "object", "goal", "long"} or \
            set(requests) != set(experts) or mass_rule not in {"relative", "uniform"}:
        raise ValueError("Four frozen experts and explicit mass rule required")
    names = [item["module"] for item in descriptors]
    if not names or len(set(names)) != len(names):
        raise ValueError("Empty or duplicate current-group Linear scope")
    component, prefixes = group_scope(group)
    if any(not name.startswith(component + ".") and name != component for name in names):
        raise ValueError("Linear outside current group component")
    modules = resolve_modules(model, names)
    saved = {key: value.detach().cpu().clone()
             for key, value in selected_state(model, group).items()}
    priors = {name: affine_parameters(modules[name]) for name in names}
    xs = {name: [] for name in names}
    expert_weights = {name: [] for name in names}
    receipts = {}
    try:
        for expert in ("spatial", "object", "goal", "long"):
            if not requests[expert] or len({row["id"] for row in requests[expert]}) != len(requests[expert]):
                raise ValueError("Empty or duplicated expert request identities")
            parts = {name: [] for name in names}
            receipts[expert] = []
            with expert_group_state(model, group, experts[expert]):
                for name in names:
                    expert_weights[name].append(affine_parameters(modules[name]))
                for record in requests[expert]:
                    features, row_receipts = capture_request_features(
                        model, descriptors, record, seed=seed, cap=cap, device=device)
                    for name in names:
                        parts[name].append(features[name])
                    receipts[expert].append({"request_id": record["id"],
                                             "modules": row_receipts})
            for name in names:
                xs[name].append(torch.cat(parts[name]))
        reports = {}
        for name in names:
            reports[name] = solve_module(
                modules[name], xs[name], expert_weights[name], priors[name],
                mass_rule=mass_rule, ridge_multiplier=ridge_multiplier,
                max_correction_ratio=max_correction_ratio, device=device)
        current = selected_state(model, group)
        solved_keys = set()
        for name in names:
            if name == component:
                prefix = ""
            elif name.startswith(component + "."):
                prefix = name[len(component) + 1:]
            else:
                raise ValueError("Solved module outside current component")
            solved_keys.add(prefix + ".weight" if prefix else "weight")
            if modules[name].bias is not None:
                solved_keys.add(prefix + ".bias" if prefix else "bias")
        if any(not torch.equal(value.detach().cpu(), saved[key])
               for key, value in current.items() if key not in solved_keys):
            raise ValueError("A non-regressed current-group tensor changed")
        return {"complete": True, "group": group, "modules": reports,
                "rows": sum(sum(len(x) for x in xs[name]) for name in names),
                "all_features_before_solve": True,
                "nonlinear_tensor_fallback_preserved": True,
                "request_receipts": receipts, "is_complete_model": False}
    except BaseException:
        with torch.no_grad():
            for key, value in selected_state(model, group).items():
                value.copy_(saved[key].to(device=value.device))
        raise
