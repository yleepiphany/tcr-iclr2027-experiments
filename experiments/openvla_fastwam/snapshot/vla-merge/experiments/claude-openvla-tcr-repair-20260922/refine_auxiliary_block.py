"""Independent, CPU-testable candidate for internal head/proprio replay.

Only the current auxiliary Linear is temporarily expert state. Earlier solved
Linears stay live, so the next Linear sees their actual merged outputs. This
does not change the frozen R1/R2 runner or start a GPU experiment.
"""

from contextlib import contextmanager

import torch

from expert_bank import resolve_block
from linear_calibration import affine_parameters, capture_block_rows, solve_module
from materialize_soup import ORDER


def native_linear_order(policy, requests, block, module_names, expected_calls):
    """Derive order from forward hooks, and reject feedback across modules."""
    modules = dict(policy.linear_modules())
    members = {id(module) for module in block.modules()}
    if (not module_names or len(module_names) != len(set(module_names))
            or set(module_names) != set(expected_calls)
            or any(name not in modules or id(modules[name]) not in members for name in module_names)):
        raise ValueError("Invalid auxiliary Linear scope")
    reference = None
    for request in requests:
        calls, handles = [], []
        try:
            for name in module_names:
                handles.append(modules[name].register_forward_pre_hook(
                    lambda _module, _args, name=name: calls.append(name)))
            with torch.no_grad():
                policy.replay(request)
        finally:
            for handle in handles:
                handle.remove()
        if {name: calls.count(name) for name in module_names} != expected_calls:
            raise ValueError("Actual auxiliary Linear call count differs")
        if reference is not None and calls != reference:
            raise ValueError("Auxiliary Linear execution order varies by request")
        reference = calls
    if reference is None:
        raise ValueError("No frozen request available to discover forward order")
    order = list(dict.fromkeys(reference))
    if any(reference.count(name) != expected_calls[name] for name in order):
        raise ValueError("Auxiliary call trace differs")
    # Repeated calls to one module are supported when contiguous. Interleaved
    # feedback needs a different optimizer and must never be silently sorted.
    if [name for name in order for _ in range(expected_calls[name])] != reference:
        raise ValueError("Interleaved shared Linear cannot be calibrated atomically")
    return order


def _module_state(block_state, block_name, name, module):
    prefix = name.removeprefix(block_name + ".") + "."
    expected = set(module.state_dict())
    state = {key: block_state[prefix + key] for key in expected
             if prefix + key in block_state}
    if set(state) != expected:
        raise ValueError("Expert auxiliary Linear state incomplete")
    return state


@contextmanager
def _temporary_module_state(module, state):
    saved = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    try:
        module.load_state_dict(state, strict=True)
        yield
    finally:
        module.load_state_dict(saved, strict=True)


def calibrate_auxiliary_block_ordered(policy, bank, block_name, request_bank, *,
                                      module_names, expected_calls,
                                      expected_rows_per_expert, cap, seed,
                                      mass_rule, ridge_multiplier,
                                      max_correction_ratio, row_mode="uniform",
                                      fixed_ridges=None,
                                      ridge_source="computed_pass_A", device="cpu"):
    """One proposed head/proprio block, preserving upstream merged activations.

    The caller must independently bind the same A/B pools, source hashes and
    numerical contract as the frozen R2 pass. This candidate makes no export.
    """
    if block_name not in ("action_head", "proprio_projector"):
        raise ValueError("Only OpenVLA auxiliary blocks may be refined")
    if set(request_bank) != set(ORDER) or any(not request_bank[key] for key in ORDER):
        raise ValueError("Four nonempty frozen expert request pools required")
    for expert in ORDER:
        requests = request_bank[expert]
        if (any(set(record) != {"id", "inputs"} or not isinstance(record["id"], str)
                or not record["id"] for record in requests)
                or len({record["id"] for record in requests}) != len(requests)):
            raise ValueError("Invalid or duplicate auxiliary request identity")
    if (set(expected_rows_per_expert) != set(module_names)
            or any(type(value) is not int or value <= 0
                   for value in expected_rows_per_expert.values())):
        raise ValueError("Missing exact per-module row contract")
    if fixed_ridges is None:
        if ridge_source != "computed_pass_A":
            raise ValueError("Pass A ridge source differs")
    elif set(fixed_ridges) != set(module_names) or ridge_source != "fixed_from_same_candidate_pass_A":
        raise ValueError("Pass B ridge map differs")
    block = resolve_block(policy, block_name)
    modules = dict(policy.linear_modules())
    order = native_linear_order(
        policy, [request_bank[key][0]["inputs"] for key in ORDER],
        block, module_names, expected_calls)
    parameters = [value for name in module_names
                  for value in modules[name].parameters(recurse=False)]
    if len({id(value) for value in parameters}) != len(parameters):
        raise ValueError("Aliased auxiliary Linear parameters require a separate contract")
    saved = {key: value.detach().cpu().clone() for key, value in block.state_dict().items()}
    reports, receipts, conversions = {}, {}, {}
    try:
        for name in order:
            target = modules[name]
            prior = affine_parameters(target)
            xs, weights = [], []
            receipts[name] = {}
            for expert in ORDER:
                expert_state, native_conversions = bank.block_state(expert, block_name, block)
                conversions.setdefault(expert, native_conversions)
                if conversions[expert] != native_conversions:
                    raise ValueError("Expert auxiliary dtype conversion changed within block")
                state = _module_state(expert_state, block_name, name, target)
                parts, receipts[name][expert] = [], []
                with _temporary_module_state(target, state):
                    weights.append(affine_parameters(target))
                    for record in request_bank[expert]:
                        features, rows = capture_block_rows(
                            policy, block, [name], record["inputs"],
                            request_id=record["id"], cap=cap, seed=seed,
                            expected_calls={name: expected_calls[name]},
                            row_mode=row_mode)
                        parts.append(features[name])
                        receipts[name][expert].append(
                            {"request_id": record["id"], "calls": rows[name]})
                x = torch.cat(parts)
                if len(x) != expected_rows_per_expert[name]:
                    raise ValueError("Realized auxiliary rows differ from frozen plan")
                xs.append(x)
            reports[name] = solve_module(
                target, xs, weights, prior, mass_rule=mass_rule,
                ridge_multiplier=ridge_multiplier,
                max_correction_ratio=max_correction_ratio,
                fixed_ridge=None if fixed_ridges is None else fixed_ridges[name],
                ridge_source=ridge_source, device=device)
        solved_keys = {name.removeprefix(block_name + ".") + "." + key
                       for name in order for key in modules[name].state_dict()}
        if any(not torch.equal(value, block.state_dict()[key].detach().cpu())
               for key, value in saved.items() if key not in solved_keys):
            raise ValueError("Non-regressed auxiliary tensor changed")
        collected_rows = sum(expected_rows_per_expert.values()) * len(ORDER)
        return {"candidate_only": True, "block": block_name,
                "actual_linear_order": order, "modules": reports,
                "row_receipts": receipts, "native_dtype_conversions": conversions,
                "collected_rows": collected_rows, "row_mode": row_mode,
                "ridge_source": ridge_source, "all_features_before_solve": False,
                "is_complete_model": False, "complete": True}
    except BaseException:
        block.load_state_dict(saved, strict=True)
        raise
