"""One complete native TCR block: collect all features, solve, commit or rollback.

The caller owns the frozen A/B manifest, forward block order and final export.
No success-based selection, full-model checkpoint, or pass-level retry is here.
"""
import torch
from expert_bank import resolve_block
from materialize_soup import ORDER
from linear_calibration import (affine_parameters, capture_block_rows,
                                expert_block_state, solve_module)


def calibrate_block(policy, bank, block_name, request_bank, *, module_names,
                    expected_calls, expected_rows_per_expert, cap, seed,
                    mass_rule, ridge_multiplier, max_correction_ratio,
                    row_mode='uniform', fixed_ridges=None,
                    ridge_source='computed_pass_A', device='cpu'):
    if set(request_bank) != set(ORDER) or len(set(module_names)) != len(module_names) or not module_names:
        raise ValueError('Exactly four expert pools and a nonempty unique module scope required')
    if set(expected_rows_per_expert) != set(module_names) or set(expected_calls) != set(module_names):
        raise ValueError('Missing frozen per-module call/row contract')
    if any(type(value) is not int or value <= 0 for value in expected_rows_per_expert.values()):
        raise ValueError('Positive actual per-expert row counts required')
    if fixed_ridges is None:
        if ridge_source != 'computed_pass_A':
            raise ValueError('Pass A ridge source differs')
    elif set(fixed_ridges) != set(module_names) or ridge_source != 'fixed_from_same_candidate_pass_A':
        raise ValueError('Pass B ridge map does not exactly cover this block')
    for expert in ORDER:
        requests = request_bank[expert]
        if not requests or any(set(record) != {'id', 'inputs'} or not isinstance(record['id'], str)
                               or not record['id'] for record in requests):
            raise ValueError('Invalid native request records')
        if len({record['id'] for record in requests}) != len(requests):
            raise ValueError('Duplicate request identities within an expert pool')
    block = resolve_block(policy, block_name)
    modules = dict(policy.linear_modules())
    members = {id(value) for value in block.modules()}
    if any(name not in modules or id(modules[name]) not in members for name in module_names):
        raise ValueError('Out-of-block module')
    parameters = [value for name in module_names for value in modules[name].parameters(recurse=False)]
    if len({id(value) for value in parameters}) != len(parameters):
        raise ValueError('Aliased regression parameters require a separate explicit contract')
    saved = {key: value.detach().cpu().clone() for key, value in block.state_dict().items()}
    priors = {name: affine_parameters(modules[name]) for name in module_names}
    xs = {name: [] for name in module_names}
    expert_weights = {name: [] for name in module_names}
    receipts, conversions = {}, {}
    try:
        # All features in this block are collected before any regression update.
        for expert in ORDER:
            state, conversions[expert] = bank.block_state(expert, block_name, block)
            parts = {name: [] for name in module_names}
            receipts[expert] = []
            with expert_block_state(block, state):
                for name in module_names:
                    expert_weights[name].append(affine_parameters(modules[name]))
                for record in request_bank[expert]:
                    features, rows = capture_block_rows(
                        policy, block, module_names, record['inputs'], request_id=record['id'],
                        cap=cap, seed=seed, expected_calls=expected_calls,
                        row_mode=row_mode)
                    for name in module_names:
                        parts[name].append(features[name])
                    receipts[expert].append({'request_id': record['id'], 'modules': rows})
            for name in module_names:
                features = torch.cat(parts[name])
                if len(features) != expected_rows_per_expert[name]:
                    raise ValueError('Realized rows differ from the frozen module contract')
                xs[name].append(features)
            del state, parts
        reports = {}
        for name in module_names:
            reports[name] = solve_module(modules[name], xs[name], expert_weights[name], priors[name],
                                         mass_rule=mass_rule, ridge_multiplier=ridge_multiplier,
                                         max_correction_ratio=max_correction_ratio,
                                         fixed_ridge=(None if fixed_ridges is None else fixed_ridges[name]),
                                         ridge_source=ridge_source, device=device)
        current = block.state_dict()
        solved_keys = set()
        for name in module_names:
            relative = name.removeprefix(block_name + '.') if name != block_name else ''
            solved_keys.update((relative + '.' if relative else '') + key
                               for key in modules[name].state_dict())
        if any(not torch.equal(current[key].detach().cpu(), value)
               for key, value in saved.items() if key not in solved_keys):
            raise ValueError('Non-regressed block tensor changed')
        return {'block': block_name, 'complete': True, 'all_features_before_solve': True,
                'modules': reports, 'native_dtype_conversions': conversions,
                'row_receipts': receipts, 'collected_rows': sum(len(x) for items in xs.values() for x in items),
                'row_mode': row_mode, 'ridge_source': ridge_source,
                'is_complete_model': False}
    except BaseException:
        block.load_state_dict(saved, strict=True)
        raise
