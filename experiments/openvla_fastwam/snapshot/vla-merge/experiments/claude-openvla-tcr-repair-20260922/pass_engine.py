"""Forward-order single-pass TCR engine; caller owns GPU gates and dispatch.

This module never starts a process, retries a failed pass, selects a recipe from
scores, or labels one pass as the final two-pass method. The caller loads a fresh
prior and verified expert/data bank, freezes this recipe, then invokes it only
after native block and export-reload validation.
"""
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open

from block_calibration import calibrate_block
from export_native import export_native
from materialize_soup import ORDER, sha, write


def verify_loaded_prior(policy, template):
    """Check actual native tensor values, allowing only the loader's dtype cast."""
    import json
    template = Path(template).resolve()
    current = policy.model.state_dict()
    mapping = json.loads((template / 'model.safetensors.index.json').read_text())['weight_map']
    if set(mapping) != set(current):
        raise ValueError('Loaded backbone differs from the fixed prior scope')
    checked = 0
    for shard in sorted(set(mapping.values())):
        if Path(shard).name != shard:
            raise ValueError('Unsafe prior shard path')
        with safe_open(str(template / shard), framework='pt', device='cpu') as handle:
            if set(handle.keys()) != {key for key, value in mapping.items() if value == shard}:
                raise ValueError('Prior shard/index inconsistency')
            for key in handle.keys():
                expected = handle.get_tensor(key).to(current[key].dtype)
                if not torch.equal(expected, current[key].detach().cpu()):
                    raise ValueError('Loaded backbone values differ from the fixed pass prior')
                checked += 1
    for component, module in (('action_head', policy.head), ('proprio_projector', policy.proprio)):
        paths = list(template.glob(component + '--*_checkpoint.pt'))
        if len(paths) != 1:
            raise ValueError('Ambiguous prior auxiliary state')
        stored = torch.load(str(paths[0]), map_location='cpu', weights_only=True, mmap=True)
        expected = {key.removeprefix('module.'): value for key, value in stored.items()}
        current = module.state_dict()
        if len(expected) != len(stored) or set(expected) != set(current):
            raise ValueError('Loaded auxiliary scope differs from the fixed prior')
        for key, value in expected.items():
            if not torch.equal(value.to(current[key].dtype), current[key].detach().cpu()):
                raise ValueError('Loaded auxiliary values differ from the fixed pass prior')
            checked += 1
    return {'native_tensor_values_match_prior': True, 'checked_tensors': checked,
            'floating_comparison': 'source cast to the actual native-loaded tensor dtype'}


def validate_recipe(recipe, plan, requests):
    required = {'candidate', 'row_mode', 'pass_id', 'pool', 'mass_rule', 'row_seed',
                'cap', 'ridge_multiplier', 'max_correction_ratio', 'input_provenance'}
    if set(recipe) != required or recipe['pass_id'] not in ('A', 'B') or recipe['pool'] != recipe['pass_id']:
        raise ValueError('Explicit frozen A or B pass recipe required')
    if recipe['candidate'] not in ('R1', 'R2'):
        raise ValueError('Only the frozen R1/R2 repair candidates are allowed')
    expected_row_mode = 'uniform' if recipe['candidate'] == 'R1' else 'llm_action_4_context_4'
    if recipe['row_mode'] != expected_row_mode:
        raise ValueError('Candidate row mode differs from its frozen repair contract')
    if recipe['mass_rule'] != ('relative' if recipe['pass_id'] == 'A' else 'uniform'):
        raise ValueError('Frozen method is relative first pass, uniform second pass')
    if type(recipe['row_seed']) is not int or type(recipe['cap']) is not int:
        raise ValueError('Explicit integer row seed/cap required')
    if recipe['cap'] != plan['row_cap_per_request_across_calls']:
        raise ValueError('Row cap differs from precomputed capacity plan')
    for key in ('ridge_multiplier', 'max_correction_ratio'):
        if not math.isfinite(recipe[key]) or recipe[key] <= 0:
            raise ValueError('Positive finite frozen numerical parameters required')
    if not isinstance(recipe['input_provenance'], dict) or not recipe['input_provenance']:
        raise ValueError('Explicit provenance required; caller verifies its identities')
    if set(requests) != set(ORDER) or any(len(requests[e]) != plan['requests_per_expert'] for e in ORDER):
        raise ValueError('Request bank differs from the frozen per-expert count')
    names = [row['module'] for rows in plan['blocks'].values() for row in rows]
    if len(names) != len(set(names)) or len(names) != plan['linear_count'] or len(plan['blocks']) != plan['block_count']:
        raise ValueError('Duplicate/missing planned block modules')
    expected = 0
    for rows in plan['blocks'].values():
        for row in rows:
            if (row['rows_per_expert'] != row['rows_per_request'] * plan['requests_per_expert']
                    or row['rows_all_experts'] != 4 * row['rows_per_expert']):
                raise ValueError('Capacity plan is internally inconsistent')
            expected += row['rows_all_experts']
    if expected != plan['planned_rows_per_pass_if_all_linears_solved']:
        raise ValueError('Total row accounting differs')


def run_pass(policy, bank, requests, plan, recipe, *, template, run,
             fixed_ridges=None, fixed_ridge_source=None, device='cpu'):
    """Calibrate all planned blocks and export a complete native checkpoint.

    The supplied policy must have been loaded at `template` without intervening
    edits. Actual values are checked; the owner additionally pins source SHA.
    Failure leaves diagnostic block reports but never a successful pass manifest.
    No inference or evaluation of the completed output is claimed here.
    """
    validate_recipe(recipe, plan, requests)
    run = Path(run).resolve()
    if run.exists():
        raise FileExistsError('No resume/retry/overwrite in a pass engine')
    prior_check = verify_loaded_prior(policy, template)
    module_map = dict(policy.linear_modules())
    planned_names = [row['module'] for rows in plan['blocks'].values() for row in rows]
    for rows in plan['blocks'].values():
        for row in rows:
            if row['module'] not in module_map or list(module_map[row['module']].weight.shape) != row['weight_shape']:
                raise ValueError('Planned module absent or wrong native shape')
    if recipe['pass_id'] == 'A':
        if fixed_ridges is not None or fixed_ridge_source is not None:
            raise ValueError('Pass A must compute its own ridge map')
        ridge_source = 'computed_pass_A'
    else:
        if not isinstance(fixed_ridges, dict) or set(fixed_ridges) != set(planned_names):
            raise ValueError('Pass B requires an exact full-module ridge map')
        if any(not math.isfinite(value) or value <= 0 for value in fixed_ridges.values()):
            raise ValueError('Pass B ridge map contains an invalid value')
        if (not isinstance(fixed_ridge_source, dict)
                or set(fixed_ridge_source) != {'candidate', 'pass_id', 'manifest', 'manifest_sha256'}
                or fixed_ridge_source['candidate'] != recipe['candidate']
                or fixed_ridge_source['pass_id'] != 'A'):
            raise ValueError('Pass B ridge source is not same-candidate pass A')
        source_manifest = Path(fixed_ridge_source['manifest']).resolve()
        if not source_manifest.is_file() or sha(source_manifest) != fixed_ridge_source['manifest_sha256']:
            raise ValueError('Pass A ridge-source manifest identity differs')
        source = json.loads(source_manifest.read_text())
        if (source.get('complete') is not True or source.get('candidate') != recipe['candidate']
                or source.get('pass_id') != 'A' or source.get('ridge_map') != fixed_ridges):
            raise ValueError('Pass A manifest does not bind this exact ridge map')
        ridge_source = 'fixed_from_same_candidate_pass_A'
    run.mkdir(parents=True)
    write(run / 'started.json', {'recipe': recipe, 'template': str(Path(template).resolve()),
          'prior_check': prior_check,
          'block_order': list(plan['blocks']), 'not_an_evaluation': True, 'no_retry': True,
          'fixed_ridge_source': fixed_ridge_source})
    reports = []
    try:
        for index, (block_name, rows) in enumerate(plan['blocks'].items()):
            report = calibrate_block(policy, bank, block_name, requests,
                module_names=[row['module'] for row in rows],
                expected_calls={row['module']: row['native_calls'] for row in rows},
                expected_rows_per_expert={row['module']: row['rows_per_expert'] for row in rows},
                cap=recipe['cap'], seed=recipe['row_seed'], mass_rule=recipe['mass_rule'],
                ridge_multiplier=recipe['ridge_multiplier'], max_correction_ratio=recipe['max_correction_ratio'],
                row_mode=recipe['row_mode'],
                fixed_ridges=(None if fixed_ridges is None else
                              {row['module']: fixed_ridges[row['module']] for row in rows}),
                ridge_source=ridge_source,
                device=device)
            write(run / ('block-%03d.json' % index), report)
            reports.append(report)
        actual_rows = sum(report['collected_rows'] for report in reports)
        if actual_rows != plan['planned_rows_per_pass_if_all_linears_solved']:
            raise ValueError('Actual full-pass rows differ from the frozen budget')
        ridge_map = {name: report['modules'][name]['ridge']
                     for report in reports for name in report['modules']}
        if set(ridge_map) != set(planned_names):
            raise ValueError('Solved ridge map does not cover the complete native scope')
        if fixed_ridges is not None and ridge_map != fixed_ridges:
            raise ValueError('Pass B did not preserve the exact pass-A ridge map')
        export = export_native({'backbone': policy.model.state_dict(), 'action_head': policy.head.state_dict(),
                                'proprio_projector': policy.proprio.state_dict()}, template, run / 'export',
                               provenance={'recipe': recipe, 'complete_blocks': len(reports), 'actual_rows': actual_rows})
        result = {'complete': True, 'candidate': recipe['candidate'],
                  'row_mode': recipe['row_mode'], 'pass_id': recipe['pass_id'], 'pool': recipe['pool'],
                  'blocks': len(reports), 'modules': sum(len(report['modules']) for report in reports),
                  'rows': actual_rows, 'recipe': recipe, 'checkpoint': export['checkpoint'],
                  'files_sha256': export['files_sha256'], 'native_reload_verified': False,
                  'ridge_map': ridge_map, 'ridge_source': ridge_source,
                  'fixed_ridge_source': fixed_ridge_source,
                  'numerical_contract': {
                      'ridge': '0.05 * max(weighted_trace / augmented_input_dim, 1e-12)',
                      'trust_radius': '3 * max(mean_expert_delta_frobenius_norm, 1e-12)',
                      'solve_dtype': 'float64', 'writeback_dtype': 'native'},
                  'success_evaluated': False, 'is_final_two_pass_method': False}
        write(run / 'manifest.json', result)
        return result
    except BaseException as error:
        write(run / 'failed.json', {'exception': type(error).__name__, 'message': str(error),
              'completed_blocks': len(reports), 'no_retry': True, 'checkpoint_accepted': False})
        raise
