"""Native block-interface validation only; no fitted checkpoint or success score."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from expert_bank import ExpertBank, resolve_block
from linear_calibration import capture_expert_block
from materialize_soup import sha, write
from native_oft import NativePolicy
from native_scope_guard import validate_native_scope


def main():
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'ledger', 'request', 'block-plan', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(274001)
    torch.set_num_threads(2)
    request = torch.load(args.request, map_location='cpu', weights_only=False)
    plan = json.loads(args.block_plan.read_text())
    policy = NativePolicy(args.checkpoint, 'libero_spatial')
    before = policy.replay(request)
    modules = dict(policy.linear_modules())
    unused = validate_native_scope(modules, {row['module'] for rows in plan['blocks'].values() for row in rows})
    all_parameters = [(prefix + '.' + name, value) for prefix, root in (
        ('backbone', policy.model), ('action_head', policy.head), ('proprio_projector', policy.proprio))
        for name, value in root.named_parameters()]
    records = []
    started = time.time()
    with ExpertBank(args.ledger) as bank:
        # The native featurizer returns intermediate tokens without forward_head.
        # Preserve the unused pool in the checkpoint; never fabricate its rows.
        for name in unused:
            for key, current in modules[name].state_dict().items():
                experts = [bank.tensor(expert, name + '.' + key) for expert in ('spatial', 'object', 'goal', 'long')]
                if not all(torch.equal(experts[0], value) for value in experts[1:]):
                    raise ValueError('Unused pooling parameters unexpectedly differ across experts')
                if not torch.equal(current.detach().cpu(), experts[0].to(current.dtype)):
                    raise ValueError('Prior did not retain unused identical parameters')
        for block_name, rows in plan['blocks'].items():
            block = resolve_block(policy, block_name)
            prior = {key: value.detach().cpu().clone() for key, value in block.state_dict().items()}
            outside = [(name, value, value._version) for name, value in all_parameters
                       if not name.startswith(block_name + '.')]
            state, conversions = bank.block_state('spatial', block_name, block)
            names = [row['module'] for row in rows]
            calls = {row['module']: row['native_calls'] for row in rows}
            features, receipts = capture_expert_block(
                policy, block, state, names, request, request_id='interface-spatial-only',
                cap=8, seed=274001, expected_calls=calls)
            for row in rows:
                name = row['module']
                x = features[name]
                expected_dim = modules[name].in_features + int(modules[name].bias is not None)
                if x.shape != (row['rows_per_request'], expected_dim) or not torch.isfinite(x).all():
                    raise ValueError('Actual native row capacity/feature shape differs')
                for receipt in receipts[name]:
                    chosen = receipt['selected_rows']
                    if len(set(chosen)) != len(chosen) or not all(0 <= j < receipt['available_rows'] for j in chosen):
                        raise ValueError('Invalid or repeated row selection')
            if any(not torch.equal(value, block.state_dict()[key].detach().cpu()) for key, value in prior.items()):
                raise ValueError('Expert block was not restored byte-exactly')
            if any(value._version != version for _, value, version in outside):
                raise ValueError('A parameter outside the current block was mutated')
            record = {'block': block_name, 'modules': len(names), 'rows': sum(len(x) for x in features.values()),
                      'native_dtype_conversions': conversions, 'restored_exactly': True,
                      'outside_parameter_versions_unchanged': True, 'row_receipts': receipts}
            records.append(record)
            write(args.output / ('block-%03d.json' % len(records)), record)
            del features, state, prior
    after = policy.replay(request)
    if not np.array_equal(before, after):
        raise ValueError('Native actions changed after restored block replay')
    result = {'complete': True, 'is_tcr': False, 'success_evaluation': False, 'episodes': 0,
              'scope': 'all native blocks, one spatial expert/request; not full calibration',
              'blocks': len(records), 'modules': sum(x['modules'] for x in records),
              'unused_identical_linears_retained': unused,
              'checkpoint': str(args.checkpoint.resolve()), 'request_sha256': sha(args.request),
              'ledger_sha256': sha(args.ledger), 'block_plan_sha256': sha(args.block_plan),
              'native_actions_restored_exactly': True, 'elapsed_seconds': time.time() - started,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'process_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'process_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    write(args.output / 'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
