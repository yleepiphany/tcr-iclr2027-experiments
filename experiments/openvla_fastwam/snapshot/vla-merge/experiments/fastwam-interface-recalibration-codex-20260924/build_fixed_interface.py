"""Offline Fast-WAM candidate: freeze Spatial interfaces, then recalibrate backbone."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'vla-merge/experiments/claude-fastwam-tcr-20260923'
BRIDGE = ROOT / 'vla-merge/experiments/fastwam-local-diagnosis-20260924'
sys.path[:0] = [str(OLD), str(BRIDGE)]
import block_calibration_v4 as block
import build_two_pass_fixed as backend

import pi05_solver_bridge as corrected


class CorrectedGroups:
    def __init__(self, output):
        self.output = Path(output)
        self.ridges = {}
        self.completed = {'A': [], 'B': []}
        self.original_group = block.calibrate_group

    def __call__(self, model, group, descriptors, experts, requests, **kwargs):
        stage = {'relative': 'A', 'uniform': 'B'}[kwargs['mass_rule']]
        names = [d['module'] for d in descriptors]
        ids = {id(model.get_submodule(n)): n for n in names}
        if group in self.completed[stage]:
            raise ValueError('Repeated group')
        if stage == 'B' and (len(self.completed['A']) != 62 or any(n not in self.ridges for n in names)):
            raise ValueError('Pass B requires this complete 62-block fixed-interface pass A')
        previous = block.solve_module

        def solve(module, xs, ws, prior, **options):
            name = ids[id(module)]
            return corrected.solve_module(module, xs, ws, prior,
                fixed_ridge=None if stage == 'A' else self.ridges[name],
                ridge_source='computed_pass_A' if stage == 'A' else 'fixed_from_same_candidate_pass_A',
                **options)

        block.solve_module = solve
        try:
            report = self.original_group(model, group, descriptors, experts, requests, **kwargs)
        finally:
            block.solve_module = previous
        for name, row in report['modules'].items():
            if stage == 'A':
                if name in self.ridges:
                    raise ValueError('Duplicate pass-A module')
                self.ridges[name] = row['ridge']
            elif row['ridge'] != self.ridges[name]:
                raise ValueError('Second-pass numerical ridge changed')
            expected = 3 * max(row['mean_expert_delta_norm'], 1e-12)
            if row['trust_limit'] != expected:
                raise ValueError('Wrong paper safeguard')
        self.completed[stage].append(group)
        backend.save(self.output / 'numerical-contract.json', {
            'ridge_map': self.ridges, 'completed_groups': self.completed,
            'second_pass_ridge': 'fixed_from_this_pass_A',
            'trust_limit': '3 * max(mean_expert_delta_norm, 1e-12)',
            'ridge_floor': '0.05 * max(weighted_trace / augmented_dim, 1e-12)',
            'replay_and_sampling': 'unchanged_from_v5',
            'fixed_interface_tensors': list(backend.FIXED_TENSORS)})
        return report


def main(args):
    if os.environ.get('FASTWAM_FIXED_INTERFACE_AUTH') != 'local-4-7-offline-v1':
        raise RuntimeError('Launch through the frozen local pipeline')
    import torch
    torch.cuda.set_per_process_memory_fraction(40 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    adapter = CorrectedGroups(args.output)
    previous = block.calibrate_group
    block.calibrate_group = adapter
    try:
        result = backend.run(args)
    finally:
        block.calibrate_group = previous
    if len(adapter.ridges) != 610 or any(len(v) != 62 for v in adapter.completed.values()):
        raise ValueError('Incomplete fixed-interface backbone coverage')
    result['numerical_repair'] = 'fixed_spatial_interfaces_A_to_B_and_main_solver'
    result['numerical_contract_sha256'] = backend.sha(args.output / 'numerical-contract.json')
    backend.save(args.output / 'final-manifest.json', result)
    print(json.dumps({'complete': True, 'checkpoint': result['checkpoint']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['acceptance', 'block-plan', 'prefix-parity', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    main(p.parse_args())
