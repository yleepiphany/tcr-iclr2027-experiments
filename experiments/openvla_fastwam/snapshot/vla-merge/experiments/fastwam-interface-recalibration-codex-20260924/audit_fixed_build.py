"""Independent, read-only audit of the fixed Spatial interface build."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'vla-merge/experiments/fastwam-local-diagnosis-20260924'))
import run_diagnosis as common

FIXED = ('mot.mixtures.action.action_encoder', 'mot.mixtures.action.head',
         'proprio_encoder')
KEYS = (('mot', 'mixtures.action.action_encoder.weight'),
        ('mot', 'mixtures.action.action_encoder.bias'),
        ('mot', 'mixtures.action.head.weight'),
        ('mot', 'mixtures.action.head.bias'),
        ('proprio_encoder', 'weight'), ('proprio_encoder', 'bias'))


def audit(run: Path, output: Path):
    import torch
    plan = common.read(run / 'plan.json')
    ended = common.read(run / 'ended.json')
    smoke = common.read(run / 'smoke.json')
    final = common.read(run / 'build/final-manifest.json')
    contract = common.read(run / 'build/numerical-contract.json')
    source = common.read(ROOT / 'vla-merge-runtime/experiments/openvla-fastwam-diagnostics-20260923/fastwam-soup/manifest.json')
    if (not ended['complete'] or plan['schema'] != 'fastwam_fixed_spatial_interface_offline_build_v1'
            or not smoke['accepted'] or not final['complete'] or final['is_tcr'] is not False
            or final['method'] != 'fixed_interface_tcr_adaptation'
            or not final['native_reload_action_exact'] or final['active_linear_modules_per_pass'] != 610
            or final['blocks_per_pass'] != 62):
        raise ValueError('Frozen candidate build gate incomplete')
    names = set()
    maps = []
    for stage in ('A', 'B'):
        path = run / 'build' / f'pass-{stage}'
        manifest = common.read(path / 'manifest.json')
        blocks = sorted(path.glob('block-*.json'))
        if len(blocks) != 64 or not manifest['complete'] or manifest['active_linears'] != 610:
            raise ValueError('Block/pass coverage incomplete')
        ridges = {}
        active = 0
        for p in blocks:
            report = common.read(p)
            if not report['complete'] or not report['all_features_before_solve'] or \
                    not report['nonlinear_tensor_fallback_preserved']:
                raise ValueError('Capture/swap/solve block gate incomplete')
            if report['modules']:
                active += 1
            elif not report.get('fixed_interface_only'):
                raise ValueError('Unexplained empty fixed group')
            for name, receipt in report['modules'].items():
                if name in ridges or name in FIXED or receipt['solver_dtype'] != 'float32':
                    raise ValueError('Duplicate/fixed/wrong-dtype solved module')
                if (receipt['solver_source_sha256'] != 'f2b2b1c1373343f00b4238bd349f06185ee08a59153503bf31dc54924eb1f740'
                        or receipt['trust_limit'] != 3 * max(receipt['mean_expert_delta_norm'], 1e-12)
                        or receipt['ridge_source'] != ('computed_pass_A' if stage == 'A' else 'fixed_from_same_candidate_pass_A')):
                    raise ValueError('Numerical solver receipt differs')
                ridges[name] = receipt['ridge']
        if active != 62 or len(ridges) != 610:
            raise ValueError('Expected 62 groups and 610 modules')
        if stage == 'A':
            names = set(ridges)
        elif set(ridges) != names:
            raise ValueError('A/B module coverage differs')
        maps.append(ridges)
        ck = manifest['checkpoint']
        if common.sha(ck['path']) != ck['sha256']:
            raise ValueError('Pass checkpoint SHA differs')
    if maps[0] != maps[1] or maps[0] != contract['ridge_map'] or \
            len(contract['completed_groups']['A']) != 62 or len(contract['completed_groups']['B']) != 62:
        raise ValueError('B does not reuse this candidate A ridge map')
    start = final['start_checkpoint']
    if common.sha(start['path']) != start['sha256'] or \
            final['checkpoint']['sha256'] != ended['checkpoint']['sha256']:
        raise ValueError('Start/final identity differs')
    soup = torch.load(source['checkpoint']['path'], map_location='cpu', weights_only=True, mmap=True)
    spatial = torch.load(source['sources']['spatial']['path'], map_location='cpu', weights_only=True, mmap=True)
    initial = torch.load(start['path'], map_location='cpu', weights_only=True, mmap=True)
    fixed_keys = {comp: {key for c, key in KEYS if c == comp} for comp in ('mot', 'proprio_encoder')}
    for comp in ('mot', 'proprio_encoder'):
        if set(initial[comp]) != set(soup[comp]):
            raise ValueError('Start checkpoint scope differs from Soup')
        for key, value in initial[comp].items():
            target = spatial[comp][key] if key in fixed_keys[comp] else soup[comp][key]
            if not torch.equal(value, target):
                raise ValueError('Start checkpoint is not Soup backbone + Spatial interface')
    del soup, initial
    for stage in ('A', 'B'):
        checkpoint = torch.load(final[f'pass_{stage}']['checkpoint']['path'], map_location='cpu',
                                weights_only=True, mmap=True)
        for comp, key in KEYS:
            if not torch.equal(checkpoint[comp][key], spatial[comp][key]):
                raise ValueError(f'Fixed interface changed after pass {stage}: {comp}.{key}')
        del checkpoint
    result = {'accepted': True, 'build_plan_sha256': common.sha(run / 'plan.json'),
              'build_ended_sha256': common.sha(run / 'ended.json'),
              'final_manifest_sha256': common.sha(run / 'build/final-manifest.json'),
              'passes': 2, 'groups_per_pass': 62, 'linears_per_pass': 610,
              'fixed_tensors': 6, 'start_equals_soup_backbone_plus_spatial_interfaces': True,
              'interfaces_equal_spatial_after_A_and_B': True,
              'B_ridges_equal_this_A': True, 'native_reload_action_exact': True,
              'checkpoint': final['checkpoint'], 'time': common.now()}
    common.write(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.run, args.output), sort_keys=True))
