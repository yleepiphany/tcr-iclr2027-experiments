"""Compare a completed fixed-interface build with the same 12 native requests."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BRIDGE = ROOT / 'vla-merge/experiments/fastwam-local-diagnosis-20260924'
OLD = ROOT / 'vla-merge/experiments/claude-fastwam-tcr-20260923'
FAST = ROOT / 'vla-merge_table4/Fast-WAM'
sys.path[:0] = [str(HERE), str(BRIDGE), str(OLD), str(FAST / '.python-packages'),
                str(FAST / 'source/src'), str(FAST / 'source')]
import build_two_pass_fixed as fixed
import run_diagnosis as common
import run_interface_ablation_v3 as old_eval


def main(args):
    if os.environ.get('FASTWAM_FIXED_INTERFACE_AUTH') != 'local-4-7-offline-v1':
        raise RuntimeError('Missing frozen local launch')
    import torch
    from replay_merged_prefix import _device, replay_selected_expert_states
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(28672 * 1024**2 /
                                               torch.cuda.get_device_properties(0).total_memory)
    plan = common.read(common.RUN.parent / 'interface-offline-attempt-03/plan.json')
    previous = common.read(common.RUN.parent / 'interface-offline-attempt-03/result.json')
    build = common.read(args.build_manifest)
    if not previous['complete'] or len(previous['rows']) != 60 or \
            plan['executed_action_prefix'] != 10 or not build['complete'] or \
            build['method'] != 'fixed_interface_tcr_adaptation':
        raise ValueError('Frozen comparison or candidate build incomplete')
    if fixed.sha(Path(build['checkpoint']['path'])) != build['checkpoint']['sha256']:
        raise ValueError('Candidate checkpoint SHA differs')
    cfg = old_eval.native_config()
    if old_eval.execution_prefix(cfg) != 10:
        raise ValueError('Native execution prefix differs')
    spatial = torch.load(plan['models']['expert-spatial']['path'], map_location='cpu',
                         weights_only=True, mmap=True)
    model = fixed.load_native_model(torch, cfg, Path(build['checkpoint']['path']))
    fixed.verify_fixed_interfaces(torch, model, spatial)
    baselines = {}
    for row in previous['rows']:
        if row['arm'] in ('corrected_B', 'soup_fixed_spatial_interfaces'):
            key = (row['arm'], row['suite'], row['request']['sha256'])
            if key in baselines:
                raise ValueError('Duplicate baseline request')
            baselines[key] = row
    if len(baselines) != 24:
        raise ValueError('Missing paired baseline requests')
    rows = []
    for suite in common.SUITES:
        for spec in plan['requests'][suite]:
            common.check(spec)
            record = torch.load(spec['path'], map_location='cpu', weights_only=True)
            target = record['trace']['native_action']
            with torch.inference_mode():
                action = model.infer_action(**_device(record['native_request'], 'cuda:0'))['action']
                predictions, _ = replay_selected_expert_states(
                    model, record['native_request'], record['trace'], device='cuda:0')
            if tuple(action.shape[-2:]) != (32, 7) or len(predictions) != 3:
                raise ValueError('Native action/flow contract differs')
            flow = [common.metrics(torch, pred, call['prediction'])
                    for pred, call in zip(predictions, record['trace']['calls'])]
            full = common.metrics(torch, action, target)
            executed = old_eval.executed_metrics(torch, action, target, 10)
            base = {arm: baselines[(arm, suite, spec['sha256'])]
                    for arm in ('corrected_B', 'soup_fixed_spatial_interfaces')}
            rows.append({'suite': suite, 'request': spec, 'request_index': record['request_index'],
                         'candidate': {'action': full, 'executed_action': executed,
                                       'flow': flow, 'native_action': action.detach().float().cpu().tolist()},
                         'baselines': base})
            common.write(args.output.with_name('partial-results.json'),
                         {'complete': False, 'rows': rows})
    if len(rows) != 12:
        raise ValueError('Missing frozen native requests')
    summary = {}
    for suite in (*common.SUITES, 'all'):
        selected = [r for r in rows if suite == 'all' or r['suite'] == suite]
        summary[suite] = {}
        for arm in ('candidate', 'corrected_B', 'soup_fixed_spatial_interfaces'):
            data = [r['candidate'] if arm == 'candidate' else r['baselines'][arm]
                    for r in selected]
            summary[suite][arm] = {
                'requests': len(data),
                'full32_continuous6_mse': sum(d['action']['continuous6_mse'] for d in data) / len(data),
                'executed10_continuous6_mse': sum(d['executed_action']['continuous6_mse'] for d in data) / len(data),
                'flow_mse': sum(sum(v['mse'] for v in d['flow']) / 3 for d in data) / len(data)}
    fixed.verify_fixed_interfaces(torch, model, spatial)
    common.write(args.output, {'complete': True, 'candidate_checkpoint': build['checkpoint'],
                 'previous_result_sha256': common.sha(common.RUN.parent / 'interface-offline-attempt-03/result.json'),
                 'calibration_requests_not_heldout': True, 'environment_episodes': 0,
                 'executed_prefix': 10, 'rows': rows, 'summary': summary,
                 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                 'time': common.now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
