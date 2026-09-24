"""One precommitted unused frame per suite; four native in-memory arms."""
import argparse
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


ARMS = ('expert', 'corrected_B', 'soup_fixed_spatial_interfaces', 'candidate')


def main(args):
    if os.environ.get('FASTWAM_FIXED_INTERFACE_AUTH') != 'local-4-7-offline-v1':
        raise RuntimeError('Missing frozen local launch')
    import torch
    from replay_merged_prefix import _device, replay_selected_expert_states

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(28672 * 1024**2 /
                                               torch.cuda.get_device_properties(0).total_memory)
    plan = common.read(args.plan)
    if (plan['schema'] != 'fastwam_same_trajectory_unused_frame_v1'
            or plan['request_count'] != 4 or len(plan['requests']) != 4):
        raise ValueError('Four precommitted extra requests required')
    prior_plan = common.read(common.RUN.parent / 'interface-offline-attempt-03/plan.json')
    built = common.read(args.build_manifest)
    if not built['complete'] or built['method'] != 'fixed_interface_tcr_adaptation' or \
            fixed.sha(Path(built['checkpoint']['path'])) != built['checkpoint']['sha256']:
        raise ValueError('Fixed-interface candidate identity differs')
    cfg = old_eval.native_config()
    if old_eval.execution_prefix(cfg) != 10:
        raise ValueError('Native execution prefix differs')
    records = {}
    for row in plan['requests']:
        spec = row['request']
        common.check(spec)
        record = torch.load(spec['path'], map_location='cpu', weights_only=True)
        if record['request_index'] != row['request_index'] or row['request_index'] in row['calibration_selected_indices']:
            raise ValueError('Extra request is a selected calibration frame')
        records[row['suite']] = (row, record)
    if set(records) != set(common.SUITES):
        raise ValueError('Missing suite coverage')
    spatial_spec = prior_plan['models']['expert-spatial']
    common.check(spatial_spec)
    spatial = torch.load(spatial_spec['path'], map_location='cpu', weights_only=True, mmap=True)
    interfaces = {component: {key: spatial[component][key].clone()
                               for key in keys}
                  for component, keys in old_eval.INTERFACES.items()}
    model = None
    rows = []
    for arm in ARMS:
        for suite in common.SUITES:
            if arm == 'expert':
                spec = prior_plan['models']['expert-' + suite]
            elif arm == 'corrected_B':
                spec = prior_plan['models']['corrected_B']
            elif arm == 'soup_fixed_spatial_interfaces':
                spec = prior_plan['models']['soup']
            else:
                spec = built['checkpoint']
            if arm != 'candidate':
                common.check(spec)
            if model is None:
                model = fixed.load_native_model(torch, cfg, Path(spec['path']))
            else:
                payload = torch.load(spec['path'], map_location='cpu', weights_only=True, mmap=True)
                model.mot.load_state_dict(payload['mot'], strict=True)
                model.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
                del payload
            if arm == 'soup_fixed_spatial_interfaces':
                old_eval.replace_interfaces(model, interfaces)
            if arm == 'candidate':
                fixed.verify_fixed_interfaces(torch, model, spatial)
            model.eval()
            source, record = records[suite]
            with torch.inference_mode():
                action = model.infer_action(**_device(record['native_request'], 'cuda:0'))['action']
                predictions, _ = replay_selected_expert_states(
                    model, record['native_request'], record['trace'], device='cuda:0')
            if tuple(action.shape[-2:]) != (32, 7) or len(predictions) != 3:
                raise ValueError('Native action/flow contract differs')
            full = common.metrics(torch, action, record['trace']['native_action'])
            executed = old_eval.executed_metrics(torch, action, record['trace']['native_action'], 10)
            flow = [common.metrics(torch, pred, call['prediction'])
                    for pred, call in zip(predictions, record['trace']['calls'])]
            if arm == 'expert' and not (full['exact'] and all(row['exact'] for row in flow)):
                raise ValueError('Extra request expert native replay differs')
            rows.append({'arm': arm, 'suite': suite, 'request': source['request'],
                         'request_index': source['request_index'],
                         'action': full, 'executed_action': executed, 'flow': flow,
                         'native_action': action.detach().float().cpu().tolist()})
            common.write(args.output.with_name('partial-results.json'),
                         {'complete': False, 'rows': rows})
            torch.cuda.empty_cache()
    if len(rows) != 16:
        raise ValueError('Missing 4 arms x 4 precommitted requests')
    summary = {}
    for arm in ARMS:
        selected = [row for row in rows if row['arm'] == arm]
        summary[arm] = {'requests': len(selected),
            'full32_continuous6_mse': sum(row['action']['continuous6_mse'] for row in selected) / 4,
            'executed10_continuous6_mse': sum(row['executed_action']['continuous6_mse'] for row in selected) / 4,
            'flow_mse': sum(sum(v['mse'] for v in row['flow']) / 3 for row in selected) / 4}
    common.write(args.output, {'complete': True, 'plan_sha256': common.sha(args.plan),
                 'candidate_checkpoint': built['checkpoint'], 'rows': rows, 'summary': summary,
                 'same_trajectory_unused_frames': True, 'independent_trajectory_heldout': False,
                 'training': False, 'environment_episodes': 0,
                 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                 'time': common.now()})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--build-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
