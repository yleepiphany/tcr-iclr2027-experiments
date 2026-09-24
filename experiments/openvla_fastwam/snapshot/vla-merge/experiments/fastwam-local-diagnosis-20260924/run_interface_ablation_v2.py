"""Fixed-request, in-memory Fast-WAM interface intervention; no rollouts/training."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import run_diagnosis as common

HERE = Path(__file__).resolve().parent
RUN = common.RUN.parent / 'interface-offline-attempt-02'
ARMS = ('expert', 'corrected_B', 'B_fixed_spatial_interfaces',
        'soup_fixed_spatial_interfaces', 'expert_B_interfaces')
INTERFACES = {
    'mot': ('mixtures.action.action_encoder.weight', 'mixtures.action.action_encoder.bias',
            'mixtures.action.head.weight', 'mixtures.action.head.bias'),
    'proprio_encoder': ('weight', 'bias'),
}


def replace_interfaces(model, payload):
    """Only the six explicit interface tensors; no backbone/module replacement."""
    import torch
    with torch.no_grad():
        for component, keys in INTERFACES.items():
            target = getattr(model, component).state_dict()
            for key in keys:
                source = payload[component][key]
                if source.shape != target[key].shape or not torch.isfinite(source).all():
                    raise ValueError('Invalid interface: ' + component + '.' + key)
                target[key].copy_(source.to(device=target[key].device, dtype=target[key].dtype))
                if not torch.equal(target[key].cpu(), source.to(dtype=target[key].dtype).cpu()):
                    raise ValueError('Interface copy differs')


def native_config():
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base='1.3', config_dir=str(common.FAST / 'source/configs')):
        return compose(config_name='sim_libero.yaml', overrides=['task=libero_uncond_2cam224_1e-4'])


def execution_prefix(cfg):
    prefix = int(cfg.EVALUATION.replan_steps)
    horizon = int(cfg.data.train.num_frames) - 1
    if bool(cfg.EVALUATION.use_action_ensembler) or not 0 < prefix <= horizon:
        raise ValueError('Unsupported native execution protocol')
    return prefix


def executed_metrics(torch, action, target, prefix):
    if action.shape != target.shape or action.ndim != 3 or not 0 < prefix <= action.shape[-2]:
        raise ValueError('Invalid executed action window')
    return common.metrics(torch, action[..., :prefix, :], target[..., :prefix, :])


def prepare():
    if RUN.exists() or socket.gethostname() != common.HOST:
        raise RuntimeError('Existing attempt or wrong host')
    failed_run = common.RUN.parent / 'interface-offline-attempt-01'
    failure = common.read(failed_run / 'ended.json')
    if failure.get('complete') is not False or 'Executed action horizon changed' not in (failed_run / 'worker.log').read_text():
        raise ValueError('Expected original horizon-check failure')
    prefix = execution_prefix(native_config())
    old = common.read(common.RUN / 'plan.json')
    corrected = common.read(common.RUN.parent / 'corrected-attempt-01/build-audit.json')
    ended = common.read(common.RUN.parent / 'corrected-attempt-01/ended.json')
    if not corrected['accepted'] or ended.get('complete') is not False:
        raise ValueError('Need accepted corrected build and stopped rollout pipeline')
    models = {k: v for k, v in old['models'].items() if k.startswith('expert-') or k == 'soup'}
    models['corrected_B'] = common.frozen(corrected['checkpoint']['path'], corrected['checkpoint']['sha256'])
    for spec in models.values():
        common.check(spec, full=False)
    for specs in old['requests'].values():
        for spec in specs:
            common.check(spec)
    sources = [r for r in old['sources'] if r['path'] != str(HERE / 'run_diagnosis.py')]
    sources += [common.frozen(HERE / 'run_diagnosis.py'), common.frozen(__file__)]
    for spec in sources:
        common.check(spec)
    plan = {'schema': 'fastwam_offline_interface_intervention_v1', 'created_at': common.now(),
        'host': common.HOST, 'gpu_uuids': old['gpu_uuids'], 'models': models,
        'requests': old['requests'], 'sources': sources, 'arms': list(ARMS),
        'interface_tensor_keys': INTERFACES, 'fixed_interface_expert': 'spatial',
        'expected_rows': 60, 'max_workers': 1, 'min_free_mib': 49152,
        'runtime_floor_mib': 16384, 'allocator_cap_mib': 28672,
        'training': False, 'environment_episodes': 0, 'export_checkpoint': False,
        'calibration_requests_not_heldout': True, 'no_retry': True,
        'executed_action_prefix': prefix,
        'supersedes_failed_attempt': str(failed_run),
        'superseded_plan_sha256': common.sha(failed_run / 'plan.json'),
        'interpretation': 'Frozen 12 calibration requests, native noise reused. Fixed Spatial interfaces across ALL suites in both hybrids. Reverse arm uses each expert backbone plus the SAME B interfaces. No routing-policy or TCR-success claim.'}
    RUN.mkdir(parents=True)
    common.write(RUN / 'plan.json', plan)
    print({'prepared': True, 'rows': 60, 'environment_episodes': 0}, flush=True)


def worker(gpu):
    plan = common.read(RUN / 'plan.json')
    if os.environ.get('FASTWAM_INTERFACE_LEASE') != common.sha(RUN / 'plan.json'):
        raise RuntimeError('Missing inherited lease')
    sys.path[:0] = [str(common.NATIVE), str(common.FAST / '.python-packages'),
                   str(common.FAST / 'source/src'), str(common.FAST / 'source')]
    import torch
    from build_two_pass_v5 import load_native_model
    from replay_merged_prefix import _device, replay_selected_expert_states
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(28672 * 1024**2 / torch.cuda.get_device_properties(0).total_memory)
    cfg = native_config()
    if execution_prefix(cfg) != plan['executed_action_prefix']:
        raise ValueError('Executed action horizon changed')
    for name, spec in plan['models'].items():
        common.write(RUN / 'progress.json', {'phase': 'verify_checkpoint', 'model': name, 'updated_at': common.now()})
        common.check(spec)
    records = {}
    for suite, specs in plan['requests'].items():
        records[suite] = []
        for spec in specs:
            common.check(spec)
            records[suite].append((spec, torch.load(spec['path'], map_location='cpu', weights_only=True)))
    interfaces = {}
    for name in ('expert-spatial', 'corrected_B'):
        payload = torch.load(plan['models'][name]['path'], map_location='cpu', weights_only=True, mmap=True)
        interfaces[name] = {component: {k: payload[component][k].clone() for k in keys}
                            for component, keys in INTERFACES.items()}
        del payload
    model = None
    rows = []
    for arm in ARMS:
        per_expert = arm in ('expert', 'expert_B_interfaces')
        for owner in (common.SUITES if per_expert else (None,)):
            base = 'expert-' + owner if per_expert else ('soup' if arm.startswith('soup') else 'corrected_B')
            donor = 'corrected_B' if arm == 'expert_B_interfaces' else (
                'expert-spatial' if 'fixed_spatial' in arm else None)
            common.write(RUN / 'progress.json', {'phase': 'load', 'arm': arm, 'base': base,
                'donor': donor, 'completed_rows': len(rows), 'updated_at': common.now()})
            path = plan['models'][base]['path']
            if model is None:
                model = load_native_model(torch, cfg, Path(path))
            else:
                payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                model.mot.load_state_dict(payload['mot'], strict=True)
                model.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
                del payload
            if donor:
                replace_interfaces(model, interfaces[donor])
            model.eval()
            with torch.inference_mode():
                for suite in ((owner,) if per_expert else common.SUITES):
                    for spec, record in records[suite]:
                        common.write(RUN / 'progress.json', {'phase': 'infer', 'arm': arm, 'suite': suite,
                            'request': record['request_index'], 'completed_rows': len(rows), 'updated_at': common.now()})
                        target = record['trace']['native_action']
                        action = model.infer_action(**_device(record['native_request'], 'cuda:0'))['action']
                        predictions, prefix = replay_selected_expert_states(model, record['native_request'],
                            record['trace'], device='cuda:0')
                        full = common.metrics(torch, action, target)
                        executed = executed_metrics(torch, action, target, plan['executed_action_prefix'])
                        flow = [common.metrics(torch, x, call['prediction'])
                                for x, call in zip(predictions, record['trace']['calls'])]
                        if arm == 'expert' and not (full['exact'] and all(x['exact'] for x in flow)):
                            raise ValueError('Expert native replay parity failed: ' + suite)
                        rows.append({'arm': arm, 'suite': suite, 'base': base, 'donor': donor,
                            'request': spec, 'request_index': record['request_index'],
                            'action': full, 'executed_action': executed, 'flow': flow,
                            'native_action': action.detach().float().cpu().tolist()})
                        common.write(RUN / 'partial-results.json', {'complete': False, 'rows': rows})
                        del action, predictions, prefix
            torch.cuda.empty_cache()
    if len(rows) != plan['expected_rows']:
        raise ValueError('Missing comparison rows')
    for spec in plan['models'].values():
        common.check(spec, full=False)
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for suite in (*common.SUITES, 'all'):
            chosen = [r for r in rows if r['arm'] == arm and (suite == 'all' or r['suite'] == suite)]
            summary[arm][suite] = {'requests': len(chosen),
                'full32_continuous6_mse': sum(r['action']['continuous6_mse'] for r in chosen) / len(chosen),
                'executed_continuous6_mse': sum(r['executed_action']['continuous6_mse'] for r in chosen) / len(chosen),
                'flow_mse': sum(sum(m['mse'] for m in r['flow']) / 3 for r in chosen) / len(chosen)}
    common.write(RUN / 'result.json', {'complete': True, 'rows': rows, 'summary': summary,
        'plan_sha256': common.sha(RUN / 'plan.json'), 'environment_episodes': 0,
        'executed_action_prefix': plan['executed_action_prefix'],
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
        'calibration_requests_not_heldout': True, 'completed_at': common.now()})


def run():
    plan = common.read(RUN / 'plan.json')
    if socket.gethostname() != plan['host']:
        raise RuntimeError('Wrong host')
    with (RUN / 'shared-lane.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / 'started.json').exists():
            raise RuntimeError('No restart')
        for spec in plan['sources']:
            common.check(spec)
        common.write(RUN / 'started.json', {'pid': os.getpid(), 'host': common.HOST,
            'time': common.now(), 'plan_sha256': common.sha(RUN / 'plan.json')})
        child = None
        def stop(sig, frame):
            raise KeyboardInterrupt('Stop signal ' + str(sig))
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while True:
                choices = common.eligible(common.gpu_rows())
                common.write(RUN / 'state.json', {'status': 'waiting_safe_shared_gpu', 'updated_at': common.now()})
                if choices:
                    admission = choices[0]
                    if plan['gpu_uuids'][str(admission['gpu'])] != admission['uuid']:
                        raise ValueError('GPU UUID changed')
                    break
                time.sleep(30)
            env = dict(os.environ)
            for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(key, None)
            env.update(CUDA_VISIBLE_DEVICES=str(admission['gpu']), FASTWAM_INTERFACE_LEASE=common.sha(RUN / 'plan.json'),
                DIFFSYNTH_MODEL_BASE_PATH=str(common.ROOT / '.datasets/FastWAM/models'), DIFFSYNTH_SKIP_DOWNLOAD='true',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            cmd = [str(common.PYTHON), '-u', str(Path(__file__).resolve()), 'worker', '--gpu', str(admission['gpu'])]
            with (RUN / 'worker.log').open('x') as log:
                child = subprocess.Popen(cmd, env=env, cwd=common.ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lease.fileno(),))
                common.write(RUN / 'launch.json', {'pid': child.pid, 'command': cmd, 'admission': admission, 'time': common.now()})
                while child.poll() is None:
                    current = next(r for r in common.gpu_rows() if r['gpu'] == admission['gpu'])
                    common.write(RUN / 'state.json', {'status': 'running', 'child_pid': child.pid,
                        'gpu': current, 'updated_at': common.now()})
                    if current['uuid'] != admission['uuid'] or current['free_mib'] < 16384:
                        raise RuntimeError('Spare memory floor violated')
                    time.sleep(10)
                if child.returncode:
                    raise RuntimeError('Worker exited ' + str(child.returncode))
            result = common.read(RUN / 'result.json')
            if not result['complete'] or len(result['rows']) != 60 or result['plan_sha256'] != common.sha(RUN / 'plan.json'):
                raise ValueError('Incomplete offline result')
            common.write(RUN / 'ended.json', {'complete': True, 'time': common.now(), 'result_sha256': common.sha(RUN / 'result.json')})
            common.write(RUN / 'state.json', {'status': 'complete', 'updated_at': common.now()})
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            common.write(RUN / 'ended.json', {'complete': False, 'error': repr(exc), 'time': common.now()})
            common.write(RUN / 'state.json', {'status': 'failed_no_retry', 'error': repr(exc), 'updated_at': common.now()})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--gpu', type=int, choices=[4, 5, 6, 7])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'worker': lambda: worker(args.gpu)}[args.mode]()
