"""Bounded shared-GPU Fast-WAM diagnosis; no training or success evaluation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNTIME = ROOT / 'vla-merge-runtime'
FAST = ROOT / 'vla-merge_table4/Fast-WAM'
NATIVE = ROOT / 'vla-merge/experiments/claude-fastwam-tcr-20260923'
BASE = RUNTIME / 'experiments/claude-fastwam-tcr-20260923'
RUN = RUNTIME / 'experiments/fastwam-local-diagnosis-20260924/attempt-01'
PYTHON = RUNTIME / 'envs/mergevla/bin/python'
HOST = 'dsw-967394-56ffd4897d-42wft'
SUITES = ('spatial', 'object', 'goal', 'long')
MIN_FREE = 48 * 1024
FLOOR = 16 * 1024
CAP = 28 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(p)


def read(p):
    return json.loads(Path(p).read_text())


def frozen(p, expected=None):
    p = Path(p).resolve()
    s = p.stat()
    return {'path': str(p), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns,
            'sha256': expected or sha(p)}


def check(row, full=True):
    p = Path(row['path'])
    s = p.stat()
    if (s.st_size, s.st_mtime_ns) != (row['size'], row['mtime_ns']):
        raise ValueError('Source changed: ' + str(p))
    if full and sha(p) != row['sha256']:
        raise ValueError('SHA mismatch: ' + str(p))


def gpu_rows():
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    rows = []
    for line in lines:
        g, uuid, free = [s.strip() for s in line.split(',')]
        if int(g) in (4, 5, 6, 7):
            rows.append({'gpu': int(g), 'uuid': uuid, 'free_mib': int(free)})
    return rows


def eligible(rows):
    # Keep a nearly empty card available to the pre-existing RoboTwin builder.
    return sorted([r for r in rows if MIN_FREE <= r['free_mib'] < 71000],
                  key=lambda r: r['free_mib'], reverse=True)


def prepare():
    if socket.gethostname() != HOST or RUN.exists():
        raise RuntimeError('Wrong host or attempt already exists')
    sm_path = RUNTIME / 'experiments/openvla-fastwam-diagnostics-20260923/fastwam-soup/manifest.json'
    tm_path = BASE / 'tcr-build-v5/final-manifest.json'
    sm, tm = read(sm_path), read(tm_path)
    if not (sm['complete'] and tm['complete'] and tm['native_reload_action_exact']):
        raise ValueError('Unaccepted model')
    models = {f'expert-{s}': frozen(sm['sources'][s]['path'], sm['sources'][s]['sha256'])
              for s in SUITES}
    models['soup'] = frozen(sm['checkpoint']['path'], sm['checkpoint']['sha256'])
    for stage in ('A', 'B'):
        ck = tm['pass_' + stage]['checkpoint']
        models[stage] = frozen(ck['path'], ck['sha256'])
    requests = {}
    receipts = [sm_path, tm_path, BASE / 'calibration-bank-v1.json']
    for s in SUITES:
        folder = BASE / 'calibration-v1' / f'A-{s}-task00'
        ep = read(folder / 'episode.json')
        audit_path = BASE / 'capture-queue-v1/audits' / f'A-{s}-task00.json'
        audit = read(audit_path)
        hashes = {r['index']: r['sha256'] for r in audit['request_files']}
        if not ep['complete'] or ep['identity']['checkpoint'] != models[f'expert-{s}']['path']:
            raise ValueError('Capture expert identity differs')
        requests[s] = []
        for i in ep['selected_request_indices']:
            item = frozen(folder / f'request-{i:06d}.pt', hashes[i])
            check(item)
            requests[s].append(item)
        receipts.extend([folder / 'episode.json', audit_path])
    sources = {Path(__file__), *NATIVE.glob('*.py')}
    for folder in [FAST / 'source/src/fastwam', FAST / 'source/configs']:
        sources.update(p for p in folder.rglob('*') if p.suffix in ('.py', '.yaml') and p.is_file())
    plan = {'schema': 'fastwam_local_diagnosis_v1', 'host': HOST, 'created_at': now(),
            'gpu_uuids': {str(r['gpu']): r['uuid'] for r in gpu_rows()},
            'min_free_mib': MIN_FREE, 'runtime_floor_mib': FLOOR, 'allocator_cap_mib': CAP,
            'max_workers': 1, 'models': models, 'requests': requests,
            'sources': [frozen(p) for p in sorted(sources)],
            'receipts': [frozen(p) for p in receipts],
            'arms': ['expert', 'soup', 'A', 'B'], 'formal_evaluation': False,
            'training': False, 'calibration_requests_not_heldout': True,
            'no_retry': True, 'authorization': 'User: Fast-WAM local GPUs 4-7, spare memory'}
    RUN.mkdir(parents=True)
    write(RUN / 'plan.json', plan)
    print(json.dumps({'prepared': True, 'models': len(models),
                      'requests': sum(map(len, requests.values()))}), flush=True)


def metrics(torch, actual, target):
    a, b = actual.detach().float().cpu(), target.detach().float().cpu()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Invalid action/prediction shape or nonfinite value')
    d = a - b
    return {'mse': float(d.square().mean()), 'max_abs_error': float(d.abs().max()),
            'rms': float(a.square().mean().sqrt()), 'max_abs': float(a.abs().max()),
            'cosine': float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
            'continuous6_mse': float(d[..., :6].square().mean()),
            'gripper_mse': float(d[..., 6].square().mean()), 'exact': bool(torch.equal(a, b))}


def worker(gpu):
    plan = read(RUN / 'plan.json')
    if os.environ.get('FASTWAM_DIAG_LEASE') != sha(RUN / 'plan.json'):
        raise RuntimeError('Missing inherited shared lane')
    sys.path[:0] = [str(NATIVE), str(FAST / '.python-packages'),
                   str(FAST / 'source/src'), str(FAST / 'source')]
    import torch
    from hydra import compose, initialize_config_dir
    from build_two_pass_v2 import load_native_model
    from replay_merged_prefix import _device, replay_selected_expert_states
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(CAP * 1024**2 / torch.cuda.get_device_properties(0).total_memory)
    with initialize_config_dir(version_base='1.3', config_dir=str(FAST / 'source/configs')):
        cfg = compose(config_name='sim_libero.yaml', overrides=['task=libero_uncond_2cam224_1e-4'])
    records = {}
    for s, specs in plan['requests'].items():
        records[s] = []
        for spec in specs:
            check(spec)
            records[s].append((spec, torch.load(spec['path'], map_location='cpu', weights_only=True)))
    model = None
    results = []
    norms = {}
    for arm in plan['arms']:
        names = list(SUITES) if arm == 'expert' else [None]
        for name in names:
            key = 'expert-' + name if name else arm
            spec = plan['models'][key]
            write(RUN / 'progress.json', {'phase': 'verify_and_load', 'model': key, 'updated_at': now()})
            check(spec)
            if model is None:
                model = load_native_model(torch, cfg, Path(spec['path']))
            else:
                payload = torch.load(spec['path'], map_location='cpu', weights_only=True, mmap=True)
                model.mot.load_state_dict(payload['mot'], strict=True)
                model.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
                del payload
            model.eval()
            with torch.inference_mode():
                norms[key] = {n: float(p.float().norm()) for n, p in model.named_parameters()
                              if n.startswith(('mot.', 'proprio_encoder.'))}
                for suite in ([name] if name else SUITES):
                    for request_spec, row in records[suite]:
                        write(RUN / 'progress.json', {'phase': 'inference', 'model': key,
                            'suite': suite, 'request': row['request_index'], 'updated_at': now()})
                        actual = model.infer_action(**_device(row['native_request'], 'cuda:0'))['action']
                        action_stats = metrics(torch, actual, row['trace']['native_action'])
                        predictions, prefix = replay_selected_expert_states(model,
                            row['native_request'], row['trace'], device='cuda:0')
                        flow_stats = [metrics(torch, p, old['prediction'])
                                      for p, old in zip(predictions, row['trace']['calls'])]
                        if arm == 'expert' and not (action_stats['exact'] and all(x['exact'] for x in flow_stats)):
                            raise ValueError('Expert native/recorded parity failed: ' + suite)
                        results.append({'arm': arm, 'suite': suite, 'request': request_spec,
                            'request_index': row['request_index'], 'action': action_stats,
                            'flow_call_indices': [0, 5, 9], 'flow': flow_stats})
                        del predictions, prefix, actual
                        write(RUN / 'partial-results.json', {'complete': False, 'rows': results})
            torch.cuda.empty_cache()
    expected = sum(map(len, plan['requests'].values())) * 4
    if len(results) != expected:
        raise ValueError('Missing arm/request coverage')
    for spec in plan['models'].values():
        check(spec, full=False)
    write(RUN / 'parameter-norms.json', norms)
    write(RUN / 'result.json', {'complete': True, 'formal_evaluation': False,
        'calibration_requests_not_heldout': True, 'plan_sha256': sha(RUN / 'plan.json'),
        'rows': results, 'parameter_norms_sha256': sha(RUN / 'parameter-norms.json'),
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2, 'completed_at': now()})


def run():
    plan = read(RUN / 'plan.json')
    if socket.gethostname() != plan['host']:
        raise RuntimeError('Wrong host')
    with (RUN / 'shared-lane.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / 'started.json').exists():
            raise RuntimeError('No restart/retry of this attempt')
        for spec in plan['sources'] + plan['receipts']:
            check(spec)
        for spec in plan['models'].values():
            check(spec, full=False)
        write(RUN / 'started.json', {'pid': os.getpid(), 'host': HOST, 'time': now(),
            'plan_sha256': sha(RUN / 'plan.json')})
        child = None
        def stop(sig, frame):
            raise KeyboardInterrupt('Supervisor received signal ' + str(sig))
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while True:
                rows = gpu_rows()
                choices = eligible(rows)
                write(RUN / 'state.json', {'status': 'waiting_safe_shared_gpu', 'gpu_rows': rows, 'updated_at': now()})
                if choices:
                    admission = choices[0]
                    if plan['gpu_uuids'][str(admission['gpu'])] != admission['uuid']:
                        raise ValueError('GPU UUID changed')
                    break
                time.sleep(30)
            env = dict(os.environ)
            for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(k, None)
            env.update(CUDA_VISIBLE_DEVICES=str(admission['gpu']), FASTWAM_DIAG_LEASE=sha(RUN / 'plan.json'),
                DIFFSYNTH_MODEL_BASE_PATH=str(ROOT / '.datasets/FastWAM/models'), DIFFSYNTH_SKIP_DOWNLOAD='true',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            cmd = [str(PYTHON), '-u', str(Path(__file__).resolve()), 'worker', '--gpu', str(admission['gpu'])]
            with (RUN / 'worker.log').open('x') as log:
                child = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=(lease.fileno(),))
                write(RUN / 'launch.json', {'pid': child.pid, 'command': cmd, 'admission': admission,
                    'host': HOST, 'time': now(), 'shared_card_authorized': True})
                while child.poll() is None:
                    current = next(r for r in gpu_rows() if r['gpu'] == admission['gpu'])
                    write(RUN / 'state.json', {'status': 'running', 'child_pid': child.pid,
                        'gpu': current, 'updated_at': now()})
                    if current['uuid'] != admission['uuid'] or current['free_mib'] < FLOOR:
                        raise RuntimeError('GPU identity/runtime spare-memory floor violated')
                    time.sleep(10)
                if child.returncode:
                    raise RuntimeError('Worker exited ' + str(child.returncode))
            result = read(RUN / 'result.json')
            if result['complete'] is not True or result['plan_sha256'] != sha(RUN / 'plan.json'):
                raise ValueError('Incomplete diagnosis')
            write(RUN / 'ended.json', {'complete': True, 'time': now(), 'result_sha256': sha(RUN / 'result.json')})
            write(RUN / 'state.json', {'status': 'complete', 'updated_at': now()})
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            write(RUN / 'ended.json', {'complete': False, 'error': repr(exc), 'time': now()})
            write(RUN / 'state.json', {'status': 'failed_no_retry', 'error': repr(exc), 'updated_at': now()})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--gpu', type=int, choices=[4, 5, 6, 7])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'worker': lambda: worker(args.gpu)}[args.mode]()
