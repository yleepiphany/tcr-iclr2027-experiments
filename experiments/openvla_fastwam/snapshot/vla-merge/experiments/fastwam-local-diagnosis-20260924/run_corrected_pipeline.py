"""One corrected build -> SAME frozen development-40, local shared GPU 4-7.

No formal evaluation, training, retries, or edits to existing queues/models.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import run_diagnosis as common

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = common.RUN.parent / 'corrected-attempt-01'
# Legacy layout used solely to reuse the unchanged native evaluator.
BUILD = RUN / 'tcr-build-v5'
OLD = common.BASE


def prepare():
    if RUN.exists() or socket.gethostname() != common.HOST:
        raise RuntimeError('Existing attempt or wrong host')
    diagnosis = common.read(common.RUN / 'result.json')
    if not diagnosis['complete'] or len(diagnosis['rows']) != 48:
        raise ValueError('Need complete prior diagnosis')
    if not all(r['action']['exact'] and all(f['exact'] for f in r['flow'])
               for r in diagnosis['rows'] if r['arm'] == 'expert'):
        raise ValueError('Expert native parity failed')
    bank = common.read(OLD / 'evaluation-bank-v1.json')
    jobs = [r for r in bank['jobs'] if r['stage'] == 'development']
    if len(jobs) != 40 or any(r['stock_indices'] != [0] for r in jobs):
        raise ValueError('Fixed development 40 changed')
    old = common.read(common.RUN / 'plan.json')
    sources = [r for r in old['sources'] if r['path'] != str(HERE / 'run_diagnosis.py')]
    paths = list(HERE.glob('*.py')) + [
        ROOT / 'vla-merge/scripts/materialize_pi05_tcr_e_peft_safe.py',
        ROOT / 'vla-merge/experiments/openvla-tcr-20260921/linear_calibration.py',
        ROOT / 'vla-merge/experiments/openvla-tcr-20260921/ridge.py',
        OLD / 'calibration-acceptance-v1.json', OLD / 'block-plan-v1.json',
        OLD / 'real-prefix-parity-v5.json', OLD / 'evaluation-bank-v1.json',
        common.RUN / 'result.json']
    sources.extend(common.frozen(p) for p in paths)
    plan = {'schema': 'fastwam_pi05_aligned_fixed40_v1', 'created_at': common.now(),
        'host': common.HOST, 'gpu_uuids': old['gpu_uuids'], 'source_files': sources,
        'model_inputs': old['models'], 'jobs': jobs, 'development_episodes': 40,
        'formal_episodes': 0, 'no_retry': True, 'training': False,
        'build_min_free_mib': 61440, 'eval_min_free_mib': 49152,
        'runtime_floor_mib': 16384, 'build_allocator_cap_mib': 40960,
        'algorithm_reference': 'pi0.5 main-result source f2b2b1c1373343f00b4238bd349f06185ee08a59153503bf31dc54924eb1f740',
        'changes': ['Use exact FP32 main-result solve_weight_multi',
                    'Pass B reuses THIS pass-A module ridge map',
                    'Safeguard uses mean expert-prior delta norm'],
        'unchanged': ['80 expert A/B captures', '64-block native dependency order',
                      '613 affine regressions', 'expert computations within current block',
                      'relative A and uniform B masses', 'row cap 6 per selected request',
                      'fixed development reset/seed/normalizer', 'native action inference'],
        'development_gate': 'Report all 40; zero success blocks formal. Nonzero alone is not proof of transfer.'}
    RUN.mkdir(parents=True)
    common.write(RUN / 'plan.json', plan)
    print('Prepared corrected build + fixed development 40; zero formal episodes', flush=True)


def build_audit():
    final = common.read(BUILD / 'final-manifest.json')
    contract = common.read(BUILD / 'numerical-contract.json')
    if not final['complete'] or not final['native_reload_action_exact']:
        raise ValueError('Native build gate failed')
    maps = []
    for stage in ('A', 'B'):
        blocks = sorted((BUILD / f'pass-{stage}').glob('block-*.json'))
        if len(blocks) != 64:
            raise ValueError('Incomplete block coverage')
        reports = {}
        for p in blocks:
            row = common.read(p)
            if not row['complete'] or not row['all_features_before_solve'] or not row['nonlinear_tensor_fallback_preserved']:
                raise ValueError('Replay/fallback gate failed')
            for name, r in row['modules'].items():
                if name in reports or r['solver_dtype'] != 'float32':
                    raise ValueError('Duplicate module or wrong numerical type')
                if r['solver_source_sha256'] != 'f2b2b1c1373343f00b4238bd349f06185ee08a59153503bf31dc54924eb1f740':
                    raise ValueError('Wrong main-result source')
                if r['trust_limit'] != 3 * max(r['mean_expert_delta_norm'], 1e-12):
                    raise ValueError('Wrong delta safeguard')
                reports[name] = r['ridge']
        if len(reports) != 613:
            raise ValueError('Incomplete module coverage')
        maps.append(reports)
    if maps[0] != maps[1] or maps[0] != contract['ridge_map']:
        raise ValueError('Pass B did not reuse pass A ridges')
    ckpt = final['checkpoint']
    if common.sha(ckpt['path']) != ckpt['sha256']:
        raise ValueError('Export model SHA differs')
    common.write(RUN / 'build-audit.json', {'accepted': True, 'modules': 613,
        'passes': 2, 'fixed_ridges_match': True, 'checkpoint': ckpt, 'time': common.now()})
    common.write(RUN / 'pipeline-v5/ended.json', {'complete': True, 'checkpoint': ckpt,
        'adapter_layout_only': True, 'algorithm': 'pi05_main_solver_aligned'})
    return ckpt


def eval_audit(job):
    folder = RUN / 'development40' / job['id']
    complete = common.read(folder / 'complete.json')
    started = common.read(folder / 'started.json')
    import json
    rows = [json.loads(s) for s in (folder / 'episodes.jsonl').read_text().splitlines()]
    if (not complete['complete'] or complete['episodes'] != 1 or len(rows) != 1
            or common.sha(folder / 'episodes.jsonl') != complete['episodes_sha256']):
        raise ValueError('Incomplete raw development receipt')
    row = rows[0]
    expected = {'job_id': job['id'], 'stage': 'development', 'suite': job['suite'],
                'task_id': job['task_id'], 'stock_state_index': 0,
                'state_sha256': job['reset_sha256'][0], 'seed': job['seed']}
    if any(row.get(k) != v for k, v in expected.items()) or type(row['success']) is not bool:
        raise ValueError('Frozen reset/seed/native result differs')
    ckpt = common.read(BUILD / 'final-manifest.json')['checkpoint']
    if started['checkpoint_sha256'] != ckpt['sha256'] or complete['checkpoint_sha256'] != ckpt['sha256']:
        raise ValueError('Evaluated wrong model')
    return row


def run():
    plan = common.read(RUN / 'plan.json')
    if socket.gethostname() != plan['host']:
        raise RuntimeError('Wrong host')
    with (RUN / 'shared-lane.lock').open('a+') as lane:
        fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / 'started.json').exists():
            raise RuntimeError('No restarts')
        for src in plan['source_files']:
            common.check(src)
        for model in plan['model_inputs'].values():
            common.check(model, full=False)
        common.write(RUN / 'started.json', {'pid': os.getpid(), 'host': common.HOST,
            'time': common.now(), 'plan_sha256': common.sha(RUN / 'plan.json')})
        child = None
        accepted = []
        def stop(sig, frame):
            raise KeyboardInterrupt('Stop signal ' + str(sig))
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

        def execute(label, command, threshold):
            nonlocal child
            while True:
                choices = [r for r in common.eligible(common.gpu_rows()) if r['free_mib'] >= threshold]
                common.write(RUN / 'state.json', {'status': 'waiting_safe_shared_gpu',
                    'stage': label, 'accepted_development': len(accepted), 'updated_at': common.now()})
                if choices:
                    admission = choices[0]
                    if admission['uuid'] != plan['gpu_uuids'][str(admission['gpu'])]:
                        raise ValueError('GPU identity changed')
                    break
                time.sleep(30)
            env = dict(os.environ)
            for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(key, None)
            env.update(CUDA_VISIBLE_DEVICES=str(admission['gpu']), FASTWAM_CORRECTED_AUTH='local-4-7-fixed40',
                FASTWAM_INHERITED_GPU_INDEX=str(admission['gpu']), FASTWAM_INHERITED_GPU_UUID=admission['uuid'],
                DIFFSYNTH_MODEL_BASE_PATH=str(ROOT / '.datasets/FastWAM/models'), DIFFSYNTH_SKIP_DOWNLOAD='true',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            command = [str(common.PYTHON), '-u', *command]
            if label != 'build':
                command += ['--gpu', str(admission['gpu'])]
            (RUN / 'logs').mkdir(exist_ok=True)
            with (RUN / 'logs' / (label + '.log')).open('x') as log:
                child = subprocess.Popen(command, env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lane.fileno(),))
                common.write(RUN / 'launches' / (label + '.json'), {'pid': child.pid,
                    'command': command, 'admission': admission, 'time': common.now()})
                while child.poll() is None:
                    current = next(r for r in common.gpu_rows() if r['gpu'] == admission['gpu'])
                    common.write(RUN / 'state.json', {'status': 'running', 'stage': label,
                        'child_pid': child.pid, 'gpu': current, 'accepted_development': len(accepted),
                        'updated_at': common.now()})
                    if current['uuid'] != admission['uuid'] or current['free_mib'] < 16384:
                        raise RuntimeError('Board spare-memory floor violated')
                    time.sleep(10)
                common.write(RUN / 'exits' / (label + '.json'), {'returncode': child.returncode, 'time': common.now()})
                if child.returncode:
                    raise RuntimeError(label + ' exited ' + str(child.returncode))

        try:
            execute('build', [str(HERE / 'build_corrected.py'),
                '--acceptance', str(OLD / 'calibration-acceptance-v1.json'),
                '--block-plan', str(OLD / 'block-plan-v1.json'),
                '--prefix-parity', str(OLD / 'real-prefix-parity-v5.json'),
                '--output', str(BUILD)], 61440)
            ckpt = build_audit()
            for job in plan['jobs']:
                execute(job['id'], [str(HERE / 'evaluate_corrected.py'),
                    '--bank', str(OLD / 'evaluation-bank-v1.json'), '--job-id', job['id'],
                    '--checkpoint', ckpt['path'], '--output', str(RUN / 'development40' / job['id'])], 49152)
                accepted.append(eval_audit(job))
                common.write(RUN / 'accepted-development.json', accepted)
            successes = sum(r['success'] for r in accepted)
            common.write(RUN / 'result.json', {'complete': True, 'episodes': 40, 'successes': successes,
                'development_only': True, 'formal_episodes': 0, 'gate_pass': successes > 0,
                'checkpoint': ckpt, 'rows': accepted, 'time': common.now(),
                'interpretation': 'Passing a >0/40 gate is not proof of useful skill retention.'})
            common.write(RUN / 'ended.json', {'complete': True, 'time': common.now()})
            common.write(RUN / 'state.json', {'status': 'complete', 'accepted_development': 40, 'updated_at': common.now()})
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
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'run'])
    args = p.parse_args()
    {'prepare': prepare, 'run': run}[args.mode]()
