"""One bounded local GPU child: native smoke, then fixed-interface A/B build."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'vla-merge/experiments/fastwam-local-diagnosis-20260924'),
                str(ROOT / 'vla-merge/experiments/claude-firstpass-cause-20260920')]
import run_diagnosis as common
import card_flock

RUN = ROOT / 'vla-merge-runtime/experiments/fastwam-interface-recalibration-codex-20260924/attempt-01'
OLD = ROOT / 'vla-merge-runtime/experiments/claude-fastwam-tcr-20260923'
AUTH = 'local-4-7-offline-v1'
MIN_FREE = 60 * 1024
FLOOR = 16 * 1024


def prepare():
    if RUN.exists() or socket.gethostname() != common.HOST:
        raise RuntimeError('Existing attempt or wrong host')
    prior = common.read(common.RUN.parent / 'interface-offline-attempt-03/result.json')
    if not prior['complete'] or len(prior['rows']) != 60 or prior['executed_action_prefix'] != 10:
        raise ValueError('Prior native 12-request comparison incomplete')
    if not all(row['action']['exact'] and all(x['exact'] for x in row['flow'])
               for row in prior['rows'] if row['arm'] == 'expert'):
        raise ValueError('Native expert parity incomplete')
    source_paths = sorted(HERE.glob('*.py')) + [
        ROOT / 'vla-merge/experiments/fastwam-local-diagnosis-20260924/pi05_solver_bridge.py',
        ROOT / 'vla-merge/scripts/materialize_pi05_tcr_e_peft_safe.py',
        OLD / 'calibration-acceptance-v1.json', OLD / 'block-plan-v1.json',
        OLD / 'real-prefix-parity-v5.json',
        ROOT / 'vla-merge-runtime/experiments/openvla-fastwam-diagnostics-20260923/fastwam-soup/manifest.json',
        common.RUN.parent / 'interface-offline-attempt-03/result.json']
    block = common.read(OLD / 'block-plan-v1.json')
    fixed = {'proprio_encoder', 'mot.mixtures.action.action_encoder', 'mot.mixtures.action.head'}
    if (sum(len([d for d in block['blocks'][g] if d['module'] not in fixed])
            for g in block['block_order']) != 610 or
            sum(any(d['module'] not in fixed for d in block['blocks'][g])
                for g in block['block_order']) != 62):
        raise ValueError('Fixed backbone scope differs')
    plan = {'schema': 'fastwam_fixed_spatial_interface_offline_build_v1',
            'created_at': common.now(), 'host': common.HOST,
            'gpu_uuids': {str(r['gpu']): r['uuid'] for r in common.gpu_rows()},
            'sources': [common.frozen(p) for p in source_paths],
            'prior_result_sha256': common.sha(common.RUN.parent / 'interface-offline-attempt-03/result.json'),
            'candidate': 'Soup backbone + six Spatial interface tensors throughout native swaps and A/B',
            'fixed_tensors': 6, 'blocks_per_pass': 62, 'linears_per_pass': 610,
            'one_request_smoke': True, 'replan_steps': 10, 'native_action_shape': [32, 7],
            'min_free_mib': MIN_FREE, 'floor_free_mib': FLOOR, 'allocator_cap_mib': 40960,
            'max_gpu_children': 1, 'gpu_scope': [4, 5, 6, 7],
            'formal_priority_admission_order': [4, 5],
            'calibration_requests_not_heldout': True,
            'training': False, 'environment_episodes': 0, 'no_retry': True,
            'not_original_complete_tcr': True}
    RUN.mkdir(parents=True)
    common.write(RUN / 'plan.json', plan)
    print({'prepared': True, 'output': str(RUN)}, flush=True)


def run():
    plan = common.read(RUN / 'plan.json')
    if socket.gethostname() != plan['host'] or plan['schema'] != 'fastwam_fixed_spatial_interface_offline_build_v1':
        raise RuntimeError('Wrong host or plan')
    with (RUN / 'supervisor.lock').open('a+') as lane:
        fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / 'started.json').exists():
            raise RuntimeError('No restart')
        for row in plan['sources']:
            common.check(row)
        common.write(RUN / 'started.json', {'pid': os.getpid(), 'time': common.now(),
                    'plan_sha256': common.sha(RUN / 'plan.json')})
        child = None
        card = legacy = None
        def stop(sig, frame):
            raise KeyboardInterrupt(f'Stop signal {sig}')
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while card is None:
                rows = {r['gpu']: r for r in common.gpu_rows()}
                for gpu in plan['formal_priority_admission_order']:
                    row = rows[gpu]
                    if row['uuid'] != plan['gpu_uuids'][str(gpu)] or row['free_mib'] < MIN_FREE:
                        continue
                    card = card_flock.take_card(gpu, row['uuid'], 'fastwam-fixed-interface-01', 'offline-build')
                    if card is None:
                        continue
                    legacy = card_flock._try_lock(common.RUNTIME / 'resource-leases' / common.HOST /
                                f'gpu-{gpu}.lock', 'fastwam-fixed-interface-01',
                                {'gpu': gpu, 'stage': 'fastwam-offline-build'})
                    if legacy is None:
                        card.release(); card = None; continue
                    again = {r['gpu']: r for r in common.gpu_rows()}[gpu]
                    if again['uuid'] != row['uuid'] or again['free_mib'] < MIN_FREE:
                        legacy.release(); card.release(); legacy = card = None; continue
                    break
                if card is None:
                    common.write(RUN / 'state.json', {'status': 'waiting_safe_gpu',
                                 'updated_at': common.now()})
                    time.sleep(30)
            env = dict(os.environ)
            for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(key, None)
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), FASTWAM_FIXED_INTERFACE_AUTH=AUTH,
                       DIFFSYNTH_MODEL_BASE_PATH=str(ROOT / '.datasets/FastWAM/models'),
                       DIFFSYNTH_SKIP_DOWNLOAD='true', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            common.write(RUN / 'admission.json', {'gpu': gpu, 'uuid': row['uuid'],
                         'free_mib': again['free_mib'], 'time': common.now()})
            for label, cmd in (
                ('smoke', [str(HERE / 'smoke_fixed_interface.py'),
                           '--acceptance', str(OLD / 'calibration-acceptance-v1.json'),
                           '--block-plan', str(OLD / 'block-plan-v1.json'),
                           '--output', str(RUN / 'smoke.json')]),
                ('build', [str(HERE / 'build_fixed_interface.py'),
                           '--acceptance', str(OLD / 'calibration-acceptance-v1.json'),
                           '--block-plan', str(OLD / 'block-plan-v1.json'),
                           '--prefix-parity', str(OLD / 'real-prefix-parity-v5.json'),
                           '--output', str(RUN / 'build')])):
                log_path = RUN / f'{label}.log'
                with log_path.open('x') as log:
                    command = [str(common.PYTHON), '-u', *cmd]
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            pass_fds=(card.handle, legacy.handle, lane.fileno()))
                    common.write(RUN / f'{label}-launch.json', {'pid': child.pid, 'gpu': gpu,
                                 'command': command, 'time': common.now()})
                    while child.poll() is None:
                        current = {r['gpu']: r for r in common.gpu_rows()}[gpu]
                        common.write(RUN / 'state.json', {'status': 'running', 'stage': label,
                                     'child_pid': child.pid, 'gpu': current,
                                     'updated_at': common.now()})
                        if current['uuid'] != row['uuid'] or current['free_mib'] < FLOOR:
                            raise RuntimeError('GPU UUID/spare-memory contract violated')
                        time.sleep(10)
                    common.write(RUN / f'{label}-exit.json', {'returncode': child.returncode,
                                 'time': common.now()})
                    if child.returncode:
                        raise RuntimeError(f'{label} exited {child.returncode}')
                if label == 'smoke' and not common.read(RUN / 'smoke.json')['accepted']:
                    raise ValueError('Native smoke not accepted')
            final = common.read(RUN / 'build/final-manifest.json')
            if not final['complete'] or final['method'] != 'fixed_interface_tcr_adaptation' or \
                    not final['native_reload_action_exact']:
                raise ValueError('Built variant incomplete')
            common.write(RUN / 'ended.json', {'complete': True, 'checkpoint': final['checkpoint'],
                         'time': common.now()})
            common.write(RUN / 'state.json', {'status': 'complete', 'updated_at': common.now()})
        except BaseException as error:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
            common.write(RUN / 'ended.json', {'complete': False, 'error': repr(error),
                         'time': common.now()})
            common.write(RUN / 'state.json', {'status': 'failed_no_retry',
                         'error': repr(error), 'updated_at': common.now()})
            raise
        finally:
            if legacy is not None:
                legacy.release()
            if card is not None:
                card.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    {'prepare': prepare, 'run': run}[parser.parse_args().mode]()
