"""Bounded one-child GPU comparison on four precommitted unused frames."""
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

BASE = ROOT / 'vla-merge-runtime/experiments/fastwam-interface-recalibration-codex-20260924'
RUN = BASE / 'extra-eval-attempt-01'
EXTRA = BASE / 'same-trajectory-extra-requests-plan-v1.json'
BUILD = BASE / 'attempt-01/build/final-manifest.json'
FIRST = BASE / 'eval-attempt-01/result.json'


def prepare():
    if RUN.exists() or socket.gethostname() != common.HOST:
        raise RuntimeError('Existing extra evaluation or wrong host')
    first = common.read(FIRST)
    audit = common.read(BASE / 'independent-audit-waiter-v1/audit.json')
    extra = common.read(EXTRA)
    if (not first['complete'] or len(first['rows']) != 12 or not audit['accepted']
            or first['candidate_checkpoint']['sha256'] != audit['checkpoint']['sha256']
            or extra['request_count'] != 4 or not extra['outcome_blind']):
        raise ValueError('First candidate or precommitted requests incomplete')
    wins = sum(row['candidate']['executed_action']['continuous6_mse'] <
               row['baselines']['corrected_B']['executed_action']['continuous6_mse']
               for row in first['rows'])
    if wins < 7:
        raise ValueError('Precondition: candidate did not improve over original TCR on most requests')
    paths = [Path(__file__), HERE / 'evaluate_extra.py', HERE / 'build_two_pass_fixed.py',
             HERE / 'prepare_extra_requests.py',
             ROOT / 'vla-merge/experiments/fastwam-local-diagnosis-20260924/run_interface_ablation_v3.py',
             ROOT / 'vla-merge/experiments/fastwam-local-diagnosis-20260924/run_diagnosis.py',
             EXTRA, BUILD, FIRST, BASE / 'independent-audit-waiter-v1/audit.json']
    plan = {'schema': 'fastwam_same_trajectory_extra_native_eval_v1',
            'created_at': common.now(), 'host': common.HOST,
            'source_files': [common.frozen(p) for p in paths],
            'gpu_uuids': {str(r['gpu']): r['uuid'] for r in common.gpu_rows()},
            'precommitted_request_sha256': common.sha(EXTRA),
            'candidate_checkpoint_sha256': audit['checkpoint']['sha256'],
            'first_12_request_result_sha256': common.sha(FIRST),
            'reason': f'Candidate wins versus original TCR B on {wins}/12 precommitted calibration requests',
            'arms': ['expert', 'corrected_B', 'soup_fixed_spatial_interfaces', 'candidate'],
            'requests': 4, 'expected_rows': 16, 'max_gpu_children': 1,
            'gpu_scope': [4, 5, 6, 7], 'formal_priority_admission_order': [4, 5],
            'min_free_mib': 48 * 1024, 'floor_free_mib': 16 * 1024,
            'allocator_cap_mib': 28672,
            'same_trajectory_unused_frames': True, 'independent_trajectory_heldout': False,
            'training': False, 'environment_episodes': 0, 'no_retry': True}
    RUN.mkdir(parents=True)
    common.write(RUN / 'plan.json', plan)
    print({'prepared': True, 'wins_vs_TCR_B': wins, 'expected_rows': 16}, flush=True)


def run():
    plan = common.read(RUN / 'plan.json')
    if socket.gethostname() != plan['host']:
        raise RuntimeError('Wrong host')
    with (RUN / 'supervisor.lock').open('a+') as lane:
        fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / 'started.json').exists():
            raise RuntimeError('No restart')
        for spec in plan['source_files']:
            common.check(spec)
        common.write(RUN / 'started.json', {'pid': os.getpid(), 'time': common.now(),
                    'plan_sha256': common.sha(RUN / 'plan.json')})
        child = card = legacy = None
        def stop(sig, frame):
            raise KeyboardInterrupt(f'Stop signal {sig}')
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while card is None:
                rows = {r['gpu']: r for r in common.gpu_rows()}
                for gpu in plan['formal_priority_admission_order']:
                    row = rows[gpu]
                    if row['uuid'] != plan['gpu_uuids'][str(gpu)] or row['free_mib'] < plan['min_free_mib']:
                        continue
                    card = card_flock.take_card(gpu, row['uuid'], 'fastwam-extra-01', 'offline-eval')
                    if card is None:
                        continue
                    legacy = card_flock._try_lock(common.RUNTIME / 'resource-leases' / common.HOST /
                                f'gpu-{gpu}.lock', 'fastwam-extra-01',
                                {'gpu': gpu, 'stage': 'fastwam-offline-extra'})
                    if legacy is None:
                        card.release(); card = None; continue
                    again = {r['gpu']: r for r in common.gpu_rows()}[gpu]
                    if again['uuid'] != row['uuid'] or again['free_mib'] < plan['min_free_mib']:
                        legacy.release(); card.release(); legacy = card = None; continue
                    break
                if card is None:
                    common.write(RUN / 'state.json', {'status': 'waiting_safe_gpu',
                                 'updated_at': common.now()})
                    time.sleep(30)
            env = dict(os.environ)
            for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(key, None)
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), FASTWAM_FIXED_INTERFACE_AUTH='local-4-7-offline-v1',
                       DIFFSYNTH_MODEL_BASE_PATH=str(ROOT / '.datasets/FastWAM/models'),
                       DIFFSYNTH_SKIP_DOWNLOAD='true', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            cmd = [str(common.PYTHON), '-u', str(HERE / 'evaluate_extra.py'),
                   '--plan', str(EXTRA), '--build-manifest', str(BUILD),
                   '--output', str(RUN / 'result.json')]
            with (RUN / 'worker.log').open('x') as log:
                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(card.handle, legacy.handle, lane.fileno()))
                common.write(RUN / 'launch.json', {'pid': child.pid, 'gpu': gpu, 'uuid': row['uuid'],
                             'command': cmd, 'time': common.now()})
                while child.poll() is None:
                    current = {r['gpu']: r for r in common.gpu_rows()}[gpu]
                    common.write(RUN / 'state.json', {'status': 'running', 'child_pid': child.pid,
                                 'gpu': current, 'updated_at': common.now()})
                    if current['uuid'] != row['uuid'] or current['free_mib'] < plan['floor_free_mib']:
                        raise RuntimeError('GPU UUID/spare-memory contract violated')
                    time.sleep(10)
                common.write(RUN / 'exit.json', {'returncode': child.returncode, 'time': common.now()})
                if child.returncode:
                    raise RuntimeError(f'Extra evaluator exited {child.returncode}')
            result = common.read(RUN / 'result.json')
            if not result['complete'] or len(result['rows']) != 16 or \
                    result['plan_sha256'] != common.sha(EXTRA):
                raise ValueError('Incomplete extra-request comparison')
            common.write(RUN / 'ended.json', {'complete': True,
                         'result_sha256': common.sha(RUN / 'result.json'), 'time': common.now()})
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
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'run'])
    {'prepare': prepare, 'run': run}[p.parse_args().mode]()
