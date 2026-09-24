"""Wait for the one candidate build, then evaluate its frozen native requests."""
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

BUILD = ROOT / 'vla-merge-runtime/experiments/fastwam-interface-recalibration-codex-20260924/attempt-01'
RUN = BUILD.parent / 'eval-attempt-01'


def prepare():
    if RUN.exists() or socket.gethostname() != common.HOST:
        raise RuntimeError('Existing evaluation attempt or wrong host')
    build_plan = common.read(BUILD / 'plan.json')
    if build_plan['schema'] != 'fastwam_fixed_spatial_interface_offline_build_v1':
        raise ValueError('Unknown fixed-interface build')
    old = common.RUN.parent / 'interface-offline-attempt-03'
    sources = [HERE / 'evaluate_fixed.py', Path(__file__),
               ROOT / 'vla-merge/experiments/fastwam-local-diagnosis-20260924/run_interface_ablation_v3.py',
               ROOT / 'vla-merge/experiments/fastwam-local-diagnosis-20260924/run_diagnosis.py',
               HERE / 'build_two_pass_fixed.py', old / 'plan.json', old / 'result.json',
               BUILD / 'plan.json']
    plan = {'schema': 'fastwam_fixed_spatial_interface_offline_eval_v1',
            'created_at': common.now(), 'host': common.HOST,
            'build_plan_sha256': common.sha(BUILD / 'plan.json'),
            'sources': [common.frozen(p) for p in sources],
            'request_count': 12, 'baseline_arms': ['corrected_B', 'soup_fixed_spatial_interfaces'],
            'gpu_scope': [4, 5, 6, 7], 'formal_priority_admission_order': [4, 5],
            'min_free_mib': 48 * 1024, 'floor_free_mib': 16 * 1024,
            'allocator_cap_mib': 28672, 'max_gpu_children': 1,
            'training': False, 'environment_episodes': 0,
            'calibration_requests_not_heldout': True, 'no_retry': True}
    RUN.mkdir(parents=True)
    common.write(RUN / 'plan.json', plan)
    print({'prepared': True, 'output': str(RUN)}, flush=True)


def run():
    plan = common.read(RUN / 'plan.json')
    if socket.gethostname() != plan['host']:
        raise RuntimeError('Wrong host')
    with (RUN / 'supervisor.lock').open('a+') as lane:
        fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / 'started.json').exists():
            raise RuntimeError('No restart')
        for row in plan['sources']:
            common.check(row)
        common.write(RUN / 'started.json', {'pid': os.getpid(), 'time': common.now(),
                    'plan_sha256': common.sha(RUN / 'plan.json')})
        child = card = legacy = None
        def stop(sig, frame):
            raise KeyboardInterrupt(f'Stop signal {sig}')
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while not (BUILD / 'ended.json').exists():
                common.write(RUN / 'state.json', {'status': 'waiting_build', 'updated_at': common.now()})
                time.sleep(30)
            ended = common.read(BUILD / 'ended.json')
            if not ended['complete'] or common.sha(BUILD / 'plan.json') != plan['build_plan_sha256']:
                raise ValueError('Candidate build failed or changed')
            manifest = BUILD / 'build/final-manifest.json'
            built = common.read(manifest)
            if not built['complete'] or built['checkpoint']['sha256'] != ended['checkpoint']['sha256'] or \
                    common.sha(built['checkpoint']['path']) != built['checkpoint']['sha256']:
                raise ValueError('Candidate identity differs')
            while card is None:
                rows = {r['gpu']: r for r in common.gpu_rows()}
                for gpu in plan['formal_priority_admission_order']:
                    row = rows[gpu]
                    if row['free_mib'] < plan['min_free_mib']:
                        continue
                    card = card_flock.take_card(gpu, row['uuid'], 'fastwam-fixed-eval-01', 'offline-eval')
                    if card is None:
                        continue
                    legacy = card_flock._try_lock(common.RUNTIME / 'resource-leases' / common.HOST /
                                f'gpu-{gpu}.lock', 'fastwam-fixed-eval-01',
                                {'gpu': gpu, 'stage': 'fastwam-offline-eval'})
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
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),
                       FASTWAM_FIXED_INTERFACE_AUTH='local-4-7-offline-v1',
                       DIFFSYNTH_MODEL_BASE_PATH=str(ROOT / '.datasets/FastWAM/models'),
                       DIFFSYNTH_SKIP_DOWNLOAD='true', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            command = [str(common.PYTHON), '-u', str(HERE / 'evaluate_fixed.py'),
                       '--build-manifest', str(manifest), '--output', str(RUN / 'result.json')]
            with (RUN / 'worker.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(card.handle, legacy.handle, lane.fileno()))
                common.write(RUN / 'launch.json', {'pid': child.pid, 'gpu': gpu, 'uuid': row['uuid'],
                             'command': command, 'checkpoint_sha256': built['checkpoint']['sha256'],
                             'time': common.now()})
                while child.poll() is None:
                    current = {r['gpu']: r for r in common.gpu_rows()}[gpu]
                    common.write(RUN / 'state.json', {'status': 'running', 'child_pid': child.pid,
                                 'gpu': current, 'updated_at': common.now()})
                    if current['uuid'] != row['uuid'] or current['free_mib'] < plan['floor_free_mib']:
                        raise RuntimeError('GPU UUID/spare-memory contract violated')
                    time.sleep(10)
                common.write(RUN / 'exit.json', {'returncode': child.returncode, 'time': common.now()})
                if child.returncode:
                    raise RuntimeError(f'Evaluation exited {child.returncode}')
            result = common.read(RUN / 'result.json')
            if not result['complete'] or len(result['rows']) != 12 or \
                    result['candidate_checkpoint']['sha256'] != built['checkpoint']['sha256']:
                raise ValueError('Incomplete native comparison')
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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'run'])
    {'prepare': prepare, 'run': run}[p.parse_args().mode]()
