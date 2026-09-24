"""Six frozen ablation checkpoints, the existing held-out native-action protocol."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
OLD = HERE.parent / 'main-action-fidelity-20260921'
spec = importlib.util.spec_from_file_location('ablation_candidate_core', OLD / 'run_candidate_actions.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
ROOT = WORK / 'vla-merge-runtime/experiments/claude-full-recipe-ablation-20260922'
FORMAL = ROOT / 'full-recipe-formal-attempt-01/plan.json'
SOLVES = ROOT / 'full-recipe-solves-attempt-02/plan.json'
RUN = WORK / 'vla-merge-runtime/experiments/main-ablation-action-metrics-20260923/local-attempt-01'


def read(path):
    return json.loads(Path(path).read_text())


def models():
    result = {}
    for job in read(FORMAL)['jobs']:
        if job['arm'] not in ('last_call', 'expert_prefix'):
            continue
        row = dict(path=job['model'], model_sha256=job['model_sha256'],
                   identity_source=str(FORMAL), group=job['arm'])
        name = job['model_name']
        if name in result and result[name] != row:
            raise ValueError('Conflicting suite checkpoint identity')
        result[name] = row
    expected = {f'{arm}-r{r:02d}' for arm in ('last_call', 'expert_prefix') for r in (1, 2, 3)}
    if set(result) != expected:
        raise ValueError('Expected exactly six ablation models')
    return result


def teacher_gate():
    audit = read(core.TEACHER_AUDIT)
    if not audit['accepted'] or audit['raw_bank_sha256'] != core.sha(core.BANK):
        raise ValueError('Teacher bank identity differs')
    bank = {r['request_id']: r for r in read(core.BANK)['requests']}
    seen = set()
    for name, digest in audit['manifest_sha256'].items():
        path = Path(name)
        if core.sha(path) != digest:
            raise ValueError('Teacher manifest changed')
        data = read(path)
        if core.sha(Path(data['actions_file'])) != data['actions_sha256']:
            raise ValueError('Teacher tensor changed')
        for row in data['rows']:
            key = row['request_id']
            if key in seen or key not in bank:
                raise ValueError('Teacher coverage differs')
            for field in ('raw_input_sha256', 'native_noise_sha256'):
                if row[field] != bank[key][field]:
                    raise ValueError('Teacher input or noise differs')
            seen.add(key)
    if seen != set(bank) or len(seen) != 400:
        raise ValueError('Incomplete teacher bank')


def reset_gate():
    import numpy as np
    from libero.libero import benchmark
    from vla_merge.libero_procedural_bank import sha256_state
    bank = read(core.BANK)
    old = read(bank['reset_content_audit'])
    if core.sha(Path(bank['reset_content_audit'])) != bank['reset_content_audit_sha256']:
        raise ValueError('Prior reset audit changed')
    old_tasks = {(r['suite'], r['task_id']): r for r in old['tasks']}
    manifests = set()
    for job in read(SOLVES)['jobs']:
        for arg in job['command']:
            if arg.startswith('--manifest='):
                manifests.add(arg.split('=', 2)[2])
    offsets, proofs, initial_obs = {}, {}, set()
    for name in sorted(manifests):
        data = read(name)
        suite = {'spatial': 'libero_spatial', 'object': 'libero_object',
                 'goal': 'libero_goal', 'long': 'libero_10'}.get(data['task'], data['task'])
        offsets.setdefault(suite, set()).add(data['init_state_offset'])
        proofs[name] = core.sha(Path(name))
        for row in data['samples']:
            if row['init_state_id'] != data['init_state_offset']:
                raise ValueError('Calibration offset metadata inconsistent')
            initial_obs.add(row['initial_observation_sha256'])
    if initial_obs & {r['initial_observation_sha256'] for r in bank['requests']}:
        raise ValueError('Held-out/calibration observations overlap')
    rows = []
    for suite_name, selected in sorted(offsets.items()):
        suite = benchmark.get_benchmark_dict()[suite_name]()
        for task in range(10):
            states = suite.get_task_init_states(task)
            heldout = {str(i): sha256_state(np.asarray(states[i])) for i in (40, 41)}
            if heldout != old_tasks[(suite_name, task)]['heldout_stock_raw_state_sha256']:
                raise ValueError('Held-out reset content has changed')
            calibration = {str(i): sha256_state(np.asarray(states[i])) for i in sorted(selected)}
            if set(heldout.values()) & set(calibration.values()):
                raise ValueError('Held-out/calibration reset content overlaps')
            rows.append(dict(suite=suite_name, task_id=task, heldout=heldout, calibration=calibration))
    if len(rows) != 40:
        raise ValueError('Incomplete reset content audit')
    return dict(accepted=True, manifest_hashes=proofs, tasks=rows, calibration_offsets={k: sorted(v) for k,v in offsets.items()})


def prepare():
    if RUN.exists():
        raise FileExistsError(RUN)
    teacher_gate()
    resets = reset_gate()
    core.model_specs = models
    core.CANDIDATE_GPUS = (2, 3)
    core.MAX_ACTIVE = 1
    core.prepare(RUN)
    core.write(RUN / 'reset-independence.json', resets)
    plan = read(RUN / 'plan.json')
    plan.update(wrapper=str(Path(__file__).resolve()), wrapper_sha256=core.sha(Path(__file__)),
                formal_plan_sha256=core.sha(FORMAL), solve_plan_sha256=core.sha(SOLVES),
                reset_audit_sha256=core.sha(RUN / 'reset-independence.json'))
    core.write(RUN / 'plan.json', plan)


def frozen():
    plan = read(RUN / 'plan.json')
    if plan['hostname'] != socket.gethostname():
        raise ValueError('Wrong host')
    for key, path in [('wrapper', Path(__file__)), ('source', OLD / 'run_candidate_actions.py'),
                      ('bank', core.BANK), ('teacher_audit', core.TEACHER_AUDIT)]:
        if core.sha(path) != plan[key + '_sha256']:
            raise ValueError(f'Changed frozen {key}')
    return plan


def dispatch():
    plan = frozen()
    if (RUN / 'started.json').exists():
        raise FileExistsError('No automatic retry/restart')
    core.write(RUN / 'started.json', dict(pid=os.getpid(), host=socket.gethostname(), started_unix=time.time()))
    child, lease, log = None, None, None
    done = []
    admission = 40 * 1024
    def stop(signum, frame):
        raise RuntimeError(f'Operator stop signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        jobs = [(next(iter(plan['models'])), True)] + [(name, False) for name in plan['models']]
        for name, smoke in jobs:
            deadline = time.monotonic() + 3600
            snapshot = None
            while lease is None:
                for row in core.gpu_rows():
                    if row['index'] not in (2, 3) or row['free_mib'] < admission:
                        continue
                    lease = core.card_flock.take_card(row['index'], row['uuid'], name, 'ablation-offline-actions')
                    if lease:
                        snapshot = row
                        if core.free_mib(row['index']) < admission:
                            lease.release()
                            lease = None
                            continue
                        break
                if lease is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError('No safe GPU 2/3 admission in one hour')
                    time.sleep(60)
            tag = ('smoke-' if smoke else '') + name
            command = [str(core.PYTHON), '-u', str(Path(__file__).resolve()), '--worker', name]
            if smoke:
                command.append('--smoke')
            log = (RUN / f'{tag}.log').open('x')
            env = core.environment(snapshot['index'])
            core.write(RUN / f'{tag}-preflight.json', dict(command=command, snapshot=snapshot,
                       plan_sha256=core.sha(RUN / 'plan.json'), admission_mib=admission))
            signals = {signal.SIGTERM, signal.SIGINT}
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
            try:
                child = subprocess.Popen(command, cwd=core.VLA, env=env, stdout=log, stderr=subprocess.STDOUT,
                          start_new_session=True, pass_fds=(lease.handle,),
                          preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, mask))
                core.write(RUN / f'{tag}-launch.json', dict(pid=child.pid, gpu=snapshot['index'],
                           started_unix=time.time(), command=command))
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            print(json.dumps(dict(started=tag, pid=child.pid, gpu=snapshot['index'])), flush=True)
            code = child.wait()
            child = None
            log.close()
            log = None
            core.write(RUN / f'{tag}-exit.json', dict(return_code=code, ended_unix=time.time()))
            if code:
                raise RuntimeError(f'{tag} failed ({code}); remaining jobs stopped, no retry')
            data = read(RUN / ('smoke' if smoke else 'actions') / name / 'manifest.json')
            if data['requests'] != (1 if smoke else 400) or not data['determinism_first_request']:
                raise ValueError('Worker output incomplete')
            if smoke:
                admission = max(admission, math.ceil((data['peak_reserved_gib'] + 12) * 1024))
                core.write(RUN / 'SMOKE-PASSED.json', dict(passed=True, admission_mib=admission, result=data))
            else:
                done.append(name)
            lease.release()
            lease = None
            core.write(RUN / 'state.json', dict(completed=done, last_finished=tag, time=time.time()))
            print(json.dumps(dict(finished=tag, requests=data['requests'], seconds=data['elapsed_seconds'])), flush=True)
        core.write(RUN / 'queue-ended.json', dict(complete=True, status='complete', models=done))
    except BaseException as exc:
        # Retain ownership and lock until our own child is reaped.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        core.write(RUN / 'queue-ended.json', dict(complete=False, status='stopped', models=done, error=repr(exc)))
        raise
    finally:
        if log:
            log.close()
        if lease:
            lease.release()


def score():
    frozen()
    teacher_gate()
    audit = core.load_module('ablation_score_core', OLD / 'audit_and_score.py')
    audit.RUN, audit.OUTPUT = RUN, RUN / 'strict-audit-and-metrics.json'
    audit.main()
    result = read(audit.OUTPUT)
    groups = {}
    for arm in ('last_call', 'expert_prefix'):
        values = [v['aggregate'] for v in result['results'].values() if v['group'] == arm]
        if len(values) != 3 or any(v['cosine_valid_requests'] != 400 for v in values):
            raise ValueError('Incomplete metric repeat/coverage')
        groups[arm] = {k: dict(mean=statistics.mean(v[k] for v in values),
                              std=statistics.stdev(v[k] for v in values)) for k in ('mse', 'cosine')}
    core.write(RUN / 'group-metrics.json', dict(accepted=True, groups=groups, audit_sha256=core.sha(audit.OUTPUT)))
    print(json.dumps(groups), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--dispatch', action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--score', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker:
        plan = frozen()
        item = plan['models'][args.worker]
        for filename, digest in item['policy_files'].items():
            if core.sha(Path(item['path']) / filename) != digest:
                raise ValueError('Processor identity changed')
        core.worker(RUN, args.worker, args.smoke)
    elif args.dispatch:
        dispatch()
        score()
    elif args.score:
        score()
    else:
        parser.error('Choose a stage')
