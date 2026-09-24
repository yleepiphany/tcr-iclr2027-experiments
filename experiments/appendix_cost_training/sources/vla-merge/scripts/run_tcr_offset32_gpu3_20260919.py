"""Explicitly migrate only two never-launched jobs from an idle queue to GPU3.

Old failures remain failures. This does not retry any killed/failed worker, and
never edits the old running dependencies. No other process is signalled.
"""
import argparse
import copy
import os
from pathlib import Path
import signal
import socket
import time

import run_tcr_night_20260919 as night
import run_tcr_pass2_controls_20260919 as controls

RUN = night.EXP / 'tcr-offset32-gpu3-20260919'
CAPTURE = night.EXP / 'tcr-e2-occupancy-20260919'
SCHEMA = 'tcr_offset32_gpu3_explicit_migration_v1'
ARMS = ('alternate', 'soup')


def process(pid, script):
    root = Path(f'/proc/{pid}')
    argv = (root / 'cmdline').read_bytes().split(b'\0')
    if str(script).encode() not in argv:
        raise ValueError('PID does not belong to the exact registered supervisor')
    ticks = (root / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    # All supervisor threads must have no child process, including waiting lanes.
    for task in (root / 'task').iterdir():
        if (task / 'children').read_text().strip():
            raise ValueError('Supervisor still has workers; do not cancel healthy work')
    return dict(pid=pid, start_ticks=ticks, argv=[v.decode() for v in argv if v],
                hostname=socket.gethostname())


def stop_idle(run, script):
    registered = night.read(run / 'started.json')
    if registered['host'] != socket.gethostname():
        raise ValueError('Cannot operate on remote PIDs')
    pid = registered['pid']
    before = process(pid, script)
    fd = os.pidfd_open(pid)
    try:
        if process(pid, script) != before:
            raise ValueError('Supervisor identity changed')
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    deadline = time.monotonic() + 15
    while Path(f'/proc/{pid}').exists() and time.monotonic() < deadline:
        time.sleep(.1)
    if Path(f'/proc/{pid}').exists():
        raise ValueError('Supervisor did not exit; no new launch')
    return dict(**before, intentional_idle_cancellation=True, worker_count=0,
                reason='Move two never-launched offset32 jobs to authorized free GPU3; preserve old queue and failures')


def validate_plan(plan):
    expected = [dict(id=f'dev-{a}-32', kind='development', model=a, offset=32,
                     deps=[], priority=1) for a in ARMS]
    if plan['schema'] != SCHEMA or plan['gpu_ids'] != [3] or plan['jobs'] != expected:
        raise ValueError('Not the exact two-job migration')
    if set(plan['models']) != set(ARMS) or plan['new_development_episodes'] != 80:
        raise ValueError('Budget/model set changed')


def prepare():
    if RUN.exists():
        raise FileExistsError('Migration already prepared; no overwrite or automatic restart')
    plan = copy.deepcopy(night.read(controls.RUN / 'plan.json'))
    model_info = {}
    for arm in ARMS:
        pending = controls.RUN / 'jobs' / f'dev-{arm}-32'
        if (pending / 'launch.json').exists() or (pending / 'worker.json').exists():
            raise ValueError('Offset32 has launched; migration no longer applicable')
        for offset in (31, 33, 34, 35, 36, 37, 38, 39):
            p = controls.RUN / 'jobs' / f'dev-{arm}-{offset}'
            if night.read(p / 'exit.json')['return_code'] != 0 or not (p / 'verified.json').exists():
                raise ValueError('Healthy local evaluations not finished')
        item = plan['models'][arm]
        old = night.read(controls.RUN / 'jobs' / f'solve-{arm}' / 'verified.json')
        night.identity(Path(item['path']) / 'model.safetensors', old['model_sha256'])
        model_info[arm] = {**item, 'sha256': old['model_sha256']}
    # Check both identities/empty child sets BEFORE cancelling either owner.
    for root, script in ((CAPTURE, night.SCRIPTS / 'run_tcr_e2_capture_20260919.py'),
                         (controls.RUN, Path(controls.__file__).resolve())):
        started = night.read(root / 'started.json')
        if started['host'] != socket.gethostname():
            raise ValueError('Unexpected host')
        process(started['pid'], script)
    plan.update(schema=SCHEMA, gpu_ids=[3], gpus=[3], models=model_info,
                new_checkpoints=0, new_development_episodes=80, new_formal_episodes=0,
                reused_baseline=[], jobs=[dict(id=f'dev-{a}-32', kind='development',
                model=a, offset=32, deps=[], priority=1) for a in ARMS],
                original_run=str(controls.RUN), explicit_migration=True)
    validate_plan(plan)
    controls.Queue(RUN, plan).check_sources()
    # Retire impossible predecessor-gated collection; it has never launched a worker.
    RUN.mkdir(parents=True, exist_ok=False)
    for label, root, script in (
        ('capture', CAPTURE, night.SCRIPTS / 'run_tcr_e2_capture_20260919.py'),
        ('controls', controls.RUN, Path(controls.__file__).resolve())):
        receipt = stop_idle(root, script)
        night.write(RUN / f'cancelled-idle-{label}.json', receipt)
    for path in (Path(__file__).resolve(), controls.WORKER):
        plan['frozen'][str(path)] = night.identity(path)
    for arm in ARMS:
        path = Path(model_info[arm]['path']) / 'model.safetensors'
        plan['frozen'][str(path)] = night.identity(path, model_info[arm]['sha256'])
    night.write(RUN / 'plan.json', plan)
    print('PREPARED: retired two idle supervisors, no workers stopped; 80 episodes on GPU3 only', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        return prepare()
    plan = night.read(RUN / 'plan.json')
    validate_plan(plan)
    for root in (controls.RUN, CAPTURE):
        if Path(f"/proc/{night.read(root / 'started.json')['pid']}").exists():
            raise ValueError('Old PID present; review before any launch')
    queue = controls.Queue(RUN, plan)
    queue.check_sources()
    night.write(RUN / 'started.json', dict(pid=os.getpid(), host=socket.gethostname(), time=time.time()))
    controls.GPUS = (3,)  # Only this new process: never changes old source files.
    queue.execute()


if __name__ == '__main__':
    main()
