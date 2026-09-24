"""Run the independent CPU build audit once the immutable build ends."""
import os
from pathlib import Path
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / 'vla-merge/experiments/fastwam-local-diagnosis-20260924')]
import audit_fixed_build
import run_diagnosis as common

BUILD = ROOT / 'vla-merge-runtime/experiments/fastwam-interface-recalibration-codex-20260924/attempt-01'
RUN = BUILD.parent / 'independent-audit-waiter-v1'


def run():
    if RUN.exists() or socket.gethostname() != common.HOST:
        raise RuntimeError('Existing audit attempt or wrong host')
    plan = {'schema': 'fastwam_fixed_interface_independent_build_audit_waiter_v1',
            'build_plan_sha256': common.sha(BUILD / 'plan.json'),
            'source': [common.frozen(HERE / 'audit_fixed_build.py'),
                       common.frozen(Path(__file__))],
            'gpu_children': 0, 'no_retry': True, 'created_at': common.now()}
    RUN.mkdir(parents=True)
    common.write(RUN / 'plan.json', plan)
    common.write(RUN / 'started.json', {'pid': os.getpid(), 'time': common.now()})
    try:
        while not (BUILD / 'ended.json').exists():
            common.write(RUN / 'state.json', {'status': 'waiting_build_terminal',
                         'updated_at': common.now()})
            time.sleep(30)
        ended = common.read(BUILD / 'ended.json')
        if not ended['complete'] or common.sha(BUILD / 'plan.json') != plan['build_plan_sha256']:
            raise ValueError('Build failed or plan changed')
        for source in plan['source']:
            common.check(source)
        result = audit_fixed_build.audit(BUILD, RUN / 'audit.json')
        if not result['accepted']:
            raise ValueError('Independent build audit not accepted')
        common.write(RUN / 'ended.json', {'complete': True, 'audit_sha256': common.sha(RUN / 'audit.json'),
                     'time': common.now()})
        common.write(RUN / 'state.json', {'status': 'complete', 'updated_at': common.now()})
    except BaseException as error:
        common.write(RUN / 'ended.json', {'complete': False, 'error': repr(error), 'time': common.now()})
        common.write(RUN / 'state.json', {'status': 'failed_no_retry',
                     'error': repr(error), 'updated_at': common.now()})
        raise


if __name__ == '__main__':
    run()
