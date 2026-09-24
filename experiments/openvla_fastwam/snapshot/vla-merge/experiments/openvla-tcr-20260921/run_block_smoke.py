"""One guarded local 2/3 worker for native block-interface validation."""
import argparse
import json
from pathlib import Path
import socket

import run_local as base
from run_dynamic import UUIDS, environment

formal = base.formal
HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
ROOT = WORK / 'vla-merge-runtime/experiments/openvla-tcr-20260921'
PRIOR = ROOT / 'soup-prior-attempt-02/checkpoint'
LEDGER = ROOT / 'local-expert-dynamic-01/identities.json'
REQUEST = ROOT / 'local-expert-dynamic-01/smoke/jobs/smoke-spatial-r01/eval/request.pt'
BLOCK_PLAN = WORK / 'coordination/2026-09-21/openvla-native-block-capacity.json'


def verify(job):
    result = json.loads((Path(job['output']) / 'validation/summary.json').read_text())
    if not result['complete'] or result['is_tcr'] or result['success_evaluation'] or result['episodes'] != 0:
        raise ValueError('Invalid block-validation result')
    plan = json.loads(BLOCK_PLAN.read_text())
    if result['blocks'] != len(plan['blocks']) or result['modules'] != sum(map(len, plan['blocks'].values())):
        raise ValueError('Incomplete block coverage')
    if (result['request_sha256'] != formal.sha(REQUEST) or result['ledger_sha256'] != formal.sha(LEDGER)
            or result['block_plan_sha256'] != formal.sha(BLOCK_PLAN)
            or result['checkpoint'] != str(PRIOR.resolve()) or not result['native_actions_restored_exactly']):
        raise ValueError('Wrong input/model or non-identical restoration')
    base.assert_unchanged(json.loads(Path(job['identity']).read_text())['files'])
    return result


def prepare(run):
    if run.exists():
        raise FileExistsError(run)
    source = ROOT / 'soup-native-smoke-attempt-01'
    if json.loads((source / 'queue-ended.json').read_text())['status'] != 'complete':
        raise ValueError('Soup native validation not complete')
    identity = json.loads((source / 'identities.json').read_text())
    base.assert_unchanged(identity['files'])
    for path in [REQUEST, BLOCK_PLAN, LEDGER, HERE / 'block_smoke.py', Path(__file__).resolve(),
                 HERE / 'expert_bank.py', HERE / 'linear_calibration.py', HERE / 'ridge.py',
                 HERE / 'materialize_soup.py', HERE / 'native_scope_guard.py']:
        identity['files'][str(path)] = base.file_identity(path)
    run.mkdir(parents=True)
    formal.save(run / 'identities.json', identity)
    old = json.loads((source / 'plan.json').read_text())
    output = run / 'jobs/native-block-interface'
    command = [str(base.PYTHON), str(HERE / 'block_smoke.py'), '--checkpoint', str(PRIOR),
               '--ledger', str(LEDGER), '--request', str(REQUEST), '--block-plan', str(BLOCK_PLAN),
               '--output', str(output / 'validation')]
    job = {'id': output.name, 'output': str(output), 'identity': str(run / 'identities.json'), 'command': command}
    plan = {**old, 'schema': 'oft_block_interface_validation_v1', 'run': str(run),
            'jobs': [job], 'max_workers': 1, 'gpus': [2, 3], 'episodes': 0,
            'identity_audit': str(run / 'identities.json'),
            'identity_audit_sha256': formal.sha(run / 'identities.json'),
            'evaluator_sha256': formal.sha(HERE / 'block_smoke.py'),
            'min_free_mib': 32768, 'runtime_floor_mib': 12288}
    formal.save(run / 'plan.json', plan)
    formal.save(run / 'CLAIM.json', {'owner': 'Codex', 'host': socket.gethostname(), 'gpus': [2, 3],
                'scope': 'native expert block swapping/capture/restoration, not calibration or evaluation',
                'episodes': 0, 'no_retry': True})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    run = args.run.resolve()
    if socket.gethostname() != base.HOST or any(formal.gpu_row(g)['uuid'] != u for g, u in UUIDS.items()):
        raise ValueError('Wrong host/card')
    if args.prepare:
        prepare(run)
        return
    base.assert_unchanged(json.loads((run / 'identities.json').read_text())['files'])
    formal.GPUS = (2, 3)
    formal.FLOOR_MIB = 12288
    formal.EVALUATOR = HERE / 'block_smoke.py'
    formal.IDENTITIES = run / 'identities.json'
    formal.scientific_environment = environment
    formal.verify_job = verify
    formal.save = base.save_with_actual_counts
    formal.run_queue(json.loads((run / 'plan.json').read_text()))


if __name__ == '__main__':
    main()
