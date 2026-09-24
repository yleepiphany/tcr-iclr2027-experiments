"""CPU-only handoff manifest, NOT a launcher or restart of the local campaign."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import socket

import run_tcr_night_20260919 as night
from run_tcr_pass2_controls_20260919 import RUN, WORKER, CONTRACT, SOLVER

REMOTE = night.EXP / 'claude-pass2-supplement-20260919'
FAILURES = ('solve-reuse', 'dev-alternate-30', 'dev-soup-30')
REMOTE_HOST = 'dsw-824375-57c745db88-n6tv9'
WRAPPER = night.ROOT / 'experiments/claude-20260917/eval_pi05_expanded_development.py'
WRAPPER_SHA = '7d6418d3addb6efb63f3b5a600416a11e56bee4db5cc50dd3f3aff241f987ac9'


def arguments(launch, kind):
    command = launch['command']
    if (command[:4] != [str(night.PYTHON), '-u', str(WORKER), '--parent'] or
            command[5:8] != ['--mode', kind, '--resources']):
        raise ValueError('Unexpected original worker command')
    return copy.deepcopy(command[9:])


def replace_one(args, key, value):
    if sum(arg.startswith(key + '=') for arg in args) != 1:
        raise ValueError(f'Missing/duplicated argument: {key}')
    return [f'{key}={value}' if arg.startswith(key + '=') else arg for arg in args]


def make_jobs(launches, models):
    jobs = []
    solve = replace_one(arguments(launches['solve-reuse'], 'solve'), '--output', models['reuse']['path'])
    jobs.append(dict(id='solve-reuse', kind='solve', model='reuse', deps=[],
                     output=models['reuse']['path'], scientific_args=solve))
    # Use the original evaluation command; change only model, offset-dependent seed
    # and output. The offset belongs in the evaluation environment, not task IDs.
    for arm, offsets in (('reuse', range(30, 40)), ('alternate', (30,)), ('soup', (30,))):
        template = launches['dev-soup-30' if arm == 'soup' else 'dev-alternate-30']
        for offset in offsets:
            key = f'dev-{arm}-{offset}'
            output = str(REMOTE / 'jobs' / key / 'eval')
            args = arguments(template, 'development')
            for arg, value in (('--policy.path', models[arm]['path']), ('--output_dir', output),
                               ('--seed', str(391600 + 1000 * (offset - 30)))):
                args = replace_one(args, arg, value)
            jobs.append(dict(id=key, kind='development', model=arm, offset=offset,
                             deps=['solve-reuse'] if arm == 'reuse' else [],
                             output=output, scientific_args=args, environment_overrides={
                                 'PI05_LIBERO_INIT_STATE_OFFSET': str(offset),
                                 'PI05_LIBERO_INIT_STATE_COUNT': '1',
                                 'ITERATION_NOISE_RECEIPT': str(Path(output) / 'paired-noise.jsonl')}))
    if len(jobs) != 13 or len({job['id'] for job in jobs}) != 13:
        raise ValueError('Unexpected supplement allocation')
    for job in jobs:
        Path(job['output']).relative_to(REMOTE)  # All output paths must be new.
    return jobs


def prepare():
    plan = night.read(RUN / 'plan.json')
    if plan['host'] != socket.gethostname():
        raise ValueError('Failure PID audit must run on the original local host')
    records, launches = {}, {}
    for key in FAILURES:
        root = RUN / 'jobs' / key
        end, worker, launch = (night.read(root / name) for name in ('exit.json', 'worker.json', 'launch.json'))
        if (end['return_code'] != 1 or worker['host'] != socket.gethostname() or
                Path(f"/proc/{worker['pid']}").exists()):
            raise ValueError('Failure is not terminal on original host; review identity')
        if (root / 'verified.json').exists():
            raise ValueError('Do not duplicate a successfully verified job')
        logs = (root / 'worker.log').read_text()
        reason = 'GPU free reserve below12GiB' if key == 'solve-reuse' else 'Shared GPU reserve low'
        if reason not in logs:
            raise ValueError('Failure cause differs from authorized memory supplement')
        records[key] = dict(exit=end, old_worker=worker, failure_reason=reason,
                            files={name: night.identity(root / name) for name in
                                   ('launch.json', 'exit.json', 'resources.json', 'failure.json', 'worker.log')})
        launches[key] = launch
    models = {'reuse': dict(path=str(REMOTE / 'models/reuse'), model_sha256=None,
                            hash_available_after_new_solve=True)}
    for arm in ('alternate', 'soup'):
        completed = night.read(RUN / 'jobs' / f'solve-{arm}' / 'verified.json')
        models[arm] = dict(path=plan['models'][arm]['path'],
                           model_sha256=completed['model_sha256'], rehash_on_remote_before_eval=True)
        if launches[f'dev-{arm}-30']['model_sha256'] != completed['model_sha256']:
            raise ValueError('Failure and exported model identities differ')
    sources = {}
    for path in (WORKER, CONTRACT, SOLVER, WRAPPER, night.SCRIPTS / 'tcr_night_worker_20260919.py',
                 night.SCRIPTS / 'run_tcr_night_20260919.py', RUN / 'configs/reuse.json',
                 RUN / 'references/reuse.json'):
        expected = plan['frozen'][str(path)]['sha256']
        sources[str(path)] = night.identity(path, expected)
    if sources[str(WRAPPER)]['sha256'] != WRAPPER_SHA:
        raise ValueError('Local evaluator is not the registered 7d6418 source')
    config = night.read(RUN / 'configs/reuse.json')
    return dict(schema='tcr_pass2_remote_supplement_request_v1', status='assigned_not_launched',
                created_utc=datetime.now(timezone.utc).isoformat(),
                authorized_by='User: 不行可以交给claude跑, 2026-09-19',
                original_run=str(RUN), original_plan_sha256=night.sha(RUN / 'plan.json'),
                output_root=str(REMOTE), allowed_host=REMOTE_HOST, allowed_gpu_ids=[1, 2, 4, 5, 6, 7],
                total_eval_workers_across_all_remote_studies=12, solve_workers_per_gpu=1,
                old_failures=records, models=models, jobs=make_jobs(launches, models),
                sources=sources, scientific_solve_config=config, scientific_solve_config_path=str(RUN / 'configs/reuse.json'),
                environment_recipe=dict(function='run_tcr_night_20260919.environment',
                                        inputs='actual remote physical GPU and job kind',
                                        wrapper_overrides='tcr_night_worker_20260919.py main',
                                        preserve_tf32_mode_rule=True, preserve_native_inputs=True),
                new_episodes=480, transferred_waiting_jobs=[], untouched_local_jobs='all other jobs',
                limitations=['This is a handoff manifest, not a launched experiment.',
                             'Large models/caches must be rehashed by remote owner before execution.',
                             'Cross-host matching does not assert bitwise closed-loop reproducibility.',
                             'No files are copied into the failed original run; its audit remains incomplete.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    evidence = prepare()
    night.write(args.output, evidence)
    print('Prepared CPU-only request:', len(evidence['jobs']), 'jobs,', evidence['new_episodes'], 'episodes; nothing launched')
