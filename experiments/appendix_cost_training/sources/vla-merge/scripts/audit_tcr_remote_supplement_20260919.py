"""Audit remote raw outcomes and export independently; disclose absent exit receipts."""
import argparse
import json
from pathlib import Path

import run_tcr_night_20260919 as night
from prepare_tcr_remote_supplement_20260919 import REMOTE, RUN, WRAPPER, WRAPPER_SHA
from audit_tcr_expanded_pairing import receipt_lines, validate_receipt
from summarize_tcr_night_20260919 import development_item
from pi05_table3_contract import validate_rows


def check_outcomes(info, batch, expected):
    raw = {}
    for row in info['per_task']:
        key = f"{row['task_group']}/{row['task_id']}"
        values = row['metrics']['successes']
        if key in raw or len(values) != 1 or type(values[0]) is not bool:
            raise ValueError('Nonboolean, duplicate or multiple raw episodes')
        raw[key] = values[0]
    recorded = {f"{row['suite']}/{row['task_id']}": row['successes'][0] for row in batch}
    if (len(raw) != 40 or raw != recorded or raw != expected['outcomes'] or
            any(type(v) is not bool for v in expected['outcomes'].values()) or
            sum(raw.values()) != expected['successes']):
        raise ValueError('Native eval/receipt/verified outcomes differ')


def audit():
    request = night.read(night.WORK / 'coordination/2026-09-19/pass2-supplement-request.json')
    checked = {}
    for path, expected in request['sources'].items():
        checked[path] = night.sha(path)
        if checked[path] != expected['sha256']:
            raise ValueError(f'Original source changed: {path}')
    if night.sha(WRAPPER) != WRAPPER_SHA:
        raise ValueError('Local evaluator identity differs')
    config = request['scientific_solve_config']
    output = Path(request['models']['reuse']['path'])
    manifest = night.read(output / 'block_regmeanpp_manifest.json')
    verified_solve = night.read(REMOTE / 'jobs/solve-reuse/verified.json')
    ridge = night.read(config['numeric_ridge_source'])
    digest = night.sha(output / 'model.safetensors')
    if (digest != manifest['model_sha256'] or digest != verified_solve['model_sha256'] or
            manifest['ablation'] != config or manifest['modified_tensor_count'] != 422 or
            set(manifest['modules']) != set(ridge['modules']) or
            manifest['inputs']['prior_model'] != config['start_point'] or
            manifest['dense_expert_bank_sha256'] != night.sha(config['bank_path']) or
            validate_rows(manifest['modules'], night.NAMES, None) != 1331600):
        raise ValueError('Remote export scope/identity differs')
    for name, module in manifest['modules'].items():
        if module['ridge'] != ridge['modules'][name]['ridge'] or \
                module['expert_objective_weights'] != {n: .25 for n in night.NAMES}:
            raise ValueError('Frozen ridge/mass mismatch')
    model_hashes = {'reuse': digest}
    for arm in ('alternate', 'soup'):
        model = request['models'][arm]
        model_hashes[arm] = night.sha(Path(model['path']) / 'model.safetensors')
        if model_hashes[arm] != model['model_sha256']:
            raise ValueError('Reference model changed')
    original_signature = night.read(RUN / 'plan.json')['native_signature']
    if any(night.native_signature(Path(item['path'])) != original_signature
           for item in request['models'].values()):
        raise ValueError('Native config/normalizers differ')
    rows, jobs, missing_exit = [], {}, []
    for job in request['jobs']:
        folder = REMOTE / 'jobs' / job['id']
        launch, resource = night.read(folder / 'launch.json'), night.read(folder / 'resources.json')
        if (launch['job_id'] != job['id'] or launch['hostname'] != request['allowed_host'] or
                launch['gpu'] not in request['allowed_gpu_ids'] or
                not resource['completed'] or resource['mode'] != job['kind'] or
                resource['runtime_config'] != str(night.DATA / 'config-standard') or
                resource['changes_policy_rng'] or resource['duty_sleep'] != 0):
            raise ValueError('Remote launch/resource binding differs')
        if launch['command'][6:] != job['scientific_args']:
            raise ValueError('Scientific CLI differs from registered handoff')
        if (folder / 'exit.json').exists():
            if night.read(folder / 'exit.json')['return_code'] != 0:
                raise ValueError('Nonzero recorded exit')
        else:
            missing_exit.append(job['id'])
        result = dict(worker_seconds=resource['wall_seconds'], gpu=launch['gpu'],
                      launch_sha256=night.sha(folder / 'launch.json'))
        if job['kind'] == 'development':
            out = Path(job['output'])
            expected = night.read(folder / 'verified.json')
            if expected['job'] != job['id'] or expected['offset'] != job['offset']:
                raise ValueError('Incorrect verified job')
            if (night.sha(out / 'eval_info.json') != expected['eval_info_sha256'] or
                    night.sha(out / 'paired-noise.jsonl') != expected['receipt_sha256']):
                raise ValueError('Raw hash changed')
            batch, _ = receipt_lines(out / 'paired-noise.jsonl', False)
            keys = [validate_receipt(row, job['offset']) for row in batch]
            if len(keys) != 40 or len(set(keys)) != 40:
                raise ValueError('Missing/duplicate episode')
            check_outcomes(night.read(out / 'eval_info.json'), batch, expected)
            if launch['environment_overrides'] != job['environment_overrides']:
                raise ValueError('Offset/noise environment changed')
            result.update(episodes=40, successes=expected['successes'],
                          raw_sha256=expected['eval_info_sha256'], receipts_sha256=expected['receipt_sha256'])
            if job['model'] == 'reuse':
                rows.extend(batch)
        jobs[job['id']] = result
    return dict(raw_and_export_checks_passed=True, models=model_hashes, jobs=jobs,
                reuse=development_item(rows), new_episodes=480,
                worker_seconds=sum(item['worker_seconds'] for item in jobs.values()),
                missing_individual_exit_receipts=missing_exit,
                lifecycle_independently_certified=not missing_exit,
                boundary='Actual export weights rehashed; original sources, native config and raw 480 episodes checked. '
                'Remote resources completed=true and owner scheduler success are evidence, but absent per-job '
                'exit receipts prevent independent exit-code/lifecycle certification. Host effect not identified.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit()
    night.write(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'jobs'}, indent=2))
