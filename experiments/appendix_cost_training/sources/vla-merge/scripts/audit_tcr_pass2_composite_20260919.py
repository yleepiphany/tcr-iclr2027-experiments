"""Explicit composite of local, remote supplement and GPU3 migration outcomes.

Does not relabel the old failed campaign as successful. Requires all400 per arm.
"""
import argparse
import json
from pathlib import Path

import run_tcr_night_20260919 as night
from run_tcr_pass2_controls_20260919 import RUN as ORIGINAL, Queue
from run_tcr_offset32_gpu3_20260919 import RUN as MIGRATION, validate_plan
from prepare_tcr_remote_supplement_20260919 import REMOTE
from audit_tcr_remote_supplement_20260919 import audit as remote_audit, check_outcomes
from audit_tcr_expanded_pairing import receipt_lines
from summarize_tcr_night_20260919 import development_rows, development_item, paired_development


def location(arm, offset):
    if arm not in ('reuse','alternate','soup') or offset not in range(30,40):
        raise ValueError('Unregistered logical episode group')
    campaign = REMOTE if arm == 'reuse' or offset == 30 else MIGRATION if offset == 32 else ORIGINAL
    return campaign / 'jobs' / f'dev-{arm}-{offset}'


def audit():
    migrated_plan, ended = night.read(MIGRATION/'plan.json'), night.read(MIGRATION/'queue-ended.json')
    validate_plan(migrated_plan)
    if ended['stopped'] or not ended['all_successful'] or ended['states'] != {
            'dev-alternate-32':'done','dev-soup-32':'done'}:
        raise ValueError('Migration incomplete; no partial composite comparison')
    start = night.read(MIGRATION/'started.json')
    if Path(f"/proc/{start['pid']}").exists():
        raise ValueError('Migration supervisor still present; wait or review PID reuse')
    remote = remote_audit()
    plan = night.read(ORIGINAL/'plan.json')
    queue = Queue(ORIGINAL,plan)
    queue.check_sources()
    for arm in ('alternate','soup'):
        job = next(j for j in plan['jobs'] if j['id']==f'solve-{arm}')
        actual = queue.verify(job,ORIGINAL/'jobs'/job['id'],Path(plan['models'][arm]['path']))
        if actual != night.read(ORIGINAL/'jobs'/job['id']/'verified.json'):
            raise ValueError('Local export changed')
    rows={name:[] for name in ('reuse','alternate','soup','unchanged')}
    sources={}
    for arm in ('reuse','alternate','soup'):
        for offset in range(30,40):
            folder=location(arm,offset)
            verified=night.read(folder/'verified.json')
            if folder.is_relative_to(REMOTE):
                batch,_=receipt_lines(folder/'eval/paired-noise.jsonl',False)
                check_outcomes(night.read(folder/'eval/eval_info.json'),batch,verified)
            else:
                launch,end,resource=(night.read(folder/name) for name in
                                      ('launch.json','exit.json','resources.json'))
                campaign_plan=migrated_plan if folder.is_relative_to(MIGRATION) else plan
                if (end['return_code']!=0 or not resource['completed'] or
                        launch['model_sha256']!=remote['models'][arm] or
                        launch['plan_sha256']!=night.sha(folder.parents[1]/'plan.json') or
                        launch['runtime_config']!=str(night.DATA/'config-standard') or
                        launch['gpu'] not in campaign_plan['gpu_ids']):
                    raise ValueError('Local evaluation identity/exit differs')
                for arg in (f"--policy.path={plan['models'][arm]['path']}",
                            f'--output_dir={folder / "eval"}', '--policy.n_action_steps=10',
                            '--eval.n_episodes=1',f'--seed={391600+1000*(offset-30)}'):
                    if launch['command'].count(arg)!=1:
                        raise ValueError('Evaluation scientific arguments changed')
                batch=development_rows(folder/'eval',offset,verified)
            rows[arm].extend(batch)
            sources[f'{arm}/{offset}']=dict(folder=str(folder),
                raw_sha256=night.sha(folder/'eval/eval_info.json'),
                noise_sha256=night.sha(folder/'eval/paired-noise.jsonl'),
                verified_sha256=night.sha(folder/'verified.json'))
    for item in plan['reused_baseline']:
        rows['unchanged'].extend(development_rows(Path(item['folder']),item['offset'],item['verified']))
    arms={arm:development_item(batch) for arm,batch in rows.items()}
    if not all(item['complete'] for item in arms.values()):
        raise ValueError('Incomplete or duplicate400-episode arm')
    contrasts=[('alternate','reuse'),('alternate','soup')]
    contrasts += [(arm,'unchanged') for arm in ('reuse','alternate','soup')]
    return dict(numeric_evidence_complete=True,arms=arms,episode_job_provenance=sources,
        comparisons={f'{a}_minus_{b}':paired_development(rows[a],rows[b]) for a,b in contrasts},
        original_campaign_terminal=night.read(ORIGINAL/'queue-ended.json'),
        original_failures_relabelled=False,
        remote_lifecycle_independently_certified=remote['lifecycle_independently_certified'],
        missing_remote_exit_receipts=remote['missing_individual_exit_receipts'],
        evaluated_episodes=1200,reused_unchanged_episodes=400,
        boundary='Explored D2 with explicit supplements; old failures retained. Alternate-reuse changes '
                 'expert cache, alternate-soup changes starting model/anchor. Comparisons against '
                 'unchanged also change calibration recipe/cost, not solely pass count. '
                 'Reuse is remote, other arms mostly local; hardware/host effects are not separately identified.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();result=audit();night.write(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('episode_job_provenance','original_campaign_terminal')},indent=2))
