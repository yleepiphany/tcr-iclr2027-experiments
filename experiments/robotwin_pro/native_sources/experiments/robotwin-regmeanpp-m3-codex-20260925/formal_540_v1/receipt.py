"""Fail-closed native 60-episode receipt audit; no performance threshold."""
from __future__ import annotations
import math
from pathlib import Path
import contract as c

def audit_episode(row,initial,action,job,q01,q99):
    import torch
    key=(row.get('task_index'),row.get('task'),row.get('seed'))
    if key not in c.expected_keys(job['tasks']):raise ValueError('Unexpected native episode identity')
    if type(row.get('success')) is not bool or type(row.get('steps')) is not int or not 0<row['steps']<=row['horizon']:
        raise ValueError('Invalid native success/step record')
    if row['horizon']!=job['horizons'][row['task']] or (not row['success'] and row['steps']!=row['horizon']):
        raise ValueError('Truncated or changed native horizon')
    if initial.get('phase')!='env_reset' or initial.get('seed')!=row['seed'] or initial.get('episode_horizon')!=row['horizon'] or initial.get('registered_episode_horizon')!=row['horizon']:
        raise ValueError('Native reset trace identity differs')
    expected=job['reference_reset_sha256'][f'{row["task_index"]}:{row["seed"]}']
    if initial.get('observation_sha256')!=row['initial_observation_sha256'] or row['initial_observation_sha256']!=expected:
        raise ValueError('Initial observation differs from frozen Experts reset fingerprint')
    if initial.get('state_shape')!=[1,14] or initial.get('camera_keys')!=['cam_high','cam_left_wrist','cam_right_wrist']:
        raise ValueError('Native observation interface differs')
    if action.get('phase')!='step_action_ready':raise ValueError('Missing native first action')
    for field in ('normalized_action','postprocessed_action','expected_executed_action'):
        if len(action.get(field,[]))!=14 or any(not math.isfinite(x) for x in action[field]):raise ValueError('Invalid 14D native action')
    denom=torch.where(q99-q01==0,torch.tensor(1e-8),q99-q01)
    reconstructed=(torch.tensor(action['normalized_action'],dtype=torch.float32)+1)*denom/2+q01
    actual=torch.tensor(action['postprocessed_action'],dtype=torch.float32)
    error=float((reconstructed-actual).abs().max())
    if error>1e-7:raise ValueError('Checkpoint action unnormalization differs from native trace')
    return key,error

def audit_job(plan,job,plan_sha):
    import json
    from safetensors.torch import load_file
    output=Path(job['output']);norm=load_file(str(c.CHECKPOINT/'policy_postprocessor_step_0_unnormalizer_processor.safetensors'))
    q01,q99=norm['action.q01'].flatten(),norm['action.q99'].flatten()
    if q01.shape!=(14,) or q99.shape!=(14,):raise ValueError('Wrong action normalizer dimensions')
    expected=c.expected_keys(job['tasks']);rows=[];seen=set();errors=[]
    for index,task,seed in sorted(expected):
        path=output/f'{task}-{seed}.json';trace=path.with_suffix('.jsonl');row=c.read(path)
        with trace.open() as f:initial=json.loads(f.readline());action=json.loads(f.readline())
        key,error=audit_episode(row,initial,action,job,q01,q99)
        if key in seen:raise ValueError('Duplicate native episode')
        seen.add(key);rows.append({'row':c.bind(path),'trace':c.bind(trace),'task_index':index,'task':task,'seed':seed,'success':row['success']});errors.append(error)
    allowed={f'{t}-{s}.json' for _,t,s in expected}|{'simulator.json'}
    if set(p.name for p in output.glob('*.json'))!=allowed or seen!=expected:raise ValueError('Native episode files differ from exact 60-row panel')
    native=c.read(output/'simulator.json');successes=sum(r['success'] for r in rows)
    task_rates={str(i):sum(r['success'] for r in rows if r['task_index']==i)/6 for i in sorted({k[0] for k in expected})}
    if native['episodes']!=60 or native['status']!='development_rollout_complete' or abs(native['macro_success']-successes/60)>1e-12:
        raise ValueError('Native summary and raw episodes disagree')
    by_name={t:sum(r['success'] for r in rows if r['task']==t)/6 for t in {k[1] for k in expected}}
    if native['task_success']!=by_name:raise ValueError('Native per-task summary disagrees')
    return {'status':'PASS','formal_result':True,'method':c.METHOD,'job_id':job['id'],'group':job['group'],'repeat':job['repeat'],
        'plan_sha256':plan_sha,'model_sha256':c.MODEL_SHA,'model_accepted_sha256':c.ACCEPTED_SHA,
        'manifest':c.bind(Path(plan['run'])/'manifests'/f'{job["id"]}.json'),'reset_bank':job['reset_bank'],
        'episodes':60,'successes':successes,'macro_success':successes/60,'task_success':task_rates,'rows':rows,
        'native_simulator_receipt':c.bind(output/'simulator.json'),'matched_expert_reset_fingerprints':60,
        'first_action_quantile_checks':60,'max_action_postprocess_error':max(errors),'full_native_horizon':True,'finished_at':c.now()}
