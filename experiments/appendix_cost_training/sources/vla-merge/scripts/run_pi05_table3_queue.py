#!/usr/bin/env python3
"""Single-repeat Table-3 DAG on GPUs 4--7, one subprocess per GPU, no process eviction."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from pi05_table3_contract import VARIANTS, validate_trace, validate_rows
from pi05_tcr_e_dense_contract import validate_dense_bank

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
SOURCE=WORK/'pi05_lora_finetune_v2_20260826'
PYTHON=SOURCE/'.venv/bin/python'
TABLE=WORK/'vla-merge-runtime/experiments/iclr2027-table1-20260910'
DEFAULT=WORK/'vla-merge-runtime/experiments/table3-ablation-20260916'
SUITES=dict(spatial='libero_spatial',object='libero_object',goal='libero_goal',long='libero_10')
BANK=TABLE/'expert-dense-bank-peft-v2.json'
REFERENCE=TABLE/'libero/tcr-e/repeat-01/merge/attempt-01-peft-safe-v1/block_regmeanpp_manifest.json'
RESET=TABLE/'reset-banks/libero-procedural-clean-v1'
PINNED=('run_pi05_table3_queue.py','pi05_table3_contract.py','collect_pi05_table3_execution.py',
        'collect_pi05_table3_demo.py','run_pi05_table3_worker.py','materialize_pi05_tcr_e_peft_safe.py',
        'collect_pi05_block_regmeanpp_calibration.py','materialize_pi05_block_regmeanpp.py',
        'pi05_replay_prefix.py','eval_pi05_policy_with_procedural_bank.py')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)


def environment(gpu):
    return {**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'MUJOCO_GL':'egl',
            'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'3','MKL_NUM_THREADS':'3',
            'OPENBLAS_NUM_THREADS':'3','TORCHINDUCTOR_COMPILE_THREADS':'1','PYTHONUNBUFFERED':'1',
            'PI05_ALLOW_GPU_SHARING':'1','PI05_TASK_TEXT_MODE':'correct',
            'PALIGEMMA_TOKENIZER_PATH':str(SOURCE/'assets/paligemma-3b-pt-224-tokenizer'),
            'PYTHONPATH':':'.join(map(str,[WORK/'vla-merge-runtime/python-overlay',SOURCE/'src',SOURCE/'scripts',ROOT/'scripts',ROOT/'src']))}


def job_plan(run, bank):
    reference=json.loads(REFERENCE.read_text())
    jobs=[]
    for name,suite in SUITES.items():
        out=run/'inputs'/name
        env={f'PI05_BLOCK_REGMEANPP_{k}':v for k,v in dict(
            TASK=name,TENSOR_OUTPUT=str(out/'raw.safetensors'),MANIFEST_OUTPUT=str(out/'raw.json'),
            CALIBRATION_POLICY=bank[name]['path'],MAX_CALLS='3840',MAX_CALLS_PER_PROMPT='1',
            REQUESTS_PER_EPISODE='128',EPISODE_AWARE='1',FULL_PREFIX='1',
            FLOW_INDICES='0,5,9',REQUEST_MODE='initial',START_SEED='271001').items()}
        env.update(PI05_LIBERO_INIT_STATE_OFFSET='0',PI05_LIBERO_INIT_STATE_COUNT='1',PI05_LIBERO_SUITE=suite)
        command=[str(PYTHON),str(ROOT/'scripts/collect_pi05_table3_execution.py'),
                 f'--output_dir={out/"rollout"}','--env.type=libero',f'--env.task={suite}',
                 '--env.task_ids=[0,1,2,3,4,5,6,7,8,9]','--env.init_states=true',
                 '--eval.batch_size=1','--eval.n_episodes=1','--seed=271001',
                 f'--policy.path={bank[name]["path"]}','--policy.device=cuda',
                 '--policy.compile_model=false','--policy.gradient_checkpointing=false','--policy.n_action_steps=10']
        jobs.append(dict(id=f'collect-{name}',stage='collect',expert=name,deps=[],command=command,env=env,reserve_mib=40*1024))
    jobs.append(dict(id='solve-full',stage='solve',variant='full',deps=[f'collect-{n}' for n in SUITES]))
    for name in SUITES:
        jobs.append(dict(id=f'demo-{name}',stage='demo',expert=name,deps=[f'collect-{name}'],
                         command=[str(PYTHON),str(ROOT/'scripts/collect_pi05_table3_demo.py'),
                                  f'--expert={name}',f'--policy={bank[name]["path"]}',
                                  f'--template={run/"inputs"/name/"across"}',f'--output={run/"inputs"/name/"demo"}'],env={},reserve_mib=40*1024))
    for variant in VARIANTS[1:]:
        deps=['solve-full']+([f'demo-{n}' for n in SUITES] if variant=='demo' else [])
        jobs.append(dict(id=f'solve-{variant}',stage='solve',variant=variant,deps=deps))
    for job in jobs:
        if job['stage']!='solve': continue
        variant=job['variant']
        command=[str(PYTHON),str(ROOT/'scripts/run_pi05_table3_worker.py'),'--table3-stage=solve',
                 f'--ablation-config={run/"configs"/(variant+".json")}',f'--dense-expert-bank={BANK}',
                 f'--base-model={reference["inputs"]["base_model"]}',f'--prior-model={reference["inputs"]["prior_model"]}',
                 f'--output={run/"checkpoints"/variant}', '--ridge-ratio=.05','--ridge-scale=feature_energy',
                 '--max-correction-ratio=3','--max-rows-per-sample=16','--device=cuda',
                 '--allow-mixed-calibration-policies','--allow-prior-calibration-mismatch',
                 '--expert-loss-normalization='+('none' if variant=='uniform' else 'prior'),
                 '--replay-prefix='+('expert' if variant=='expert_prefix' else 'merged')]
        for name in SUITES:
            kind='demo' if variant=='demo' else 'early' if variant=='early' else 'across'
            directory=run/'inputs'/name/kind
            command += [f'--expert={name}={reference["experts"][name]}',
                        f'--calibration={name}={directory/"replay.safetensors"}',
                        f'--manifest={name}={directory/"replay.json"}']
        job.update(command=command,env={},reserve_mib=44*1024)
    for variant in VARIANTS:
        for name,suite in SUITES.items():
            out=run/'eval'/variant/suite
            command=[str(PYTHON),str(ROOT/'scripts/run_pi05_table3_worker.py'),'--table3-stage=eval',
                     f'--procedural-bank={RESET}',f'--procedural-selection={RESET/"selections/repeat-01.json"}',
                     f'--output_dir={out}','--env.type=libero',f'--env.task={suite}',
                     '--env.init_states=true','--env.hard_reset=true','--eval.batch_size=1','--eval.n_episodes=10',
                     '--seed=274001',f'--policy.path={run/"checkpoints"/variant}',
                     '--policy.device=cuda','--policy.compile_model=false','--policy.gradient_checkpointing=false',
                     '--policy.n_action_steps=10']
            jobs.append(dict(id=f'eval-{variant}-{name}',stage='eval',variant=variant,expert=name,
                             deps=[f'solve-{variant}'],command=command,env={'PI05_LIBERO_SUITE':suite},reserve_mib=32*1024))
    return jobs


def gpu_available(gpu, reserve):
    used,total=map(float,subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
    # Other local pi05 workers own their lane until they exit. Host training is
    # left untouched; sharing spare memory is explicitly authorized.
    blockers=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            command=(proc/'cmdline').read_bytes().replace(b'\0',b' ')
            if not any(s in command for s in (b'eval_pi05_',b'materialize_pi05_',b'collect_pi05_')): continue
            variables=(proc/'environ').read_bytes().split(b'\0')
            if f'CUDA_VISIBLE_DEVICES={gpu}'.encode() in variables:
                blockers.append(int(proc.name))
        except (OSError,PermissionError): pass
    return total-used>=reserve and not blockers,dict(free_mib=total-used,required_mib=reserve,other_local_worker_pids=blockers)


def verify_job(job,run,bank):
    stage=job['stage']
    if stage in ('collect','demo'):
        name=job['expert']; receipts={}
        for kind in (('across','early') if stage=='collect' else ('demo',)):
            directory=run/'inputs'/name/kind
            manifest=json.loads((directory/'replay.json').read_text())
            validate_trace(manifest,bank[name]['path'],name)
            receipts[kind]=dict(manifest_sha256=digest(directory/'replay.json'),tensor_sha256=digest(directory/'replay.safetensors'))
        return receipts
    if stage=='solve':
        variant=job['variant']; directory=run/'checkpoints'/variant
        m=json.loads((directory/'block_regmeanpp_manifest.json').read_text())
        total=validate_rows(m['modules'],list(SUITES),variant)
        if digest(directory/'model.safetensors')!=m['model_sha256']: raise ValueError('Export hash mismatch')
        if m['manual_native_block_max_error']>1e-4: raise ValueError('Manual/native block mismatch')
        if variant!='full':
            full=json.loads((run/'checkpoints/full/block_regmeanpp_manifest.json').read_text())
            for module,entry in m['modules'].items():
                if entry.get('kind')!='frozen_prior' and entry['ridge']!=full['modules'][module]['ridge']:
                    raise ValueError('Numeric ridge differs from Full')
        if variant in ('action_only','conditioning_action'):
            import torch
            from safetensors import safe_open
            with safe_open(directory/'model.safetensors',framework='pt',device='cpu') as output, \
                 safe_open(Path(m['inputs']['prior_model'])/'model.safetensors',framework='pt',device='cpu') as prior:
                for key in output.keys():
                    excluded='.vision_tower.' in key or (variant=='action_only' and '.language_model.' in key)
                    if excluded and not torch.equal(output.get_tensor(key),prior.get_tensor(key)):
                        raise ValueError(f'Off-scope tensor changed: {key}')
        return dict(model_sha256=m['model_sha256'],realized_rows=total,
                    calibrated_modules=sum(v.get('kind')!='frozen_prior' for v in m['modules'].values()))
    path=run/'eval'/job['variant']/SUITES[job['expert']]/'eval_info.json'
    data=json.loads(path.read_text()); rows=data['per_task']
    if len(rows)!=10 or {r['task_id'] for r in rows}!=set(range(10)):
        raise ValueError('Incomplete suite tasks')
    for row in rows:
        values=row['metrics']['successes']
        if len(values)!=10 or any(type(v) is not bool for v in values): raise ValueError('Incomplete episode outcomes')
    successes=sum(sum(row['metrics']['successes']) for row in rows)
    return dict(successes=successes,episodes=100,eval_info_sha256=digest(path))


def summary(states,jobs):
    rows={}
    for variant in VARIANTS:
        evaluations=[states[f'eval-{variant}-{name}'] for name in SUITES]
        if all(s['status']=='complete' for s in evaluations):
            rows[variant]=dict(successes=sum(s['receipt']['successes'] for s in evaluations),episodes=400,
                               success_percent=sum(s['receipt']['successes'] for s in evaluations)/4,
                               per_suite={name:s['receipt']['successes'] for name,s in zip(SUITES,evaluations)})
    if 'full' in rows:
        for row in rows.values(): row['delta_vs_full_pp']=row['success_percent']-rows['full']['success_percent']
    return dict(single_repeat=True,standard_deviation=None,independent_expert_training=False,
                historical_76_58_not_used_as_reference=True,results=rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,default=DEFAULT)
    p.add_argument('--launch',action='store_true'); args=p.parse_args()
    run=args.run.resolve()
    bank=validate_dense_bank(BANK,verify_weights=args.launch)
    jobs=job_plan(run,bank)
    assets={str(ROOT/'scripts'/s):digest(ROOT/'scripts'/s) for s in PINNED}
    assets[str(BANK)]=digest(BANK); assets[str(RESET/'selections/repeat-01.json')]=digest(RESET/'selections/repeat-01.json')
    plan=dict(schema='table3_one_repeat_v1',host=socket.gethostname(),gpus=[4,5,6,7],
              variants=list(VARIANTS),merges_per_variant=1,eval_episodes_per_variant=400,total_episodes=3200,
              calibration_episodes_per_task=1,requests_per_episode=5,flow_indices=[0,5,9],
              row_cap_per_request_module=16,time_projection_rows_per_request=1,
              vision_quota_shared_across_cameras=True,expected_full_rows=1331600,
              calibration_seed=271001,generation_seed=272001,eval_seed=274001,
              status='single_repeat_execution_check_not_final_repeated_evidence',
              differences_from_planned_repeated_paper_protocol=['one episode/task and one build, not three calibration subsamples',
                  'camera and flow states share one per-request/module cap; new Full required'],
              source_assets=assets,dense_bank=bank,jobs=jobs)
    if not args.launch:
        print(json.dumps(plan,indent=2)); return
    run.mkdir(parents=True,exist_ok=False)
    lock=(run/'supervisor.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(run/'plan.json',plan)
    for variant in VARIANTS:
        save(run/'configs'/(variant+'.json'),dict(schema='table3_one_repeat_v1',variant=variant,repeat=1,
             row_cap_per_request_module=16,full_manifest=None if variant=='full' else str(run/'checkpoints/full/block_regmeanpp_manifest.json')))
    states={j['id']:dict(status='pending') for j in jobs}
    active={}; started=time.time()
    print(f'Supervisor pid={os.getpid()} run={run} jobs={len(jobs)}',flush=True)
    while True:
        for gpu,(job,child,log) in list(active.items()):
            code=child.poll()
            if code is None: continue
            log.close(); state=states[job['id']]
            state.update(return_code=code,finished_at=time.time())
            try:
                if code: raise RuntimeError(f'Child exited {code}; output preserved, no automatic seed change')
                state['receipt']=verify_job(job,run,bank)
                state['status']='complete'
            except Exception as error:
                state.update(status='failed',error=str(error))
            print(job['id'],state['status'],flush=True); del active[gpu]
        for job in jobs:
            state=states[job['id']]
            if state['status']=='pending' and any(states[d]['status'] in ('failed','blocked') for d in job['deps']):
                state.update(status='blocked',reason='prerequisite failed; no incompatible fallback')
        for gpu in [4,5,6,7]:
            if gpu in active: continue
            eligible=[j for j in jobs if states[j['id']]['status']=='pending' and all(states[d]['status']=='complete' for d in j['deps'])]
            if not eligible: continue
            job=eligible[0]
            allowed,check=gpu_available(gpu,job['reserve_mib'])
            states[job['id']]['resource_check']=check
            if not allowed: continue
            if any(digest(Path(path))!=sha for path,sha in assets.items()):
                raise RuntimeError('Pinned source changed; stop scheduling rather than silently change experiment')
            logpath=run/'logs'/(job['id']+'.log'); logpath.parent.mkdir(parents=True,exist_ok=True)
            log=logpath.open('x')
            child=subprocess.Popen(job['command'],env={**environment(gpu),**job['env']},cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            states[job['id']].update(status='running',gpu=gpu,pid=child.pid,started_at=time.time(),log=str(logpath))
            active[gpu]=(job,child,log)
            print(f'START {job["id"]} gpu={gpu} pid={child.pid}',flush=True)
        terminal=all(s['status'] in ('complete','failed','blocked') for s in states.values())
        save(run/'status.json',dict(supervisor_pid=os.getpid(),updated_at=time.time(),elapsed_seconds=time.time()-started,
                                  terminal=terminal,jobs=states))
        save(run/'summary.json',summary(states,jobs))
        if terminal: break
        time.sleep(20)
    print('QUEUE FINISHED',flush=True)


if __name__=='__main__': main()
