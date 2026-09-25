#!/usr/bin/env python3
"""Isolated 5-suite/500-episode TIES formal runner; prepare/bind never use CUDA."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

WORK = Path('/mnt/workspace/Wilson/parameter-fusion')
REPO = WORK/'vla-merge'
ROOT = WORK/'vla-merge-runtime/experiments/fig4-baselines-codex-20260925'
HERE = Path(__file__).resolve().parent
RUN = ROOT/'ties-formal-v1'
CPU_ROOT = ROOT/'ties-adapter-v2'
CPU_DONE = CPU_ROOT/'cpu-queue-01/DONE.json'
PARENT_SHA = '62ff0891df02a75d20a8b724f70f93a8373f75852a7385c88034993c326d7875'
CPU_PLAN_SHA = 'e9b38b1e7793b31876c7c7804e0bc446e6287ec91f34d33444eca818432c407b'
PYTHON = WORK/'pi05_lora_finetune_v2_20260826/.venv/bin/python'
CONFIG = HERE/'libero-config'
TOKENIZER = WORK/'pi05_lora_finetune_v2_20260826/assets/paligemma-3b-pt-224-tokenizer'
AUDIT_SOURCE = REPO/'experiments/claude-pi05-fixed-appendix-20260923/run_union_eval_v8.py'
FLOCK_SOURCE = REPO/'experiments/claude-firstpass-cause-20260920/card_flock.py'
HOST = 'dsw-824375-57c745db88-n6tv9'
GPUS = (1,4,5,6)
MIN_FREE = 78000
SUBSETS = ('spatial-goal','spatial-object-goal')
sys.path[:0] = [str(HERE),str(REPO/'scripts'),str(REPO/'src'),
               str(WORK/'pi05_lora_finetune_v2_20260826/src'),
               str(REPO/'experiments/claude-firstpass-cause-20260920')]
stopping = False

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    with Path(p).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,msg):
    if not ok:raise ValueError(msg)
def exclusive(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(json.dumps(data,sort_keys=True,indent=2)+'\n')
def state(data):
    p=RUN/'STATE.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n');os.replace(t,p)
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def stat_identity(p):
    s=Path(p).stat();return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def cpu_env():
    e=os.environ.copy()
    e.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',LIBERO_CONFIG_PATH=str(CONFIG),HF_HUB_OFFLINE='1',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for key in ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','PI05_LIBERO_INIT_STATE_OFFSET','PI05_LIBERO_INIT_STATE_COUNT','LIBERO_PRO_REPO','LIBERO_PRO_ASSET_DIR']:
        e.pop(key,None)
    return e

def preflight():
    require(sha(ROOT/'plan.json')==PARENT_SHA,'Parent plan changed')
    require(sha(CPU_ROOT/'plan.json')==CPU_PLAN_SHA,'CPU build plan changed')
    parent=read(ROOT/'plan.json')
    rows=[j for j in parent['new_evaluations'] if j['method']=='ties']
    require(len(rows)==5 and len({j['id'] for j in rows})==5 and sum(j['episodes'] for j in rows)==500,'TIES 5/500 matrix differs')
    os.environ.update(cpu_env())
    from vla_merge.libero_procedural_bank import load_selection
    selection=Path(parent['evaluation_selection']['path'])
    require(sha(selection)==parent['evaluation_selection']['sha256'],'Selection changed')
    loaded=load_selection(selection.parent.parent,selection,verify_source_files=True)
    from dataclasses import asdict
    # The native loader checks source BDDL/init files, stored bank arrays, indices,
    # and every selected raw state; no policy or CUDA context is instantiated.
    return {'status':'pass_cpu_native_reset_preflight','jobs':5,'episodes':500,
            'selection':str(selection),'selection_sha256':sha(selection),
            'bank':str(selection.parent.parent),'bank_manifest_sha256':sha(selection.parent.parent/'manifest.json'),
            'verify_source_files':True,'model_binding':'pending_cpu_queue_DONE',
            'gpu_jobs_launched':0}

def prepare():
    require(not RUN.exists(),'Formal root already exists; do not duplicate')
    check=preflight();parent=read(ROOT/'plan.json')
    jobs=[]
    for j in parent['new_evaluations']:
        if j['method']!='ties':continue
        row=dict(j);row['full_suite']=row['suite'];row['output']=str(RUN/'jobs'/row['id'])
        old=next(x for x in row['command_template'] if x.startswith('--output_dir='))
        row['command']=[f'--output_dir={row["output"]}/eval' if x==old else x for x in row.pop('command_template')]
        row['status']='model_sha_pending_DONE';jobs.append(row)
    sources=[Path(__file__),AUDIT_SOURCE,FLOCK_SOURCE,REPO/'scripts/eval_tcr_10k_formal_bounded.py',
             REPO/'scripts/eval_pi05_policy_with_procedural_bank.py',REPO/'src/vla_merge/libero_procedural_bank.py',
             HERE/'ties_subset_v2.py',CONFIG/'config.yaml']
    template={'schema':'fig4_ties_formal_template_v1','created_at':now(),'parent_plan_sha256':PARENT_SHA,
              'cpu_build_plan_sha256':CPU_PLAN_SHA,'native_reset_preflight':check,'jobs':jobs,
              'source_sha256':{str(p):sha(p) for p in sources},'models':{s:None for s in SUBSETS},
              'episodes':500,'formal_repeat':'repeat-01','eval_seed':274001,'host':HOST,'permitted_gpus':list(GPUS),
              'max_active':4,'min_free_mib':MIN_FREE,'require_no_compute_app':True,
              'leases':'UUID card + host/index legacy + historical flat host/index compatibility guard',
              'evaluator_resource_behavior':{'allocator_fraction':.20,'duty_fraction':.25,'runtime_free_floor_mib':12288,'source_unchanged':True},
              'native_action_gate':'One fixed Spatial preprocessed observation; finite native sample_actions result only, no success threshold.',
              'native_action_input':parent['inputs']['featcal']['spatial']['observations'],
              'no_retry':True,'no_training':True,'no_score_based_scheduling':True,
              'template_is_not_permission_to_launch':True}
    RUN.mkdir(parents=True)
    exclusive(RUN/'template.json',template)
    exclusive(RUN/'TEMPLATE-SHA256.json',{'sha256':sha(RUN/'template.json')})
    exclusive(RUN/'CPU-PREFLIGHT.json',check)
    return {'status':'template_frozen','template_sha256':sha(RUN/'template.json'),'jobs':5,'episodes':500,'gpu_jobs_launched':0}

def template():
    require(sha(RUN/'template.json')==read(RUN/'TEMPLATE-SHA256.json')['sha256'],'Formal template changed')
    t=read(RUN/'template.json')
    for p,s in t['source_sha256'].items():require(sha(p)==s,f'Frozen source changed: {p}')
    return t

def bind():
    t=template();require(not (RUN/'plan.json').exists(),'Final model-bound plan already exists')
    done=read(CPU_DONE)
    require(done['status']=='complete_cpu_builds_and_native_loads' and set(done['completed'])==set(SUBSETS),'Both CPU builds and native loads must complete')
    models={}
    for subset in SUBSETS:
        export=ROOT/'checkpoints'/f'ties-{subset}'/'pretrained_model'
        m=read(export.parent/'candidate-manifest.json');native=read(CPU_ROOT/subset/'NATIVE-CPU-LOAD.json')
        require(m['experts']==read(CPU_ROOT/'plan.json')['jobs'][list(SUBSETS).index(subset)]['experts'],'Subset changed')
        require(m['adapter_plan_sha256']==CPU_PLAN_SHA and m['density']==.3 and m['alpha']==.9 and m['finite'],'TIES export recipe differs')
        require(native['status']=='pass_native_cpu_load' and native['model_sha256']==m['model_sha256']==done['completed'][subset]['model_sha256'],'Native load binding differs')
        require(sha(export/'model.safetensors')==m['model_sha256'],'Completed model bytes changed')
        sidecars={str(p.relative_to(export)):sha(p) for p in export.rglob('*') if p.is_file() and p.name!='model.safetensors'}
        models[subset]={'path':str(export),'sha256':m['model_sha256'],'stat':stat_identity(export/'model.safetensors'),
                        'export_manifest_sha256':sha(export.parent/'candidate-manifest.json'),'native_cpu_receipt_sha256':sha(CPU_ROOT/subset/'NATIVE-CPU-LOAD.json'),
                        'sidecar_sha256':sidecars}
    final={**t,'schema':'fig4_ties_frozen_formal_v1','bound_at':now(),'models':models,
           'template_sha256':sha(RUN/'template.json'),'cpu_done_sha256':sha(CPU_DONE),
           'status':'frozen_not_launched_native_action_gate_pending'}
    for row in final['jobs']:row['model_sha256']=models[row['subset']]['sha256'];row['status']='ready_after_native_action_gate'
    exclusive(RUN/'plan.json',final);exclusive(RUN/'PLAN-SHA256.json',{'sha256':sha(RUN/'plan.json')})
    return {'status':'final_plan_bound_no_GPU_launch','plan_sha256':sha(RUN/'plan.json'),'models':models,'jobs':5,'episodes':500}

def wait_bind():
    template();p=RUN/'bind-waiter'
    p.mkdir(exist_ok=False)
    with (p/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        exclusive(p/'STARTED.json',{'pid':os.getpid(),'start_ticks':int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),'time':now(),'cpu_only':True,'gpu_dispatch':False})
        try:
            while not CPU_DONE.exists():
                require(not (CPU_DONE.parent/'FAILED.json').exists(),'CPU build queue failed; no automatic recovery')
                time.sleep(30)
            exclusive(p/'DONE.json',bind())
        except BaseException as exc:
            exclusive(p/'FAILED.json',{'time':now(),'error':repr(exc),'gpu_jobs_launched':0});raise

def final_plan():
    require(sha(RUN/'plan.json')==read(RUN/'PLAN-SHA256.json')['sha256'],'Final plan changed')
    p=read(RUN/'plan.json');template()
    require(p['template_sha256']==sha(RUN/'template.json') and p['cpu_done_sha256']==sha(CPU_DONE),'Binding provenance changed')
    for m in p['models'].values():
        require(stat_identity(Path(m['path'])/'model.safetensors')==m['stat'],'Immutable model stat changed; re-audit required')
        for rel,s in m['sidecar_sha256'].items():require(sha(Path(m['path'])/rel)==s,'Policy sidecar changed')
    return p

def gpu_inventory():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    rows={}
    for line in out.splitlines():
        index,u,free=[x.strip() for x in line.split(',')];rows[int(index)]={'uuid':u,'free_mib':int(free)}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'],text=True)
    occupied={x.strip() for x in apps.splitlines() if x.strip().startswith('GPU-')}
    for row in rows.values():row['has_compute_app']=row['uuid'] in occupied
    return rows

def eval_env(gpu):
    env=cpu_env();env.update(CUDA_VISIBLE_DEVICES=str(gpu),ITERATION_PHYSICAL_GPU=str(gpu),
        MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=str(gpu),PALIGEMMA_TOKENIZER_PATH=str(TOKENIZER),PYTHONUNBUFFERED='1',
        PYTHONPATH=':'.join([str(REPO/'scripts'),str(REPO/'src'),str(WORK/'pi05_lora_finetune_v2_20260826/src'),str(WORK/'pi05_lora_finetune_v2_20260826/lerobot/src')]))
    return env

def native_worker(subset):
    plan=final_plan();model=plan['models'][subset]
    require(socket.gethostname()==HOST and os.environ.get('CUDA_VISIBLE_DEVICES') in {str(g) for g in GPUS},'Native GPU worker not on allowed card')
    import torch
    from safetensors import safe_open
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20,0)
    cfg=PreTrainedConfig.from_pretrained(model['path'],local_files_only=True)
    cfg.device='cuda';cfg.compile_model=False;cfg.gradient_checkpointing=False;cfg.n_action_steps=10
    policy=PI05Policy.from_pretrained(model['path'],config=cfg,local_files_only=True,strict=True)
    source=plan['native_action_input'];require(sha(source['path'])==source['sha256'],'Native smoke input changed')
    with safe_open(source['path'],framework='pt',device='cpu') as data:
        batch={k[len('observation_000.'):]:data.get_tensor(k).to('cuda') for k in data.keys() if k.startswith('observation_000.')}
    torch.manual_seed(274001);torch.cuda.manual_seed_all(274001)
    with torch.inference_mode():
        output=policy.model.sample_actions([batch[f'image_{i}'] for i in range(3)],
            [batch[f'image_mask_{i}'] for i in range(3)],batch['tokens'],batch['masks'])
    require(output.ndim==3 and output.shape[0]==1 and torch.isfinite(output).all().item(),'Native action output invalid')
    exclusive(RUN/'native'/f'{subset}.json',{'status':'pass_finite_native_actions','model_sha256':model['sha256'],
        'shape':list(output.shape),'input_sha256':source['sha256'],'seed':274001,'success_values_used':False,
        'time':now(),'peak_allocator_gib':torch.cuda.max_memory_allocated()/2**30})

def stop_signal(*_):
    global stopping
    stopping=True
def run():
    plan=final_plan();require(socket.gethostname()==HOST,'Wrong host')
    require(not (RUN/'STARTED.json').exists(),'No retry/restart of this formal attempt')
    import card_flock
    audit=load_module(AUDIT_SOURCE,'fig4_original_formal_audit')
    with (RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
        exclusive(RUN/'STARTED.json',{'pid':os.getpid(),'time':now(),'plan_sha256':sha(RUN/'plan.json')})
        pending=[{'id':f'native-{s}','phase':'native','subset':s} for s in SUBSETS]
        pending += [{**j,'phase':'formal'} for j in plan['jobs']]
        active={};accepted={};native={}
        try:
            while pending or active:
                if stopping:raise RuntimeError('Owner stopped this formal batch')
                for jid,x in list(active.items()):
                    code=x['proc'].poll()
                    if code is None:continue
                    x['proc'].wait();x['log'].close()
                    for lease in x['leases']:lease.release()
                    active.pop(jid)
                    exclusive(RUN/'exits'/f'{jid}.json',{'pid':x['proc'].pid,'returncode':code,'time':now()})
                    require(code==0,f'{jid} failed; no retry')
                    job=x['job']
                    if job['phase']=='native':
                        receipt=read(RUN/'native'/f'{job["subset"]}.json')
                        require(receipt['status']=='pass_finite_native_actions' and receipt['model_sha256']==plan['models'][job['subset']]['sha256'],'Native action receipt differs')
                        native[job['subset']]=receipt
                    else:
                        audit_plan={'selection':{'sha256':plan['native_reset_preflight']['selection_sha256'],'eval_seed':274001,
                            'selection':plan['native_reset_preflight']['selection'],'bank':plan['native_reset_preflight']['bank']},
                            'bank_manifest_sha256':plan['native_reset_preflight']['bank_manifest_sha256'],
                            'procedural_entrypoint_sha256':plan['source_sha256'][str(REPO/'scripts/eval_pi05_policy_with_procedural_bank.py')]}
                        accepted[jid]=audit.audit_result(job,job['model_sha256'],audit_plan)
                        exclusive(RUN/'accepted'/f'{jid}.json',accepted[jid])
                inventory=gpu_inventory()
                for gpu in GPUS:
                    if len(active)>=plan['max_active']:break
                    if any(x['gpu']==gpu for x in active.values()):continue
                    ready=[j for j in pending if j['phase']=='native' or j['subset'] in native]
                    if not ready:break
                    row=inventory[gpu]
                    if row['free_mib']<MIN_FREE or row['has_compute_app']:continue
                    job=ready[0];leases=[]
                    card=card_flock.take_card(gpu,row['uuid'],job['id'],'fig4-ties-formal')
                    if card is None:continue
                    leases.append(card)
                    for path in [WORK/'vla-merge-runtime/resource-leases'/HOST/f'gpu-{gpu}.lock',
                                 WORK/'vla-merge-runtime/resource-leases'/f'{HOST}-gpu-{gpu}.lock']:
                        lease=card_flock._try_lock(path,job['id'],{'job':job['id'],'stage':'fig4-ties-formal','gpu':gpu})
                        if lease is None:break
                        leases.append(lease)
                    again=gpu_inventory()[gpu]
                    if len(leases)!=3 or again['uuid']!=row['uuid'] or again['free_mib']<MIN_FREE or again['has_compute_app']:
                        for lease in leases:lease.release()
                        continue
                    proc=None;log=None
                    try:
                        final_plan()
                        require(not (RUN/'launches'/f'{job["id"]}.json').exists(),'Duplicate launch ID')
                        path=RUN/'logs'/f'{job["id"]}.log';path.parent.mkdir(parents=True,exist_ok=True);log=path.open('x')
                        command=([str(PYTHON),'-u',str(Path(__file__)),'native-worker','--subset',job['subset']]
                                 if job['phase']=='native' else job['command'])
                        proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                            cwd=WORK,env=eval_env(gpu),start_new_session=True,pass_fds=tuple(x.handle for x in leases))
                        active[job['id']]={'job':job,'proc':proc,'log':log,'leases':leases,'gpu':gpu}
                        exclusive(RUN/'launches'/f'{job["id"]}.json',{'pid':proc.pid,'gpu':gpu,'uuid':row['uuid'],'admission':again,
                            'command':command,'model_sha256':plan['models'][job['subset']]['sha256'],'time':now()})
                        pending.remove(job)
                    except BaseException:
                        if proc is not None and proc.poll() is None:
                            proc.terminate()
                            try:proc.wait(timeout=30)
                            except subprocess.TimeoutExpired:proc.kill();proc.wait()
                        if log:log.close()
                        for lease in leases:lease.release()
                        active.pop(job['id'],None)
                        raise
                state({'status':'running_or_waiting_owned_idle_cards','time':now(),'active':{k:{'pid':v['proc'].pid,'gpu':v['gpu'],'phase':v['job']['phase']} for k,v in active.items()},
                       'pending':[j['id'] for j in pending],'native':native,'accepted':accepted})
                if pending or active:time.sleep(20)
            require(len(accepted)==5 and sum(x['episodes'] for x in accepted.values())==500,'Formal denominator incomplete')
            by_subset={s:{'episodes':sum(v['episodes'] for k,v in accepted.items() if next(j for j in plan['jobs'] if j['id']==k)['subset']==s),
                          'successes':sum(v['successes'] for k,v in accepted.items() if next(j for j in plan['jobs'] if j['id']==k)['subset']==s)} for s in SUBSETS}
            exclusive(RUN/'DONE.json',{'status':'complete','time':now(),'episodes':500,'jobs':5,'accepted':accepted,'by_subset':by_subset,'native_checks_excluded_from_formal':True})
        except BaseException as exc:
            for x in active.values():
                if x['proc'].poll() is None:x['proc'].terminate()
            for x in active.values():
                try:x['proc'].wait(timeout=30)
                except subprocess.TimeoutExpired:x['proc'].kill();x['proc'].wait()
                x['log'].close()
                for lease in x['leases']:lease.release()
            exclusive(RUN/'FAILED.json',{'status':'failed_no_retry','time':now(),'error':repr(exc),'accepted':accepted,'pending':[j['id'] for j in pending],
                'partial_results_excluded':True,'signals_scope':'only this supervisor own Popen objects'})
            raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','preflight','bind','wait-bind','run','native-worker'])
    parser.add_argument('--subset',choices=SUBSETS)
    args=parser.parse_args()
    if args.action=='native-worker':require(args.subset is not None,'Native worker requires subset');native_worker(args.subset)
    elif args.action=='run':run()
    elif args.action=='wait-bind':wait_bind()
    else:print(json.dumps(prepare() if args.action=='prepare' else preflight() if args.action=='preflight' else bind(),indent=2))
