#!/usr/bin/env python3
"""RoboTwin FeatCal formal panel on local GPU 5/7, sharing only with known PRO owners.

Reuses the exact existing nine-job formal panel and unchanged native receipt
validator. No training, new fusion, resampling, horizon change or automatic retry.
"""
import argparse
import copy
from datetime import datetime, timezone
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
from types import SimpleNamespace

W = Path(__file__).resolve().parents[3]
V = W/'vla-merge'; R = W/'vla-merge-runtime'
RUN = R/'experiments/robotwin-featcal-local-20260925/formal-v1'
BUILD = R/'experiments/robotwin-static-baselines-readiness-20260925/featcal-m3-tf32-v2'
MODEL = BUILD/'checkpoint/pretrained_model'
MODEL_SHA = 'd2b6d99abcea01877045b98c6e704b552a0dee804c70bebdbb1519a0b71e0228'
SOURCE = R/'experiments/robotwin-regmeanpp-m3-codex-20260925/formal-540-v1/plan.json'
SOURCE_SHA = 'b59c96eb520874e1083bc4bf30e6027773a8135e7b44075a6f7ad04281acd60d'
AUDITOR = V/'experiments/robotwin-regmeanpp-m3-codex-20260925/formal_540_v1/receipt.py'
NATIVE = V/'scripts/run_iclr2027_robotwin_checkpoint_development.py'
SLICER = V/'scripts/run_iclr2027_robotwin_development_slices.py'
PYTHON = R/'envs/iclr2027-robotwin2-py312-mplib-curobo-v3/bin/python'
HOST = 'dsw-967394-56ffd4897d-42wft'
GPUS = {5:'GPU-059c0ed7-bcd0-3a71-69c1-2282316cddc8',7:'GPU-e027de95-4c80-d8ec-d650-30da9b6eb135'}
OWNERS = {5:(3307656,'soup'),7:(3309292,'ties')}
sys.path.insert(0,str(V/'experiments/claude-firstpass-cause-20260920'))
import card_flock
now = lambda: datetime.now(timezone.utc).isoformat()
read = lambda p: json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def bind(p):
    p=Path(p).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def process_identity(pid):
    text=Path(f'/proc/{pid}/stat').read_text();tail=text[text.rindex(')')+2:].split()
    return {'pid':pid,'startticks':int(tail[19])}
def expected_keys(tasks):
    values=[(int(t['task_index']),t['task'],int(s)) for t in tasks for s in t['seeds']]
    assert len(tasks)==10 and len(values)==len(set(values))==60 and all(len(t['seeds'])==6 for t in tasks)
    return set(values)
def owner_alive(gpu,plan):
    owner=plan['owners'][str(gpu)];pid=owner['pid']
    try:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return (process_identity(pid)['startticks']==owner['startticks']
                and f"resume_matched_pro.py run --method {owner['method']} --gpu {gpu}" in cmd)
    except (OSError,ValueError): return False
def board(gpu):
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    uuid,free=raw.strip().split(',');return {'uuid':uuid.strip(),'free_mib':int(free)}
def prepare():
    assert socket.gethostname()==HOST and not RUN.exists()
    assert sha(SOURCE)==SOURCE_SHA
    source=read(SOURCE);built=read(BUILD/'complete.json')
    assert built['reload_bitwise_equal'] and built['checkpoint']['model_sha256']==MODEL_SHA
    assert sha(MODEL/'model.safetensors')==MODEL_SHA
    cfg=read(MODEL/'config.json');assert not cfg['use_peft'] and cfg['output_features']['action']['shape']==[14]
    import yaml
    jobs=[]
    for original in source['jobs']:
        job=copy.deepcopy(original);expected_keys(job['tasks'])
        job['output']=str(RUN/'jobs'/job['id']/'attempt-01')
        job['runtime']['checkpoint']=str(MODEL.parent)
        job['runtime']['purpose']='robotwin_featcal_m3_formal_evaluation'
        assert job['runtime']['formal_result'] and not job['runtime']['plateau_eligible']
        jobs.append(job)
    assert len(jobs)==9
    owners={str(g):{**process_identity(pid),'method':method} for g,(pid,method) in OWNERS.items()}
    plan={'host':HOST,'created_at':now(),'run':str(RUN),'jobs':jobs,'episodes':540,
          'source_plan':bind(SOURCE),'build_complete':bind(BUILD/'complete.json'),
          'model_sha256':MODEL_SHA,'model':str(MODEL),'owners':owners,'gpu_uuids':GPUS,
          'sources':[bind(p) for p in (Path(__file__),NATIVE,SLICER,AUDITOR)],
          'model_sidecars':[bind(p) for p in MODEL.rglob('*') if p.is_file() and p.name!='model.safetensors'],
          'native_frozen_files':source['frozen_files_sha256'],
          'memory_fraction':0.35,'minimum_free_mib':49152,'runtime_free_floor_mib':12288,
          'max_active':2,'no_retry':True,'resource_policy':'One added worker per GPU; share only with identity-bound PRO owner, otherwise require exclusive original lease.'}
    for p,h in plan['native_frozen_files'].items():assert sha(p)==h,p
    for job in jobs:save(RUN/'manifests'/f"{job['id']}.json",job)
    save(RUN/'plan.json',plan);save(RUN/'PLAN-SHA256.json',{'sha256':sha(RUN/'plan.json')})
    print(json.dumps({'prepared':True,'jobs':9,'episodes':540,'model_sha256':MODEL_SHA}),flush=True)
def validate():
    p=read(RUN/'plan.json');assert socket.gethostname()==HOST
    assert sha(RUN/'plan.json')==read(RUN/'PLAN-SHA256.json')['sha256']
    for b in p['sources']+[p['source_plan'],p['build_complete']]+p['model_sidecars']:
        assert sha(b['path'])==b['sha256'],b['path']
    return p
def worker(jid,gpu):
    plan=validate();job=next(j for j in plan['jobs'] if j['id']==jid)
    grant=read(RUN/'jobs'/jid/'LEASE.json')
    assert grant['gpu']==gpu and grant['uuid']==GPUS[gpu]
    for x in grant['leases']:
        a=os.fstat(x['fd']);b=Path(x['path']).stat();assert (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)==tuple(x['identity'])
    assert sha(MODEL/'model.safetensors')==MODEL_SHA
    assert board(gpu)['free_mib']>=plan['minimum_free_mib']
    output=Path(job['output']);assert not output.exists()
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    slicer=load('featcal_local_native_setup',SLICER);slicer.runtime_setup(job['runtime'])
    import torch
    assert torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(plan['memory_fraction'],0)
    torch.cuda.reset_peak_memory_stats();output.mkdir(parents=True,exist_ok=False)
    save(output.parent/'STARTED.json',{'pid':os.getpid(),'at':now(),'job':jid,'model_sha256':MODEL_SHA,'gpu':gpu})
    start=time.monotonic()
    native=load('featcal_local_native_engine',NATIVE)
    native.simulator_audit(job['runtime'],output,False)
    # Supply method-specific constants to the unchanged, method-independent validator.
    # All 60 reset, horizon, action-normalizer and raw-outcome checks are retained.
    contract=SimpleNamespace(CHECKPOINT=MODEL,MODEL_SHA=MODEL_SHA,METHOD='FeatCal (static M3)',
        ACCEPTED_SHA=plan['build_complete']['sha256'],expected_keys=expected_keys,read=read,bind=bind,now=now)
    sys.modules['contract']=contract
    receipt=load('featcal_local_receipt',AUDITOR)
    result=receipt.audit_job(plan,job,sha(RUN/'plan.json'))
    assert sha(MODEL/'model.safetensors')==MODEL_SHA
    validate()
    result.update(seconds=time.monotonic()-start,peak_cuda_memory_mib=torch.cuda.max_memory_allocated()/1024**2)
    save(output.parent/'ACCEPTED.json',result)
def run():
    plan=validate();assert not (RUN/'STARTED.json').exists()
    ownerlock=card_flock._try_lock(RUN/'supervisor.lock','robotwin-featcal-formal',{'host':HOST})
    assert ownerlock is not None
    save(RUN/'STARTED.json',{'pid':os.getpid(),'at':now()})
    pending=list(plan['jobs']);active={};accepted={};failed={};stop=False
    def halt(*_):
        nonlocal stop
        stop=True  # Drain own workers; never signal existing PRO/TCR jobs.
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    while pending or active:
        for jid,x in list(active.items()):
            code=x['child'].poll()
            if code is None:
                if board(x['gpu'])['free_mib']<plan['runtime_free_floor_mib'] and not x.get('floor'):
                    x['floor']=True;os.killpg(x['child'].pid,signal.SIGTERM)
                continue
            x['log'].close()
            for lock in x['locks']:lock.release()
            try:
                assert code==0 and not x.get('floor'),f'exit {code}, resource_floor={x.get("floor")}'
                a=read(RUN/'jobs'/jid/'ACCEPTED.json');assert a['episodes']==60 and a['model_sha256']==MODEL_SHA
                accepted[jid]=a
            except Exception as exc:failed[jid]=repr(exc)
            save(RUN/'exits'/f'{jid}.json',{'code':code,'error':failed.get(jid),'at':now()});del active[jid]
        if not stop and not failed:
            for gpu in GPUS:
                if not pending or len(active)>=2:break
                if gpu in [x['gpu'] for x in active.values()]:continue
                row=board(gpu)
                if row['uuid']!=GPUS[gpu] or row['free_mib']<plan['minimum_free_mib']:continue
                shared=owner_alive(gpu,plan);job=pending[0];locks=[]
                paths=[card_flock.card_path(gpu,GPUS[gpu]),R/'resource-leases'/f'{HOST}-gpu-{gpu}.lock',
                       R/'resource-leases'/HOST/(f'gpu-{gpu}-featcal-shared.lock' if shared else f'gpu-{gpu}.lock')]
                for path in paths:
                    lock=card_flock._try_lock(path,job['id'],{'gpu':gpu,'job':job['id'],'shared_known_pro':shared})
                    if lock is None:break
                    locks.append(lock)
                if len(locks)!=3 or board(gpu)['free_mib']<plan['minimum_free_mib'] or (shared and not owner_alive(gpu,plan)):
                    for lock in locks:lock.release()
                    continue
                root=RUN/'jobs'/job['id'];assert not root.exists()
                grant={'gpu':gpu,'uuid':GPUS[gpu],'shared_known_pro':shared,'owner':plan['owners'][str(gpu)],
                       'leases':[{'fd':l.handle,'path':str(l.path),'identity':[os.fstat(l.handle).st_dev,os.fstat(l.handle).st_ino]} for l in locks]}
                save(root/'LEASE.json',grant);(RUN/'logs').mkdir(exist_ok=True)
                log=(RUN/'logs'/f"{job['id']}.log").open('x')
                env=dict(os.environ)
                for k in ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','NVIDIA_TF32_OVERRIDE','CUBLAS_WORKSPACE_CONFIG']:
                    env.pop(k,None)
                env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                command=[str(PYTHON),'-u',str(Path(__file__)),'worker','--job-id',job['id'],'--gpu',str(gpu)]
                try:
                    child=subprocess.Popen(command,cwd=W,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                           start_new_session=True,pass_fds=tuple(l.handle for l in locks))
                except Exception:
                    log.close()
                    for lock in locks:lock.release()
                    raise
                active[job['id']]={'child':child,'gpu':gpu,'locks':locks,'log':log}
                save(RUN/'launches'/f"{job['id']}.json",{'pid':child.pid,'gpu':gpu,'command':command,'at':now(),'shared_known_pro':shared})
                pending.pop(0)
        save(RUN/'state.json',{'accepted':accepted,'failed':failed,'active':{k:{'pid':v['child'].pid,'gpu':v['gpu']} for k,v in active.items()},
                             'pending':[j['id'] for j in pending],'stopped':stop,'updated_at':now()})
        if (stop or failed) and not active:break
        if pending or active:time.sleep(20)
    save(RUN/'queue-ended.json',{'complete':len(accepted)==9 and not failed,'accepted':accepted,'failed':failed,
                               'pending':[j['id'] for j in pending],'stopped':stop,'at':now()})
    ownerlock.release()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker']);p.add_argument('--job-id');p.add_argument('--gpu',type=int,choices=[5,7]);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='run':run()
    else:worker(a.job_id,a.gpu)
