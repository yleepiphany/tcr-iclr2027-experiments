#!/usr/bin/env python3
"""Exact transfer of unstarted static-v2 PRO jobs; no algorithm changes.

prepare/validate are CPU-only. arm reserves the source ownership only when
explicitly invoked. run requires that reservation and acquires both GPU leases.
"""
from __future__ import annotations
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

WORK = Path('/mnt/workspace/Wilson/parameter-fusion')
VLA = WORK/'vla-merge'
RUNTIME = WORK/'vla-merge-runtime'
HERE = Path(__file__).resolve().parent
RUN = RUNTIME/'experiments/pi05-pro-static1016-codex-20260925/attempt-01'
OLD_ROOT = RUNTIME/'experiments/pi05-pro-baselines-codex-20260924'
OLD_CODE = VLA/'experiments/pi05-pro-baselines-codex-20260924/run_baseline_formal_static_v2.py'
FLOCK = VLA/'experiments/claude-firstpass-cause-20260920/card_flock.py'
HOST = 'dsw-824375-57c745db88-n6tv9'
SOURCE_HOST = 'dsw-967394-56ffd4897d-42wft'
METHODS = ('regmean_pp', 'featcal')
IDENTITIES = {
    'regmean_pp': ('fea99a8c09f8ca16bf8a2889bac659aba86bcbf831ef422e8c0a30115bccffb2', '09d1fc54e8f30a8fa985146307baff779a77ed88ca943d2a3e91fbfadab62038', 'regmeanpp-spectral-smoothing-main10k-static-v2'),
    'featcal': ('b0da07eda2a9ab4485f555f694d31afe08c40a39d6e3bfd44e9b9d8d006b0114', '2f7723bc65ec51c5059d819305d5f9ba9f933d3acb9d1469e717c648354dd456', 'featcal-observation-only-main10k-static-v2'),
}
GPUS = {1:'GPU-c18a3bf3-c47c-9a8a-75b0-5a0f37b33a65', 4:'GPU-48baba1c-f0e7-93b7-4a30-b3c21d20e61a', 5:'GPU-4ed28198-b742-ee3c-2acb-4183dc944c81', 6:'GPU-da625da2-f504-7927-4025-907e876718aa'}
MIN_FREE = 40960
FLOOR = 12288

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def require(ok,message):
    if not ok: raise ValueError(message)
def write(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    else:
        temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def require_unstarted(root):
    for name in ('started.json','state.json','queue-ended.json','results','launches','exits'):
        require(not (root/name).exists(), f'Source is not unstarted: {root/name}')

def translate_job(method,job,output_root):
    new=copy.deepcopy(job)
    output=Path(output_root)/'methods'/method/'results'/job['id']
    old_token='--output_dir='+job['output']
    require(job['command'].count(old_token)==1,'Exactly one original output argument required')
    new['command']=[('--output_dir='+str(output)) if x==old_token else x for x in job['command']]
    changed=[i for i,(a,b) in enumerate(zip(job['command'],new['command'])) if a!=b]
    require(len(changed)==1,'Only the output argument may change')
    new['output']=str(output);new['method_key']=method;new['transfer_id']=method+'/'+job['id']
    return new

def source_bindings(verify_weights=False):
    sources={}
    for method in METHODS:
        model_sha,plan_sha,method_id=IDENTITIES[method]
        root=OLD_ROOT/method/'formal-static-v2';require_unstarted(root)
        require(sha(root/'plan.json')==plan_sha,'Frozen source plan SHA differs '+method)
        plan=read(root/'plan.json');claim=read(root/'CLAIM.json')
        require(claim['host']==SOURCE_HOST and claim['owner']=='Codex' and claim['plan_sha256']==plan_sha,'Source owner claim differs')
        require(plan['model_sha256']==model_sha and plan['method']==method_id and plan['host']==SOURCE_HOST,'Current source model or host differs')
        require(plan['jobs_expected']==480 and plan['episodes_expected']==4800 and len(plan['jobs'])==480,'Source panel count differs')
        require(len({j['id'] for j in plan['jobs']})==480,'Duplicate source job')
        require(plan['runner_sha256']==sha(OLD_CODE),'Source native runner changed')
        require(plan['evaluator_sha256']==sha(VLA/'scripts/eval_pi05_policy_with_extension_selection.py'),'Native evaluator changed')
        require(plan['native_runner_sha256']==sha(VLA/'scripts/run_pi05_libero_pro_evaluation.py'),'Native planning code changed')
        selection_path=Path(plan['static_selection_manifest']);selection=read(selection_path)
        require(sha(selection_path)==plan['source_manifest_sha256'],'Static selection manifest changed')
        require(selection['model']['model_sha256']==model_sha and selection['old_formal_v1_results_reusable'] is False,'Static source identity differs')
        proofs=selection['model']
        for label in ('build_manifest','clean_formal_summary'):
            require(sha(proofs[label]['path'])==proofs[label]['sha256'],'Current model proof changed '+label)
        for rep in ('repeat-01','repeat-02','repeat-03'):
            item=selection['selection_index'][rep]
            require(sha(item['path'])==item['sha256'],'Repeat selection hash changed')
            require(item['jobs']==160 and item['episodes']==1600,'Repeat scope differs')
        for job in plan['jobs']:
            require(len(job['state_ids'])==10 and len(job['state_raw_sha256'])==10,'Ten fixed states required')
            require('--policy-sha256='+model_sha in job['command'],'Command model differs')
            require('--seed='+str(job['eval_seed']) in job['command'],'Command seed differs')
            require(job['eval_seed']==274000+int(job['repeat'][-2:]),'Repeat seed differs')
        if verify_weights: require(sha(Path(proofs['path'])/'model.safetensors')==model_sha,'Model file differs')
        sources[method]={'directory':str(root),'plan_sha256':plan_sha,'claim_sha256':sha(root/'CLAIM.json'),
                         'model_sha256':model_sha,'method_id':method_id,'selection_manifest_sha256':sha(selection_path),
                         'model_path':proofs['path'],'original_plan':plan}
    return sources

def build_plan(verify_weights=False):
    sources=source_bindings(verify_weights)
    jobs=[]
    for index in range(480):
        for method in METHODS:jobs.append(translate_job(method,sources[method]['original_plan']['jobs'][index],RUN))
    return {'schema':'pi05_pro_static1016_exact_transfer_v1','host':HOST,'source_host':SOURCE_HOST,
            'methods':list(METHODS),'jobs':jobs,'job_count':960,'episodes':9600,
            'sources':{m:{k:v for k,v in x.items() if k!='original_plan'} for m,x in sources.items()},
            'native_plans':{m:sources[m]['original_plan'] for m in METHODS},
            'implementation':{str(p):sha(p) for p in (Path(__file__),OLD_CODE,FLOCK)},
            'resource':{'gpu_uuids':{str(k):v for k,v in GPUS.items()},'minimum_free_mib':MIN_FREE,'maximum_idle_used_mib':64,'require_no_compute_pid':True,'runtime_floor_mib':FLOOR,'max_active':4,'one_worker_per_gpu':True,'double_leases':'host/UUID card_flock and host/index resource-leases'},
            'ownership_fence':str(OLD_ROOT/'gpu-7-baseline-shared-lane.lock'),
            'command_delta':'Exactly one --output_dir token; model/selection/seed/task/episodes/evaluator flags unchanged',
            'no_retry':True,'training':False,'old_plan_or_results_modified':False}

def prepare():
    require(not RUN.exists(),'New run directory must not exist')
    plan=build_plan(True);write(RUN/'plan.json',plan,True)
    return {'status':'PREPARED_NO_GPU','plan':str(RUN/'plan.json'),'sha256':sha(RUN/'plan.json'),'jobs':960,'episodes':9600}

def validate():
    plan=read(RUN/'plan.json');expected=build_plan(False)
    require(plan==expected,'Exact transfer plan reconstruction differs')
    return plan

def marker(method):return OLD_ROOT/method/'formal-static-v2/TRANSFERRED-TO-STATIC1016.json'

def arm():
    plan=validate();card=load_module('transfer_flock',FLOCK)
    held=card._try_lock(Path(plan['ownership_fence']),'static1016-transfer',{'stage':'source-owner-transfer'})
    require(held is not None,'Source legacy PRO baseline owner is active')
    try:
        for method in METHODS:
            require_unstarted(Path(plan['sources'][method]['directory']))
            require(not marker(method).exists(),'Transfer already marked; no duplicate arming')
        value={'schema':'pi05_pro_static1016_owner_transfer_v1','target_plan':str(RUN/'plan.json'),'target_plan_sha256':sha(RUN/'plan.json'),'from_host':SOURCE_HOST,'to_host':HOST,'methods':list(METHODS),'sources':plan['sources'],'armed_at':now()}
        for method in METHODS:write(marker(method),{**value,'method':method},True)
        write(RUN/'ARMED.json',value,True)
    finally:held.release()
    return {'status':'ARMED_NO_GPU','plan_sha256':sha(RUN/'plan.json')}

def board(gpu):
    out=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    uuid,free,used,util=[x.strip() for x in out.split(',')]
    pids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    return {'gpu':gpu,'uuid':uuid,'free_mib':int(free),'used_mib':int(used),'utilization':int(util),'compute_pids':pids.splitlines() if pids else []}
def idle(row):return row['uuid']==GPUS[row['gpu']] and row['free_mib']>=MIN_FREE and row['used_mib']<=64 and row['utilization']==0 and not row['compute_pids']

def run():
    require(socket.gethostname()==HOST,'Execution requires frozen host1016')
    plan=validate();require(not(RUN/'STARTED.json').exists(),'No automatic restart within attempt')
    armed=read(RUN/'ARMED.json');require(armed['target_plan_sha256']==sha(RUN/'plan.json'),'Missing exact source-owner handoff')
    for m in METHODS:require(read(marker(m))['target_plan_sha256']==sha(RUN/'plan.json'),'Source transfer marker differs')
    flocks=load_module('transfer_runtime_flock',FLOCK)
    fence=flocks._try_lock(Path(plan['ownership_fence']),'static1016-owner',{'stage':'pi05-pro-static1016'})
    require(fence is not None,'Old source runner holds ownership fence')
    pending=list(plan['jobs']);active={};accepted={};failed={};stop=False
    modules={}
    for method in METHODS:
        mod=load_module('static_native_'+method,OLD_CODE);mod.select_method(method);modules[method]=mod
    def on_signal(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    write(RUN/'STARTED.json',{'pid':os.getpid(),'host':HOST,'plan_sha256':sha(RUN/'plan.json'),'started_at':now()},True)
    try:
        for m in METHODS:require_unstarted(Path(plan['sources'][m]['directory']))
        while pending or active:
            for gpu,item in list(active.items()):
                child=item['child'];code=child.poll()
                if code is None:
                    row=board(gpu)
                    if (row['uuid']!=GPUS[gpu] or row['free_mib']<FLOOR) and not item.get('floor'):
                        # Only the process group created by this parent for this job.
                        os.killpg(child.pid,signal.SIGTERM);item['floor']=True
                    continue
                child.wait();item['log'].close();item['legacy'].release();item['card'].release();active.pop(gpu)
                job=item['job'];key=job['transfer_id'];artifact=None;error=None
                try:
                    require(code==0 and not item.get('floor'),'Own worker exited or hit resource floor')
                    artifact=modules[job['method_key']].verify_job(job,plan['native_plans'][job['method_key']])
                except Exception as exc:error=repr(exc);failed[key]=error;stop=True
                if artifact:accepted[key]=artifact
                write(RUN/'methods'/job['method_key']/'exits'/(job['id']+'.json'),{'id':key,'returncode':code,'accepted':artifact is not None,'artifact':artifact,'error':error,'finished_at':now()},True)
            if not stop:
                for gpu in GPUS:
                    if gpu in active or not pending:continue
                    row=board(gpu)
                    if not idle(row):continue
                    job=pending[0];key=job['transfer_id']
                    card=flocks.take_card(gpu,row['uuid'],key,'pi05-pro-static1016')
                    if card is None:continue
                    legacy=flocks._try_lock(RUNTIME/'resource-leases'/HOST/f'gpu-{gpu}.lock',key,{'stage':'pi05-pro-static1016','gpu':gpu})
                    if legacy is None:card.release();continue
                    if not idle(board(gpu)):legacy.release();card.release();continue
                    require(not Path(job['output']).exists(),'Refusing preexisting output '+job['output'])
                    logpath=RUN/'methods'/job['method_key']/'logs'/(job['id']+'.log');logpath.parent.mkdir(parents=True,exist_ok=True);log=logpath.open('x')
                    env=modules[job['method_key']].environment();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),PYTHONDONTWRITEBYTECODE='1')
                    try:child=subprocess.Popen(job['command'],cwd=WORK,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(card.handle,legacy.handle,fence.handle))
                    except BaseException:log.close();legacy.release();card.release();raise
                    write(RUN/'methods'/job['method_key']/'launches'/(job['id']+'.json'),{'id':key,'pid':child.pid,'gpu':gpu,'uuid':row['uuid'],'admission':row,'command':job['command'],'model_sha256':plan['sources'][job['method_key']]['model_sha256'],'plan_sha256':sha(RUN/'plan.json'),'started_at':now()},True)
                    active[gpu]={'job':job,'child':child,'card':card,'legacy':legacy,'log':log};pending.pop(0)
            write(RUN/'state.json',{'accepted':accepted,'failed':failed,'active':{str(g):{'id':x['job']['transfer_id'],'pid':x['child'].pid} for g,x in active.items()},'pending':[x['transfer_id'] for x in pending],'stopped':stop,'updated_at':now()})
            if stop and not active:break
            if pending or active:time.sleep(15)
    finally:
        if active:
            # Unexpected supervisor errors must not release ownership while its
            # already-launched children still run. Drain only these children;
            # do not start another job or signal somebody else's process.
            write(RUN/'SUPERVISOR-DRAIN.json',{'active':{str(g):{'id':x['job']['transfer_id'],'pid':x['child'].pid} for g,x in active.items()},'created_at':now()})
            for gpu,item in list(active.items()):
                code=item['child'].wait();item['log'].close();job=item['job'];key=job['transfer_id']
                artifact=None;error=None
                try:
                    require(code==0 and not item.get('floor'),'Own worker failed during supervisor drain')
                    artifact=modules[job['method_key']].verify_job(job,plan['native_plans'][job['method_key']]);accepted[key]=artifact
                except Exception as exc:error=repr(exc);failed[key]=error
                write(RUN/'methods'/job['method_key']/'exits'/(job['id']+'.json'),{'id':key,'returncode':code,'accepted':artifact is not None,'artifact':artifact,'error':error,'supervisor_drained':True,'finished_at':now()},True)
                item['legacy'].release();item['card'].release();active.pop(gpu)
            stop=True
        if not active:
            write(RUN/'queue-ended.json',{'complete':len(accepted)==960 and not failed,'accepted':accepted,'failed':failed,'pending':[x['transfer_id'] for x in pending],'episodes':10*len(accepted),'stopped':stop,'ended_at':now()})
        fence.release()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','validate','arm','run'));args=parser.parse_args()
    if args.action=='prepare':result=prepare()
    elif args.action=='validate':p=validate();result={'status':'PASS_CPU_ONLY','jobs':len(p['jobs']),'episodes':p['episodes'],'plan_sha256':sha(RUN/'plan.json')}
    elif args.action=='arm':result=arm()
    else:run();return
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
