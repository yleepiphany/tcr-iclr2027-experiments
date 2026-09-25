#!/usr/bin/env python3
"""Durable one-shot native-smoke waiter; never builds, retries, or signals jobs."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,socket,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
WORK=Path('/mnt/workspace/Wilson/parameter-fusion')
RUNTIME=WORK/'vla-merge-runtime';REPO=WORK/'vla-merge'
CORE=RUNTIME/'experiments/robotwin-regmeanpp-m3-codex-20260925/cpu-plan-v1'
CORE_SHA='bb25730b256669eecd1180026f1339cbb6b52d16342d9e65750e521e1b39ad0b'
HOST='dsw-824375-57c745db88-n6tv9'
GPUS={1:'GPU-c18a3bf3-c47c-9a8a-75b0-5a0f37b33a65',4:'GPU-48baba1c-f0e7-93b7-4a30-b3c21d20e61a',
      5:'GPU-4ed28198-b742-ee3c-2acb-4183dc944c81',6:'GPU-da625da2-f504-7927-4025-907e876718aa'}
PY=WORK/'pi05_lora_finetune_v2_20260826/.venv/bin/python'
FLOCK_FILE=REPO/'experiments/claude-firstpass-cause-20260920/card_flock.py'

def sha(file):
    h=hashlib.sha256()
    with Path(file).open('rb') as f:
        while part:=f.read(8*1024**2):h.update(part)
    return h.hexdigest()
def read(file):return json.loads(Path(file).read_text())
def write(file,value):
    file=Path(file);file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sources():
    return {str(p.resolve()):sha(p) for p in (Path(__file__),Path(__file__).with_name('test_supervisor_cpu.py'),ROOT/'run_regmeanpp_m3.py',FLOCK_FILE)}

def command(plan,gpu,permit):
    if gpu not in GPUS:raise ValueError('Forbidden smoke GPU')
    return [plan['python'],plan['runner'],'--stage','smoke','--run',plan['core_run'],
        '--gpu',str(gpu),'--execute','--permit',str(Path(permit).resolve())]

def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh supervisor plan directory required')
    if sha(CORE/'plan.json')!=CORE_SHA:raise ValueError('Original M3 plan changed')
    if any((CORE/name).exists() for name in ('smoke-PERMIT-CONSUMED.json','native-smoke.json','smoke-FAILED.json')):
        raise ValueError('Original smoke already attempted')
    plan={'schema':'robotwin_regmeanpp_m3_one_shot_native_smoke_supervisor_v1','status':'CPU_PREPARED_REQUIRES_EXTERNAL_PERMIT',
        'created_at':now(),'output':str(output),'core_run':str(CORE),'core_plan_sha256':CORE_SHA,
        'host':HOST,'allowed_gpus':GPUS,'python':str(PY),'runner':str(ROOT/'run_regmeanpp_m3.py'),
        'sources_sha256':sources(),'poll_seconds':15,'maximum_wait_seconds':86400,
        'only_stage':'smoke','maximum_child_launches':1,'retry_after_failure':False,'materialize_after_success':False,
        'resource_gate':{'minimum_free_mib':71680,'maximum_used_mib':64,'utilization':0,'no_compute_pids':True,
            'uuid_and_legacy_flocks':True,'build_slot':True,'final_atomic_admission':'original guarded child runner'},
        'permit_rule':'External single-use smoke permit binds original plan SHA, core run, host, chosen GPU and UUID; never generated here',
        'race_policy':'Supervisor probes all leases then releases them; child independently reacquires all leases and rechecks. Any child rejection is a recorded single failure, never retried.',
        'gpu_used':False,'signals_sent':0}
    write(output/'plan.json',plan)
    return {'status':'CPU_PREPARED_REQUIRES_EXTERNAL_PERMIT','plan':str(output/'plan.json'),'plan_sha256':sha(output/'plan.json'),'gpu_used':False}

def validate_plan(file):
    plan=read(file)
    if plan['sources_sha256']!=sources() or plan['core_plan_sha256']!=CORE_SHA or sha(Path(plan['core_run'])/'plan.json')!=CORE_SHA:
        raise ValueError('Supervisor or original plan/source drift')
    if plan['only_stage']!='smoke' or plan['maximum_child_launches']!=1 or plan['retry_after_failure'] or plan['materialize_after_success']:
        raise ValueError('Only one native smoke is allowed')
    if Path(plan['runner']).resolve()!=ROOT/'run_regmeanpp_m3.py' or Path(plan['python'])!=PY or Path(plan['core_run'])!=CORE:
        raise ValueError('Unexpected executable or output target')
    if plan['host']!=HOST or {int(g):u for g,u in plan['allowed_gpus'].items()}!=GPUS:
        raise ValueError('Resource allowlist changed')
    if plan['poll_seconds']!=15 or plan['maximum_wait_seconds']!=86400:raise ValueError('Wait budget changed')
    if Path(file).resolve()!=Path(plan['output'])/'plan.json':raise ValueError('Supervisor output identity changed')
    # Avoid a second 37GB hash pass here; the child fully hashes the four model files.
    core=read(CORE/'plan.json')
    for path,digest in core['implementations'].items():
        if sha(path)!=digest:raise ValueError('Frozen materializer dependency changed')
    return plan

def probe_leases(flocks,gpu,uuid):
    """Availability probe only. The GPU child obtains final exclusive ownership."""
    held=[]
    try:
        for take in (
            lambda:flocks.take_card(gpu,uuid,'robotwin-regmeanpp-m3-smoke-probe','smoke-probe'),
            lambda:flocks._try_lock(RUNTIME/'resource-leases'/HOST/f'gpu-{gpu}.lock','robotwin-regmeanpp-m3-smoke-probe',{'stage':'smoke-probe'}),
            lambda:flocks.take_build_slot('robotwin-regmeanpp-m3-smoke-probe')):
            lock=take()
            if lock is None:return False
            held.append(lock)
        return True
    finally:
        for lock in reversed(held):lock.release()

def wait_for_admission(gpu,uuid,*,board_fn,admissible_fn,leases_fn,check_fn,sleep_fn=time.sleep,clock=time.monotonic,poll_seconds=15,maximum_wait_seconds=86400):
    start=clock();checks=0
    while True:
        check_fn();row=board_fn(gpu);checks+=1
        if row['uuid']!=uuid:raise ValueError('GPU identity changed while waiting')
        if admissible_fn(row,GPUS) and leases_fn(gpu,uuid):
            row=board_fn(gpu);checks+=1
            if row['uuid']!=uuid:raise ValueError('GPU identity changed after lease probe')
            if admissible_fn(row,GPUS):return {'board':row,'wait_seconds':clock()-start,'board_checks':checks}
        if clock()-start>=maximum_wait_seconds:raise TimeoutError('No admissible empty leased card before deadline')
        sleep_fn(min(poll_seconds,maximum_wait_seconds-(clock()-start)))

def check_result(core,plan_sha):
    result=read(Path(core)/'native-smoke.json')
    if result.get('status')!='PASS' or result.get('plan_sha256')!=plan_sha or result.get('formal_episodes')!=0:
        raise ValueError('Native smoke result identity/status differs')
    rows=result.get('reports',[])
    if len(rows)!=3 or {r.get('group') for r in rows}!={'coordination','receptacle','precision'}:
        raise ValueError('Native smoke M3 report coverage differs')
    for row in rows:
        if row.get('replica')!=0 or row.get('physical_timestep')!=1. or row.get('velocity_shape')!=[1,50,32] or row.get('sampled_inputs_bitwise')!=418 or row.get('native_velocity_bitwise') is not True:
            raise ValueError('Native graph parity evidence incomplete')
        if row.get('candidate_blocks_bitwise')!=['vision0','language0','action0']:
            raise ValueError('Candidate-block smoke coverage differs')
    return result

def run(file,permit_path,execute):
    if not execute:raise ValueError('Explicit --execute required')
    plan=validate_plan(file);root=Path(plan['output'])
    if socket.gethostname()!=HOST:raise ValueError('Wrong host')
    permit_path=Path(permit_path).resolve();permit=read(permit_path);permit_sha=sha(permit_path)
    gpu=permit.get('gpu')
    if gpu not in GPUS:raise ValueError('Permit GPU is outside allowlist')
    guard=load('_robotwin_regmeanpp_m3_smoke_guard',ROOT/'run_regmeanpp_m3.py')
    guard.check_permit(permit,'smoke',CORE_SHA,CORE,gpu,GPUS[gpu],HOST)
    flocks=load('_robotwin_regmeanpp_m3_smoke_flocks',FLOCK_FILE)
    owner=flocks._try_lock(root/'supervisor.lock','robotwin-regmeanpp-m3-smoke-supervisor',{'stage':'smoke-wait'})
    if owner is None:raise RuntimeError('Supervisor already owned')
    try:
        if (root/'STARTED.json').exists():raise FileExistsError('Supervisor is single-use; no automatic retry')
        def still_frozen():
            if sha(permit_path)!=permit_sha:raise ValueError('External permit changed while waiting')
            if sha(CORE/'plan.json')!=CORE_SHA:raise ValueError('Original plan changed while waiting')
            if any((CORE/n).exists() for n in ('smoke-PERMIT-CONSUMED.json','native-smoke.json','smoke-FAILED.json')):
                raise ValueError('Original smoke acquired by another owner or already attempted')
        still_frozen()
        write(root/'STARTED.json',{'started_at':now(),'supervisor_pid':os.getpid(),'host':HOST,'gpu':gpu,'uuid':GPUS[gpu],
            'supervisor_plan_sha256':sha(file),'core_plan_sha256':CORE_SHA,'permit_sha256':permit_sha,'stage':'waiting','signals_sent':0})
        admission=wait_for_admission(gpu,GPUS[gpu],board_fn=guard.board,admissible_fn=guard.admissible,
            leases_fn=lambda g,u:probe_leases(flocks,g,u),check_fn=still_frozen,
            poll_seconds=plan['poll_seconds'],maximum_wait_seconds=plan['maximum_wait_seconds'])
        still_frozen();validate_plan(file)
        cmd=command(plan,gpu,permit_path)
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['TOKENIZERS_PARALLELISM']='false'
        # Child decides CUDA visibility only after its own full admission checks.
        env.pop('CUDA_VISIBLE_DEVICES',None)
        with (root/'native-smoke.log').open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            except FileNotFoundError:start_ticks=None
            write(root/'CHILD-STARTED.json',{'started_at':now(),'pid':child.pid,'start_ticks':start_ticks,'command':cmd,
                'admission':admission,'maximum_launches':1,'retry_allowed':False})
            code=child.wait()
        write(root/'CHILD-EXIT.json',{'exited_at':now(),'pid':child.pid,'exit_code':code})
        if code!=0:raise RuntimeError(f'Native smoke child failed with exit {code}; no retry')
        result=check_result(CORE,CORE_SHA)
        write(root/'COMPLETE.json',{'status':'PASS','completed_at':now(),'core_plan_sha256':CORE_SHA,
            'native_smoke_receipt':str(CORE/'native-smoke.json'),'native_smoke_receipt_sha256':sha(CORE/'native-smoke.json'),
            'parity_reports':result['reports'],'child_launches':1,'materializer_started':False,'formal_episodes':0,'signals_sent':0})
    except BaseException as exc:
        if (root/'STARTED.json').exists() and not (root/'FAILED.json').exists():
            write(root/'FAILED.json',{'status':'FAILED','at':now(),'error':repr(exc),'automatic_retry':False,'signals_sent':0})
        raise
    finally:owner.release()

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('run');p.add_argument('--plan',required=True,type=Path);p.add_argument('--permit',required=True,type=Path);p.add_argument('--execute',action='store_true')
    a=parser.parse_args()
    if a.command=='prepare':print(json.dumps(prepare(a.output)))
    else:run(a.plan,a.permit,a.execute)
if __name__=='__main__':main()
