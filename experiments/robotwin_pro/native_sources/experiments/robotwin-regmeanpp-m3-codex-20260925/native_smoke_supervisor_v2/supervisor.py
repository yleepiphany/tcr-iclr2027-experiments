#!/usr/bin/env python3
"""One-shot smoke recovery with uninterrupted inherited flock ownership."""
from __future__ import annotations
import argparse,fcntl,hashlib,importlib.util,json,os,socket,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
v1=load('_robotwin_m3_smoke_v1_helpers',ROOT/'native_smoke_supervisor_v1/supervisor.py')
CORE=v1.CORE;CORE_SHA=v1.CORE_SHA;HOST=v1.HOST;GPUS=v1.GPUS;PY=v1.PY
RUNTIME=v1.RUNTIME;FLOCK_FILE=v1.FLOCK_FILE
sha=v1.sha;read=v1.read;write=v1.write;now=v1.now
PRIOR=CORE.parent/'native-smoke-supervisor-v1'

def sources():
    paths=[HERE/'supervisor.py',HERE/'child_guard.py',HERE/'test_supervisor_cpu.py',
        ROOT/'native_smoke_supervisor_v1/supervisor.py',ROOT/'run_regmeanpp_m3.py',FLOCK_FILE]
    return {str(p):sha(p) for p in paths}

def verify_no_gpu_attempt():
    for name in ('smoke-PERMIT-CONSUMED.json','native-smoke.json','smoke-FAILED.json'):
        if (CORE/name).exists():raise ValueError('Core smoke already entered/finished/failed GPU; no recovery/retry allowed')

def prior_failure():
    names=('plan.json','STARTED.json','CHILD-STARTED.json','CHILD-EXIT.json','FAILED.json','native-smoke.log')
    records={str(PRIOR/n):sha(PRIOR/n) for n in names}
    started=read(PRIOR/'CHILD-STARTED.json');done=read(PRIOR/'CHILD-EXIT.json');log=(PRIOR/'native-smoke.log').read_text()
    if started['pid']!=3073204 or done['pid']!=3073204 or done['exit_code']!=1:raise ValueError('Unexpected prior failure identity')
    if 'line 44, in gpu_stage' not in log or not log.rstrip().endswith('RuntimeError: BLOCKED_RESOURCE_NOT_EMPTY_OR_LOW_MEMORY'):
        raise ValueError('Prior failure was not the inspected pre-CUDA admission failure')
    verify_no_gpu_attempt()
    # Never mistake a live/reused PID for the exited prior child.
    stat=Path('/proc/3073204/stat')
    if stat.exists() and stat.read_text().rsplit(')',1)[1].split()[19]==str(started['start_ticks']):
        raise ValueError('Prior child identity is still live')
    return records

def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh v2 supervisor directory required')
    if sha(CORE/'plan.json')!=CORE_SHA:raise ValueError('Original model/algorithm plan changed')
    evidence=prior_failure()
    plan={'schema':'robotwin_regmeanpp_m3_one_shot_smoke_inherited_leases_v2','status':'CPU_PREPARED_REQUIRES_NEW_EXTERNAL_PERMIT',
        'created_at':now(),'output':str(output),'core_run':str(CORE),'core_plan_sha256':CORE_SHA,
        'host':HOST,'allowed_gpus':GPUS,'python':str(PY),'child_guard':str(HERE/'child_guard.py'),
        'sources_sha256':sources(),'prior_pre_cuda_failure_files_sha256':evidence,
        'poll_seconds':15,'maximum_wait_seconds':86400,'maximum_child_launches':1,'stage':'smoke',
        'retry_after_gpu_entry':False,'automatic_retry':False,'materialize':False,
        'permit_requires_supervisor_plan_sha256':True,
        'lease_transfer':'same open-file descriptions inherited with subprocess pass_fds; supervisor keeps all three until child exit',
        'release_rule':'close own descriptors only, never LOCK_UN; inherited child descriptors preserve locks if supervisor disappears',
        'resource_gate':{'minimum_free_mib':71680,'maximum_used_mib':64,'utilization':0,'compute_pids':[],
            'uuid_and_legacy_flocks':True,'build_slot':True,'full_model_hashes_and_fresh_card_check_under_held_leases':True},
        'gpu_used':False,'signals_sent':0}
    write(output/'plan.json',plan)
    return {'status':plan['status'],'plan':str(output/'plan.json'),'plan_sha256':sha(output/'plan.json'),'gpu_used':False}

def validate_plan(file):
    plan=read(file)
    if plan['sources_sha256']!=sources() or plan['core_plan_sha256']!=CORE_SHA or sha(CORE/'plan.json')!=CORE_SHA:
        raise ValueError('Frozen original/v2 source or plan changed')
    if Path(plan['output'])/'plan.json'!=Path(file).resolve() or Path(plan['core_run'])!=CORE:
        raise ValueError('Output/core identity differs')
    if plan['stage']!='smoke' or plan['maximum_child_launches']!=1 or plan['retry_after_gpu_entry'] or plan['automatic_retry'] or plan['materialize']:
        raise ValueError('Only one smoke is authorized')
    if plan['host']!=HOST or {int(g):u for g,u in plan['allowed_gpus'].items()}!=GPUS:
        raise ValueError('Host/GPU allowlist differs')
    if Path(plan['python'])!=PY or Path(plan['child_guard'])!=HERE/'child_guard.py':raise ValueError('Unexpected child executable')
    if plan['poll_seconds']!=15 or plan['maximum_wait_seconds']!=86400:raise ValueError('Wait budget differs')
    for path,digest in plan['prior_pre_cuda_failure_files_sha256'].items():
        if sha(path)!=digest:raise ValueError('Prior failure evidence changed')
    for path,digest in read(CORE/'plan.json')['implementations'].items():
        if sha(path)!=digest:raise ValueError('Original algorithm dependency changed')
    return plan

def validate_permit(permit,file,gpu):
    guard=load('_robotwin_m3_original_guard',ROOT/'run_regmeanpp_m3.py')
    if gpu not in GPUS:raise ValueError('Forbidden GPU')
    guard.check_permit(permit,'smoke',CORE_SHA,CORE,gpu,GPUS[gpu],HOST)
    if permit.get('supervisor_plan_sha256')!=sha(file):raise ValueError('New v2-bound external permit required; v1 permit cannot be reused')
    return guard

def close_own(held):
    """Do not call flock(LOCK_UN): that would also unlock an inherited OFD."""
    for lock in reversed(held):
        if not getattr(lock,'_released',False):
            os.close(lock.handle);lock._released=True

def acquire_all(flocks,gpu,uuid):
    held=[]
    try:
        for take in (
            lambda:flocks.take_card(gpu,uuid,'robotwin-regmeanpp-m3-smoke-v2','smoke-handoff'),
            lambda:flocks._try_lock(RUNTIME/'resource-leases'/HOST/f'gpu-{gpu}.lock','robotwin-regmeanpp-m3-smoke-v2',{'stage':'smoke-handoff'}),
            lambda:flocks.take_build_slot('robotwin-regmeanpp-m3-smoke-v2')):
            lock=take()
            if lock is None:close_own(held);return None
            held.append(lock)
        return held
    except BaseException:close_own(held);raise

def wait_and_hold(gpu,uuid,*,board_fn,admissible_fn,acquire_fn,check_fn,sleep_fn=time.sleep,clock=time.monotonic,timeout=86400,poll=15):
    start=clock();checks=0
    while True:
        check_fn();row=board_fn(gpu);checks+=1
        if row['uuid']!=uuid:raise ValueError('GPU UUID changed')
        if admissible_fn(row,GPUS):
            held=acquire_fn(gpu,uuid)
            if held is not None:
                try:
                    row=board_fn(gpu);checks+=1
                    if row['uuid']!=uuid:raise ValueError('GPU UUID changed under leases')
                    if admissible_fn(row,GPUS):
                        return held,{'board':row,'wait_seconds':clock()-start,'board_checks':checks,'leases_retained':True}
                except BaseException:close_own(held);raise
                close_own(held)
        if clock()-start>=timeout:raise TimeoutError('No safe card before deadline')
        sleep_fn(min(poll,timeout-(clock()-start)))

def lease_records(held):
    result=[]
    for lock in held:
        st=os.fstat(lock.handle)
        result.append({'fd':lock.handle,'path':str(lock.path),'device':st.st_dev,'inode':st.st_ino})
    return result

def verify_inherited_leases(records,expected_paths):
    if len(records)!=3 or len({r['fd'] for r in records})!=3 or {r['path'] for r in records}!=set(expected_paths):
        raise ValueError('Expected distinct UUID/index/build-slot lease descriptors')
    for row in records:
        fd=row['fd'];actual=os.fstat(fd);pathstat=Path(row['path']).stat()
        if (actual.st_dev,actual.st_ino)!=(row['device'],row['inode']) or (actual.st_dev,actual.st_ino)!=(pathstat.st_dev,pathstat.st_ino):
            raise ValueError('Inherited lease descriptor/path identity mismatch')
        # A new file description must be excluded, and the inherited one must own the lock.
        probe=os.open(row['path'],os.O_RDWR)
        try:
            try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:
                fcntl.flock(probe,fcntl.LOCK_UN)
                raise ValueError('Lease lost before handoff')
        finally:os.close(probe)
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise ValueError('Inherited descriptor does not own lease') from e

def run(file,permit_path,execute):
    if not execute:raise ValueError('Explicit execute required')
    plan=validate_plan(file);root=Path(plan['output']);permit_path=Path(permit_path).resolve()
    if socket.gethostname()!=HOST:raise ValueError('Wrong host')
    permit=read(permit_path);permit_sha=sha(permit_path);gpu=permit.get('gpu');guard=validate_permit(permit,file,gpu)
    flocks=load('_robotwin_m3_v2_flocks',FLOCK_FILE)
    owner=flocks._try_lock(root/'supervisor.lock','robotwin-regmeanpp-m3-smoke-v2',{'stage':'waiting'})
    if owner is None:raise RuntimeError('Supervisor already owned')
    held=[];child=None
    try:
        if (root/'STARTED.json').exists():raise FileExistsError('V2 is single-use, no automatic retry')
        def check():
            if sha(permit_path)!=permit_sha or sha(CORE/'plan.json')!=CORE_SHA:raise ValueError('Permit/core drift while waiting')
            verify_no_gpu_attempt()
        check();write(root/'STARTED.json',{'started_at':now(),'pid':os.getpid(),'gpu':gpu,'uuid':GPUS[gpu],
            'supervisor_plan_sha256':sha(file),'core_plan_sha256':CORE_SHA,'permit_sha256':permit_sha,'signals_sent':0})
        held,admission=wait_and_hold(gpu,GPUS[gpu],board_fn=guard.board,admissible_fn=guard.admissible,
            acquire_fn=lambda g,u:acquire_all(flocks,g,u),check_fn=check,timeout=plan['maximum_wait_seconds'],poll=plan['poll_seconds'])
        check();validate_plan(file)
        grant=root/'LEASE-HANDOFF.json'
        write(grant,{'schema':'robotwin_m3_inherited_flock_handoff_v2','supervisor_plan_sha256':sha(file),'core_plan_sha256':CORE_SHA,
            'permit_sha256':permit_sha,'host':HOST,'gpu':gpu,'uuid':GPUS[gpu],'parent_pid':os.getpid(),
            'leases':lease_records(held),'admission':admission})
        cmd=[str(PY),str(HERE/'child_guard.py'),'--supervisor-plan',str(Path(file).resolve()),'--permit',str(permit_path),
             '--grant',str(grant),'--grant-sha256',sha(grant),'--execute']
        env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None);env['PYTHONDONTWRITEBYTECODE']='1'
        with (root/'native-smoke.log').open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,
                pass_fds=tuple(lock.handle for lock in held))
            try:ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            except FileNotFoundError:ticks=None
            write(root/'CHILD-STARTED.json',{'pid':child.pid,'start_ticks':ticks,'command':cmd,'started_at':now(),
                'lease_fds_inherited':True,'parent_retains_leases_until_child_exit':True,'maximum_launches':1})
            code=child.wait()
        write(root/'CHILD-EXIT.json',{'pid':child.pid,'exit_code':code,'exited_at':now()})
        if code!=0:raise RuntimeError(f'Native smoke child exit {code}; preserve evidence and stop, no retry')
        result=v1.check_result(CORE,CORE_SHA)
        write(root/'COMPLETE.json',{'status':'PASS','completed_at':now(),'core_plan_sha256':CORE_SHA,
            'native_smoke_receipt':str(CORE/'native-smoke.json'),'native_smoke_receipt_sha256':sha(CORE/'native-smoke.json'),
            'parity_reports':result['reports'],'child_launches':1,'materializer_started':False,'signals_sent':0})
    except BaseException as exc:
        if (root/'STARTED.json').exists() and not (root/'FAILED.json').exists():
            write(root/'FAILED.json',{'status':'FAILED','at':now(),'error':repr(exc),'automatic_retry':False,'signals_sent':0,
                'child_may_continue_with_inherited_leases':child is not None and child.poll() is None})
        raise
    finally:
        close_own(held)  # Safe even if child is alive: it retains the same OFDs.
        owner.release()

def main():
    p=argparse.ArgumentParser();subs=p.add_subparsers(dest='command',required=True)
    a=subs.add_parser('prepare');a.add_argument('--output',type=Path,required=True)
    a=subs.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--permit',type=Path,required=True);a.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.command=='prepare':print(json.dumps(prepare(a.output)))
    else:run(a.plan,a.permit,a.execute)
if __name__=='__main__':main()
