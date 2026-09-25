#!/usr/bin/env python3
"""One separately permitted M3 materialization with continuous resource leases."""
from __future__ import annotations
import argparse,importlib.util,json,os,socket,subprocess,time
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
v2=load('_robotwin_m3_continuous_leases',ROOT/'native_smoke_supervisor_v2/supervisor.py')
CORE=v2.CORE;CORE_SHA=v2.CORE_SHA;HOST=v2.HOST;GPUS=v2.GPUS;PY=v2.PY
RUNTIME=v2.RUNTIME;FLOCK_FILE=v2.FLOCK_FILE
sha=v2.sha;read=v2.read;write=v2.write;now=v2.now
SMOKE_SHA='454a48b6e639020a0ec93604831cc2bb1d001cb076f29800c168197b2ca85e91'
PROTOCOL=RUNTIME/'experiments/iclr2027-table5-20260910/evaluation-queues/robotwin-three-expert-formal-v2/protocol.json'
PROTOCOL_SHA='ed47fc5340cfb53a4a1c497a08ef3aca984c0ad50182748ac3df3ca7e060d0de'
REPO=v2.v1.REPO

def sources():
    paths=list(HERE.glob('*.py'))+[ROOT/'native_smoke_supervisor_v2/supervisor.py',ROOT/'native_smoke_supervisor_v1/supervisor.py',
        ROOT/'run_regmeanpp_m3.py',FLOCK_FILE,REPO/'scripts/watch_robotwin_model_soups_formal_v1.py',
        REPO/'scripts/run_iclr2027_robotwin_checkpoint_development.py']
    return {str(p):sha(p) for p in sorted(paths)}

def verify_smoke():
    if sha(CORE/'native-smoke.json')!=SMOKE_SHA:raise ValueError('Accepted native smoke SHA differs')
    return v2.v1.check_result(CORE,CORE_SHA)

def no_build_attempt(core=CORE):
    for name in ('materialize-PERMIT-CONSUMED.json','materialize-FAILED.json','BUILD-COMPLETE.json','build-progress.jsonl','checkpoint'):
        if (Path(core)/name).exists():raise ValueError('Materialize already claimed/started/completed; no automatic retry')

def freeze_formal_protocol():
    if sha(PROTOCOL)!=PROTOCOL_SHA:raise ValueError('RoboTwin formal protocol changed')
    p=read(PROTOCOL);files={str(PROTOCOL):PROTOCOL_SHA};jobs=[]
    if p['episodes_total']!=540 or p['repeats']!=3 or len(p['jobs'])!=9:raise ValueError('Formal budget changed')
    for ref in [*p['jobs'],*p['reset_banks']]:
        if sha(ref['path'])!=ref['sha256']:raise ValueError('Formal job/reset source changed')
        files[ref['path']]=ref['sha256']
    for ref in p['jobs']:
        job=read(ref['path'])
        if job['episodes']!=60 or len(job['tasks'])!=10 or any(len(t['seeds'])!=6 for t in job['tasks']):raise ValueError('Formal job budget changed')
        jobs.append({'group':job['group'],'repeat':job['repeat'],'source_job':ref,'tasks':job['tasks'],
            'reset_bank':job['reset_bank'],'episodes':60,'full_native_horizon':True,'action_mode':'joint','condition':'demo_clean'})
    if {(j['group'],j['repeat']) for j in jobs}!={(g,r) for g in ('coordination','receptacle','precision') for r in (1,2,3)}:
        raise ValueError('Formal group/repeat coverage differs')
    return {'status':'BLOCKED_UNTIL_ACCEPTED_MODEL_SHA_AND_ISOLATED_DENSE_RUNNER','source_files_sha256':files,
        'jobs':jobs,'episodes':540,'model_sha256':None,'checkpoint':str(CORE/'checkpoint'),'autolaunch':False,
        'reusable_engine':str(REPO/'scripts/run_iclr2027_robotwin_checkpoint_development.py'),
        'reference_dense_queue':str(REPO/'scripts/watch_robotwin_model_soups_formal_v1.py'),
        'next_step':'Clone dense queue into an independent RegMean++ adapter; bind all jobs to MODEL-ACCEPTED.json SHA without changing tasks/seeds/reset banks.'}

def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh materialize supervisor root required')
    if sha(CORE/'plan.json')!=CORE_SHA:raise ValueError('Original M3 plan changed')
    verify_smoke();no_build_attempt()
    plan={'schema':'robotwin_regmeanpp_m3_materialize_inherited_leases_v1','status':'CPU_PREPARED_REQUIRES_MATERIALIZE_PERMIT',
        'created_at':now(),'output':str(output),'core_run':str(CORE),'core_plan_sha256':CORE_SHA,'native_smoke_sha256':SMOKE_SHA,
        'host':HOST,'allowed_gpus':GPUS,'python':str(PY),'child_guard':str(HERE/'child_guard.py'),'sources_sha256':sources(),
        'poll_seconds':15,'maximum_wait_seconds':86400,'stage':'materialize','maximum_child_launches':1,
        'automatic_retry':False,'automatic_formal_evaluation':False,'original_kernel_and_67_stages_unchanged':True,
        'resource_gate':{'minimum_free_mib':71680,'maximum_used_mib':64,'utilization':0,'compute_pids':[],
            'allocator_fraction':.70,'minimum_runtime_free_mib':12288,'runtime_poll_seconds':5,
            'uuid_and_legacy_flocks':True,'build_slot':True,'continuous_pass_fds':True},
        'acceptance':{'saved_model_sha':True,'manifest_scope_and_rows':418,'saved_and_loaded_tensor_bitwise_checks':813,
            'native_reload_static_replica0_observations':3,'native_velocity_shape':[1,50,32],
            'native_velocity_finite':True,'formal_performance_not_claimed':True},
        'formal_binding':freeze_formal_protocol(),'gpu_used':False,'signals_sent':0}
    write(output/'plan.json',plan)
    return {'status':plan['status'],'plan':str(output/'plan.json'),'plan_sha256':sha(output/'plan.json'),'gpu_used':False}

def validate_plan(file):
    p=read(file)
    if p['sources_sha256']!=sources() or p['core_plan_sha256']!=CORE_SHA or sha(CORE/'plan.json')!=CORE_SHA or p['native_smoke_sha256']!=SMOKE_SHA:
        raise ValueError('Frozen source/plan/smoke identity drift')
    if Path(file).resolve()!=Path(p['output'])/'plan.json' or Path(p['core_run'])!=CORE or Path(p['python'])!=PY or Path(p['child_guard'])!=HERE/'child_guard.py':
        raise ValueError('Unexpected executable/output')
    if p['stage']!='materialize' or p['maximum_child_launches']!=1 or p['automatic_retry'] or p['automatic_formal_evaluation']:
        raise ValueError('Only one independent materialization is allowed')
    if p['host']!=HOST or {int(g):u for g,u in p['allowed_gpus'].items()}!=GPUS:raise ValueError('Host/GPU allowlist changed')
    if p['poll_seconds']!=15 or p['maximum_wait_seconds']!=86400:raise ValueError('Wait budget changed')
    verify_smoke()
    for path,digest in read(CORE/'plan.json')['implementations'].items():
        if sha(path)!=digest:raise ValueError('Original model implementation changed')
    for path,digest in p['formal_binding']['source_files_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen formal source changed')
    return p

def validate_permit(permit,file,gpu):
    guard=load('_robotwin_m3_original_guard',ROOT/'run_regmeanpp_m3.py')
    if gpu not in GPUS:raise ValueError('Forbidden GPU')
    guard.check_permit(permit,'materialize',CORE_SHA,CORE,gpu,GPUS[gpu],HOST)
    if permit.get('supervisor_plan_sha256')!=sha(file) or permit.get('native_smoke_sha256')!=SMOKE_SHA:
        raise ValueError('New materialize permit must bind supervisor plan and accepted smoke SHA')
    return guard

def acquire_all(flocks,gpu,uuid):
    held=[]
    try:
        for take in (
            lambda:flocks.take_card(gpu,uuid,'robotwin-regmeanpp-m3-materialize','materialize'),
            lambda:flocks._try_lock(RUNTIME/'resource-leases'/HOST/f'gpu-{gpu}.lock','robotwin-regmeanpp-m3-materialize',{'stage':'materialize'}),
            lambda:flocks.take_build_slot('robotwin-regmeanpp-m3-materialize')):
            lock=take()
            if lock is None:v2.close_own(held);return None
            held.append(lock)
        return held
    except BaseException:v2.close_own(held);raise

def run(file,permit_path,execute):
    if not execute:raise ValueError('Explicit execute required')
    plan=validate_plan(file);root=Path(plan['output']);permit_path=Path(permit_path).resolve()
    if socket.gethostname()!=HOST:raise ValueError('Wrong host')
    permit=read(permit_path);permit_sha=sha(permit_path);gpu=permit.get('gpu');guard=validate_permit(permit,file,gpu)
    flocks=load('_robotwin_m3_build_flocks',FLOCK_FILE);owner=flocks._try_lock(root/'supervisor.lock','robotwin-regmeanpp-m3-materialize',{'stage':'waiting'})
    if owner is None:raise RuntimeError('Supervisor already owned')
    held=[];child=None
    try:
        if (root/'STARTED.json').exists():raise FileExistsError('Single-use supervisor already started')
        def check():
            if sha(permit_path)!=permit_sha:raise ValueError('Permit changed while waiting')
            no_build_attempt();verify_smoke()
        check();write(root/'STARTED.json',{'pid':os.getpid(),'started_at':now(),'gpu':gpu,'uuid':GPUS[gpu],
            'supervisor_plan_sha256':sha(file),'core_plan_sha256':CORE_SHA,'permit_sha256':permit_sha,'signals_sent':0})
        held,admission=v2.wait_and_hold(gpu,GPUS[gpu],board_fn=guard.board,admissible_fn=guard.admissible,
            acquire_fn=lambda g,u:acquire_all(flocks,g,u),check_fn=check,timeout=plan['maximum_wait_seconds'],poll=plan['poll_seconds'])
        check();validate_plan(file);grant=root/'LEASE-HANDOFF.json'
        write(grant,{'schema':'robotwin_m3_materialize_inherited_flock_handoff_v1','supervisor_plan_sha256':sha(file),
            'core_plan_sha256':CORE_SHA,'native_smoke_sha256':SMOKE_SHA,'permit_sha256':permit_sha,'host':HOST,
            'gpu':gpu,'uuid':GPUS[gpu],'parent_pid':os.getpid(),'leases':v2.lease_records(held),'admission':admission})
        cmd=[str(PY),str(HERE/'child_guard.py'),'--supervisor-plan',str(Path(file).resolve()),'--permit',str(permit_path),
             '--grant',str(grant),'--grant-sha256',sha(grant),'--execute']
        env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None);env['PYTHONDONTWRITEBYTECODE']='1'
        with (root/'materialize.log').open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,pass_fds=tuple(x.handle for x in held))
            try:ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            except FileNotFoundError:ticks=None
            write(root/'CHILD-STARTED.json',{'pid':child.pid,'start_ticks':ticks,'command':cmd,'started_at':now(),
                'inherited_leases':3,'parent_retains_leases_until_child_exit':True,'maximum_launches':1})
            code=child.wait()
        write(root/'CHILD-EXIT.json',{'pid':child.pid,'exit_code':code,'exited_at':now()})
        if code!=0:raise RuntimeError(f'Materialize child exit {code}; preserve partial evidence, no retry')
        accepted=read(root/'MODEL-ACCEPTED.json')
        if accepted.get('status')!='PASS' or accepted.get('supervisor_plan_sha256')!=sha(file) or accepted.get('core_plan_sha256')!=CORE_SHA:
            raise ValueError('Accepted checkpoint identity differs')
        if sha(CORE/'checkpoint/model.safetensors')!=accepted['model_sha256']:raise ValueError('Accepted model bytes changed')
        write(root/'COMPLETE.json',{'status':'PASS','completed_at':now(),'model_accepted':str(root/'MODEL-ACCEPTED.json'),
            'model_accepted_sha256':sha(root/'MODEL-ACCEPTED.json'),'model_sha256':accepted['model_sha256'],
            'checkpoint':str(CORE/'checkpoint'),'formal_evaluation_started':False,'signals_sent':0})
    except BaseException as exc:
        if (root/'STARTED.json').exists() and not (root/'FAILED.json').exists():
            write(root/'FAILED.json',{'status':'FAILED','at':now(),'error':repr(exc),'automatic_retry':False,'signals_sent':0,
                'child_may_continue_with_inherited_leases':child is not None and child.poll() is None})
        raise
    finally:v2.close_own(held);owner.release()

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--permit',type=Path,required=True);a.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.command=='prepare':print(json.dumps(prepare(a.output)))
    else:run(a.plan,a.permit,a.execute)
if __name__=='__main__':main()
