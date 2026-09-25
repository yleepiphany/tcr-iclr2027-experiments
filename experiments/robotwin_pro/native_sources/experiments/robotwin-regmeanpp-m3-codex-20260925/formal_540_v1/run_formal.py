#!/usr/bin/env python3
"""Single-use nine-job formal queue; no resampling, retries or process signals."""
from __future__ import annotations
import argparse,json,os,socket,subprocess,time
from pathlib import Path
import contract as c

def admissible(row):
    return row['gpu'] in c.GPUS and row['uuid']==c.GPUS[row['gpu']] and row['free_mib']>=40960 and row['used_mib']<=64 and row['utilization']==0 and not row['compute_pids']

def take_leases(flocks,gpu):
    held=[]
    try:
        for take in (
            lambda:flocks.take_card(gpu,c.GPUS[gpu],'robotwin-regmeanpp-m3-formal','formal'),
            lambda:flocks._try_lock(c.RUNTIME/'resource-leases'/c.HOST/f'gpu-{gpu}.lock','robotwin-regmeanpp-m3-formal',{'stage':'formal'}),
            lambda:flocks._try_lock(c.RUNTIME/'resource-leases'/f'{c.HOST}-gpu-{gpu}.lock','robotwin-regmeanpp-m3-formal',{'stage':'formal'})):
            lock=take()
            if lock is None:c.leases.close_own(held);return None
            held.append(lock)
        return held
    except BaseException:c.leases.close_own(held);raise

def write_state(root,pending,active,completed,failed):
    target=root/'state.json';tmp=root/f'.state-{os.getpid()}.json'
    tmp.write_text(json.dumps({'updated_at':c.now(),'pending':[j['id'] for j in pending],
        'active':[{'job':x['job']['id'],'gpu':x['gpu'],'pid':x['process'].pid} for x in active],
        'accepted':list(completed),'failed':failed},indent=2)+'\n');tmp.replace(target)

def run(planfile,permit_path,execute):
    if not execute or socket.gethostname()!=c.HOST:raise ValueError('Explicit execution on approved host required')
    plan,digest=c.validate(planfile,full_model=True);permit_path=Path(permit_path).resolve()
    permit=c.read(permit_path);gpus=c.validate_permit(permit,plan,digest);permit_sha=c.sha(permit_path);root=Path(plan['run'])
    flocks=c.load('_robotwin_formal_queue_flocks',c.FLOCK_FILE);guard=c.load('_robotwin_formal_queue_board',c.ROOT/'run_regmeanpp_m3.py')
    owner=flocks._try_lock(root/'supervisor.lock','robotwin-regmeanpp-m3-formal',{'stage':'formal-queue'})
    if owner is None:raise RuntimeError('Formal queue already owned')
    active=[];pending=list(plan['jobs']);completed={};failed={};stop=False;began=time.monotonic()
    try:
        if (root/'STARTED.json').exists() or (root/'jobs').exists():raise FileExistsError('Queue is single-use; no automatic retries/resume')
        c.write(root/'STARTED.json',{'pid':os.getpid(),'host':c.HOST,'started_at':c.now(),'plan_sha256':digest,
            'model_sha256':c.MODEL_SHA,'permit_sha256':permit_sha,'gpus':gpus,'expected_jobs':9,'expected_episodes':540,'signals_sent':0})
        while pending or active:
            if c.sha(permit_path)!=permit_sha:stop=True;failed['permit']='External permit changed during execution'
            for item in list(active):
                code=item['process'].poll()
                if code is None:continue
                job=item['job'];jobroot=root/'jobs'/job['id']
                c.write(jobroot/'CHILD-EXIT.json',{'pid':item['process'].pid,'exit_code':code,'exited_at':c.now()})
                item['log'].close();c.leases.close_own(item['held']);active.remove(item)
                if code==0:
                    try:
                        receipt=c.read(jobroot/'ACCEPTED.json')
                        if receipt['status']!='PASS' or receipt['plan_sha256']!=digest or receipt['job_id']!=job['id'] or receipt['model_sha256']!=c.MODEL_SHA or receipt['episodes']!=60:
                            raise ValueError('Accepted native job binding mismatch')
                        completed[job['id']]=receipt
                    except BaseException as exc:failed[job['id']]=repr(exc);stop=True
                else:failed[job['id']]=f'worker exit {code}; no retry';stop=True
            if not stop and pending:
                if time.monotonic()-began>plan['maximum_wait_seconds']:stop=True;failed['deadline']='Queue wait deadline reached'
                else:
                    for gpu in gpus:
                        if not pending or any(x['gpu']==gpu for x in active):continue
                        row=guard.board(gpu)
                        if not admissible(row):continue
                        held=take_leases(flocks,gpu)
                        if held is None:continue
                        if not admissible(guard.board(gpu)):c.leases.close_own(held);continue
                        job=pending.pop(0);jobroot=root/'jobs'/job['id'];process=None;log=None
                        try:
                            jobroot.mkdir(parents=True,exist_ok=False)
                            grant=jobroot/'LEASE-HANDOFF.json'
                            c.write(grant,{'plan_sha256':digest,'permit_sha256':permit_sha,'job_id':job['id'],'model_sha256':c.MODEL_SHA,
                                'host':c.HOST,'gpu':gpu,'uuid':c.GPUS[gpu],'parent_pid':os.getpid(),'leases':c.leases.lease_records(held)})
                            cmd=[plan['python'],str(c.HERE/'worker.py'),'--plan',str(Path(planfile).resolve()),'--permit',str(permit_path),
                                '--grant',str(grant),'--grant-sha256',c.sha(grant),'--job-id',job['id'],'--execute']
                            env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None);env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1'
                            log=(jobroot/'native.log').open('x')
                            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,
                                pass_fds=tuple(x.handle for x in held))
                            item={'job':job,'gpu':gpu,'held':held,'process':process,'log':log};active.append(item)
                            try:ticks=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
                            except FileNotFoundError:ticks=None
                            c.write(jobroot/'CHILD-STARTED.json',{'pid':process.pid,'start_ticks':ticks,'started_at':c.now(),
                                'command':cmd,'plan_sha256':digest,'gpu':gpu,'uuid':c.GPUS[gpu],'inherited_leases':3,'maximum_attempts':1})
                        except BaseException as exc:
                            failed[job['id']]=repr(exc);stop=True
                            if process is None:
                                if log:log.close()
                                c.leases.close_own(held)
                            break
            write_state(root,pending,active,completed,failed)
            if stop and not active:break
            if pending or active:time.sleep(plan['poll_seconds'])
        if failed:
            c.write(root/'FAILED.json',{'status':'FAILED','at':c.now(),'failed':failed,'accepted_jobs':list(completed),
                'unstarted_jobs':[j['id'] for j in pending],'automatic_retry':False,'signals_sent':0})
            raise RuntimeError('Formal queue stopped after failure; preserved accepted/partial outputs')
        summary=c.aggregate(list(completed.values()))
        c.write(root/'AGGREGATE.json',{'status':'PASS','formal_result':True,'method':c.METHOD,'plan_sha256':digest,
            'model_sha256':c.MODEL_SHA,'model_accepted_sha256':c.ACCEPTED_SHA,'receipts':[c.bind(root/'jobs'/j['id']/'ACCEPTED.json') for j in plan['jobs']],
            **summary,'completed_at':c.now()})
        c.write(root/'COMPLETE.json',{'status':'PASS','jobs':9,'episodes':540,'aggregate':c.bind(root/'AGGREGATE.json'),'signals_sent':0})
    except BaseException as exc:
        if (root/'STARTED.json').exists() and not (root/'FAILED.json').exists():c.write(root/'FAILED.json',{'status':'FAILED','at':c.now(),
            'error':repr(exc),'automatic_retry':False,'running_children_retain_inherited_leases':bool(active),'signals_sent':0})
        raise
    finally:
        for item in active:
            item['log'].close();c.leases.close_own(item['held'])
        owner.release()

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--run',type=Path,required=True)
    a=sub.add_parser('validate');a.add_argument('--plan',type=Path,required=True)
    a=sub.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--permit',type=Path,required=True);a.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.command=='prepare':
        plan=c.prepare(a.run);print(json.dumps({'status':'CPU_PREPARED','plan':str(Path(plan['run'])/'plan.json'),'plan_sha256':c.sha(Path(plan['run'])/'plan.json'),'gpu_used':False}))
    elif a.command=='validate':c.validate(a.plan,full_model=True);print(json.dumps({'status':'PASS','gpu_used':False}))
    else:run(a.plan,a.permit,a.execute)
if __name__=='__main__':main()
