#!/usr/bin/env python3
"""One native group/repeat, no retries; consumes inherited triple leases."""
from __future__ import annotations
import argparse,os,socket,time
from pathlib import Path
import contract as c

def lease_paths(gpu):
    flocks=c.load('_robotwin_formal_flocks',c.FLOCK_FILE)
    return {str(flocks.card_path(gpu,c.GPUS[gpu])),
        str(c.RUNTIME/'resource-leases'/c.HOST/f'gpu-{gpu}.lock'),
        str(c.RUNTIME/'resource-leases'/f'{c.HOST}-gpu-{gpu}.lock')}

def no_previous_attempt(jobroot,output):
    if Path(output).exists() or any((Path(jobroot)/n).exists() for n in ('STARTED.json','FAILED.json','ACCEPTED.json')):
        raise FileExistsError('Job already attempted; automatic retry/resume forbidden')

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--permit',type=Path,required=True)
    p.add_argument('--grant',type=Path,required=True);p.add_argument('--grant-sha256',required=True);p.add_argument('--job-id',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute or socket.gethostname()!=c.HOST:raise ValueError('Explicit execute and correct host required')
    plan,digest=c.validate(a.plan,full_model=True);permit=c.read(a.permit);gpus=c.validate_permit(permit,plan,digest)
    job=next(j for j in plan['jobs'] if j['id']==a.job_id);jobroot=Path(plan['run'])/'jobs'/job['id'];output=Path(job['output'])
    if a.grant.resolve()!=jobroot/'LEASE-HANDOFF.json' or c.sha(a.grant)!=a.grant_sha256:raise ValueError('Handoff grant differs')
    grant=c.read(a.grant);gpu=grant['gpu']
    if gpu not in gpus:raise ValueError('Unpermitted GPU')
    for key,value in {'plan_sha256':digest,'permit_sha256':c.sha(a.permit),'job_id':job['id'],
        'model_sha256':c.MODEL_SHA,'host':c.HOST,'uuid':c.GPUS[gpu]}.items():
        if grant.get(key)!=value:raise ValueError('Worker/permit/model handoff mismatch: '+key)
    c.leases.verify_inherited_leases(grant['leases'],lease_paths(gpu));no_previous_attempt(jobroot,output)
    guard=c.load('_robotwin_formal_board',c.ROOT/'run_regmeanpp_m3.py')
    from run_formal import admissible
    row=guard.board(gpu)
    if not admissible(row):raise RuntimeError('Formal GPU became nonempty or below 40GiB under leases')
    c.write(jobroot/'STARTED.json',{'pid':os.getpid(),'started_at':c.now(),'plan_sha256':digest,'job_id':job['id'],
        'model_sha256':c.MODEL_SHA,'model_accepted_sha256':c.ACCEPTED_SHA,'permit_sha256':c.sha(a.permit),
        'manifest':c.bind(Path(plan['run'])/'manifests'/f'{job["id"]}.json'),'admission':row,
        'inherited_lease_paths':sorted(lease_paths(gpu)),'expected_episodes':60,'automatic_retry':False})
    begin=time.monotonic()
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['TOKENIZERS_PARALLELISM']='false'
        base=c.load('_robotwin_regmeanpp_native_engine',c.NATIVE);slicer=c.load('_robotwin_regmeanpp_runtime_setup',c.SLICER)
        slicer.runtime_setup(job['runtime'])
        import torch
        if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required')
        torch.cuda.set_per_process_memory_fraction(.50,0);torch.cuda.reset_peak_memory_stats()
        output.mkdir(parents=False,exist_ok=False)
        # Same native dense engine and complete-task arguments as the accepted TIES/Soups panel.
        result=base.simulator_audit(job['runtime'],output,False)
        from receipt import audit_job
        accepted=audit_job(plan,job,digest)
        if c.sha(c.CHECKPOINT/'model.safetensors')!=c.MODEL_SHA:raise ValueError('Model changed during native evaluation')
        for path,sha in plan['model_sidecar_sha256'].items():
            if c.sha(path)!=sha:raise ValueError('Model sidecar changed during evaluation')
        accepted.update(seconds=time.monotonic()-begin,peak_cuda_memory_mib=torch.cuda.max_memory_allocated()/1024**2,
            native_return=result,started_receipt_sha256=c.sha(jobroot/'STARTED.json'))
        c.write(jobroot/'ACCEPTED.json',accepted)
    except BaseException as exc:
        if not (jobroot/'FAILED.json').exists():c.write(jobroot/'FAILED.json',{'status':'FAILED','at':c.now(),
            'error':repr(exc),'model_sha256':c.MODEL_SHA,'job_id':job['id'],'automatic_retry':False,'signals_sent':0})
        raise
    # Inherited descriptors remain open until process exit; never LOCK_UN.
if __name__=='__main__':main()
