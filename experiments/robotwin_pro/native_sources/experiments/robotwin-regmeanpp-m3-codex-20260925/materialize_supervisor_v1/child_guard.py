#!/usr/bin/env python3
"""Continuous lease and runtime-memory shim around frozen original materialize."""
from __future__ import annotations
import argparse,os,socket,sys,time
from pathlib import Path
import supervisor as s

def claim_once(core,receipt):
    s.no_build_attempt(core)
    s.write(Path(core)/'materialize-PERMIT-CONSUMED.json',receipt)

def lease_paths(gpu,uuid,records):
    flocks=s.load('_robotwin_m3_build_child_flocks',s.FLOCK_FILE)
    card=str(flocks.card_path(gpu,uuid));legacy=str(s.RUNTIME/'resource-leases'/s.HOST/f'gpu-{gpu}.lock')
    slots={str(flocks.DEFAULT_ROOT/s.HOST/f'build-slot-{i}.lock') for i in range(flocks.BUILD_SLOTS)}
    supplied={r['path'] for r in records};found=supplied&slots
    if len(found)!=1 or supplied!={card,legacy,*found}:raise ValueError('Wrong inherited resource leases')
    return supplied

class RuntimeFloor:
    def __init__(self,board_fn,gpu,uuid,clock=time.monotonic):self.board_fn=board_fn;self.gpu=gpu;self.uuid=uuid;self.clock=clock;self.last=None;self.minimum_seen=None
    def __call__(self,force=False):
        now=self.clock()
        if not force and self.last is not None and now-self.last<5:return
        row=self.board_fn(self.gpu);self.last=now
        self.minimum_seen=row['free_mib'] if self.minimum_seen is None else min(self.minimum_seen,row['free_mib'])
        if row['uuid']!=self.uuid or row['free_mib']<12288:raise RuntimeError('Own materialize stopped: runtime free below 12GiB or GPU UUID changed: '+str(row))

def main():
    p=argparse.ArgumentParser();p.add_argument('--supervisor-plan',type=Path,required=True);p.add_argument('--permit',type=Path,required=True)
    p.add_argument('--grant',type=Path,required=True);p.add_argument('--grant-sha256',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:raise ValueError('Explicit execute required')
    plan=s.validate_plan(a.supervisor_plan);root=Path(plan['output'])
    if socket.gethostname()!=s.HOST:raise ValueError('Wrong host')
    if a.grant.resolve()!=root/'LEASE-HANDOFF.json' or s.sha(a.grant)!=a.grant_sha256:raise ValueError('Frozen handoff changed')
    grant=s.read(a.grant);permit=s.read(a.permit);gpu=permit.get('gpu');guard=s.validate_permit(permit,a.supervisor_plan,gpu)
    for key,value in {'supervisor_plan_sha256':s.sha(a.supervisor_plan),'core_plan_sha256':s.CORE_SHA,'native_smoke_sha256':s.SMOKE_SHA,
        'permit_sha256':s.sha(a.permit),'gpu':gpu,'uuid':s.GPUS[gpu],'host':s.HOST}.items():
        if grant.get(key)!=value:raise ValueError('Materialize handoff identity differs: '+key)
    records=grant['leases'];paths=lease_paths(gpu,s.GPUS[gpu],records)
    s.v2.verify_inherited_leases(records,paths);s.no_build_attempt();s.verify_smoke()
    sys.path.insert(0,str(s.ROOT))
    import torch
    from contract_regmeanpp_m3 import validate
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before materialize admission')
    original,original_sha=validate(s.CORE,full_models=True)
    if original_sha!=s.CORE_SHA:raise ValueError('Core algorithm plan changed')
    s.v2.verify_inherited_leases(records,paths);s.no_build_attempt();s.verify_smoke();fresh=guard.board(gpu)
    if not guard.admissible(fresh,s.GPUS):raise RuntimeError('Resource changed under inherited materialize leases')
    claim_once(s.CORE,{'plan_sha256':s.CORE_SHA,'permit_sha256':s.sha(a.permit),'supervisor_plan_sha256':s.sha(a.supervisor_plan),
        'native_smoke_sha256':s.SMOKE_SHA,'stage':'materialize','pid':os.getpid(),'host':s.HOST,'board':fresh,
        'leases':[r['path'] for r in records],'lease_fds_inherited':True,'signals_sent':0})
    floor=RuntimeFloor(guard.board,gpu,s.GPUS[gpu])
    try:
        s.write(root/'GPU-ADMISSION.json',{'status':'PASS','pid':os.getpid(),'board':fresh,'inherited_leases_verified':3,
            'model_files_fully_hashed_under_same_leases':4,'cuda_initialized_before_admission':torch.cuda.is_initialized()})
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['TOKENIZERS_PARALLELISM']='false'
        torch.cuda.set_per_process_memory_fraction(.70,0)
        import materialize_regmeanpp_m3 as algorithm
        original_batch=algorithm.batch_for
        def monitored_batch(*args,**kwargs):
            floor();return original_batch(*args,**kwargs)
        # Resource-only callback; sampler, data, solver and block schedule are unchanged.
        algorithm.batch_for=monitored_batch
        try:algorithm.materialize(original,s.CORE_SHA,s.CORE)
        finally:algorithm.batch_for=original_batch
        from accept_model import accept
        accept(original,a.supervisor_plan,root,floor)
        s.write(root/'RESOURCE-FINAL.json',{'minimum_runtime_free_mib_observed':floor.minimum_seen,'floor_mib':12288,'signals_sent':0})
    except BaseException as exc:
        if not (s.CORE/'materialize-FAILED.json').exists():
            s.write(s.CORE/'materialize-FAILED.json',{'status':'FAILED','error':repr(exc),'supervisor_plan_sha256':s.sha(a.supervisor_plan),
                'minimum_runtime_free_mib_observed':floor.minimum_seen,'automatic_retry':False,'signals_sent':0})
        raise
    # Close inherited descriptors only on process exit; never LOCK_UN them.
if __name__=='__main__':main()
