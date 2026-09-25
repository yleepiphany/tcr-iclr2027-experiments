#!/usr/bin/env python3
"""Resource-only shim; calls the original frozen M3 smoke function unchanged."""
from __future__ import annotations
import argparse,os,socket,sys
from pathlib import Path
import supervisor as s

def claim_once(core,receipt):
    """Preserve the original core stage identity with an exclusive file create."""
    core=Path(core)
    for name in ('smoke-PERMIT-CONSUMED.json','native-smoke.json','smoke-FAILED.json'):
        if (core/name).exists():raise FileExistsError('Original smoke already claimed or attempted')
    s.write(core/'smoke-PERMIT-CONSUMED.json',receipt)

def expected_lease_paths(gpu,uuid,records):
    flocks=s.load('_robotwin_m3_child_flocks',s.FLOCK_FILE)
    card=str(flocks.card_path(gpu,uuid));legacy=str(s.RUNTIME/'resource-leases'/s.HOST/f'gpu-{gpu}.lock')
    slots={str(flocks.DEFAULT_ROOT/s.HOST/f'build-slot-{i}.lock') for i in range(flocks.BUILD_SLOTS)}
    supplied={r['path'] for r in records};found=supplied&slots
    if len(found)!=1 or supplied!={card,legacy,*found}:raise ValueError('Handoff does not name approved resource leases')
    return supplied

def main():
    p=argparse.ArgumentParser();p.add_argument('--supervisor-plan',type=Path,required=True);p.add_argument('--permit',type=Path,required=True)
    p.add_argument('--grant',type=Path,required=True);p.add_argument('--grant-sha256',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:raise ValueError('Explicit execute required')
    plan=s.validate_plan(a.supervisor_plan);root=Path(plan['output'])
    if socket.gethostname()!=s.HOST:raise ValueError('Wrong host')
    if a.grant.resolve()!=root/'LEASE-HANDOFF.json' or s.sha(a.grant)!=a.grant_sha256:raise ValueError('Frozen handoff grant differs')
    grant=s.read(a.grant);permit=s.read(a.permit);gpu=permit.get('gpu')
    guard=s.validate_permit(permit,a.supervisor_plan,gpu)
    for key,value in {'supervisor_plan_sha256':s.sha(a.supervisor_plan),'core_plan_sha256':s.CORE_SHA,
        'permit_sha256':s.sha(a.permit),'gpu':gpu,'uuid':s.GPUS[gpu],'host':s.HOST}.items():
        if grant.get(key)!=value:raise ValueError('Handoff identity differs: '+key)
    records=grant['leases'];paths=expected_lease_paths(gpu,s.GPUS[gpu],records)
    s.verify_inherited_leases(records,paths);s.verify_no_gpu_attempt()
    sys.path.insert(0,str(s.ROOT))
    import torch
    from contract_regmeanpp_m3 import validate
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before inherited-lease admission')
    original,original_sha=validate(s.CORE,full_models=True)
    if original_sha!=s.CORE_SHA:raise ValueError('Original materializer plan changed')
    s.verify_inherited_leases(records,paths);s.verify_no_gpu_attempt()
    fresh=guard.board(gpu)
    if not guard.admissible(fresh,s.GPUS):raise RuntimeError('BLOCKED_RESOURCE_CHANGED_UNDER_INHERITED_LEASES')
    # Atomic single-use marker is written before any CUDA initialization.
    claim_once(s.CORE,{'plan_sha256':s.CORE_SHA,'permit_sha256':s.sha(a.permit),
        'supervisor_plan_sha256':s.sha(a.supervisor_plan),'stage':'smoke','pid':os.getpid(),'host':s.HOST,'board':fresh,
        'leases':[r['path'] for r in records],'lease_fds_inherited':True,'resource_shim_only':True,'signals_sent':0})
    s.write(root/'GPU-ADMISSION.json',{'status':'PASS','pid':os.getpid(),'board':fresh,'inherited_leases_verified':3,
        'model_files_fully_hashed_under_same_leases':4,'cuda_initialized_before_admission':torch.cuda.is_initialized()})
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['TOKENIZERS_PARALLELISM']='false'
        torch.cuda.set_per_process_memory_fraction(.70,0)
        from materialize_regmeanpp_m3 import smoke
        smoke(original,s.CORE_SHA,s.CORE)
    except BaseException as exc:
        if not (s.CORE/'smoke-FAILED.json').exists():
            s.write(s.CORE/'smoke-FAILED.json',{'status':'FAILED','error':repr(exc),'supervisor_plan_sha256':s.sha(a.supervisor_plan),'signals_sent':0})
        raise
    # Never explicitly unlock inherited descriptors. Process exit closes them;
    # supervisor closes its copies only after wait() or its own termination.
if __name__=='__main__':main()
