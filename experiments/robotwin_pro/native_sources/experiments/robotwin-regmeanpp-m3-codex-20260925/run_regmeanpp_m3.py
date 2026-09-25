#!/usr/bin/env python3
"""Prepare on CPU; an explicit single-use permit gates each GPU stage.

No queue, process signaling, automatic formal evaluation, or permit generation.
Busy cards fail closed. A future authorized caller must provide the permit.
"""
from __future__ import annotations
import argparse,importlib.util,json,os,socket,subprocess,sys
from pathlib import Path

def board(gpu):
    line=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).strip()
    uuid,free,used,util=[x.strip() for x in line.split(',')]
    # Filter by GPU UUID ourselves as well as passing -i: foreign-card rows cannot admit this card.
    out=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    pids=[]
    for line in out.strip().splitlines():
        card,pid=[x.strip() for x in line.split(',')]
        if card==uuid:pids.append(int(pid))
    return {'gpu':gpu,'uuid':uuid,'free_mib':int(free),'used_mib':int(used),'utilization':int(util),'compute_pids':pids}

def admissible(row,gpus):
    return row['gpu'] in gpus and row['uuid']==gpus[row['gpu']] and row['free_mib']>=71680 and row['used_mib']<=64 and row['utilization']==0 and not row['compute_pids']

def check_permit(permit,stage,plan_sha,run,gpu,uuid,host):
    expected={'schema':'robotwin_regmeanpp_m3_single_use_execution_permit_v1','allowed':True,'stage':stage,
        'plan_sha256':plan_sha,'run':str(Path(run).resolve()),'gpu':gpu,'gpu_uuid':uuid,'host':host}
    if any(permit.get(k)!=v for k,v in expected.items()):raise ValueError('Explicit stage/run/plan/host/GPU permit mismatch')

def gpu_stage(args):
    from contract_regmeanpp_m3 import HOST,GPUS,REPO,RUNTIME,sha,read,write,validate
    import torch
    if not args.execute or not args.permit:raise ValueError('GPU stage requires --execute and an externally authorized single-use --permit')
    if socket.gethostname()!=HOST or args.gpu not in GPUS:raise ValueError('Unapproved host or GPU')
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before resource admission')
    plan,plan_sha=validate(args.run,full_models=True)
    permit=read(args.permit);check_permit(permit,args.stage,plan_sha,args.run,args.gpu,GPUS[args.gpu],HOST)
    run=Path(args.run);consumed=run/f'{args.stage}-PERMIT-CONSUMED.json'
    if consumed.exists():raise FileExistsError('Execution permit already consumed; preserve this attempt')
    if args.stage=='materialize':
        receipt=read(run/'native-smoke.json')
        if receipt.get('status')!='PASS' or receipt.get('plan_sha256')!=plan_sha:raise ValueError('Matching native GPU parity smoke required')
    if not admissible(board(args.gpu),GPUS):raise RuntimeError('BLOCKED_RESOURCE_NOT_EMPTY_OR_LOW_MEMORY')
    file=REPO/'experiments/claude-firstpass-cause-20260920/card_flock.py'
    spec=importlib.util.spec_from_file_location('_robotwin_regmeanpp_m3_flocks',file);flocks=importlib.util.module_from_spec(spec);spec.loader.exec_module(flocks)
    held=[]
    try:
        for lock in (
            lambda:flocks.take_card(args.gpu,GPUS[args.gpu],'robotwin-regmeanpp-m3',args.stage),
            lambda:flocks._try_lock(RUNTIME/'resource-leases'/HOST/f'gpu-{args.gpu}.lock','robotwin-regmeanpp-m3',{'stage':args.stage}),
            lambda:flocks.take_build_slot('robotwin-regmeanpp-m3')):
            value=lock()
            if value is None:raise RuntimeError('BLOCKED_RESOURCE_LEASE_HELD')
            held.append(value)
        fresh=board(args.gpu)
        if not admissible(fresh,GPUS):raise RuntimeError('GPU changed during dual-lease admission')
        write(consumed,{'plan_sha256':plan_sha,'permit_sha256':sha(args.permit),'stage':args.stage,
            'pid':os.getpid(),'host':HOST,'board':fresh,'leases':[str(x.path) for x in held],'signals_sent':0})
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
        os.environ['TOKENIZERS_PARALLELISM']='false'
        torch.cuda.set_per_process_memory_fraction(.70,0)
        from materialize_regmeanpp_m3 import smoke,materialize
        (smoke if args.stage=='smoke' else materialize)(plan,plan_sha,run)
    except BaseException as exc:
        if consumed.exists() and not (run/f'{args.stage}-FAILED.json').exists():
            write(run/f'{args.stage}-FAILED.json',{'status':'FAILED','error':repr(exc),'signals_sent':0})
        raise
    finally:
        # This process owns its CUDA context; no other process is signaled.
        for lock in reversed(held):lock.release()

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','validate','smoke','materialize'],required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--bank',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--permit',type=Path);p.add_argument('--gpu',type=int)
    args=p.parse_args()
    if args.stage in ('prepare','validate'):
        os.environ['CUDA_VISIBLE_DEVICES']=''
        from contract_regmeanpp_m3 import READY,prepare,validate
        if args.stage=='prepare':prepare(args.run,args.bank or READY/'static-bank-v1/bank.json')
        else:validate(args.run,full_models=True)
        print(json.dumps({'status':'PASS','stage':args.stage,'gpu_used':False,'run':str(args.run)}))
    else:gpu_stage(args)
if __name__=='__main__':main()
