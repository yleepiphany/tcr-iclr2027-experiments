"""Internal isolated one-request inference worker; launched by cli.py with leases."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from cli import checkpoint, sha, read, SNAPSHOT, FAST_CONTRACT, BANK, SUITES, rebase


def openvla(args):
    source=SNAPSHOT/'vla-merge/experiments/openvla-tcr-20260921/evaluate.py'
    spec=importlib.util.spec_from_file_location('released_oft_evaluate',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.argv=[str(source),'--checkpoint',str(checkpoint('openvla',args.suite,args.host_root)),
              '--suite',SUITES[args.suite],'--bank',str(args.host_root/BANK),
              '--selection',str(args.host_root/BANK/'selections/repeat-01.json'),
              '--output',str(args.output),'--preflight-only' if args.preflight_only else '--smoke']
    module.main()


def move(value,torch):
    if torch.is_tensor(value):return value.to('cuda:0')
    if isinstance(value,dict):return {k:move(v,torch) for k,v in value.items()}
    if isinstance(value,list):return [move(v,torch) for v in value]
    if isinstance(value,tuple):return tuple(move(v,torch) for v in value)
    return value


def fastwam(args):
    import torch
    from hydra import compose,initialize_config_dir
    from hydra.utils import instantiate
    root=args.host_root
    fast=root/'vla-merge_table4/Fast-WAM'
    accepted=read(SNAPSHOT/'vla-merge-runtime/experiments/claude-fastwam-tcr-20260923/calibration-acceptance-v1.json')
    job=next(r for r in accepted['jobs_detail'] if r['expert']==args.suite and r['round']=='A' and r['task_id']==0)
    index=job['selected_request_indices'][0]
    request=root/'vla-merge-runtime/experiments/claude-fastwam-tcr-20260923/calibration-v1'/job['id']/f'request-{index:06d}.pt'
    expected=next(r['sha256'] for r in job['request_file_sha256'] if r['index']==index)
    if sha(request)!=expected:raise ValueError('Frozen native request identity differs')
    if args.preflight_only:
        import numpy as np
        weights=torch.load(str(checkpoint('fastwam',args.suite,root)),map_location='cpu',weights_only=True,mmap=True)
        if len(weights.get('mot',{}))!=1649 or len(weights.get('proprio_encoder',{}))!=2:
            raise ValueError('Fast-WAM checkpoint structure differs from accepted experts')
        for component in ('mot','proprio_encoder'):
            if not all(torch.is_tensor(v) and v.numel()>0 for v in weights[component].values()):
                raise ValueError('Invalid model tensor metadata')
        # Validate every selected source state without creating any CUDA context.
        from vla_merge.libero_procedural_bank import load_selection
        selection=load_selection(root/BANK,root/BANK/'selections/repeat-01.json',verify_source_files=True)
        count=sum(len(v.indices) for k,v in selection.tasks.items() if k[0]==SUITES[args.suite])
        if count!=100:raise ValueError('Expected 100 selected states')
        args.output.mkdir(parents=True,exist_ok=False)
        result={'status':'PASS','backend':'fastwam','checkpoint_mmap_tensors':1651,'selected_states_verified':count,
                'native_request_sha256_verified':expected,'gpu_used':False}
        (args.output/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.65,0)
    with initialize_config_dir(version_base='1.3',config_dir=str(fast/'source/configs')):
        cfg=compose(config_name='sim_libero.yaml',overrides=['task=libero_uncond_2cam224_1e-4'])
    model=instantiate(cfg.model,model_dtype=torch.bfloat16,device='cuda:0')
    model_path=checkpoint('fastwam',args.suite,root)
    weights=torch.load(str(model_path),map_location='cpu',weights_only=True,mmap=True)
    model.mot.load_state_dict(weights['mot'],strict=True)
    model.proprio_encoder.load_state_dict(weights['proprio_encoder'],strict=True)
    del weights
    model=model.to('cuda:0').eval()
    payload=torch.load(request,map_location='cpu',weights_only=True)
    started=time.monotonic()
    with torch.inference_mode():actions=model.infer_action(**move(payload['native_request'],torch))['action']
    if tuple(actions.shape[-2:])!=(32,7) or not torch.isfinite(actions).all():raise ValueError('Native Fast-WAM action must be finite [32,7]')
    result={'status':'PASS','backend':'fastwam','native_action_shape':list(actions.shape),'finite':True,
            'native_requests':1,'evaluation_episodes':0,'is_success_evaluation':False,
            'request_sha256':expected,'checkpoint_sha256':sha(model_path),
            'inference_seconds':time.monotonic()-started,'max_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'smoke.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['openvla','fastwam'],required=True)
    p.add_argument('--host-root',type=Path,required=True);p.add_argument('--suite',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    (openvla if args.backend=='openvla' else fastwam)(args)


if __name__=='__main__':main()
