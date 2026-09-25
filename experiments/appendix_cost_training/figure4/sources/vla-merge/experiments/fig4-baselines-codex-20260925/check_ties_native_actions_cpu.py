#!/usr/bin/env python3
"""One unchanged native ten-step PI05 action generation per TIES subset, CPU only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import time
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_ties_formal_v1 as formal
import ties_subset_v2 as adapter

def main(subset):
    formal.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU check requires CUDA hidden')
    mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.split()[1].isdigit()}
    formal.require(mem['MemAvailable']>=128*2**30,'Insufficient spare host RAM for CPU check')
    plan=formal.final_plan();model=plan['models'][subset]
    out=formal.RUN/'native-cpu-actions-v1'/subset
    out.mkdir(parents=True,exist_ok=False)
    formal.exclusive(out/'STARTED.json',{'pid':os.getpid(),'time':formal.now(),'cpu_only':True,'nice':os.getpriority(os.PRIO_PROCESS,0),
        'source_sha256':formal.sha(__file__),'formal_plan_sha256':formal.sha(formal.RUN/'plan.json'),'model_sha256':model['sha256']})
    try:
        np,torch,_,_=adapter.cpu_imports()
        from safetensors import safe_open
        from safetensors.torch import save_file
        with adapter.cpu_native_imports():
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        cfg=PreTrainedConfig.from_pretrained(model['path'],local_files_only=True)
        formal.require(cfg.num_inference_steps==10 and cfg.n_action_steps==10,'Original ten-step configuration differs')
        cfg.device='cpu';cfg.compile_model=False;cfg.gradient_checkpointing=False
        policy=PI05Policy.from_pretrained(model['path'],config=cfg,local_files_only=True,strict=True)
        source=plan['native_action_input']
        formal.require(formal.sha(source['path'])==source['sha256'],'Native input changed')
        with safe_open(source['path'],framework='pt',device='cpu') as h:
            batch={k[len('observation_000.'):]:h.get_tensor(k) for k in h.keys() if k.startswith('observation_000.')}
        # Instrument call count only; preserve the bound native implementation.
        original=policy.model.denoise_step
        calls=0
        def count_step(*args,**kwargs):
            nonlocal calls
            calls+=1
            return original(*args,**kwargs)
        policy.model.denoise_step=count_step
        torch.manual_seed(274001)
        started=time.monotonic()
        with torch.inference_mode():
            actions=policy.model.sample_actions([batch[f'image_{i}'] for i in range(3)],
                [batch[f'image_mask_{i}'] for i in range(3)],batch['tokens'],batch['masks'])
        elapsed=time.monotonic()-started
        formal.require(calls==10,'Native denoiser was not called exactly ten times')
        formal.require(list(actions.shape)==[1,cfg.chunk_size,cfg.max_action_dim] and torch.isfinite(actions).all().item(),'Invalid native output')
        formal.require(actions.device.type=='cpu' and not torch.cuda.is_initialized(),'Unexpected CUDA activity')
        save_file({'native_actions':actions.contiguous()},str(out/'actions.safetensors'))
        receipt={'status':'pass_finite_native_actions','time':formal.now(),'subset':subset,'model_sha256':model['sha256'],
            'formal_plan_sha256':formal.sha(formal.RUN/'plan.json'),'source_sha256':formal.sha(__file__),
            'input_sha256':source['sha256'],'input_record':'spatial/observation_000','seed':274001,
            'native_denoising_calls':calls,'native_num_inference_steps':cfg.num_inference_steps,'native_execution_window':cfg.n_action_steps,
            'output_shape':list(actions.shape),'output_dtype':str(actions.dtype),'all_finite':True,
            'output_sha256':formal.sha(out/'actions.safetensors'),'sample_actions_seconds':elapsed,
            'device':'cpu','gpu_initialized':False,'policy_weights_changed':False,'success_values_used':False,
            'formal_episodes':0,'runtime_modules':'Original PI05 sample_actions/euler_integrate/denoise_step; count wrapper has no numerical modification',
            'cpu_import_workaround':'Scoped optional Transformer Engine discovery suppression only'}
        formal.exclusive(out/'DONE.json',receipt)
        print(json.dumps(receipt,indent=2))
    except BaseException as exc:
        formal.exclusive(out/'FAILED.json',{'time':formal.now(),'error':repr(exc),'gpu_initialized_by_design':False,'no_retry':True});raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subset',choices=formal.SUBSETS,required=True)
    main(p.parse_args().subset)
