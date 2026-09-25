#!/usr/bin/env python3
"""CPU-only RoboTwin static observation bank, using native preprocessing.

Reads only the explicit observation/identity columns from parquet; action,
reward and success columns cannot enter preprocessing. Uses neither simulator
rollouts nor TCR execution/prefix features. No GPU is exposed or initialized.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ['TOKENIZERS_PARALLELISM']='false'
from prepare_readiness import sha

def tensor_hash(value):
    import torch
    x=value.detach().cpu().contiguous()
    header=json.dumps({'dtype':str(x.dtype),'shape':list(x.shape)},sort_keys=True).encode()
    return hashlib.sha256(header+b'\0'+x.view(torch.uint8).numpy().tobytes()).hexdigest()

def write(path,data):
    with Path(path).open('x') as stream:json.dump(data,stream,indent=2,sort_keys=True);stream.write('\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root',type=Path,required=True)
    parser.add_argument('--selection-plan',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();work=args.workspace_root.resolve();plan=json.loads(args.selection_plan.read_text());out=args.output_root.resolve()
    if out.exists():raise FileExistsError('A new isolated bank directory is required')
    if plan['gpu_used'] or plan['observation_count']!=150:raise ValueError('Unexpected CPU selection plan')
    for filename,digest in plan['frozen_files'].items():
        if sha(filename)!=digest:raise ValueError(f'Frozen source changed: {filename}')
    source=work/'pi05_lora_finetune_v2_20260826'
    sys.path[:0]=[str(source/'lerobot/src'),str(source/'src')]
    import torch
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from safetensors.torch import save_file
    from static_bank import DATA_COLUMNS,GROUPS,MODEL_FIELDS,initial_noise,validate_model_batch,load_batch
    torch.set_num_threads(4);pa.set_cpu_count(4);pa.set_io_thread_count(4)
    # PEFT's optional TransformerEngine availability check imports an installed
    # CUDA-only Triton autotuner even for CPU preprocessing. Hide only that
    # unrelated optional acceleration module in this process, not model code.
    original_find=importlib.util.find_spec
    with patch('importlib.util.find_spec',side_effect=lambda name,*a,**kw:None if name=='transformer_engine' or name.startswith('transformer_engine.') else original_find(name,*a,**kw)):
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.datasets.video_utils import decode_video_frames
    if torch.cuda.is_initialized():raise RuntimeError('CPU collector initialized CUDA')
    dataset=Path(plan['dataset_root']);info=json.loads((dataset/'meta/info.json').read_text())
    cameras=['observation.images.cam_high','observation.images.cam_left_wrist','observation.images.cam_right_wrist']
    meta_columns=['episode_index','length','data/chunk_index','data/file_index','dataset_from_index','dataset_to_index']
    for key in cameras:meta_columns += [f'videos/{key}/chunk_index',f'videos/{key}/file_index',f'videos/{key}/from_timestamp']
    meta_paths=sorted((dataset/'meta/episodes').rglob('*.parquet'))
    metadata={int(r['episode_index']):r for p in meta_paths for r in pq.read_table(p,columns=meta_columns).to_pylist()}
    selected=plan['observations'];episode_ids=sorted({r['episode_index'] for r in selected});frame_ids=sorted({r['frame_index'] for r in selected})
    if not set(episode_ids)<=set(metadata):raise ValueError('Selected calibration episode is absent')
    data_files=sorted({dataset/info['data_path'].format(chunk_index=metadata[e]['data/chunk_index'],file_index=metadata[e]['data/file_index']) for e in episode_ids})
    table=ds.dataset([str(p) for p in data_files],format='parquet').to_table(columns=list(DATA_COLUMNS),
        filter=ds.field('episode_index').isin(episode_ids)&ds.field('frame_index').isin(frame_ids))
    raw={}
    for row in table.to_pylist():
        key=(int(row['episode_index']),int(row['frame_index']))
        if key in raw:raise ValueError('Duplicate raw observation key')
        raw[key]=row
    task_table=pd.read_parquet(dataset/'meta/tasks.parquet')
    tasks={int(row.task_index):str(name) for name,row in task_table.iterrows()}
    cfg=PreTrainedConfig.from_pretrained(plan['soup']['path']);cfg.device='cpu';cfg.compile_model=False
    pre,_=make_pre_post_processors(policy_cfg=cfg,pretrained_path=plan['soup']['path'],preprocessor_overrides={'device_processor':{'device':'cpu'}})
    shell=torch.nn.Module();shell.register_parameter('device_anchor',torch.nn.Parameter(torch.zeros(()),requires_grad=False));shell.config=cfg
    out.mkdir(parents=True)
    source_hashes={str(p):sha(p) for p in [*meta_paths,*data_files,dataset/'meta/tasks.parquet']}
    groups={};total=0;seed_seen=set();noise_hash_seen=set()
    for group in GROUPS:
        tensors={};records=[]
        for task_slot in range(10):
            chosen=sorted([r for r in selected if r['group']==group and r['task_slot']==task_slot],key=lambda x:x['request'])
            if len(chosen)!=5 or len({r['episode_index'] for r in chosen})!=1:raise ValueError('Require one episode and five requests per task')
            episode=chosen[0]['episode_index'];ep=metadata[episode]
            raw_rows=[raw[(episode,r['frame_index'])] for r in chosen]
            if any(r['index']!=ep['dataset_from_index']+s['frame_index'] for r,s in zip(raw_rows,chosen,strict=True)):
                raise ValueError('Episode global/local frame mapping differs')
            raw_batch={'observation.state':torch.tensor([r['observation.state'] for r in raw_rows],dtype=torch.float32),
                       'task':[tasks[int(r['task_index'])] for r in raw_rows]}
            if raw_batch['observation.state'].shape!=(5,14):raise ValueError('RoboTwin state must be14D')
            video_refs=[]
            for key in cameras:
                video=dataset/info['video_path'].format(video_key=key,chunk_index=ep[f'videos/{key}/chunk_index'],file_index=ep[f'videos/{key}/file_index'])
                if str(video) not in source_hashes:source_hashes[str(video)]=sha(video)
                times=[float(ep[f'videos/{key}/from_timestamp'])+float(r['timestamp']) for r in raw_rows]
                # Per-frame native decode avoids holding a full episode of RGB
                # images in memory and matches DatasetReader's single-frame path.
                frames=torch.cat([decode_video_frames(video,[t],tolerance_s=1e-4,backend='pyav',return_uint8=True) for t in times],dim=0)
                if frames.shape!=(5,3,480,640) or frames.dtype!=torch.uint8:raise ValueError('Native RGB layout differs')
                raw_batch[key]=frames.float()/255.0
                video_refs.append({'key':key,'path':str(video),'timestamps':times,'rgb_sha256':[tensor_hash(x.unsqueeze(0)) for x in frames]})
                del frames
            # Raw input allowlist contains no target labels even before calling
            # the native processor. Its empty action=None output is discarded.
            if set(raw_batch)!={'observation.state','task',*cameras}:raise ValueError('Raw batch has unexpected fields')
            processed=pre(raw_batch)
            if any(v is not None for k,v in processed.items() if 'action' in k.lower()):raise ValueError('Native preprocessing generated an action target')
            processed={k:v for k,v in processed.items() if 'action' not in k.lower()}
            images,masks=PI05Policy._preprocess_images(shell,processed)
            for request,row in enumerate(chosen):
                index=task_slot*5+request;prefix=f'observation_{index:03d}.'
                record={'tokens':processed['observation.language.tokens'][request:request+1].cpu().contiguous(),
                        'masks':processed['observation.language.attention_mask'][request:request+1].cpu().contiguous()}
                record.update({f'image_{i}':images[i][request:request+1].cpu().contiguous() for i in range(3)})
                record.update({f'image_mask_{i}':masks[i][request:request+1].cpu().contiguous() for i in range(3)})
                tensors.update({prefix+k:v for k,v in record.items()})
                noise_records=[]
                for replica in range(3):
                    candidate=next(x for x in plan['noise_candidates'] if x['group']==group and x['task_slot']==task_slot and x['request']==request and x['noise_replica']==replica)
                    seed=candidate['noise_seed'];noise=initial_noise(seed);noise_hash=tensor_hash(noise)
                    if seed in seed_seen or noise_hash in noise_hash_seen:raise ValueError('Duplicate noise replica')
                    seed_seen.add(seed);noise_hash_seen.add(noise_hash)
                    model_input={**record,'x_t':noise,'time':torch.ones(1,dtype=torch.float32)};validate_model_batch(model_input)
                    noise_prefix=f'noise_{index*3+replica:03d}.'
                    tensors[noise_prefix+'x_t']=noise;tensors[noise_prefix+'time']=model_input['time']
                    noise_records.append({'replica':replica,'seed':seed,'x_t_sha256':noise_hash})
                records.append({**row,'observation_index':index,'instruction':raw_batch['task'][request],
                    'raw_state_sha256':tensor_hash(raw_batch['observation.state'][request:request+1]),
                    'processed_tensor_sha256':{k:tensor_hash(v) for k,v in record.items()},'noise':noise_records,
                    'videos':[{**v,'timestamp':v['timestamps'][request],'rgb_sha256':v['rgb_sha256'][request]} for v in video_refs]})
                total+=1
            print(json.dumps({'event':'cpu_task_collected','group':group,'task_slot':task_slot,'observations':total,'cuda_initialized':torch.cuda.is_initialized()}),flush=True)
        file=out/f'{group}.safetensors';save_file(tensors,str(file),metadata={'format':'pt','source':'static_observations_no_action_labels'})
        groups[group]={'observations':50,'noise_states':150,'tensor_file':file.name,'tensor_sha256':sha(file),'records':records}
    if total!=150 or len(seed_seen)!=450 or torch.cuda.is_initialized():raise ValueError('Final count/device contract differs')
    bank={'schema':'robotwin_static_observation_initial_noise_bank_v1','status':'CPU_BANK_COMPLETE_GPU_UNVERIFIED',
        'created_at':datetime.now(timezone.utc).isoformat(),'selection_plan':str(args.selection_plan.resolve()),'selection_plan_sha256':sha(args.selection_plan),
        'collector_sha256':sha(Path(__file__)),'loader_sha256':sha(Path(__file__).with_name('static_bank.py')),
        'observation_count':150,'noise_state_count':450,'action_labels_used':False,'reward_or_success_fields_read':False,
        'parquet_columns_read':list(DATA_COLUMNS),'expert_rollouts_used':False,'tcr_execution_features_used':False,
        'physical_timestep':1.0,'latent_shape':[50,32],'state_dimensions':14,'camera_count':3,
        'native_preprocessing_checkpoint':plan['soup']['path'],'native_cpu_optional_te_discovery_disabled':True,
        'gpu_used':False,'cuda_initialized':False,'groups':groups,'source_files_sha256':source_hashes,
        'method_views':{'regmeanpp':{'replicas':[0],'native_calls':150},'featcal':{'replicas':[0,1,2],'native_calls':450}},
        'not_yet_validated':['M=3 model feature collectors','M=3 method materializers','native GPU/reload parity','formal 540-episode method manifests']}
    write(out/'bank.json',bank)
    for group in GROUPS:
        for replica in range(3):load_batch(out/'bank.json',group,0,replica,verify=True)
    print(json.dumps({'status':bank['status'],'bank':str(out/'bank.json'),'bank_sha256':sha(out/'bank.json'),'observations':150,'noise_states':450,'cuda_initialized':False}),flush=True)

if __name__=='__main__':main()
