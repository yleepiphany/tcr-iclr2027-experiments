"""Read-only view of the original Table1 static inputs for frozen Figure4 subsets."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open

SUBSETS={'spatial-goal':('spatial','goal'), 'spatial-object-goal':('spatial','object','goal')}
ALL_EXPERTS=('spatial','object','goal','long')
FLOW_SLOTS=(0,5,9)
OBS_KEYS={'tokens','masks'}|{f'image_{i}' for i in range(3)}|{f'image_mask_{i}' for i in range(3)}
EXPECTED_SOUP_SHA={'spatial-goal':'74b007ca88a7e3070e500b78562e0618bf398eae4f23e0543ef046829d0f0189',
 'spatial-object-goal':'c49afabe84c7ad31ae657c7369c7c7cd562ef4b0de7133487d6584607ba36cf1'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def tensor_sha(tensor):
    value=tensor.detach().cpu().contiguous()
    header=json.dumps({'dtype':str(value.dtype),'shape':list(value.shape)},sort_keys=True).encode()
    return hashlib.sha256(header+b'\0'+value.view(torch.uint8).numpy().tobytes()).hexdigest()

def initial_noise(group,task,request,replica):
    # These are ORIGINAL four-expert IDs. Goal remains ordinal 2 in a 2-expert run.
    if group not in ALL_EXPERTS or not 0<=task<10 or not 0<=request<5 or replica not in range(3):
        raise ValueError('Invalid original observation identity')
    seed=202609220000+ALL_EXPERTS.index(group)*100000+task*1000+request*10+replica
    return torch.randn((1,50,32),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32),seed

def validate_model_batch(batch):
    if set(batch)!=OBS_KEYS|{'x_t','time'}:raise ValueError('Unexpected fields: actions/targets are prohibited')
    n=batch['x_t'].shape[0]
    if batch['x_t'].shape!=(n,50,32) or batch['time'].shape!=(n,) or not torch.equal(batch['time'],torch.ones(n)):
        raise ValueError('Expected native t=1 Gaussian flow input')
    if batch['tokens'].ndim!=2 or batch['masks'].shape!=batch['tokens'].shape or batch['masks'].dtype!=torch.bool:
        raise ValueError('Token/mask dimensions differ')
    for i in range(3):
        if batch[f'image_{i}'].shape!=(n,3,224,224) or batch[f'image_mask_{i}'].shape!=(n,) or batch[f'image_mask_{i}'].dtype!=torch.bool:
            raise ValueError('Native image dimensions differ')
        expected=torch.full((n,),i<2,dtype=torch.bool)
        if not torch.equal(batch[f'image_mask_{i}'],expected):raise ValueError('LIBERO has two valid cameras and a masked third camera')
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in batch.values()):raise ValueError('Nonfinite input')
    return batch

class SubsetInputs:
    def __init__(self,master,subset):
        if subset not in SUBSETS:raise ValueError('Only frozen 2/3 subsets; four-expert endpoint is read-only reuse')
        self.groups=SUBSETS[subset];self.entries={g:master['inputs']['featcal'][g] for g in self.groups};self.manifests={}
        for group,entry in self.entries.items():
            if sha(entry['manifest']['path'])!=entry['manifest']['sha256']:raise ValueError('Original input manifest SHA changed')
            m=read(entry['manifest']['path']);self.manifests[group]=m
            if m['expert']!=group or m['observation_count']!=50 or m['sample_count']!=150 or len(m['samples'])!=150:
                raise ValueError('Input expert or calibration quota differs')
            if any(m[k] for k in ('demonstration_actions_used','labels_used','rollouts_used','success_filtering')):
                raise ValueError('Input bank contains labels or rollout/success selection')
            if m['tensor_sha256']!=entry['observations']['sha256'] or Path(m['tensor_file'])!=Path(entry['observations']['path']):
                raise ValueError('Input bytes/path identity differs')
            covered=set()
            for s in m['samples']:
                task=s['task_slot'];request=s['task_ordinal']//3;replica=s['noise_replica']
                _,seed=initial_noise(group,task,request,replica)
                identity=(task,request,replica)
                if identity in covered or s['index']!=task*15+request*3+replica or s['task_ordinal']!=request*3+replica or s['observation_index']!=task*5+request or s['flow_index']!=FLOW_SLOTS[replica] or s['noise_seed']!=seed or s['timestep']!=1.:
                    raise ValueError('Input call identity changed')
                covered.add(identity)
            if len(covered)!=150:raise ValueError('Incomplete calls')

    def verify(self):
        for group,entry in self.entries.items():
            if sha(entry['manifest']['path'])!=entry['manifest']['sha256'] or sha(entry['observations']['path'])!=entry['observations']['sha256']:
                raise ValueError('Input bytes changed')
            with safe_open(entry['observations']['path'],framework='pt',device='cpu') as f:
                if set(f.keys())!={f'observation_{i:03d}.{k}' for i in range(50) for k in OBS_KEYS}:raise ValueError('Unexpected observation tensor keys')
        return {'observations':50*len(self.groups),'noise_calls':150*len(self.groups),'labels_used':False,'rollouts_used':False}

    @lru_cache(maxsize=90)
    def batch(self,group,task,replica):
        if group not in self.groups or not 0<=task<10 or replica not in range(3):raise ValueError('Out-of-subset input requested')
        rows=sorted([s for s in self.manifests[group]['samples'] if s['task_slot']==task and s['noise_replica']==replica],key=lambda s:s['task_ordinal'])
        if len(rows)!=5:raise ValueError('Expected five observations per task/replica')
        records=[]
        with safe_open(self.entries[group]['observations']['path'],framework='pt',device='cpu') as f:
            for row in rows:
                prefix=f"observation_{row['observation_index']:03d}."
                record={k:f.get_tensor(prefix+k) for k in OBS_KEYS}
                noise,seed=initial_noise(group,task,row['task_ordinal']//3,replica)
                if tensor_sha(noise)!=row['x_t_sha256'] or seed!=row['noise_seed']:raise ValueError('Original noise bytes differ')
                record.update(x_t=noise,time=torch.ones(1,dtype=torch.float32));records.append(validate_model_batch(record))
        return validate_model_batch({k:torch.cat([r[k] for r in records]) for k in records[0]})
