"""Shared, label-free RoboTwin model inputs; no feature statistics or solver."""
from __future__ import annotations
import json
from pathlib import Path
import torch

DATA_COLUMNS=('observation.state','timestamp','frame_index','episode_index','index','task_index')
GROUPS=('coordination','receptacle','precision')
MODEL_FIELDS={'tokens','masks','x_t','time'}|{f'image_{i}' for i in range(3)}|{f'image_mask_{i}' for i in range(3)}

def initial_noise(seed):
    return torch.randn((1,50,32),generator=torch.Generator(device='cpu').manual_seed(int(seed)),dtype=torch.float32)

def validate_model_batch(batch):
    if set(batch)!=MODEL_FIELDS:raise ValueError(f'Unexpected model fields: {set(batch)^MODEL_FIELDS}')
    n=batch['x_t'].shape[0]
    if tuple(batch['x_t'].shape)!=(n,50,32) or tuple(batch['time'].shape)!=(n,):raise ValueError('Latent/time shape differs')
    if not torch.equal(batch['time'],torch.ones_like(batch['time'])):raise ValueError('Only native Gaussian initial t=1 allowed')
    if batch['tokens'].dtype!=torch.int64 or batch['masks'].dtype!=torch.bool or batch['tokens'].shape!=batch['masks'].shape:
        raise ValueError('Token/mask contract differs')
    for i in range(3):
        image=batch[f'image_{i}'];mask=batch[f'image_mask_{i}']
        if tuple(image.shape)!=(n,3,224,224) or image.dtype!=torch.float32:raise ValueError('Three native RGB tensors required')
        if tuple(mask.shape)!=(n,) or mask.dtype!=torch.bool or not mask.all():raise ValueError('RoboTwin requires three actual camera inputs')
        if image.min() < -1.00001 or image.max()>1.00001:raise ValueError('Image normalization outside [-1,1]')
    if any(x.is_floating_point() and not torch.isfinite(x).all() for x in batch.values()):raise ValueError('Nonfinite input')
    return batch

def load_batch(bank_path,group,task_slot,replica=0,*,verify=True):
    """Five observations for one task and one noise replica; CPU only.

    RegMean++ consumes replica=0. FeatCal consumes replicas=0,1,2 separately.
    This returns model inputs, never labels or cached expert/student features.
    """
    from safetensors import safe_open
    from prepare_readiness import sha
    bank_path=Path(bank_path);bank=json.loads(bank_path.read_text())
    if group not in GROUPS or task_slot not in range(10) or replica not in range(3):raise ValueError('Invalid view')
    entry=bank['groups'][group];path=bank_path.parent/entry['tensor_file']
    if verify and sha(path)!=entry['tensor_sha256']:raise ValueError('Frozen bank tensor hash differs')
    items=[]
    with safe_open(str(path),framework='pt',device='cpu') as handle:
        for request in range(5):
            index=task_slot*5+request;prefix=f'observation_{index:03d}.';noise=f'noise_{index*3+replica:03d}.'
            row={k:handle.get_tensor(prefix+k) for k in MODEL_FIELDS-{'x_t','time'}}
            row['x_t']=handle.get_tensor(noise+'x_t');row['time']=handle.get_tensor(noise+'time');items.append(row)
    return validate_model_batch({k:torch.cat([item[k] for item in items],dim=0) for k in MODEL_FIELDS})
