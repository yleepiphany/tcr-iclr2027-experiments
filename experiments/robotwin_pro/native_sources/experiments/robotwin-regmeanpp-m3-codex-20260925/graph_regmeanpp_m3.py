"""RoboTwin M=3 original-Gram spectral smoothing on the native PI0.5 graph.

The collection schedule is Table1's complete-block merged-prefix schedule.
This adapter reruns the native prefix instead of consuming saved execution
features. The solver is imported from the frozen, unmodified Table1 source.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import importlib.util
from pathlib import Path
import sys
import torch

GROUPS = ('coordination', 'receptacle', 'precision')
VISION = 'model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers'
LANGUAGE = 'model.paligemma_with_expert.paligemma.model.language_model.layers'
ACTION = 'model.paligemma_with_expert.gemma_expert.model.layers'
VISION_SUFFIX = ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.out_proj','mlp.fc1','mlp.fc2')
TOWER_SUFFIX = ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj')
INTERFACES = ('model.action_in_proj','model.time_mlp_in','model.time_mlp_out','model.action_out_proj')
TOTAL_ROWS = 1_776_300
KERNEL_SHA = '55b06cc4b6a43c029ade1a5471e3ca1259eb5ec1234420a454333b98411ee3fc'

@dataclass(frozen=True)
class Stage:
    index: int
    family: str
    targets: tuple[str, ...]
    repetitions: int = 1
    rows_per_observation: int = 16

    def to_dict(self):
        return {**asdict(self), 'expected_trace': list(self.targets)*self.repetitions,
                'rows_per_expert_per_target': 50*self.repetitions*self.rows_per_observation,
                'update': 'simultaneous after all three expert block captures and solves'}

def schedule():
    stages=[]
    for family, root, count, suffixes, repeats in (
        ('vision',VISION,27,VISION_SUFFIX,3),('language',LANGUAGE,18,TOWER_SUFFIX,1)):
        for block in range(count):
            stages.append(Stage(len(stages),family,tuple(f'{root}.{block}.{x}' for x in suffixes),repeats))
    for path in INTERFACES[:3]:
        stages.append(Stage(len(stages),'interface',(path,),1,1 if 'time_mlp' in path else 16))
    for block in range(18):
        stages.append(Stage(len(stages),'action',tuple(f'{ACTION}.{block}.{x}' for x in TOWER_SUFFIX)))
    stages.append(Stage(len(stages),'output',(INTERFACES[-1],)))
    if len(stages)!=67 or len({p for s in stages for p in s.targets})!=418:
        raise ValueError('Unexpected Table1 RegMean++ block topology')
    if sum(len(s.targets)*s.repetitions*s.rows_per_observation*50*3 for s in stages)!=TOTAL_ROWS:
        raise ValueError('Unexpected M3 row budget')
    return tuple(stages)

def sample_observation_rows(value, limit=16):
    """Apply the original linspace/round sampler separately to every observation.

    Table1 replays B=1. Native B=5 must not flatten the batch before sampling.
    No padding mask or data-dependent selection is added to the Table1 rule.
    """
    if value.ndim<2 or value.shape[0]<1:raise ValueError('Missing observation batch dimension')
    rows=[]
    for sample in value:
        flat=sample.reshape(-1,sample.shape[-1])
        if flat.shape[0]>limit:
            idx=torch.linspace(0,flat.shape[0]-1,limit,device=flat.device).round().long()
            flat=flat.index_select(0,idx)
        rows.append(flat.detach().to(device='cpu',dtype=torch.float32))
    return torch.cat(rows,0)

class CaptureComplete(Exception):
    pass

def capture_stage(policy, stage, forward, *, stop_early=True):
    """Hooks only the complete candidate block; never modifies any weights."""
    modules=dict(policy.named_modules());captured={p:[] for p in stage.targets};trace=[]
    expected=list(stage.targets)*stage.repetitions;handles=[]
    def hook(path):
        def record(_module,args):
            if len(trace)>=len(expected) or path!=expected[len(trace)]:
                raise ValueError('Native candidate block call order differs')
            value=args[0]
            rows=sample_observation_rows(value,16)
            expected_rows=value.shape[0]*stage.rows_per_observation
            if rows.shape[0]!=expected_rows:raise ValueError('Native input row capacity differs from frozen budget')
            captured[path].append(rows);trace.append(path)
            if stop_early and len(trace)==len(expected):raise CaptureComplete()
        return record
    try:
        for p in stage.targets:handles.append(modules[p].register_forward_pre_hook(hook(p)))
        try:forward()
        except CaptureComplete:
            if not stop_early:raise
        if trace!=expected:raise ValueError('Native candidate block did not execute completely')
    finally:
        for h in handles:h.remove()
    return {p:torch.cat(v,0) for p,v in captured.items()}

@contextmanager
def candidate_block(policy, stage, weights, biases=None):
    """All candidate block parameters change together; merged prefix stays fixed."""
    modules=dict(policy.named_modules());saved={};biases=biases or {}
    try:
        for p in stage.targets:
            m=modules[p];saved[p]=(m.weight.detach().clone(), None)
            m.weight.data.copy_(weights[p].to(m.weight))
            if p in biases:
                saved[p]=(saved[p][0],m.bias.detach().clone())
                m.bias.data.copy_(biases[p].to(m.bias))
        yield
    finally:
        for p,(w,b) in saved.items():
            modules[p].weight.data.copy_(w)
            if b is not None:modules[p].bias.data.copy_(b)

@torch.inference_mode()
def native_velocity(policy,batch):
    """Same explicit single-time native graph as frozen featcal_execution.py.

    Only native embedding/transformer/action-head code is reused. No FeatCal
    teacher/student regression, TCR features, actions or trajectory is consumed.
    """
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks,prepare_attention_masks_4d
    model=policy.model
    prefix,prefix_pad,prefix_att=model.embed_prefix(
        [batch[f'image_{i}'] for i in range(3)],
        [batch[f'image_mask_{i}'] for i in range(3)],batch['tokens'],batch['masks'],None,None)
    suffix,suffix_pad,suffix_att,cond=model.embed_suffix(batch['x_t'],batch['time'])
    if model.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype==torch.bfloat16:
        prefix,suffix=prefix.to(torch.bfloat16),suffix.to(torch.bfloat16)
    pad=torch.cat([prefix_pad,suffix_pad],1);att=torch.cat([prefix_att,suffix_att],1)
    positions=torch.cumsum(pad,1)-1;mask=prepare_attention_masks_4d(make_att_2d_masks(pad,att))
    (_,out),_=model.paligemma_with_expert.forward(attention_mask=mask,position_ids=positions,
        past_key_values=None,inputs_embeds=[prefix,suffix],use_cache=False,adarms_cond=[None,cond])
    velocity=model.action_out_proj(out[:,-model.config.chunk_size:].to(torch.float32))
    if not torch.isfinite(velocity).all():raise ValueError('Nonfinite native velocity')
    return velocity

def load_kernel(path,sha):
    if sha(path)!=KERNEL_SHA:raise ValueError('Table1 smoothing source drift')
    spec=importlib.util.spec_from_file_location('_robotwin_regmeanpp_m3_smoothing',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

def solve_complete_block(kernel,stage,inputs,weights,device='cpu'):
    if tuple(inputs)!=GROUPS or tuple(weights)!=GROUPS:raise ValueError('M3 expert order differs')
    result={};metrics={}
    for p in stage.targets:
        x={g:inputs[g][p] for g in GROUPS};w={g:weights[g][p] for g in GROUPS}
        result[p],metrics[p]=kernel.solve_smoothed_weight(x,w,offdiag_scale=.3,strategy='auto',
            filter_kind='smooth',power_iterations=64,output_chunk_size=128,device=device)
    return result,metrics

def commit_block(policy,stage,weights):
    if set(weights)!=set(stage.targets):raise ValueError('Incomplete atomic block solution')
    modules=dict(policy.named_modules())
    for p in stage.targets:
        if weights[p].shape!=modules[p].weight.shape or not torch.isfinite(weights[p]).all():
            raise ValueError('Invalid block output; no block parameters committed')
    for p in stage.targets:modules[p].weight.data.copy_(weights[p].to(modules[p].weight))
