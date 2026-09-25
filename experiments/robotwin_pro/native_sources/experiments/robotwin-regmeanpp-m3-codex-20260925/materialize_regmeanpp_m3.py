"""Native graph smoke and materializer; called only after external resource gate."""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import gc,json,shutil,sys,time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from graph_regmeanpp_m3 import *
from contract_regmeanpp_m3 import WORK,REPO,READY,KERNEL,sha,read,write,load_static

def setup_native_imports():
    source=WORK/'pi05_lora_finetune_v2_20260826'
    sys.path[:0]=[str(WORK/'vla-merge-runtime/python-overlay'),str(source/'src'),
        str(source/'lerobot/src'),str(REPO/'scripts'),str(REPO/'src')]

def batch_for(plan,group,task,*,one=False):
    batch=load_static().load_batch(plan['bank'],group,task,replica=0,verify=False)
    return {k:(v[:1] if one else v).to('cuda') for k,v in batch.items()}

def read_stage_weights(handles,stage):
    return {g:{p:handles[g].get_tensor(p+'.weight').float() for p in stage.targets} for g in GROUPS}

def read_stage_biases(handles,stage):
    return {g:{p:handles[g].get_tensor(p+'.bias').float() for p in stage.targets if p in INTERFACES} for g in GROUPS}

def initialize(plan,handles):
    setup_native_imports()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    cfg=PreTrainedConfig.from_pretrained(plan['models'][GROUPS[0]]['path'])
    cfg.device='cuda';cfg.compile_model=False;cfg.gradient_checkpointing=False;cfg.use_peft=False
    cfg.pretrained_path=Path(plan['base']['path'])
    policy=PI05Policy.from_pretrained(plan['base']['path'],config=cfg).to('cuda').eval()
    modules=dict(policy.named_modules())
    for s in schedule():
        for p in s.targets:
            mean=sum(handles[g].get_tensor(p+'.weight').float() for g in GROUPS)/3
            modules[p].weight.data.copy_(mean.to(modules[p].weight))
            if p in INTERFACES:
                bias=sum(handles[g].get_tensor(p+'.bias').float() for g in GROUPS)/3
                modules[p].bias.data.copy_(bias.to(modules[p].bias))
    return policy

def all_inputs(policy,forward):
    modules=dict(policy.named_modules());paths={p:s for s in schedule() for p in s.targets}
    rows={p:[] for p in paths};hooks=[]
    def hook(p):
        def record(_m,args):rows[p].append(sample_observation_rows(args[0]))
        return record
    try:
        for p in paths:hooks.append(modules[p].register_forward_pre_hook(hook(p)))
        velocity=forward()
    finally:
        for h in hooks:h.remove()
    for p,s in paths.items():
        if len(rows[p])!=s.repetitions:raise ValueError('Native target repetition count differs')
    return velocity,{p:torch.cat(v,0) for p,v in rows.items()}

def original_training_interface_velocity(policy,batch):
    """Independent native model.forward oracle with synthetic fixed x_t/time.

    The dummy actions are zeros solely to enter the native training interface.
    The flow-construction helper is replaced before it can interpolate actions;
    no demonstration actions or rewards are read. Returned value is head output.
    """
    from unittest.mock import patch
    from lerobot.policies.pi05 import modeling_pi05
    output=[];handle=policy.model.action_out_proj.register_forward_hook(lambda m,a,out:output.append(out.detach().clone()))
    try:
        with patch.object(modeling_pi05,'_build_flow_matching_inputs',return_value=(batch['x_t'],batch['time'])):
            policy.model.forward(images=[batch[f'image_{i}'] for i in range(3)],
                img_masks=[batch[f'image_mask_{i}'] for i in range(3)],tokens=batch['tokens'],masks=batch['masks'],
                actions=torch.zeros_like(batch['x_t']),noise=torch.zeros_like(batch['x_t']),time=batch['time'])
    finally:handle.remove()
    if len(output)!=1:raise ValueError('Native action head must execute exactly once')
    return output[0]

@torch.inference_mode()
def smoke(plan,plan_sha,run):
    run=Path(run)
    if (run/'native-smoke.json').exists():raise FileExistsError('Smoke receipt already exists')
    reports=[]
    with ExitStack() as stack:
        handles={g:stack.enter_context(safe_open(str(Path(plan['models'][g]['path'])/'model.safetensors'),framework='pt',device='cpu')) for g in GROUPS}
        policy=initialize(plan,handles)
        for group in GROUPS:
            batch=batch_for(plan,group,0,one=True)
            native,rows=all_inputs(policy,lambda:native_velocity(policy,batch))
            oracle,oracle_rows=all_inputs(policy,lambda:original_training_interface_velocity(policy,batch))
            if native.shape!=(1,50,32) or not torch.equal(native,oracle):raise ValueError('Native fixed-time velocity parity failed')
            if set(rows)!=set(oracle_rows) or any(not torch.equal(v,oracle_rows[p]) for p,v in rows.items()):
                raise ValueError('All-target native input parity failed')
            for s in (schedule()[0],schedule()[27],schedule()[48]):
                weights=read_stage_weights(handles,s);biases=read_stage_biases(handles,s)
                with candidate_block(policy,s,weights[group],biases[group]):
                    a=capture_stage(policy,s,lambda:native_velocity(policy,batch),stop_early=True)
                    b=capture_stage(policy,s,lambda:native_velocity(policy,batch),stop_early=False)
                if any(not torch.equal(a[p],b[p]) for p in s.targets):raise ValueError('Candidate early-stop parity failed')
            reports.append({'group':group,'observation':0,'replica':0,'physical_timestep':1.,'velocity_shape':list(native.shape),
                'native_velocity_bitwise':True,'sampled_inputs_bitwise':418,'candidate_blocks_bitwise':['vision0','language0','action0']})
            del batch,native,oracle,rows,oracle_rows,a,b;gc.collect();torch.cuda.empty_cache()
        del policy;gc.collect();torch.cuda.empty_cache()
    write(run/'native-smoke.json',{'status':'PASS','plan_sha256':plan_sha,'reports':reports,
        'formal_episodes':0,'calibration_static_observations':3,'gpu_used':True,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})

def export(policy,plan,run,metrics):
    output=Path(run)/'checkpoint'
    if output.exists():raise FileExistsError('Checkpoint output must be new')
    output.mkdir()
    modified={p+'.weight' for s in schedule() for p in s.targets}|{p+'.bias' for p in INTERFACES}
    live=policy.state_dict();tensors={}
    with safe_open(str(Path(plan['base']['path'])/'model.safetensors'),framework='pt',device='cpu') as base:
        if set(live)!=set(base.keys()):raise ValueError('Export native key coverage mismatch')
        for key in base.keys():
            reference=base.get_tensor(key)
            tensors[key]=(live[key].detach().to(device='cpu',dtype=reference.dtype).contiguous() if key in modified else reference)
            if tensors[key].shape!=reference.shape or not torch.isfinite(tensors[key]).all():raise ValueError('Invalid exported tensor')
    save_file(tensors,str(output/'model.safetensors'));del tensors
    support=Path(plan['models'][GROUPS[0]]['path'])
    for name in ('policy_preprocessor.json','policy_postprocessor.json','policy_preprocessor_step_3_normalizer_processor.safetensors',
                 'policy_postprocessor_step_0_unnormalizer_processor.safetensors'):
        shutil.copy2(support/name,output/name)
    shutil.copytree(support/'tokenizer',output/'tokenizer')
    cfg=read(support/'config.json');cfg['use_peft']=False;cfg['pretrained_path']=str(output);write(output/'config.json',cfg)
    model_sha=sha(output/'model.safetensors')
    manifest={'schema':'robotwin_regmeanpp_m3_native_graph_spectral_v1','status':'MATERIALIZED_GPU_PARITY_PRECHECK_PASSED',
        'plan_sha256':sha(Path(run)/'plan.json'),'model_sha256':model_sha,'groups':list(GROUPS),'modules':metrics,
        'total_regression_rows':TOTAL_ROWS,'target_weights':418,'modified_tensors':422,'total_tensors':813,
        'replay_prefix':'merged','noise_replica':0,'physical_timestep':1.,'action_labels_used':False,'expert_rollouts_used':False,
        'strict_original_equivalence':False,'method':'original_gram_mean_centered_weak_spectrum_smoothing_v1',
        'claim':'Table1 spectral smoothing extension on RoboTwin native graph; deployment success not yet evaluated.'}
    write(output/'block_regmeanpp_manifest.json',manifest)
    # Reopen every tensor after save, without a second GPU model allocation.
    with safe_open(str(output/'model.safetensors'),framework='pt',device='cpu') as f:
        if len(f.keys())!=813:raise ValueError('Saved tensor count differs')
        for key in f.keys():
            if not torch.isfinite(f.get_tensor(key)).all():raise ValueError('Nonfinite reloaded tensor')
    write(Path(run)/'BUILD-COMPLETE.json',{'status':'PASS','model_sha256':model_sha,'checkpoint':str(output),
        'checkpoint_tensors_reopened':813,'model_class_reload_required_before_formal':True,'formal_evaluation_authorized':False})

@torch.inference_mode()
def materialize(plan,plan_sha,run):
    run=Path(run);receipt=read(run/'native-smoke.json')
    if receipt['status']!='PASS' or receipt['plan_sha256']!=plan_sha:raise ValueError('Matching native GPU smoke required')
    if (run/'build-progress.jsonl').exists() or (run/'checkpoint').exists():raise FileExistsError('New independent build attempt required')
    kernel=load_kernel(KERNEL,sha);metrics={}
    with ExitStack() as stack:
        handles={g:stack.enter_context(safe_open(str(Path(plan['models'][g]['path'])/'model.safetensors'),framework='pt',device='cpu')) for g in GROUPS}
        policy=initialize(plan,handles)
        for stage in schedule():
            started=time.monotonic();weights=read_stage_weights(handles,stage);biases=read_stage_biases(handles,stage);inputs={}
            for group in GROUPS:
                collected={p:[] for p in stage.targets}
                with candidate_block(policy,stage,weights[group],biases[group]):
                    for task in range(10):
                        batch=batch_for(plan,group,task)
                        rows=capture_stage(policy,stage,lambda:native_velocity(policy,batch))
                        for p in stage.targets:collected[p].append(rows[p])
                        del batch,rows
                inputs[group]={p:torch.cat(v,0) for p,v in collected.items()}
                if any(x.shape[0]!=50*stage.repetitions*stage.rows_per_observation for x in inputs[group].values()):
                    raise ValueError('Realized per-expert row budget differs')
            merged,diagnostics=solve_complete_block(kernel,stage,inputs,weights,device='cuda')
            commit_block(policy,stage,merged);metrics.update(diagnostics)
            for p in stage.targets:
                if p in INTERFACES:metrics[p]['bias_semantics']='uniform_expert_mean_not_augmented_regression'
            event={'stage':stage.index,'family':stage.family,'targets':len(stage.targets),
                'elapsed_seconds':time.monotonic()-started,'max_smooth_residual':max(x['smooth_normal_equation_relative_residual'] for x in diagnostics.values())}
            with (run/'build-progress.jsonl').open('a') as f:f.write(json.dumps(event)+'\n');f.flush()
            print(json.dumps(event),flush=True)
            del inputs,collected,weights,biases,merged,diagnostics;gc.collect();torch.cuda.empty_cache()
        if len(metrics)!=418 or sum(sum(m['rows_by_expert'].values()) for m in metrics.values())!=TOTAL_ROWS:
            raise ValueError('Final complete scope/row-budget mismatch')
        export(policy,plan,run,metrics)
