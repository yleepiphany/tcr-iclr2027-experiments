"""Strict saved-weight identity, original manifest and native model reload gate."""
from __future__ import annotations
import gc,hashlib,math
from pathlib import Path
import supervisor as s

GROUPS=('coordination','receptacle','precision')
INTERFACES={'model.action_in_proj','model.action_out_proj','model.time_mlp_in','model.time_mlp_out'}

def validate_manifest(manifest,core_plan,model_sha):
    expected={'schema':'robotwin_regmeanpp_m3_native_graph_spectral_v1','plan_sha256':s.CORE_SHA,'model_sha256':model_sha,
        'groups':list(GROUPS),'total_regression_rows':1776300,'target_weights':418,'modified_tensors':422,'total_tensors':813,
        'replay_prefix':'merged','noise_replica':0,'physical_timestep':1.,'action_labels_used':False,'expert_rollouts_used':False,
        'strict_original_equivalence':False,'method':'original_gram_mean_centered_weak_spectrum_smoothing_v1'}
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Checkpoint manifest core identity/recipe differs')
    quotas={q['module']:q for q in core_plan['quotas']};metrics=manifest.get('modules',{})
    if len(quotas)!=418 or set(metrics)!=set(quotas):raise ValueError('Expected 418 exact module identities')
    for path,q in quotas.items():
        m=metrics[path]
        expected_m={'method':'original_gram_mean_centered_weak_spectrum_smoothing_v1','strict_original_equivalence':False,
            'filter':'smooth','offdiag_scale':.3,'solve_dtype':'float64/complex128','input_width':q['weight_shape'][1],
            'output_width':q['weight_shape'][0],'rows_by_expert':{g:q['rows_per_expert'] for g in GROUPS},
            'teacher_output_targets_used':False,'success_based_selection':False,'final_weight_clipping':False,
            'nonzero_feature_energy_floor':False,'trust_cap':False}
        if any(m.get(k)!=v for k,v in expected_m.items()):raise ValueError('Module solver/row identity mismatch: '+path)
        residue=m.get('smooth_normal_equation_relative_residual')
        if not isinstance(residue,(float,int)) or not math.isfinite(residue) or not 0<=residue<=1e-7:
            raise ValueError('Invalid smoothed equation residual: '+path)
        original=m.get('original_centered_equation_relative_residual')
        if not isinstance(original,(float,int)) or not math.isfinite(original) or original<0 or 'relative_residual' in m:
            raise ValueError('Original versus smooth residual semantics invalid: '+path)
        if path in INTERFACES and m.get('bias_semantics')!='uniform_expert_mean_not_augmented_regression':
            raise ValueError('Interface bias rule differs')
    return {'modules':418,'regression_rows':sum(sum(m['rows_by_expert'].values()) for m in metrics.values()),
        'max_smooth_relative_residual':max(m['smooth_normal_equation_relative_residual'] for m in metrics.values()),
        'max_original_centered_relative_residual_report_only':max(m['original_centered_equation_relative_residual'] for m in metrics.values())}

def accept(core_plan,supervisor_plan_path,root,resource_check):
    import torch
    from safetensors import safe_open
    import materialize_regmeanpp_m3 as original
    checkpoint=s.CORE/'checkpoint';model_file=checkpoint/'model.safetensors';model_sha=s.sha(model_file)
    built=s.read(s.CORE/'BUILD-COMPLETE.json')
    if built.get('status')!='PASS' or built.get('model_sha256')!=model_sha or Path(built['checkpoint'])!=checkpoint:
        raise ValueError('Build completion model SHA/path binding differs')
    diagnostics=validate_manifest(s.read(checkpoint/'block_regmeanpp_manifest.json'),core_plan,model_sha)
    original.setup_native_imports()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    gc.collect();torch.cuda.empty_cache();resource_check(force=True)
    cfg=PreTrainedConfig.from_pretrained(checkpoint)
    if cfg.use_peft or cfg.max_action_dim!=32 or cfg.chunk_size!=50 or list(cfg.output_features['action'].shape)!=[14]:
        raise ValueError('Exported native action/deployment config differs')
    cfg.device='cuda';cfg.compile_model=False;cfg.gradient_checkpointing=False;cfg.pretrained_path=checkpoint
    policy=PI05Policy.from_pretrained(checkpoint,config=cfg).to('cuda').eval()
    with torch.inference_mode():
        live=policy.state_dict();checked=0
        with safe_open(str(model_file),framework='pt',device='cpu') as f:
            if set(live)!=set(f.keys()) or len(live)!=813:raise ValueError('Native reload tensor coverage differs')
            for key in f.keys():
                saved=f.get_tensor(key);loaded=live[key].detach().cpu()
                if loaded.shape!=saved.shape or loaded.dtype!=saved.dtype or not torch.equal(loaded,saved) or not torch.isfinite(saved).all():
                    raise ValueError('Native reload tensor mismatch/nonfinite: '+key)
                checked+=1;del saved,loaded
        reports=[]
        for group in GROUPS:
            resource_check(force=True);batch=original.batch_for(core_plan,group,0,one=True)
            velocity=original.native_velocity(policy,batch)
            if tuple(velocity.shape)!=(1,50,32) or not torch.isfinite(velocity).all():raise ValueError('Invalid native reloaded velocity')
            value=velocity.detach().cpu().float().contiguous()
            reports.append({'group':group,'observation':0,'replica':0,'physical_timestep':1.,'velocity_shape':list(value.shape),
                'velocity_sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest(),'max_abs_velocity':float(value.abs().max()),'finite':True})
            del batch,velocity,value
    del live,policy;gc.collect();torch.cuda.empty_cache()
    if s.sha(model_file)!=model_sha:raise ValueError('Checkpoint changed during native reload')
    receipt={'status':'PASS','supervisor_plan_sha256':s.sha(supervisor_plan_path),'core_plan_sha256':s.CORE_SHA,
        'native_prebuild_smoke_sha256':s.SMOKE_SHA,'model_sha256':model_sha,'checkpoint':str(checkpoint),
        'model_manifest_sha256':s.sha(checkpoint/'block_regmeanpp_manifest.json'),'build_complete_sha256':s.sha(s.CORE/'BUILD-COMPLETE.json'),
        'saved_and_loaded_tensors_bitwise':checked,'native_reload_reports':reports,'diagnostics':diagnostics,
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'formal_episodes':0,'formal_performance_claim':False}
    s.write(Path(root)/'MODEL-ACCEPTED.json',receipt)
    return receipt
