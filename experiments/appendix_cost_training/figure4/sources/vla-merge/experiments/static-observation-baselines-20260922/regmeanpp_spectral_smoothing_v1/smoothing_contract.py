"""Frozen disclosed spectral-smoothing recipe, not strict original equivalence."""
from collections import Counter
import hashlib
import json
from pathlib import Path

NAMES=('spatial','object','goal','long')
SCHEMA='regmeanpp_original_gram_alpha03_static_t1_spectral_smoothing_v1'
BANK_SHA='d61a5f9e56bb0f76d8186a33dc26fe283cf8cfab220e903a460f92e4507f209b'
RECIPE=dict(offdiag_scale=.3,ridge_ratio=0.,max_correction_ratio=0.,prior_model=None,
    max_rows_per_sample=16,expert_loss_normalization='none',expert_loss_normalization_power=1.,
    expert_aggregation='mean',expert_weight_plan=None,objective_grouping='expert',
    replay_prefix='merged',require_objective_improvement=False,freeze_prefix_from_prior=False,
    freeze_action_interface_from_prior=False,action_block_solver='independent_linear',
    max_states_per_calibration_source=0,calibration_source_weight=[])

SMOOTHING=dict(filter='lambda/(lambda^2+tau^2)',centering='arithmetic_expert_weight_mean',
    tau_rule='input_width * eps(float32) * deterministic_power_lambda_max_estimate',
    power_iterations=64,power_seed=20260922,numerical_rank_dtype='torch.float32',
    solve_dtype='float64/complex128',strategy='auto_dense_or_complex_Woodbury',
    output_chunk_size=128,strict_original_equivalence=False,
    threshold_basis='FP32 numerical-rank-inspired engineering rule, not a statistical precision theorem',
    success_based_selection=False,final_weight_clipping=False,nonzero_feature_energy_floor=False)

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def validate_recipe(args):
    for key,expected in RECIPE.items():
        if getattr(args,key)!=expected:raise ValueError(f'Frozen original-Gram recipe drift: {key}')
    path=Path(args.experiment_manifest).resolve()
    if digest(path)!=args.expected_experiment_sha256:raise ValueError('Experiment manifest hash differs')
    plan=json.loads(path.read_text())
    if plan['schema']!=SCHEMA or plan['recipe']!=RECIPE or plan['formal_outcomes_used_for_recipe_selection'] is not False:
        raise ValueError('Not the frozen alpha-.3 static observation plan')
    if plan.get('spectral_smoothing')!=SMOOTHING:raise ValueError('Smoothing rule differs')
    if digest(args.dense_expert_bank)!=BANK_SHA:raise ValueError('Dense expert bank changed')
    for raw in args.expert:
        name,value=raw.split('=',1)
        if name not in NAMES or str(Path(value).resolve())!=plan['inputs'][name]['adapter_path']:raise ValueError('Expert differs from plan')
    if [raw.split('=',1)[0] for raw in args.expert]!=list(NAMES):raise ValueError('Expert ordering differs')
    for key,field in [('calibration','replay_path'),('manifest','manifest_path')]:
        if len(getattr(args,key))!=4:raise ValueError('Exactly one static source per expert required')
        for raw in getattr(args,key):
            name,value=raw.split('=',1);identity=plan['inputs'][name]
            if str(Path(value).resolve())!=identity[field] or digest(value)!=identity[field.replace('_path','_sha256')]:raise ValueError('Static source binding differs')
    for raw,expected in plan['implementation_sha256'].items():
        if digest(raw)!=expected:raise ValueError(f'Implementation changed: {raw}')
    if str(Path(args.output).absolute())!=plan['output']:raise ValueError('Output path differs from plan')
    if str(Path(args.base_model).resolve())!=plan['base_model']:raise ValueError('Base model differs from plan')
    return plan

def validate_trace(manifest,dense_path,name):
    if manifest.get('task')!=name or Path(manifest['calibration_policy']).resolve()!=Path(dense_path).resolve():raise ValueError('Static source expert differs')
    required=dict(sample_count=50,prompt_count=10,flow_indices=[0],
        source_kind='static_demo_observation_pure_noise_t1',demonstration_actions_used=False,
        static_repair_schema=SCHEMA,method='pi05_full_vision_language_action_block_regmeanpp_replay_calibration')
    for key,expected in required.items():
        if manifest.get(key)!=expected:raise ValueError(f'Static source contract differs: {key}')
    samples=manifest['samples']
    if len(samples)!=50 or set(Counter(s['prompt_signature'] for s in samples).values())!={5}:raise ValueError('Require five observations per each of ten tasks')
    if any(s.get('flow_index')!=0 or s.get('vision_count')!=3 or s.get('simulator_seed') is not None for s in samples):raise ValueError('Rollout or denoised state entered static source')
    if len({(s['prompt_signature'],s['selected_request_slot']) for s in samples})!=50:raise ValueError('Duplicate static observations')

def validate_realized_rows(metrics):
    classes=Counter();total=0
    for name,entry in metrics.items():
        count=50 if name in ('model.time_mlp_in','model.time_mlp_out') else 2400 if '.vision_tower.' in name else 800
        if entry.get('rows_by_expert')!={n:count for n in NAMES}:raise ValueError(f'Row budget differs: {name}')
        if entry.get('offdiag_scale')!=.3 or entry.get('soup_centered_ridge') is not False or entry.get('correction_cap') is not False or entry.get('solve_dtype')!='float64/complex128':raise ValueError('Frozen smoothing solver drift')
        if entry.get('strict_original_equivalence') is not False or entry.get('spectral_regularization_applied') is not True:raise ValueError('Undisclosed spectral extension')
        if entry.get('tau_rule')!=SMOOTHING['tau_rule'] or entry.get('filter')!='smooth':raise ValueError('Smoothing filter differs')
        if 'relative_residual' in entry:raise ValueError('Ambiguous old residual field is forbidden')
        residual=entry.get('smooth_normal_equation_relative_residual')
        original=entry.get('original_centered_equation_relative_residual')
        if not isinstance(residual,(int,float)) or not 0<=residual<=1e-7:raise ValueError('Smoothed equation failed its residual check')
        if not isinstance(original,(int,float)) or not 0<=original<float('inf'):raise ValueError('Original equation deviation is missing/nonfinite')
        classes[count]+=1;total+=4*count
    if classes!={50:2,800:254,2400:162} or total!=2368400:raise ValueError('Full418 row budget differs')
    return total
