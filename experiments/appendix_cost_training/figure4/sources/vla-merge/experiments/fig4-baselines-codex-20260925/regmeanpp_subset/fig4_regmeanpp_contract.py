"""Selected-expert binding and budgets; preserves the Table-1 RegMean++ kernel."""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

WORK=Path('/mnt/workspace/Wilson/parameter-fusion')
FIG4=WORK/'vla-merge-runtime/experiments/fig4-baselines-codex-20260925/plan.json'
FIG4_SHA='62ff0891df02a75d20a8b724f70f93a8373f75852a7385c88034993c326d7875'
SCHEMA='fig4_regmeanpp_selected_experts_static_smoothing_v1'
ACTIVE=None

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())

def frozen_figure():
    require(digest(FIG4)==FIG4_SHA,'Figure 4 CPU plan identity differs')
    p=read(FIG4)
    require(p['algorithms']['regmeanpp']['recipe']['replay_prefix']=='merged','RegMean++ requires merged prefix')
    return p

def validate_adapter_plan(plan,check_large_files=False):
    f=frozen_figure()
    job=next(x for x in f['new_builds'] if x['id']==plan['job_id'])
    require(job['method']=='regmeanpp' and plan['schema']==SCHEMA,'Wrong Figure 4 method')
    require(plan['fig4_plan_sha256']==FIG4_SHA and plan['fig4_plan']==str(FIG4),'Wrong parent plan')
    require(plan['experts']==job['experts'] and 2<=len(plan['experts'])<=3,'Selected expert order/count differs')
    require(plan['recipe']==f['algorithms']['regmeanpp']['recipe'],'Table 1 recipe changed')
    require(plan['spectral_smoothing']==f['algorithms']['regmeanpp']['spectral_smoothing'],'Smoothing rule changed')
    require(plan['output']==job['checkpoint'],'Output differs from frozen Figure 4 job')
    require(plan['expected_realized_rows']==job['calibration_rows']==592100*len(job['experts']),'Per-expert row budget changed')
    require(plan['observations_total']==50*len(job['experts']),'Observation budget changed')
    require(plan['subset_bank']==job['subset_bank'],'Subset-bank binding changed')
    require(digest(plan['subset_bank']['path'])==plan['subset_bank']['sha256'],'Subset-bank bytes changed')
    subset=read(plan['subset_bank']['path'])
    require(subset['subset_derivation']['expert_names']==plan['experts'],'Subset-bank projection differs')
    members=subset['expert_bank']['experts']
    require([x['name'] for x in members]==plan['experts'],'Subset bank includes excluded expert')
    require(subset['adaptation_domain']['expected']['logical_tensor_count']==422,'Logical tensor coverage differs')
    require(set(plan['inputs'])==set(plan['experts']),'Excluded or missing source expert')
    require(plan['dense_bank']==f['source_binding']['dense_bank'],'Dense bank identity differs')
    require(digest(plan['dense_bank']['path'])==plan['dense_bank']['sha256'],'Dense bank bytes changed')
    for n in plan['experts']:
        a=plan['inputs'][n];b=f['inputs']['regmeanpp'][n]
        require(a==b,'Selected static input binding differs: '+n)
        m=read(a['manifest_path'])
        require(digest(a['manifest_path'])==a['manifest_sha256'],'Source manifest bytes differ')
        require(m['source_kind']=='demo_observation_expert_generation' and m['demonstration_actions_used'] is False,'Not static observation input')
        selected=[x for x in m['samples'] if x['flow_index']==0]
        require(len(selected)==50 and [x['index'] for x in selected]==a['selected_source_indices'],'Flow0 indices differ')
        require(Path(a['replay_path']).stat().st_size==a['replay']['bytes'],'Source replay size differs')
        if check_large_files:require(digest(a['replay_path'])==a['replay_sha256'],'Source replay bytes differ')
        old=next(x for x in members if x['name']==n)
        require(old['path']==a['adapter_path'] and old['training_step']==10000,'Wrong source expert')
    for path,expected in plan['implementation_sha256'].items():
        require(digest(path)==expected,'Adapter/algorithm source differs: '+path)
    return plan

def validate_recipe(args):
    global ACTIVE
    require(digest(args.experiment_manifest)==args.expected_experiment_sha256,'Build plan SHA differs')
    plan=validate_adapter_plan(read(args.experiment_manifest),check_large_files=True)
    for k,v in plan['recipe'].items():require(getattr(args,k)==v,'Frozen solver argument differs: '+k)
    require([x.split('=',1)[0] for x in args.expert]==plan['experts'],'Expert CLI order differs')
    for field,pathkey in [('expert','adapter_path'),('calibration','replay_path'),('manifest','manifest_path')]:
        actual=[tuple(x.split('=',1)) for x in getattr(args,field)]
        require(actual==[(n,plan['inputs'][n][pathkey]) for n in plan['experts']],'CLI source binding differs')
    require(str(Path(args.output).absolute())==plan['output'],'Output CLI changed')
    require(str(Path(args.base_model).resolve())==plan['base_model'],'Base changed')
    require(str(Path(args.dense_expert_bank).resolve())==plan['dense_bank']['path'],'Dense bank CLI changed')
    # The parent plan is CPU-only. A later resource-allocated execution plan
    # must explicitly name this exact build-plan hash before any GPU work.
    permit_path=os.environ.get('FIG4_REGMEANPP_EXECUTION_PERMIT')
    require(bool(permit_path),'GPU blocked: no independent Figure 4 execution permit')
    permit=read(permit_path)
    require(permit.get('schema')=='fig4_regmeanpp_execution_permit_v1' and permit.get('allowed') is True,'Invalid execution permit')
    require(permit.get('build_plan_sha256')==args.expected_experiment_sha256 and permit.get('fig4_plan_sha256')==FIG4_SHA,'Execution permit identity differs')
    require(permit.get('job_id')==plan['job_id'],'Execution permit job differs')
    ACTIVE=plan
    return plan

def validate_dense_bank(path,experts=None,verify_weights=False):
    require(ACTIVE is not None,'Subset recipe must be validated first')
    bank=read(path)
    require(bank['kind']=='iclr2027_table1_peft_safe_dense_expert_bank' and bank['status']=='passed_parameter_exact','Invalid source dense bank')
    require(bank['deployment_policy']=='dense_only_no_unmerged_adapter_fallback','Dense-only expert representation required')
    rows={x['name']:x for x in bank['experts']};result={}
    require(list(experts)==ACTIVE['experts'],'Excluded expert entered materialization')
    for n in ACTIVE['experts']:
        row=rows[n];d=row['dense_checkpoint'];binding=ACTIVE['inputs'][n]
        require(str(Path(experts[n]).resolve())==row['source_adapter']['path']==binding['adapter_path'],'Expert adapter differs')
        require(d['path']==binding['dense_path'] and d['model_sha256']==binding['dense_sha256'],'Dense expert differs')
        for key in ('manifest','verification'):require(digest(d[key]['path'])==d[key]['sha256'],'Dense expert proof differs')
        require(Path(d['path'],'model.safetensors').is_file(),'Dense model absent')
        if verify_weights:require(digest(Path(d['path'],'model.safetensors'))==d['model_sha256'],'Dense model hash differs')
        result[n]={'path':d['path'],'model_sha256':d['model_sha256']}
    return result

def validate_trace(manifest,dense_path,name):
    require(ACTIVE is not None and name in ACTIVE['experts'],'Wrong subset trace')
    require(manifest['task']==name and Path(manifest['calibration_policy']).resolve()==Path(dense_path).resolve(),'Trace expert mismatch')
    for k,v in {'sample_count':50,'prompt_count':10,'flow_indices':[0],'source_kind':'static_demo_observation_pure_noise_t1','demonstration_actions_used':False,'static_repair_schema':SCHEMA}.items():require(manifest.get(k)==v,'Static trace mismatch: '+k)
    samples=manifest['samples'];require(len(samples)==50 and set(Counter(x['prompt_signature'] for x in samples).values())=={5},'Task allocation differs')
    require(all(x['flow_index']==0 and x['vision_count']==3 and x.get('simulator_seed') is None for x in samples),'Nonstatic trace')
    require([x['index'] for x in samples]==ACTIVE['inputs'][name]['selected_source_indices'],'Selected source rows changed')

def validate_rows_for_names(metrics,names):
    require(2<=len(names)<=3 and len(set(names))==len(names),'Wrong subset size')
    require(len(metrics)==418,'Full 418-module scope required')
    counts=Counter();total=0
    for name,row in metrics.items():
        n=50 if name in ('model.time_mlp_in','model.time_mlp_out') else 2400 if '.vision_tower.' in name else 800
        require(row['rows_by_expert']=={e:n for e in names},'Module quota/expert contamination: '+name)
        require(row.get('strict_original_equivalence') is False and row.get('spectral_regularization_applied') is True,'Undisclosed smoothing')
        require(row.get('offdiag_scale')==.3 and row.get('solve_dtype')=='float64/complex128','Changed solver')
        require(row.get('filter')=='smooth' and row.get('tau_rule')=='input_width * eps(float32) * deterministic_power_lambda_max_estimate','Changed filter')
        require(row.get('soup_centered_ridge') is False and row.get('correction_cap') is False,'Forbidden enhancement')
        require('relative_residual' not in row,'Ambiguous residual label')
        require(0<=row['smooth_normal_equation_relative_residual']<=1e-7 and math.isfinite(row['original_centered_equation_relative_residual']),'Numerical residual invalid')
        require(row['expert_objective_weights']=={e:1/len(names) for e in names},'Nonuniform/excluded expert masses')
        counts[n]+=1;total+=len(names)*n
    require(counts=={50:2,800:254,2400:162} and total==592100*len(names),'Total budget mismatch')
    return total

def validate_realized_rows(metrics):
    require(ACTIVE is not None,'Missing subset plan context')
    return validate_rows_for_names(metrics,ACTIVE['experts'])
