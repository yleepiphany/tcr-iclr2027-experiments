"""Portable CPU checks of Figure 4 source snapshots and frozen evidence."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
SOURCES=HERE/'sources'
ORIGINAL_ROOT='/mnt/workspace/Wilson/parameter-fusion/'
CODE=SOURCES/'vla-merge/experiments/fig4-baselines-codex-20260925'
RUNTIME=SOURCES/'vla-merge-runtime/experiments/fig4-baselines-codex-20260925'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,message):
    if not ok:raise ValueError(message)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def source_parity():
    baseline=SOURCES/'vla-merge/experiments/static-observation-baselines-20260922/regmeanpp_spectral_smoothing_v1/materialize_smoothed.py'
    current=CODE/'regmeanpp_subset/materialize_regmeanpp_subset.py'
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef)}
    old,new=functions(baseline),functions(current)
    require(old.keys()==new.keys(),'RegMean++ function set changed')
    changed=sorted(k for k in old if old[k]!=new[k])
    require(changed==['load_replay_states','main'],'RegMean++ numerical or capture function changed')
    require(len(old)-len(changed)==37,'Unexpected unchanged function count')
    return {'unchanged_functions':37,'changed_functions':changed}

def test_ties_oracle():
    import numpy as np
    core=load(SOURCES/'vla-merge/scripts/ta_ties_core.py','released_fig4_ties_core')
    rng=np.random.default_rng(215)
    for count in [2,3]:
        vectors=[rng.integers(-5,6,257).astype(np.float32) for _ in range(count)]
        vectors[1][:24]=-vectors[0][:24]
        trimmed=[]
        for vector in vectors:
            threshold=sorted(abs(float(x)) for x in vector)[len(vector)-int(len(vector)*.3)-1]
            row=vector.astype(np.float64);trimmed.append(np.where(np.abs(row)>=threshold,row,0.))
        matrix=np.stack(trimmed);sign=np.sign(matrix.sum(axis=0));fallback=np.sign(sign.sum())
        if fallback:sign[sign==0]=fallback
        picked=np.where(np.where(sign>0,matrix>0,matrix<0),matrix,0.)
        expected=picked.sum(axis=0)/np.maximum(np.count_nonzero(picked,axis=0),1)
        actual,_=core.ties_direction(vectors,keep_fraction=.3,chunk_size=31)
        require(np.array_equal(actual,expected),'TIES differs from independent global oracle')
    return {'expert_counts':[2,3],'independent_numpy_oracle':'bitwise_equal'}

def test_regmean_contract():
    contract=load(CODE/'regmeanpp_subset/fig4_regmeanpp_contract.py','released_fig4_regmean_contract')
    modules=read(HERE.parent/'configs/module_scope_418.json')['modules']
    totals={}
    for names in [['spatial','goal'],['spatial','object','goal']]:
        metrics={}
        for name in modules:
            rows=50 if name in ['model.time_mlp_in','model.time_mlp_out'] else 2400 if '.vision_tower.' in name else 800
            metrics[name]={'rows_by_expert':{n:rows for n in names},'strict_original_equivalence':False,
                'spectral_regularization_applied':True,'offdiag_scale':.3,'solve_dtype':'float64/complex128',
                'filter':'smooth','tau_rule':'input_width * eps(float32) * deterministic_power_lambda_max_estimate',
                'soup_centered_ridge':False,'correction_cap':False,'smooth_normal_equation_relative_residual':1e-12,
                'original_centered_equation_relative_residual':.02,'expert_objective_weights':{n:1/len(names) for n in names}}
        total=contract.validate_rows_for_names(metrics,names)
        require(total==592100*len(names),'Selected-expert row budget differs')
        totals[str(len(names))]=total
        bad=copy.deepcopy(metrics);bad[modules[0]]['rows_by_expert']['excluded-expert']=800
        try:contract.validate_rows_for_names(bad,names)
        except ValueError:pass
        else:raise ValueError('Excluded expert accepted')
        bad=copy.deepcopy(metrics);bad[modules[0]]['soup_centered_ridge']=True
        try:contract.validate_rows_for_names(bad,names)
        except ValueError:pass
        else:raise ValueError('Unregistered ridge accepted')
    return {'rows':totals,'excluded_expert_and_ridge_rejected':True}

def check_release():
    provenance=read(HERE/'PROVENANCE.json')
    for row in provenance['files']:
        path=HERE/row['packaged_path']
        require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],f'Snapshot drift: {path}')
        require(path.suffix not in ['.safetensors','.pt','.bin','.mp4'],'Binary asset entered source package')
        if path.suffix=='.py':ast.parse(path.read_text())
    parent=read(RUNTIME/'plan.json');parent_sha=read(RUNTIME/'PLAN-SHA256.json')['plan_sha256']
    require(sha(RUNTIME/'plan.json')==parent_sha,'Parent Figure 4 plan changed')
    require(len(parent['new_builds'])==6 and len(parent['new_evaluations'])==15 and parent['new_episode_total']==1500,'Parent matrix differs')
    reuse=read(RUNTIME/'four-expert-reuse.json')
    require(reuse['accepted'] and reuse['total_jobs']==36 and reuse['total_episodes']==3600,'Four-expert reuse differs')
    ties=read(RUNTIME/'ties-formal-v1/plan.json')
    require(sha(RUNTIME/'ties-formal-v1/plan.json')==read(RUNTIME/'ties-formal-v1/PLAN-SHA256.json')['sha256'],'TIES formal plan changed')
    require(len(ties['jobs'])==5 and sum(j['episodes'] for j in ties['jobs'])==500,'TIES formal denominator differs')
    for subset,model in ties['models'].items():
        native=read(RUNTIME/'ties-formal-v1/native-cpu-actions-v1'/subset/'DONE.json')
        require(native['model_sha256']==model['sha256'] and native['native_denoising_calls']==10 and native['all_finite'] and not native['gpu_initialized'],'Native CPU receipt differs')
    reg=read(RUNTIME/'regmeanpp-queue-v3/plan.json')
    require(reg['fig4_plan_sha256']==parent_sha and len(reg['builds'])==2 and len(reg['evaluations'])==5 and reg['episodes']==500,'RegMean++ queue differs')
    for subset in ['spatial-goal','spatial-object-goal']:
        p=read(RUNTIME/'regmeanpp-subset-cpu-v1'/subset/'plan.json')
        require(p['recipe']['replay_prefix']=='merged' and p['recipe']['offdiag_scale']==.3 and p['recipe']['ridge_ratio']==0 and p['recipe']['max_correction_ratio']==0,'RegMean++ fixed recipe differs')
        require(p['expected_realized_rows']==592100*len(p['experts']),'RegMean++ plan row count differs')
    return {'status':'PASS','snapshot_files':len(provenance['files']),
        'snapshot_bytes':sum(x['bytes'] for x in provenance['files']),
        'plan_and_recorded_readiness_bindings':'PASS','regmean_function_parity':source_parity(),
        'ties_kernel':test_ties_oracle(),'regmean_contract':test_regmean_contract(),
        'full_native_checks_rerun_locally':False,
        'native_check_evidence':'Byte-preserved original CPU receipts; full PyTorch/safetensors/native assets required to rerun them.',
        'gpu_or_training_started':False}

if __name__=='__main__':print(json.dumps(check_release(),indent=2))
