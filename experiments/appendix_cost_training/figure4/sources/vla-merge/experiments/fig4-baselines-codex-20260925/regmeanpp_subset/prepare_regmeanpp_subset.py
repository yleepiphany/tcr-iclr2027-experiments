#!/usr/bin/env python3
"""Freeze two subset build adapters without changing the Figure 4 parent plan."""
import argparse
import ast
import json
from pathlib import Path
import sys

from fig4_regmeanpp_contract import FIG4,FIG4_SHA,SCHEMA,digest,frozen_figure,validate_adapter_plan,require

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/workspace/Wilson/parameter-fusion')
OUTPUT=WORK/'vla-merge-runtime/experiments/fig4-baselines-codex-20260925/regmeanpp-subset-cpu-v1'
SOURCE=WORK/'pi05_lora_finetune_v2_20260826'

def source_parity(figure):
    original=next(Path(x['path']) for x in figure['algorithm_sources'] if x['path'].endswith('regmeanpp_spectral_smoothing_v1/materialize_smoothed.py'))
    old=ast.parse(original.read_text());new=ast.parse((HERE/'materialize_regmeanpp_subset.py').read_text())
    functions=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    a,b=functions(old),functions(new)
    changed=sorted(k for k in a.keys()&b.keys() if a[k]!=b[k])
    require(changed==['load_replay_states','main'] and a.keys()==b.keys(),'Unexpected numerical/replay function modification')
    return {'changed_functions':changed,'unchanged_functions':len(a)-2,'all_nested_numerical_and_capture_functions_unchanged':True}

def plan_for(job,figure):
    source_plan_path=Path(figure['source_binding']['regmeanpp_plan']['path'])
    require(digest(source_plan_path)==figure['source_binding']['regmeanpp_plan']['sha256'],'Original method plan changed')
    old=json.loads(source_plan_path.read_text())
    implementation=dict(old['implementation_sha256'])
    for name in ('materialize_regmeanpp_subset.py','fig4_regmeanpp_contract.py','prepare_regmeanpp_subset.py'):
        implementation[str(HERE/name)]=digest(HERE/name)
    return {'schema':SCHEMA,'fig4_plan':str(FIG4),'fig4_plan_sha256':FIG4_SHA,
            'job_id':job['id'],'subset':job['subset'],'experts':job['experts'],
            'subset_bank':job['subset_bank'],'dense_bank':figure['source_binding']['dense_bank'],
            'base_model':old['base_model'],'inputs':{n:figure['inputs']['regmeanpp'][n] for n in job['experts']},
            'recipe':figure['algorithms']['regmeanpp']['recipe'],
            'spectral_smoothing':figure['algorithms']['regmeanpp']['spectral_smoothing'],
            'expected_realized_rows':job['calibration_rows'],'observations_total':50*len(job['experts']),
            'output':job['checkpoint'],'implementation_sha256':implementation,
            'initialization':'Arithmetic mean of selected dense experts only; mean bias. No four-expert checkpoint or Soup is used.',
            'parameter_search':False,'training':False,'gpu_launch_allowed_by_this_plan':False,
            'execution_requires':'A separately frozen resource/owner-approved execution permit naming this exact build-plan SHA, plus native/GPU admission before launch',
            'evaluation':{'repeat':1,'episodes':100*len(job['experts']),'suites':job['experts'],'across_build_error_bar':False}}

def command(plan,path):
    cmd=[str(SOURCE/'.venv/bin/python'),'-u',str(HERE/'materialize_regmeanpp_subset.py'),
         '--experiment-manifest='+str(path),'--expected-experiment-sha256='+digest(path),
         '--dense-expert-bank='+plan['dense_bank']['path'],'--base-model='+plan['base_model'],
         '--output='+plan['output'],'--offdiag-scale=.3','--ridge-ratio=0','--max-correction-ratio=0',
         '--max-rows-per-sample=16','--expert-loss-normalization=none','--replay-prefix=merged',
         '--allow-mixed-calibration-policies','--device=cuda']
    for n in plan['experts']:
        x=plan['inputs'][n]
        cmd += ['--expert='+n+'='+x['adapter_path'],'--calibration='+n+'='+x['replay_path'],'--manifest='+n+'='+x['manifest_path']]
    return cmd

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','preflight'));parser.add_argument('--rehash-large-sources',action='store_true');args=parser.parse_args()
    figure=frozen_figure();parity=source_parity(figure);rows=[]
    for job in figure['new_builds']:
        if job['method']!='regmeanpp':continue
        expected=plan_for(job,figure);path=OUTPUT/job['subset']/'plan.json'
        if args.action=='prepare':
            require(not path.exists(),'Refusing to overwrite frozen subset adapter plan')
            validate_adapter_plan(expected,args.rehash_large_sources)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x') as f:json.dump(expected,f,indent=2,sort_keys=True);f.write('\n')
        observed=json.loads(path.read_text());require(observed==expected,'Subset adapter plan drift')
        validate_adapter_plan(observed,args.rehash_large_sources)
        rows.append({'job_id':job['id'],'plan':str(path),'plan_sha256':digest(path),
                     'experts':observed['experts'],'observations':observed['observations_total'],
                     'regression_rows':observed['expected_realized_rows'],
                     'replay_prefix':observed['recipe']['replay_prefix'],
                     'gpu_execution':'blocked_until_separate_execution_permit_and_resource_admission',
                     'launch_command_template':command(observed,path)})
    result={'status':'PASS_CPU_SUBSET_ADAPTER','fig4_parent_plan_sha256':FIG4_SHA,'source_function_parity':parity,
            'subsets':rows,'large_source_verification':'rehashed' if args.rehash_large_sources else 'sizes and frozen hashes; exact replay and dense model hashes required again by execution contract',
            'gpu_jobs_launched':0,'existing_queues_modified':False,'four_expert_endpoint_rebuilt':False}
    if args.action=='prepare':
        with (OUTPUT/'CPU-ADAPTER-RECEIPT.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
