"""CPU release smoke: real source hashes/imports/contracts, never starts a GPU."""
from __future__ import annotations
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES']=''
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
from workflow import digest
from costs import extract
from continued_training import inspect as training_status
from action_metrics import describe as action_status

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root',type=Path,required=True)
    parser.add_argument('--mode',choices=['cpu'],default='cpu');parser.add_argument('--output',type=Path)
    args=parser.parse_args();checks=[]
    provenance=json.loads((HERE/'PROVENANCE.json').read_text())
    for row in provenance['files']:
        p=HERE/row['packaged_path']
        if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise ValueError(f'Packaged source/config drift: {p}')
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
    checks.append({'name':'actual_source_and_config_hashes','pass':True,'files':len(provenance['files'])})
    source=HERE/'sources/vla-merge';appendix=source/'experiments/claude-pi05-fixed-appendix-20260923'
    sys.path[:0]=[str(appendix),str(source/'scripts'),str(source/'experiments/claude-three-level-main-20260921')]
    import appendix_contract,appendix_row_budget,appendix_followon_contract_v8,appendix_cache_followon_contract_v8
    import appendix_uniform_contract_v8,appendix_scope_contract_v8,appendix_union_contract
    names=set(json.loads((HERE/'configs/module_scope_418.json').read_text())['modules'])
    actual={tier:appendix_row_budget.realized_budget(names,tier) for tier in ('low','base','high')}
    if actual!={'low':666000,'base':1331600,'high':1997200}:raise ValueError('Realized row budgets differ')
    if appendix_row_budget.single_union_budget(names)!=5772800:raise ValueError('Union budget differs')
    for nrows,cameras in [(50,1),(256,3),(968,1)]:
        for request in range(5):
            sets={tier:{(flow,camera,row) for flow in (0,5,9) for camera in range(cameras)
                for row in appendix_row_budget.nested_row_indices(nrows,tier,flow,request,cameras=cameras,camera=camera)}
                for tier in ('low','base','high')}
            if not sets['low']<sets['base']<sets['high']:raise ValueError('Row subsets lost nesting')
    checks.append({'name':'real_contract_imports_and_nested_row_budgets','pass':True,'module_count':len(names),'second_pass_rows':actual})
    recipes=json.loads((HERE/'workflows.json').read_text())
    for name,spec in recipes['workflows'].items():
        for stage in ('build','evaluate'):
            if not (appendix/spec[stage]).is_file():raise FileNotFoundError(spec[stage])
    checks.append({'name':'appendix_entrypoints_present','pass':True,'workflows':len(recipes['workflows'])})
    with tempfile.TemporaryDirectory(prefix='appendix-cost-smoke-') as temp:
        root=Path(temp);manifest=root/'manifest.json';resource=root/'resource.json'
        manifest.write_text(json.dumps({'method':'fixture','module_count':1,'realized_row_total':12,
            'modules':{'x':{'rows_by_expert':{'a':5,'b':7}}},'gradient_or_backward':False}))
        result=extract(manifest)
        if result['realized_rows']!=12 or result['worker_wall_seconds'] is not None:raise ValueError('Missing cost was invented')
        resource.write_text(json.dumps({'wall_seconds':3.,'peak_allocator_gib':2.,'return_code':0}))
        measured=extract(manifest,[resource])
        if measured['worker_wall_seconds']!=3. or measured['resources'][0]['peak_allocated_mib']!=2048.:raise ValueError('Cost unit conversion differs')
        try:extract(manifest,[resource,resource])
        except ValueError:pass
        else:raise ValueError('Duplicate cost record accepted')
    checks.append({'name':'cost_null_semantics_units_and_duplicate_guard','pass':True})
    figure_path=HERE/'figure4/check.py'
    spec=importlib.util.spec_from_file_location('figure4_release_check',figure_path)
    figure_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(figure_module)
    figure_result=figure_module.check_release()
    checks.append({'name':'figure4_source_plans_numpy_oracle_and_subset_contract','pass':figure_result['status']=='PASS','details':figure_result})
    blocked=training_status(HERE/'configs/continued_training.blocked.json')
    if blocked['status']!='BLOCKED' or not blocked['missing_protocol_fields']:raise ValueError('Missing training protocol was invented')
    assets=json.loads((HERE/'ASSETS.json').read_text())['assets'];root=args.workspace_root.resolve()
    missing=[row['id'] for row in assets if row.get('required') and not (root/row['workspace_relative_path']).exists()]
    result={'status':'PASS','mode':'cpu','checks':checks,'gpu_started':False,'training_started':False,
        'source_repository_head':provenance['source_repository_head'],'workspace_root':str(root),
        'external_asset_presence':{'required':sum(bool(r.get('required')) for r in assets),'missing':missing,
                                  'large_asset_hashes_recomputed':False},
        'continued_training':blocked,
        'table11_regmean_a03':action_status(root,root/'release-output/table11-regmean-a03'),
        'claim_boundary':'PASS means packaged source/contracts and CPU extraction smoke passed. It does not claim GPU experiments, missing measurements or continued training have been completed.'}
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text,end='')

if __name__=='__main__':main()
