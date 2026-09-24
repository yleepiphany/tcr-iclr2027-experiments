"""Derive block-atomic replay order and feasible row quotas from native traces."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import re

def block_of(name):
    patterns=(r'^(backbone\.vision_backbone\.[^.]+\.blocks\.\d+)\.',
              r'^(backbone\.language_model\.model\.layers\.\d+)\.',
              r'^(backbone\.projector)\.',r'^(action_head)\.',r'^(proprio_projector)\.')
    for pattern in patterns:
        found=re.match(pattern,name)
        if found:return found.group(1)
    if name=='backbone.language_model.lm_head':return name
    raise ValueError(f'Unclassified native Linear: {name}')

def make_plan(traces,*,cap=8,requests_per_expert=50):
    if len(traces)!=4 or cap<=0 or requests_per_expert<=0:raise ValueError('Invalid plan inputs')
    orders=[[r['name'] for r in t] for t in traces]
    if any(order!=orders[0] for order in orders[1:]):raise ValueError('Expert native call orders differ')
    names=list(dict.fromkeys(orders[0]));grouped=OrderedDict();rows_total=0
    for name in names:
        calls=[[r for r in t if r['name']==name] for t in traces]
        count=len(calls[0])
        if cap<count:raise ValueError('Cannot cover every native call')
        reference=calls[0][0]['weight_shape']
        if any(r['weight_shape']!=reference for c in calls for r in c):raise ValueError('Module shapes differ')
        available=[]
        for ordinal in range(count):
            sizes=[]
            for records in calls:
                shape=records[ordinal]['input_shape']
                if shape[-1]!=reference[1] or shape[0]!=1:raise ValueError('Unexpected native input shape')
                n=1
                for d in shape[:-1]:n*=d
                sizes.append(n)
            available.append(min(sizes))
        per_call=[min(n,cap//count+(i<cap%count)) for i,n in enumerate(available)]
        total=sum(per_call)*requests_per_expert*4;rows_total+=total
        grouped.setdefault(block_of(name),[]).append({'module':name,'native_calls':count,
               'weight_shape':reference,'rows_per_request':sum(per_call),'per_call_rows':per_call,
               'rows_per_expert':total//4,'rows_all_experts':total,
               'bias_column':'augment iff present; determine from native module',
               'solver_system_upper_bound':min(total,reference[1]+1)})
    # First occurrence determines a valid order only if blocks do not interleave
    # inside one camera/branch in a way not covered by the recognized architecture.
    return {'schema':'oft_block_atomic_calibration_plan_v1','blocks':grouped,
            'block_count':len(grouped),'linear_count':len(names),'native_call_count':len(orders[0]),
            'row_cap_per_request_across_calls':cap,'requests_per_expert':requests_per_expert,
            'planned_rows_per_pass_if_all_linears_solved':rows_total,
            'within_block':'temporarily install full expert block state, collect all its Linear inputs in one replay; restore then solve/write all block Linears before advancing',
            'unchanged_parameters':'retain unchanged; log if a Linear is identical across experts and therefore skipped',
            'scope_status':'trace-derived capacity plan; final scope requires expert tensor identity and supported non-Linear fallback audit',
            'not_a_completed_merge':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trace-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();traces=[]
    for suite in ('spatial','object','goal','long'):
        path=a.trace_root/'jobs'/f'soup-smoke-{suite}-r01/eval/smoke.json'
        d=json.loads(path.read_text())
        if d['identity_max_abs']!=0:raise ValueError('Unverified native trace')
        traces.append(d['linear_call_order'])
    plan=make_plan(traces)
    with a.output.open('x') as f:json.dump(plan,f,indent=2)
    print(json.dumps({k:v for k,v in plan.items() if k!='blocks'},indent=2))
