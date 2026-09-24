"""CPU-only Table14 extraction from supplied receipts; missing measurements stay null."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite_nonnegative(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0

def extract(manifest_path,resource_paths=()):
    path=Path(manifest_path);m=json.loads(path.read_text());by_expert={}
    for module in (m.get('modules') or {}).values():
        rows=module.get('rows_by_expert')
        if isinstance(rows,dict):
            for name,count in rows.items():
                if type(count) is not int or count<0:raise ValueError('Invalid recorded row count')
                by_expert[name]=by_expert.get(name,0)+count
        else:
            for key,count in module.items():
                if key.endswith('_rows') and type(count) is int:by_expert[key[:-5]]=by_expert.get(key[:-5],0)+count
    realized=sum(by_expert.values()) if by_expert else m.get('realized_row_total')
    if by_expert and m.get('realized_row_total') not in (None,realized):raise ValueError('Declared versus per-module row totals differ')
    resources=[];seen=set()
    for item in resource_paths:
        p=Path(item).resolve()
        if p in seen:raise ValueError('Repeated resource receipt would double-count worker time')
        seen.add(p);r=json.loads(p.read_text())
        seconds=r.get('wall_seconds',r.get('elapsed_seconds'))
        if seconds is not None and not finite_nonnegative(seconds):raise ValueError('Invalid recorded elapsed time')
        peak=r.get('peak_allocated_mib')
        if peak is None and r.get('peak_allocator_gib') is not None:peak=1024*r['peak_allocator_gib']
        if peak is not None and not finite_nonnegative(peak):raise ValueError('Invalid recorded allocated memory')
        resources.append({'path':str(p),'sha256':digest(p),'worker_wall_seconds':seconds,
                          'peak_allocated_mib':peak,'peak_reserved_mib':r.get('peak_reserved_mib'),
                          'board_peak_used_mib':(r.get('gpu') or {}).get('peak_memory_used_mib'),
                          'return_code':r.get('return_code',r.get('returncode')),'completed':r.get('completed')})
    measured=[x['worker_wall_seconds'] for x in resources]
    return {'schema':'recorded_construction_cost_v1','method':m.get('method'),'manifest':str(path.resolve()),
            'manifest_sha256':digest(path),'model_sha256_declared':m.get('model_sha256'),
            'module_count':m.get('module_count',len(m['modules']) if isinstance(m.get('modules'),dict) else None),
            'realized_rows':realized,'rows_by_expert':by_expert or None,
            'gradient_or_backward':m.get('gradient_or_backward'),
            'worker_wall_seconds':sum(measured) if measured and all(x is not None for x in measured) else None,
            'resources':resources,'missing_values_are_not_zero':True,
            'scope':'Only supplied construction/resource receipts. Wall time is worker occupancy, not GPU busy time, energy, FLOPs or whole-study compute. Training and data collection are not implicitly included.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--resource',action='append',default=[],type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();result=extract(args.manifest,args.resource)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream:stream.write(text)
    print(text,end='')

if __name__=='__main__':main()
