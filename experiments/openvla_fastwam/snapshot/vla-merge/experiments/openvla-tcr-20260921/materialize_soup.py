"""CPU-only dense expert mean, an initialization for OFT TCR, not TCR itself."""
from contextlib import ExitStack
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import torch
from safetensors import safe_open
from safetensors.torch import save_file

WORK=Path(__file__).resolve().parents[3]
LEDGER=WORK/'vla-merge-runtime/experiments/openvla-tcr-20260921/local-expert-dynamic-01/identities.json'
RUN=WORK/'vla-merge-runtime/experiments/openvla-tcr-20260921/soup-prior-attempt-01'
ORDER=('spatial','object','goal','long')
SHARED=('config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json','special_tokens_map.json')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True)

def mean_tensors(tensors):
    if len(tensors)!=4:raise ValueError('Exactly four experts required')
    first=tensors[0]
    if any(t.shape!=first.shape or t.dtype!=first.dtype or t.device.type!='cpu' for t in tensors):
        raise ValueError('Expert tensor structure mismatch')
    if not first.is_floating_point():
        if not all(torch.equal(t,first) for t in tensors):raise ValueError('Nonfloating buffers disagree')
        return first.clone()
    dtype=torch.float64 if first.dtype==torch.float64 else torch.float32
    result=torch.zeros(first.shape,dtype=dtype)
    for t in tensors:
        if not torch.isfinite(t).all():raise ValueError('Nonfinite expert tensor')
        result.add_(t.to(dtype=dtype),alpha=.25)
    result=result.to(first.dtype).contiguous()
    if not torch.isfinite(result).all():raise ValueError('Nonfinite averaged tensor')
    return result

def merge_stats(stats):
    result={}
    for d in stats:
        for key,value in d.items():
            if key in result and result[key]!=value:raise ValueError('Conflicting normalizer key')
            result[key]=value
    return result

def verify_sources(ledger):
    for name,record in ledger['files'].items():
        path=Path(name);s=path.stat()
        if (s.st_size,s.st_mtime_ns)!=(record['size'],record['mtime_ns']):
            raise ValueError(f'Frozen expert/source changed: {path}')
        if s.st_size<64*1024**2 and sha(path)!=record['sha256']:
            raise ValueError(f'Frozen file content changed: {path}')

def build(ledger_path,run,*,expected_keys=982):
    if run.exists():raise FileExistsError('No overwrite/resume: use an explicitly reviewed new attempt')
    ledger=json.loads(ledger_path.read_text());verify_sources(ledger)
    roots=[Path(ledger['experts'][name]['local_path']) for name in ORDER]
    indexes=[json.loads((p/'model.safetensors.index.json').read_text()) for p in roots]
    maps=[d['weight_map'] for d in indexes]
    if len(maps[0])!=expected_keys or any(set(m)!=set(maps[0]) for m in maps):
        raise ValueError('Backbone key sets differ')
    for filename in SHARED:
        if filename=='config.json':
            configs=[json.loads((p/filename).read_text()) for p in roots]
            if any(c!=configs[0] for c in configs[1:]):raise ValueError('Model configs differ semantically')
        elif len({sha(p/filename) for p in roots})!=1:
            raise ValueError(f'Processor mismatch: {filename}')
    stats=merge_stats([json.loads((p/'dataset_statistics.json').read_text()) for p in roots])
    components={}
    for component in ('action_head','proprio_projector'):
        paths=[]
        for p in roots:
            found=list(p.glob(component+'--*_checkpoint.pt'))
            if len(found)!=1:raise ValueError(f'Expected exactly one {component}')
            paths.append(found[0])
        components[component]=paths
    run.mkdir(parents=True,exist_ok=False)
    out=run/'checkpoint';out.mkdir()
    write(run/'started.json',{'host':socket.gethostname(),'pid':os.getpid(),
          'kernel_start':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
          'source_sha256':sha(Path(__file__)),
          'created_utc':datetime.now(timezone.utc).isoformat(),'device':'cpu','is_tcr':False,
          'source_ledger':str(ledger_path),'source_ledger_sha256':sha(ledger_path),
          'expert_order':ORDER,'aggregation':'equal dense parameter mean; FP32 accumulation except FP64 tensors'})
    written={};numel=0
    with ExitStack() as stack:
        handles=[]
        for p,m in zip(roots,maps):
            per={}
            for shard in sorted(set(m.values())):
                path=(p/shard).resolve()
                if path.parent!=p.resolve():raise ValueError('Invalid shard path')
                per[shard]=stack.enter_context(safe_open(str(path),framework='pt',device='cpu'))
            actual={key for h in per.values() for key in h.keys()}
            if actual!=set(m) or sum(len(h.keys()) for h in per.values())!=len(m):
                raise ValueError('Shard keys differ from index')
            handles.append(per)
        for shard in sorted(set(maps[0].values())):
            result={}
            for key in sorted(k for k,v in maps[0].items() if v==shard):
                values=[h[m[key]].get_tensor(key) for h,m in zip(handles,maps)]
                result[key]=mean_tensors(values);numel+=result[key].numel()
            save_file(result,str(out/shard),metadata={'format':'pt'})
            written[shard]=sha(out/shard)
            print(json.dumps({'shard_saved':shard,'keys':len(result)}),flush=True)
            del result
    write(out/'model.safetensors.index.json',indexes[0])
    for filename in SHARED:shutil.copyfile(roots[0]/filename,out/filename)
    # Keep any shared ancillary tokenizer files, never copy training weights twice.
    for filename in ('tokenizer.model','added_tokens.json','generation_config.json'):
        found=[(p/filename).exists() for p in roots]
        if any(found):
            if not all(found) or len({sha(p/filename) for p in roots})!=1:raise ValueError(filename)
            shutil.copyfile(roots[0]/filename,out/filename)
    write(out/'dataset_statistics.json',stats)
    counts={}
    for component,paths in components.items():
        states=[torch.load(str(p),map_location='cpu',weights_only=True,mmap=True) for p in paths]
        if any(set(s)!=set(states[0]) for s in states):raise ValueError('Component key mismatch')
        result={key:mean_tensors([s[key] for s in states]) for key in sorted(states[0])}
        filename=component+'--mean_checkpoint.pt'
        torch.save(result,out/filename);written[filename]=sha(out/filename)
        counts[component]=len(result);del result,states
    verify_sources(ledger)
    files={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    report={'complete':True,'is_tcr':False,'role':'Soup prior for subsequent TCR regression',
            'checkpoint':str(out.resolve()),'expert_order':ORDER,'backbone_keys':expected_keys,
            'backbone_numel':numel,'component_keys':counts,'files_sha256':files,
            'normalization':'Union of unchanged per-suite expert statistics; suite metadata chooses normalization, not expert networks.',
            'source_ledger_sha256':sha(ledger_path),'loader':'native_oft.NativePolicy (explicit pinned native source classes)',
            'source_sha256':sha(Path(__file__)),
            'gpu_forward_validated':False,'success_evaluated':False}
    write(run/'manifest.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger',type=Path,default=LEDGER)
    parser.add_argument('--run',type=Path,default=RUN)
    args=parser.parse_args()
    torch.set_num_threads(2)
    build(args.ledger.resolve(),args.run.resolve())
