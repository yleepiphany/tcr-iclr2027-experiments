#!/usr/bin/env python3
"""Portable launcher for pinned Table-4 preflight and one-request inference checks.

The package contains experiment code, not third-party model distributions. Point
--host-root at the mounted runtime tree; all outputs go to a new caller directory.
No command launches training, a queue, or a full formal evaluation.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path('/mnt/workspace/Wilson/parameter-fusion')
SNAPSHOT = HERE / 'snapshot'
SUITES = {'spatial':'libero_spatial','object':'libero_object','goal':'libero_goal','long':'libero_10'}
OFT_ID = Path('vla-merge-runtime/experiments/claude-openvla-tcr-repair-20260922/expert-formal-repeats23-attempt-03/identities.json')
FAST_CONTRACT = Path('vla-merge-runtime/experiments/iclr2027-fastwam-experts-formal-v2-20260924/contract.json')
BANK = Path('vla-merge-runtime/experiments/iclr2027-table1-20260910/reset-banks/libero-procedural-clean-v1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rebase(value, host_root):
    path = Path(value)
    try: return Path(host_root) / path.relative_to(SOURCE_ROOT)
    except ValueError: raise ValueError(f'External path outside declared source root: {path}')


def package_check():
    import ast
    provenance = read(HERE/'provenance.json')
    checks = []
    for entry in provenance['files']:
        path = SNAPSHOT/entry['path']
        if not path.is_file() or sha(path) != entry['sha256']:
            raise ValueError(f'Snapshot identity mismatch: {entry["path"]}')
        if path.suffix == '.py': ast.parse(path.read_text(), filename=str(path))
        if path.suffix == '.json': read(path)
        checks.append(entry['path'])
    for path in HERE.glob('*.py'): ast.parse(path.read_text(), filename=str(path))
    return {'status':'PASS','check':'snapshot SHA256 + Python syntax + JSON parse', 'files':len(checks),
            'gpu_used':False, 'weights_in_git':False}


def checkpoint(backend, suite, root):
    if backend == 'openvla':
        info = read(SNAPSHOT/OFT_ID)['experts'][suite]
        return rebase(info['local_path'],root)
    job = next(j for j in read(SNAPSHOT/FAST_CONTRACT)['jobs'] if j['expert']==suite and j['repeat']=='repeat-01')
    return rebase(job['checkpoint']['path'],root)


def runtime_environment(backend, root, gpu=None):
    root=Path(root); oft=root/'vla-merge_table4/OpenVLA-OFT';fast=root/'vla-merge_table4/Fast-WAM'
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2',TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    for key in ('LIBERO_PRO_REPO','LIBERO_PRO_ASSET_DIR','MUJOCO_EGL_DEVICE_ID'):
        env.pop(key,None)
    if gpu is not None: env['CUDA_VISIBLE_DEVICES']=str(gpu)
    else: env['CUDA_VISIBLE_DEVICES']=''
    paths=[SNAPSHOT/'vla-merge/src',root/'vla-merge/src',root/'vla-merge-runtime/references/LIBERO-MergeVLA']
    if backend=='openvla':
        paths=[oft/'dependencies/openvla_transformers_runtime',oft/'dependencies/transformers-openvla-oft/src',
               oft/'python_overlay',oft/'source',root/'vla-merge-runtime/references/dlimp-openvla',
               SNAPSHOT/'vla-merge/experiments/openvla-tcr-20260921',*paths]
        env.update(LIBERO_CONFIG_PATH=str(root/'.datasets/LIBERO/20260919/config-standard'),
                   HF_HOME=str(oft/'cache/huggingface'))
    else:
        paths=[fast/'.python-packages',fast/'source/src',fast/'source',fast/'source/experiments/libero',*paths]
        env.update(LIBERO_CONFIG_PATH=str(root/'vla-merge_table4/DreamZero/run/libero_config'),
                   DIFFSYNTH_MODEL_BASE_PATH=str(root/'.datasets/FastWAM/models'),DIFFSYNTH_SKIP_DOWNLOAD='true')
    env['PYTHONPATH']=':'.join(map(str,paths))
    return env


def runtime_preflight(backend, suite, root, verify_weights=False):
    root=Path(root); model=checkpoint(backend,suite,root); required=[root/BANK/'manifest.json',root/BANK/'selections/repeat-01.json',model]
    required += [root/'vla-merge-runtime/envs/mergevla/bin/python',root/'vla-merge-runtime/references/LIBERO-MergeVLA']
    identities=[]
    if backend=='openvla':
        required += [root/'vla-merge_table4/OpenVLA-OFT/source',root/'.datasets/LIBERO/20260919/config-standard/config.yaml']
        frozen=read(SNAPSHOT/OFT_ID)['files']
        selected={p:v for p,v in frozen.items() if Path(p).parent == SOURCE_ROOT/model.relative_to(root)}
        if not selected: raise ValueError('No frozen OpenVLA checkpoint identity records')
        for name,entry in selected.items():
            path=rebase(name,root);required.append(path)
            if not path.is_file():continue
            if path.stat().st_size != entry['size']:raise ValueError(f'Checkpoint size differs: {path}')
            if verify_weights or path.stat().st_size < 64*1024**2:
                if sha(path)!=entry['sha256']:raise ValueError(f'Checkpoint hash differs: {path}')
                identities.append(str(path.relative_to(root)))
    else:
        job=next(j for j in read(SNAPSHOT/FAST_CONTRACT)['jobs'] if j['expert']==suite and j['repeat']=='repeat-01')
        stats=rebase(job['normalizer']['path'],root);required += [stats,root/'vla-merge_table4/Fast-WAM/source/configs/sim_libero.yaml',root/'.datasets/FastWAM/models']
        if stats.is_file() and sha(stats)!=job['normalizer']['sha256']:raise ValueError('Normalizer SHA differs')
        if model.is_file():
            if model.stat().st_size!=job['checkpoint']['size']:raise ValueError('Fast-WAM checkpoint size differs')
            if verify_weights and sha(model)!=job['checkpoint']['sha256']:raise ValueError('Fast-WAM model SHA differs')
        identities.append(str(stats.relative_to(root)))
        if verify_weights:identities.append(str(model.relative_to(root)))
    missing=[str(p) for p in required if not p.exists()]
    return {'status':'BLOCKED' if missing else 'PASS','backend':backend,'suite':suite,'checkpoint':str(model),
            'missing':missing,'verified_sha256_files':identities,'full_weight_hashes_verified':verify_weights,'gpu_used':False}


def gpu_snapshot(index):
    row=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    uuid,used,free=row[0].strip(),int(row[1]),int(row[2])
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    pids=[line.split(',')[1].strip() for line in raw.splitlines() if line.split(',')[0].strip()==uuid]
    return {'gpu':index,'uuid':uuid,'used_mib':used,'free_mib':free,'compute_pids':pids}


def smoke(args):
    root=args.host_root.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('Smoke output must be a new directory')
    report=runtime_preflight(args.backend,args.suite,root,verify_weights=False)
    if report['status']!='PASS':return report
    before=gpu_snapshot(args.gpu)
    if before['compute_pids'] or before['used_mib']>512 or before['free_mib']<40000:
        return {'status':'BLOCKED','reason':'No safely idle GPU: require zero compute PIDs, <=512 MiB used and >=40000 MiB free','snapshot':before}
    host=socket.gethostname();uuid=before['uuid']
    locks=[root/'vla-merge-runtime/resource-leases'/host/f'gpu-{args.gpu}.lock',
           root/'vla-merge-runtime/experiments/claude-card-flocks'/host/f'gpu-{args.gpu}-{uuid}.lock']
    with ExitStack() as stack:
        for path in locks:
            path.parent.mkdir(parents=True,exist_ok=True)
            handle=stack.enter_context(path.open('a+'))
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return {'status':'BLOCKED','reason':'Existing runtime GPU lease is held','lease':str(path)}
        after=gpu_snapshot(args.gpu)
        if after['compute_pids'] or after['used_mib']>512:return {'status':'BLOCKED','reason':'GPU acquired by another job before launch','snapshot':after}
        # Reserve first: a slow filesystem weight hash must not leave the card
        # available for another queued controller between admission and launch.
        report=runtime_preflight(args.backend,args.suite,root,verify_weights=True)
        if report['status']!='PASS':return report
        output.mkdir(parents=True)
        python=root/'vla-merge-runtime/envs/mergevla/bin/python'
        command=[str(python),str(HERE/'worker.py'),'--backend',args.backend,'--host-root',str(root),
                 '--suite',args.suite,'--output',str(output/'result')]
        receipt={'backend':args.backend,'suite':args.suite,'mode':'one-native-request-only',
                 'evaluation_episodes':0,'training':False,'resource_before':after,
                 'command':command,'package_provenance_sha256':sha(HERE/'provenance.json'),
                 'full_checkpoint_hash_preflight':report}
        (output/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
        with (output/'worker.log').open('x') as log:
            result=subprocess.run(command,env=runtime_environment(args.backend,root,args.gpu),stdout=log,stderr=subprocess.STDOUT,timeout=900)
        receipt.update(status='PASS' if result.returncode==0 else 'FAILED',returncode=result.returncode)
        if (output/'result/smoke.json').is_file():
            native=read(output/'result/smoke.json')
            calls=native.pop('linear_call_order',None)
            if calls is not None:native['linear_call_count']=len(calls)
            receipt['native_result']=native
            receipt['native_result_sha256']=sha(output/'result/smoke.json')
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['package-check','preflight','smoke'])
    p.add_argument('--host-root',type=Path,default=Path(os.environ.get('TCR_HOST_ROOT',str(SOURCE_ROOT))))
    p.add_argument('--backend',choices=['openvla','fastwam'],default='openvla')
    p.add_argument('--suite',choices=SUITES,default='spatial')
    p.add_argument('--verify-weights',action='store_true')
    p.add_argument('--gpu',type=int)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    try:
        if args.action=='package-check':result=package_check()
        elif args.action=='preflight':result=runtime_preflight(args.backend,args.suite,args.host_root,args.verify_weights)
        else:
            if args.gpu is None or args.output is None:p.error('smoke requires --gpu and a new --output')
            result=smoke(args)
    except Exception as exc:
        result={'status':'FAILED','error':str(exc),'error_type':type(exc).__name__}
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 2)


if __name__=='__main__':main()
