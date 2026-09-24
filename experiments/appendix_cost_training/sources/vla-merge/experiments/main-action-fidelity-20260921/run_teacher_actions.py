#!/usr/bin/env python3
"""Generate expert teacher actions for the frozen 400-request held-out bank."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
VLA = WORK/'vla-merge'
SOURCE = WORK/'pi05_lora_finetune_v2_20260826'
PYTHON = SOURCE/'.venv/bin/python'
BANK = WORK/'vla-merge-runtime/experiments/main-action-fidelity-20260921/heldout-raw-attempt-02/bank-index.json'
DEFAULT_RUN = WORK/'vla-merge-runtime/experiments/main-action-fidelity-20260921/teacher-actions-attempt-06'
SMOKE_RUN = WORK/'vla-merge-runtime/experiments/main-action-fidelity-20260921/teacher-actions-attempt-05'
BASE_PATH = VLA/'experiments/tcr-mainline-local-20260919/mainline_capture.py'
FLOCK_DIR = VLA/'experiments/claude-firstpass-cause-20260920'
SUITES = ('spatial','object','goal','long')
CANDIDATE_GPUS = tuple(range(8))
RUNTIME_FLOOR_MIB = 12*1024
SMOKE_ADMISSION_MIB = 32*1024
MAX_ACTIVE = 4


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module)
    return module


base=load_module('main_fidelity_teacher_base',BASE_PATH)
sys.path.insert(0,str(FLOCK_DIR))
import card_flock  # noqa:E402


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def write(path:Path,value:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); tmp.replace(path)


def tensor_hash(value)->str:
    value=value.detach().cpu().contiguous(); h=hashlib.sha256()
    h.update(str(value.dtype).encode()); h.update(json.dumps(list(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def free_mib(gpu:int)->int:
    return int(float(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()))


def worker(suite:str,run:Path,smoke:bool):
    physical=int(os.environ['ITERATION_PHYSICAL_GPU'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(physical): raise ValueError('GPU visibility mismatch')
    bank=json.loads(BANK.read_text())
    if bank.get('accepted') is not True or bank.get('request_count') != 400: raise ValueError('Held-out bank not accepted')
    rows=[row for row in bank['requests'] if row['suite_short']==suite]
    if len(rows)!=100: raise ValueError('Suite must contain 100 requests')
    if smoke: rows=rows[:1]
    model=Path(rows[0]['teacher_expert_path'])
    model_sha=rows[0]['teacher_expert_sha256']
    if any(row['teacher_expert_path']!=str(model) or row['teacher_expert_sha256']!=model_sha for row in rows):
        raise ValueError('Teacher expert identity varies within suite')
    if sha(model/'model.safetensors') != model_sha: raise ValueError('Teacher model hash mismatch')
    if free_mib(physical) < RUNTIME_FLOOR_MIB: raise RuntimeError('GPU below runtime floor before load')

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from scripts.run_pi05_featcal_execution_pilot import load_policy
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    # Match the frozen raw-capture process.  Changing the allocator workspace can
    # change the selected CUDA matmul kernel and defeats the bitwise replay gate.
    torch.cuda.set_per_process_memory_fraction(.35,0)
    # The raw bank was produced through lerobot_eval.main(), which explicitly
    # enables TF32 before evaluation.  The offline worker bypasses that entry
    # point, so reproduce its backend state here before loading the model.
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    policy=load_policy(str(model))
    output=run/('smoke' if smoke else 'teachers')/suite
    output.mkdir(parents=True,exist_ok=False)
    actions={}; receipts=[]; parity=0; start=time.monotonic()
    opened={}
    try:
        for index,row in enumerate(rows):
            if free_mib(physical)<RUNTIME_FLOOR_MIB: raise RuntimeError('GPU dropped below 12GiB runtime floor')
            path=row['tensor_file']
            if sha(Path(path))!=row['tensor_file_sha256']: raise ValueError('Raw tensor file changed')
            handle=opened.get(path)
            if handle is None:
                handle=safe_open(path,framework='pt'); opened[path]=handle
            prefix=f"sample_{row['sample_index']:03d}."
            keys=[key for key in handle.keys() if key.startswith(prefix)]
            cpu={key[len(prefix):]:handle.get_tensor(key) for key in keys}
            if tensor_hash(cpu['x_t']) != row['native_noise_sha256']: raise ValueError('Native noise hash mismatch')
            batch={key:value.to('cuda') for key,value in cpu.items() if key not in ('native_velocity','time')}
            captured=[]; captured_inputs=[]; original=policy.model.denoise_step
            def denoise(*args,**kwargs):
                if not captured_inputs:
                    if args:
                        raise ValueError('Unexpected positional denoise call')
                    captured_inputs.append({
                        'x_t': kwargs['x_t'].detach().cpu().clone(),
                        'time': kwargs['timestep'].detach().cpu().clone(),
                    })
                value=original(*args,**kwargs)
                if not captured: captured.append(value.detach().cpu().clone())
                return value
            policy.model.denoise_step=denoise
            try:
                action=policy.model.sample_actions(
                    images=[batch[f'image_{i}'] for i in range(3)],
                    img_masks=[batch[f'image_mask_{i}'] for i in range(3)],
                    tokens=batch['tokens'],masks=batch['masks'],
                    states=batch.get('states'),state_masks=batch.get('state_masks'),
                    noise=batch['x_t'].clone(),num_steps=10).detach().cpu()
            finally:
                policy.model.denoise_step=original
            if len(captured)!=1:
                raise ValueError(f"Captured expert velocity count failed: {row['request_id']}")
            difference=(captured[0].float()-cpu['native_velocity'].float()).abs()
            maximum=float(difference.max())
            relative=float((difference/(cpu['native_velocity'].float().abs()+1e-12)).max())
            if not torch.equal(captured[0],cpu['native_velocity']):
                raise ValueError(
                    f"Captured expert velocity parity failed: {row['request_id']} "
                    f"max_abs={maximum:.12g} max_relative={relative:.12g} "
                    f"replayed_dtype={captured[0].dtype} captured_dtype={cpu['native_velocity'].dtype} "
                    f"x_equal={torch.equal(captured_inputs[0]['x_t'],cpu['x_t'])} "
                    f"time_equal={torch.equal(captured_inputs[0]['time'],cpu['time'])} "
                    f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32} "
                    f"matmul_precision={torch.get_float32_matmul_precision()}"
                )
            parity+=1
            if index==0:
                repeated=policy.model.sample_actions(
                    images=[batch[f'image_{i}'] for i in range(3)],
                    img_masks=[batch[f'image_mask_{i}'] for i in range(3)],
                    tokens=batch['tokens'],masks=batch['masks'],
                    states=batch.get('states'),state_masks=batch.get('state_masks'),
                    noise=batch['x_t'].clone(),num_steps=10).detach().cpu()
                if not torch.equal(action,repeated): raise ValueError('Native teacher generation is not deterministic')
            if tuple(action.shape)!=(1,50,32) or not torch.isfinite(action).all(): raise ValueError('Teacher action shape/finite failure')
            actions[row['request_id']]=action.contiguous()
            receipts.append({'request_id':row['request_id'],'raw_input_sha256':row['raw_input_sha256'],
                             'native_noise_sha256':row['native_noise_sha256'],
                             'action_sha256':tensor_hash(action)})
            del batch,action
        tensor=output/'actions.safetensors'; save_file(actions,tensor,metadata={'format':'pt'})
        manifest={'schema':'main_fidelity_teacher_actions_v1','suite':suite,'smoke':smoke,
                  'model':str(model),'model_sha256':model_sha,'bank':str(BANK),'bank_sha256':sha(BANK),
                  'requests':len(rows),'velocity_parity_passed':parity,
                  'determinism_first_request':True,'native_steps':10,
                  'actions_file':str(tensor),'actions_sha256':sha(tensor),'rows':receipts,
                  'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                  'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
                  'elapsed_seconds':time.monotonic()-start,'runtime_floor_mib':RUNTIME_FLOOR_MIB}
        write(output/'manifest.json',manifest)
        print(json.dumps({k:manifest[k] for k in ('suite','smoke','requests','peak_allocated_gib','peak_reserved_gib','elapsed_seconds')}),flush=True)
    finally:
        try: policy.to('cpu')
        except Exception: pass
        gc.collect(); torch.cuda.empty_cache()


def environment(gpu:int):
    env=base.environment(gpu)
    env.update({'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
                'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1',
                'PALIGEMMA_TOKENIZER_PATH':str(SOURCE/'assets/paligemma-3b-pt-224-tokenizer'),
                'PYTHONPATH':':'.join([str(WORK/'vla-merge-runtime/python-overlay'),str(SOURCE/'src'),str(SOURCE/'lerobot/src'),str(VLA),str(VLA/'src')])})
    return env


def gpu_rows():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in out.strip().splitlines():
        i,u,f,m,v=[x.strip() for x in line.split(',')]
        rows.append({'index':int(i),'uuid':u,'free_mib':int(float(f)),'used_mib':int(float(m)),'utilization_gpu_percent':int(float(v))})
    return sorted(rows,key=lambda row:(-row['free_mib'],row['index']))


def smoke_dispatch(run:Path):
    run.mkdir(parents=True,exist_ok=False)
    source_hashes={str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(BANK):sha(BANK)}
    write(run/'smoke-plan.json',{'schema':'main_fidelity_teacher_smoke_v1','created_at':datetime.now(timezone.utc).isoformat(),
          'hostname':socket.gethostname(),'suite':'spatial','bank':str(BANK),'source_hashes':source_hashes,
          'admission_mib':SMOKE_ADMISSION_MIB,'runtime_floor_mib':RUNTIME_FLOOR_MIB})
    uuids=card_flock.gpu_uuids(); lease=None
    for row in gpu_rows():
        if row['free_mib']<SMOKE_ADMISSION_MIB or uuids.get(row['index'])!=row['uuid']: continue
        lease=card_flock.take_card(row['index'],row['uuid'],'main-fidelity-teacher-smoke','offline-forward')
        if lease is not None: gpu=row['index']; snapshot=row; break
    if lease is None: raise RuntimeError('No admissible GPU for teacher smoke')
    try:
        command=[str(PYTHON),'-u',str(Path(__file__).resolve()),'--worker','--suite','spatial','--run',str(run),'--smoke']
        log=run/'smoke.log'
        with log.open('x') as handle:
            process=subprocess.run(command,cwd=VLA,env=environment(gpu),stdout=handle,stderr=subprocess.STDOUT)
        write(run/'smoke-exit.json',{'return_code':process.returncode,'gpu':gpu,'snapshot':snapshot})
        if process.returncode: raise RuntimeError(f'Teacher smoke failed exit {process.returncode}')
        manifest=json.loads((run/'smoke/spatial/manifest.json').read_text())
        if manifest.get('requests')!=1 or manifest.get('velocity_parity_passed')!=1: raise ValueError('Teacher smoke audit failed')
        admission=max(SMOKE_ADMISSION_MIB,math.ceil((manifest['peak_reserved_gib']+12)*1024))
        write(run/'SMOKE-PASSED.json',{'passed':True,'gpu':gpu,'manifest_sha256':sha(run/'smoke/spatial/manifest.json'),
              'measured_peak_reserved_gib':manifest['peak_reserved_gib'],'full_admission_mib':admission,
              'runtime_floor_mib':RUNTIME_FLOOR_MIB})
        print(json.dumps({'passed':True,'gpu':gpu,'peak_reserved_gib':manifest['peak_reserved_gib'],'full_admission_mib':admission}))
    finally: lease.release()


def full_dispatch(run:Path):
    """Run one suite per dynamically selected, currently admissible GPU."""
    smoke_receipt=SMOKE_RUN/'SMOKE-PASSED.json'
    if not smoke_receipt.is_file(): raise ValueError('Strict teacher smoke has not passed')
    smoke=json.loads(smoke_receipt.read_text())
    if smoke.get('passed') is not True: raise ValueError('Strict teacher smoke receipt is not accepted')
    admission=int(smoke['full_admission_mib'])
    if admission < SMOKE_ADMISSION_MIB: raise ValueError('Full admission fell below smoke floor')
    run.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve()
    plan={'schema':'main_fidelity_teacher_actions_v1','created_at':datetime.now(timezone.utc).isoformat(),
          'hostname':socket.gethostname(),'suites':list(SUITES),'requests_per_suite':100,
          'total_requests':400,'bank':str(BANK),'bank_sha256':sha(BANK),
          'smoke_receipt':str(smoke_receipt),'smoke_receipt_sha256':sha(smoke_receipt),
          'source':str(source),'source_sha256':sha(source),'candidate_gpus':list(CANDIDATE_GPUS),
          'dynamic_selection':'highest current free memory among admissible UUID-locked cards',
          'admission_mib':admission,'runtime_floor_mib':RUNTIME_FLOOR_MIB,
          'max_active':MAX_ACTIVE,'one_worker_per_card':True,'no_retry':True,
          'teacher_definition':'full native 10-step expert sampling on frozen raw request and native noise'}
    write(run/'plan.json',plan)
    pending=deque(SUITES); active={}; states={suite:{'status':'pending'} for suite in SUITES}
    uuids=card_flock.gpu_uuids(); failed=False; started=time.time(); deadline=started+6*3600
    def checkpoint():
        write(run/'state.json',{'updated_at':datetime.now(timezone.utc).isoformat(),
              'states':states,'pending':list(pending),'active':[v['suite'] for v in active.values()],
              'failed_latch':failed})
    try:
        while pending or active:
            if time.time() >= deadline: raise TimeoutError('Teacher action queue deadline reached')
            for pid,item in list(active.items()):
                code=item['process'].poll()
                if code is None: continue
                item['log_handle'].close(); item['lease'].release(); del active[pid]
                states[item['suite']].update(status='complete' if code==0 else 'failed',
                    return_code=code,ended_unix=time.time())
                write(run/'exits'/f"{item['suite']}.json",{'suite':item['suite'],'gpu':item['gpu'],
                      'pid':pid,'return_code':code,'started_unix':item['started_unix'],
                      'ended_unix':time.time(),'log':str(item['log'])})
                if code: failed=True
            if not failed:
                occupied={item['gpu'] for item in active.values()}
                for row in gpu_rows():
                    if not pending or len(active)>=MAX_ACTIVE: break
                    gpu=row['index']
                    if gpu in occupied or row['free_mib']<admission or uuids.get(gpu)!=row['uuid']: continue
                    lease=card_flock.take_card(gpu,row['uuid'],'main-fidelity-teacher-actions','offline-native-forward')
                    if lease is None: continue
                    suite=pending.popleft(); log=run/'logs'/f'{suite}.log'; log.parent.mkdir(parents=True,exist_ok=True)
                    handle=log.open('x')
                    command=[str(PYTHON),'-u',str(source),'--worker','--suite',suite,'--run',str(run)]
                    try:
                        process=subprocess.Popen(command,cwd=VLA,env=environment(gpu),stdout=handle,
                                                 stderr=subprocess.STDOUT,start_new_session=True)
                    except Exception:
                        handle.close(); lease.release(); pending.appendleft(suite); raise
                    now=time.time(); active[process.pid]={'process':process,'suite':suite,'gpu':gpu,
                        'lease':lease,'log_handle':handle,'log':log,'started_unix':now}
                    occupied.add(gpu)
                    states[suite]={'status':'running','gpu':gpu,'pid':process.pid,'started_unix':now,
                        'admission_snapshot':row,'log':str(log)}
                    write(run/'launches'/f'{suite}.json',{'suite':suite,'gpu':gpu,'pid':process.pid,
                          'started_unix':now,'snapshot':row,'command':command})
            checkpoint()
            if failed and not active: break
            time.sleep(5)
        status='complete' if not failed and all(v.get('status')=='complete' for v in states.values()) else 'failed'
        result={'status':status,'complete':status=='complete','states':states,
                'elapsed_seconds':time.time()-started,'finished_at':datetime.now(timezone.utc).isoformat()}
        write(run/'queue-ended.json',result); print(json.dumps(result),flush=True)
        if status!='complete': raise RuntimeError('Teacher action queue did not complete')
    finally:
        for item in active.values():
            if item['process'].poll() is None: item['process'].terminate()
            item['log_handle'].close(); item['lease'].release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--smoke-dispatch',action='store_true')
    parser.add_argument('--full-dispatch',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--suite',choices=SUITES)
    args=parser.parse_args()
    if sum((args.smoke_dispatch,args.full_dispatch,args.worker)) != 1:
        parser.error('Choose exactly one dispatch or worker mode')
    if args.smoke_dispatch:
        smoke_dispatch(args.run.resolve())
    elif args.full_dispatch:
        full_dispatch(args.run.resolve())
    elif args.worker and args.suite:
        worker(args.suite,args.run.resolve(),args.smoke)
    else: parser.error('Worker mode requires --suite')


if __name__=='__main__': main()
