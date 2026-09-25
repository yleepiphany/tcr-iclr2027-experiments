#!/usr/bin/env python3
"""TF32-bound v2 successor for fresh RoboTwin FeatCal M3 construction.

Default preflight is CPU-only. wait requires explicit launch authorization;
never stops, pauses, signals, or preempts another job. Native smoke gates build.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack,contextmanager
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

EXPECTED_PLAN_SHA='987e95f0db4708079fbfab59e96e5147181619f6ebf9e337b9abb4c0e691495f'
EXPECTED_BACKEND_SHA='c5d726340ee390118910e661cceb4f037316fe3ab0dca24334e886d8a3a64ada'
MIN_FREE_MIB=49152
MIN_RUNTIME_FREE_MIB=12288
MIN_DISK_BYTES=120*1024**3

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):digest.update(b)
    return digest.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
def event(queue,event,**data):
    value={'at':datetime.now(timezone.utc).isoformat(),'event':event,**data}
    with (queue/'progress.jsonl').open('a') as f:f.write(json.dumps(value)+'\n');f.flush()
    atomic_json(queue/'state.json',value);print(json.dumps(value),flush=True)
def build_environment(parent,gpu):
    env={**parent,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
        'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':'1'}
    env.pop('NVIDIA_TF32_OVERRIDE',None)
    return env

def require_build_backend(torch_module=None):
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')!='1' or os.environ.get('NVIDIA_TF32_OVERRIDE') is not None:
        raise ValueError('Construction requires TF32 override=1 and no NVIDIA global override')
    if torch_module is None:import torch as torch_module
    if not torch_module.backends.cuda.matmul.allow_tf32:raise ValueError('Effective CUDA matmul TF32 is disabled')
    return {'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':'1','NVIDIA_TF32_OVERRIDE':None,'torch_cuda_matmul_allow_tf32':True,
        'torch_version':torch_module.__version__}

def stage_artifact(run,stage):
    if stage=='smoke':return run/'native-smoke.json'
    if stage=='solve':return run/'complete.json'
    if stage in ['teachers-coordination','teachers-receptacle','teachers-precision']:return run/(stage+'.json')
    raise ValueError('Unrecognized construction stage')

def record_backend_stage(run,stage,torch_module):
    path=run/'backend-stage-receipts'/f'{stage}.json';path.parent.mkdir(parents=True,exist_ok=True)
    artifact=stage_artifact(run,stage)
    value={'stage':stage,'at':datetime.now(timezone.utc).isoformat(),'backend_contract_sha256':EXPECTED_BACKEND_SHA,
        'core_plan_sha256':EXPECTED_PLAN_SHA,'runner_sha256':sha(__file__),'backend':require_build_backend(torch_module),
        'artifact':str(artifact),'artifact_sha256':sha(artifact),'formal_episodes':0}
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')

def require_backend_stage(run,stage):
    value=read(run/'backend-stage-receipts'/f'{stage}.json');artifact=stage_artifact(run,stage)
    if value['stage']!=stage or value['backend_contract_sha256']!=EXPECTED_BACKEND_SHA or value['core_plan_sha256']!=EXPECTED_PLAN_SHA or value['runner_sha256']!=sha(__file__):
        raise ValueError('Construction stage lacks exact backend/plan/runner binding')
    if value['backend']['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']!='1' or value['backend']['NVIDIA_TF32_OVERRIDE'] is not None or not value['backend']['torch_cuda_matmul_allow_tf32']:
        raise ValueError('Construction stage used another precision backend')
    if value['artifact']!=str(artifact) or value['artifact_sha256']!=sha(artifact):raise ValueError('Native artifact changed since bound stage')
    return value

def snapshot(gpu):
    def query(field,kind='gpu'):
        return subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-'+kind+'='+field,'--format=csv,noheader,nounits'],text=True,timeout=15).strip()
    row=query('uuid,memory.free').split(',')
    if len(row)!=2 or not row[0].strip().startswith('GPU-'):raise ValueError('Unsupported GPU identity')
    return {'gpu':gpu,'uuid':row[0].strip(),'free_mib':int(row[1]),'compute_pids':query('pid','compute-apps').splitlines()}
def eligible(value):return value['free_mib']>=MIN_FREE_MIB and not value['compute_pids']
def lock_paths(work,gpu,uuid):
    runtime=Path(work)/'vla-merge-runtime';host=socket.gethostname()
    return [runtime/'resource-leases'/host/f'gpu-{gpu}.lock',
            runtime/'experiments/claude-card-flocks'/host/f'gpu-{gpu}-{uuid}.lock']

@contextmanager
def acquire_empty_card(work,gpu,queue):
    """Recheck compute contexts after both flock acquisitions, before any CUDA import."""
    first=snapshot(gpu)
    if not eligible(first):yield None;return
    with ExitStack() as stack:
        handles=[]
        try:
            for path in lock_paths(work,gpu,first['uuid']):
                path.parent.mkdir(parents=True,exist_ok=True);handle=stack.enter_context(path.open('a'))
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(handle)
        except BlockingIOError:yield None;return
        second=snapshot(gpu)
        if not eligible(second) or second['uuid']!=first['uuid']:yield None;return
        if shutil.disk_usage(queue).free<MIN_DISK_BYTES:raise RuntimeError('Less than 120GiB free disk; no GPU stage started')
        yield {'snapshot':second,'fds':tuple(h.fileno() for h in handles)}

@contextmanager
def stage_guard(run,stage,group='all'):
    # Same per-stage filenames as the frozen CLI, plus a single successor owner.
    with (Path(run)/f'{stage}-{group}.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def paths(work):
    root=Path(work)/'vla-merge-runtime/experiments/robotwin-static-baselines-readiness-20260925'
    return root,root/'source-v1',root/'static-bank-v1/bank.json',root/'featcal-m3-tf32-v2'

def preflight(work):
    root,source,bank,run=paths(work);plan=read(run/'plan.json')
    if sha(run/'plan.json')!=EXPECTED_PLAN_SHA:raise ValueError('Frozen v2 RoboTwin M3 plan changed')
    backend=read(run/'backend-contract.json')
    if sha(run/'backend-contract.json')!=EXPECTED_BACKEND_SHA or backend['core_plan_sha256']!=EXPECTED_PLAN_SHA:raise ValueError('Frozen backend contract changed')
    if backend['required_environment']!={'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':'1','NVIDIA_TF32_OVERRIDE':None} or backend['required_torch_cuda_matmul_allow_tf32'] is not True:raise ValueError('Required construction backend differs')
    if plan['groups']!=['coordination','receptacle','precision'] or plan['stages']!=47 or plan['linear_weights']!=418:
        raise ValueError('Wrong experiment contract')
    for filename,digest in plan['implementations'].items():
        if sha(filename)!=digest:raise ValueError('Frozen native implementation changed: '+filename)
    if sha(bank)!=plan['bank_sha256']:raise ValueError('Static input bank changed')
    manifest=read(bank)
    for entry in manifest['groups'].values():
        if sha(bank.parent/entry['tensor_file'])!=entry['tensor_sha256']:raise ValueError('Static input data changed')
    for item in plan['model_files']:
        st=Path(item['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(item['size'],item['mtime_ns']):raise ValueError('Model changed since full-hash preparation')
    for filename,digest in plan['deployment_files_sha256'].items():
        if sha(filename)!=digest:raise ValueError('Deployment sidecar changed')
    return {'status':'CPU_PREFLIGHT_PASS_NATIVE_GPU_NOT_RUN','plan':str(run/'plan.json'),'plan_sha256':EXPECTED_PLAN_SHA,'backend_contract_sha256':EXPECTED_BACKEND_SHA,
        'source':str(source),'run':str(run),'bank':str(bank),'groups':plan['groups'],
        'gpu_gate':{'empty_compute_only':True,'minimum_free_mib':MIN_FREE_MIB,'allocator_fraction':.5,
                    'minimum_runtime_free_mib':MIN_RUNTIME_FREE_MIB,'minimum_disk_gib':120,'two_nonblocking_leases':True},
        'formal_evaluation_launched':False}

def worker(args):
    """Only the supervisor's child with the two inherited lease FDs may execute."""
    queue=args.queue_root.resolve();preflight(args.workspace_root)
    owner=read(queue/'admission.json');gpu=args.worker_gpu
    if owner['plan_sha256']!=EXPECTED_PLAN_SHA or owner['backend_contract_sha256']!=EXPECTED_BACKEND_SHA or owner['runner_sha256']!=sha(__file__) or owner['target']!=args.target:
        raise ValueError('Worker source/target admission changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu) or owner['gpu']!=gpu or owner['controller_pid']!=os.getppid():
        raise ValueError('Worker lacks matching owner/GPU admission')
    expected=lock_paths(args.workspace_root,gpu,owner['uuid']);fds=[int(x) for x in args.lease_fds.split(',')]
    if len(fds)!=2:raise ValueError('Two inherited leases required')
    for fd,path in zip(fds,expected,strict=True):
        a,b=os.fstat(fd),path.stat()
        if (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino):raise ValueError('Inherited lease identity mismatch')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    current=snapshot(gpu)
    if current['uuid']!=owner['uuid'] or not eligible(current):raise RuntimeError('Card changed before CUDA initialization')
    root,source,bank,run=paths(args.workspace_root);sys.path.insert(0,str(source))
    require_build_backend()
    import torch
    from featcal_m3 import API
    torch.set_num_threads(4);api=API(args.workspace_root,bank);api.active_gpu=gpu
    native=api.baseline.featcal_hybrid_linear_weight
    def checked(*a,**kw):
        require_build_backend(torch);api.check_runtime_memory();value=native(*a,**kw);api.check_runtime_memory();return value
    api.baseline.featcal_hybrid_linear_weight=checked
    original_forward=api.forward
    def bound_forward(*a,**kw):
        require_build_backend(torch);return original_forward(*a,**kw)
    api.forward=bound_forward
    torch.cuda.set_per_process_memory_fraction(.5,0)
    if (run/'complete.json').exists():
        require_backend_stage(run,'solve')
        receipt=read(run/'complete.json')
        if receipt['status']!='complete' or receipt['checkpoint']['plan_sha256']!=EXPECTED_PLAN_SHA or not receipt['reload_bitwise_equal']:
            raise ValueError('Existing complete marker is invalid')
        if sha(Path(receipt['checkpoint']['path'])/'model.safetensors')!=receipt['checkpoint']['model_sha256']:
            raise ValueError('Completed checkpoint bytes changed')
        event(queue,'already_complete',checkpoint=receipt['checkpoint']);return
    _,digest,_,_=api.contract(run)
    if not (run/'native-smoke.json').exists():
        event(queue,'native_smoke_started',gpu=gpu,pid=os.getpid())
        with stage_guard(run,'smoke'):
            require_build_backend(torch);api.smoke(run);record_backend_stage(run,'smoke',torch)
    require_backend_stage(run,'smoke');api.require_smoke(run,digest);event(queue,'native_smoke_pass',receipt=str(run/'native-smoke.json'))
    if args.target=='smoke':return
    for group in ('coordination','receptacle','precision'):
        marker=run/f'teachers-{group}.json'
        if marker.exists():
            old=read(marker)
            if old['status']!='complete' or old['plan_sha256']!=digest:raise ValueError('Invalid teacher completion marker')
        else:
            event(queue,'teacher_started',group=group)
            with stage_guard(run,'teachers',group):
                require_build_backend(torch);api.teachers(run,group);record_backend_stage(run,'teachers-'+group,torch)
        require_backend_stage(run,'teachers-'+group)
    event(queue,'sequential_solve_started',stages=47)
    with stage_guard(run,'solve'):
        require_build_backend(torch);api.solve(run);record_backend_stage(run,'solve',torch)
    event(queue,'build_complete',receipt=str(run/'complete.json'),formal_episodes=0)

def supervise(args):
    queue=args.queue_root.resolve();queue.mkdir(parents=True,exist_ok=True)
    with (queue/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=preflight(args.workspace_root);receipt['runner_sha256']=sha(__file__)
        receipt['selected_gpus']=args.gpus;receipt['target']=args.target;atomic_json(queue/'preflight.json',receipt)
        if args.mode=='preflight':print(json.dumps(receipt,indent=2));return
        if not args.authorize_gpu_run:raise ValueError('wait mode needs explicit --authorize-gpu-run from resource owner')
        started=time.monotonic();last=None
        event(queue,'waiting_for_empty_card',gpus=args.gpus,controller_pid=os.getpid(),target=args.target)
        while time.monotonic()-started<args.max_wait_hours*3600:
            if (queue/'STOP').exists():event(queue,'stopped_before_admission');return
            for gpu in args.gpus:
                try:
                    with acquire_empty_card(args.workspace_root,gpu,queue) as held:
                        if held is None:continue
                        preflight(args.workspace_root)
                        admission={**held['snapshot'],'controller_pid':os.getpid(),'plan_sha256':EXPECTED_PLAN_SHA,'backend_contract_sha256':EXPECTED_BACKEND_SHA,
                            'runner_sha256':sha(__file__),'target':args.target,'at':datetime.now(timezone.utc).isoformat()}
                        atomic_json(queue/'admission.json',admission)
                        command=[sys.executable,str(Path(__file__).resolve()),'--workspace-root',str(args.workspace_root),
                            '--queue-root',str(queue),'--mode','worker','--worker-gpu',str(gpu),
                            '--lease-fds',','.join(str(fd) for fd in held['fds']),'--target',args.target]
                        env=build_environment(os.environ,gpu)
                        with (queue/'worker.log').open('a') as log:
                            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=held['fds'])
                            event(queue,'worker_started',gpu=gpu,pid=child.pid,target=args.target)
                            code=child.wait()
                        event(queue,'worker_finished' if code==0 else 'worker_failed',returncode=code,gpu=gpu,
                            worker_log=str(queue/'worker.log'),formal_episodes=0)
                        if code:raise SystemExit(code)
                        return
                except (OSError,subprocess.SubprocessError) as error:
                    message=f'{type(error).__name__}: {error}'
                    if message!=last:event(queue,'resource_query_or_lock_deferred',gpu=gpu,reason=message);last=message
            time.sleep(args.poll_seconds)
        event(queue,'wait_timeout_no_gpu_stage_started',elapsed_seconds=time.monotonic()-started)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--queue-root',type=Path,required=True);p.add_argument('--mode',choices=['preflight','wait','worker'],default='preflight')
    p.add_argument('--gpus',type=lambda s:[int(x) for x in s.split(',')],default=[]);p.add_argument('--target',choices=['smoke','build'],default='smoke')
    p.add_argument('--authorize-gpu-run',action='store_true');p.add_argument('--max-wait-hours',type=float,default=24)
    p.add_argument('--poll-seconds',type=float,default=60);p.add_argument('--worker-gpu',type=int);p.add_argument('--lease-fds')
    args=p.parse_args()
    if args.mode=='worker':worker(args);return
    if args.mode=='wait' and (not args.gpus or len(set(args.gpus))!=len(args.gpus) or any(g<0 or g>7 for g in args.gpus)):
        raise ValueError('An explicit unique host GPU list 0..7 is required')
    if args.poll_seconds<10 or args.poll_seconds>60:raise ValueError('Polling interval must be 10..60 seconds')
    if args.max_wait_hours<=0:raise ValueError('Positive maximum wait required')
    if args.mode=='wait':
        with stage_guard(paths(args.workspace_root)[3],'supervisor','successor'):supervise(args)
    else:supervise(args)
if __name__=='__main__':main()
