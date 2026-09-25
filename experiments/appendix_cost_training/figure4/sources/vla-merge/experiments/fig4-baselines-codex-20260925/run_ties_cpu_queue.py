#!/usr/bin/env python3
"""Single persistent owner for two low-priority CPU TIES builds and native loads."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time

WORK = Path('/mnt/workspace/Wilson/parameter-fusion')
CODE = WORK/'vla-merge/experiments/fig4-baselines-codex-20260925'
ROOT = WORK/'vla-merge-runtime/experiments/fig4-baselines-codex-20260925/ties-adapter-v2'
RUN = ROOT/'cpu-queue-01'
PYTHON = WORK/'pi05_lora_finetune_v2_20260826/.venv/bin/python'
ADAPTER = CODE/'ties_subset_v2.py'
PLAN_SHA = 'e9b38b1e7793b31876c7c7804e0bc446e6287ec91f34d33444eca818432c407b'
SUBSETS = ['spatial-goal','spatial-object-goal']
stopping = False

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    with Path(p).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def write(name,d):
    p=RUN/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n');os.replace(t,p)
def resources():
    m={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.split()[1].isdigit()}
    disk=os.statvfs(ROOT)
    return {'memory_available_bytes':m['MemAvailable'],'cpu_affinity_count':len(os.sched_getaffinity(0)),
            'load1':os.getloadavg()[0],'disk_available_bytes':disk.f_bavail*disk.f_frsize}
def stop_handler(*_):
    global stopping
    stopping=True
def terminate_owned(active):
    for x in active.values():
        if x['process'].poll() is None:x['process'].terminate()
    for x in active.values():
        try:x['process'].wait(timeout=30)
        except subprocess.TimeoutExpired:
            x['process'].kill();x['process'].wait()
        x['log'].close()

def main():
    assert sha(ROOT/'plan.json')==PLAN_SHA
    plan=json.loads((ROOT/'plan.json').read_text())
    assert sha(ADAPTER)==plan['adapter_sha256']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    RUN.mkdir(exist_ok=False)
    with (RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
        write('STARTED.json',{'pid':os.getpid(),'start_ticks':int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
              'started_at':now(),'plan_sha256':PLAN_SHA,'source_sha256':sha(__file__),'nice':os.getpriority(os.PRIO_PROCESS,0),
              'cuda_visible_devices':'','cpu_only':True,'max_active':2,'resource_admission':resources()})
        pending=[(s,'build') for s in SUBSETS];active={};completed={};failure=None
        try:
            while pending or active:
                if stopping:raise RuntimeError('CPU queue received stop signal; no retry')
                resource=resources()
                if resource['memory_available_bytes']<32*2**30:raise RuntimeError('Host CPU memory reserve below 32 GiB')
                for subset,x in list(active.items()):
                    code=x['process'].poll()
                    if code is None:continue
                    x['process'].wait();x['log'].close();active.pop(subset)
                    write(f'{subset}-{x["phase"]}-EXIT.json',{'returncode':code,'finished_at':now(),'pid':x['process'].pid,'phase':x['phase']})
                    if code!=0:raise RuntimeError(f'{subset} {x["phase"]} exited {code}')
                    if x['phase']=='build':pending.insert(0,(subset,'native-load'))
                    else:
                        receipt=json.loads((ROOT/subset/'NATIVE-CPU-LOAD.json').read_text())
                        assert receipt['status']=='pass_native_cpu_load'
                        completed[subset]=receipt
                admitted=resource['memory_available_bytes']>=128*2**30 and resource['disk_available_bytes']>=100*2**30 and resource['load1']<.9*resource['cpu_affinity_count']
                while pending and len(active)<2 and admitted:
                    subset,phase=pending.pop(0)
                    log=(RUN/f'{subset}-{phase}.log').open('x')
                    command=[str(PYTHON),'-u',str(ADAPTER),phase,'--subset',subset]
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1'}
                    proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=WORK,env=env,start_new_session=True)
                    active[subset]={'phase':phase,'process':proc,'log':log}
                    write(f'{subset}-{phase}-LAUNCH.json',{'pid':proc.pid,'command':command,'started_at':now(),'resource_admission':resource,'cpu_only':True})
                write('STATE.json',{'updated_at':now(),'status':'running_or_waiting_cpu_resources','pending':pending,
                      'active':{k:{'pid':v['process'].pid,'phase':v['phase']} for k,v in active.items()},'completed':completed,'resource':resource})
                if pending or active:time.sleep(15)
            write('DONE.json',{'status':'complete_cpu_builds_and_native_loads','completed_at':now(),'completed':completed,
                  'gpu_jobs_launched':0,'formal_jobs_launched':0,'remaining_before_formal':'native action finite check plus owned GPU evaluation admission'})
        except BaseException as exc:
            terminate_owned(active)
            write('FAILED.json',{'status':'failed_no_retry','time':now(),'error':repr(exc),'completed':completed,'pending':pending,
                  'partial_artifacts_preserved':True,'signals_scope':'only this supervisor Popen child objects'})
            raise

if __name__=='__main__':main()
