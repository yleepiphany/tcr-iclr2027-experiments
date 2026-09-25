#!/usr/bin/env python3
"""Single-permit FeatCal M2/M3 queue. Preparation/tests never dispatch CUDA work."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import ctypes
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

WORK=Path('/mnt/workspace/Wilson/parameter-fusion')
REPO=WORK/'vla-merge'
FIG=WORK/'vla-merge-runtime/experiments/fig4-baselines-codex-20260925'
HERE=Path(__file__).resolve().parent
API_SOURCE=REPO/'experiments/fig4-baselines-codex-20260925/featcal_subset_v1/featcal_subset.py'
ADAPTER=FIG/'featcal-adapter-v1'
RUN=FIG/'featcal-queue-v2'
PYTHON=WORK/'pi05_lora_finetune_v2_20260826/.venv/bin/python'
MASTER_SHA='62ff0891df02a75d20a8b724f70f93a8373f75852a7385c88034993c326d7875'
SUBSETS={'spatial-goal':['spatial','goal'],'spatial-object-goal':['spatial','object','goal']}
SUBSET_SHAS={'spatial-goal':'d1c5791ca7cd32e2b87ab9e4cdf490836f45f07591b4d9f754a178809b54e8e4',
             'spatial-object-goal':'31e63c6b83e2a95b87d8d393b2d978ac0a8ba5711a84fb5d2853bb97a870b205'}
HOST='dsw-824375-57c745db88-n6tv9'
GPUS=(1,4,5,6)
CONFIG=HERE/'libero-config'
RESOURCE={'minimum_free_mib':49152,'maximum_idle_used_mib':64,'idle_utilization_percent':0,
 'minimum_disk_bytes':120*2**30,'runtime_free_mib':12288,'runtime_disk_bytes':32*2**30,
 'allocator_fraction':.5,'solver_workspace_bytes':2*2**30,'max_workers':4,'max_build_slots':2,
 'context_release_timeout_seconds':60,'poll_seconds':2,'allow_shared':False}
sys.path[:0]=[str(HERE),str(API_SOURCE.parent),str(REPO),str(REPO/'src'),str(REPO/'scripts'),
 str(REPO/'experiments/claude-firstpass-cause-20260920'),str(WORK/'pi05_lora_finetune_v2_20260826/src')]
STOP=False

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def write(p,x,exclusive=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
    else:
        t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(t,p)
def stamp(pid):
    try:return int(Path(f'/proc/{pid}/stat').read_text().split()[21])
    except (OSError,ValueError):return None
def stat_id(path):
    s=Path(path).stat();return {'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev}
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def gpu_rows():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    out={}
    for line in text.splitlines():
        i,u,free,used,util=[x.strip() for x in line.split(',')]
        out[int(i)]={'uuid':u,'free_mib':int(free),'used_mib':int(used),'utilization':int(util),'compute':[]}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    for line in apps.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if len(parts)!=2:continue
        for row in out.values():
            if row['uuid']==parts[0]:row['compute'].append(parts[1])
    return out
def resource_ok(row,disk_bytes):
    return (row['free_mib']>=49152 and row['used_mib']<=64 and row['utilization']==0
            and not row['compute'] and disk_bytes>=120*2**30)
def wait_context_clear(gpu,uuid,snapshot=gpu_rows,clock=time.monotonic,sleep=time.sleep):
    deadline=clock()+60;observed=[]
    while True:
        row=snapshot()[gpu];require(row['uuid']==uuid,'GPU UUID changed while leases held')
        observed.append(row)
        if not row['compute'] and row['used_mib']<=64 and row['free_mib']>=49152:return observed
        require(clock()<deadline,'CUDA context did not clear within 60s; no foreign PID is signalled')
        sleep(2)

def make_jobs(master):
    jobs=[]
    for subset,groups in SUBSETS.items():
        smoke=f'{subset}-smoke';jobs.append({'id':smoke,'subset':subset,'stage':'smoke','needs':[]})
        teachers=[]
        for group in groups:
            jid=f'{subset}-teacher-{group}';teachers.append(jid)
            jobs.append({'id':jid,'subset':subset,'stage':'teachers','group':group,'needs':[smoke]})
        solve=f'{subset}-solve';jobs.append({'id':solve,'subset':subset,'stage':'solve','needs':teachers})
        for original in master['new_evaluations']:
            if original['method']=='featcal' and original['subset']==subset:
                row=dict(original);row.update(stage='formal',needs=[solve],full_suite=original['suite'])
                row['output']=str(RUN/'formal'/row['id'])
                row['command']=[f'--output_dir={row["output"]}/eval' if x.startswith('--output_dir=') else x for x in row.pop('command_template')]
                jobs.append(row)
    require(len(jobs)==14 and len({j['id'] for j in jobs})==14,'Unexpected stage coverage')
    require(sum(j.get('episodes',0) for j in jobs)==500,'Formal denominator differs')
    return jobs

def cpu_preflight():
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['LIBERO_CONFIG_PATH']=str(CONFIG)
    require(sha(FIG/'plan.json')==MASTER_SHA,'Master changed')
    master=read(FIG/'plan.json');assets={str(FIG/'plan.json'):MASTER_SHA}
    original=master['source_binding']['featcal_plan']
    require(sha(original['path'])==original['sha256'] and read(original['path'])['backend']=={'tf32_override':'1'},'Original Table1 build backend differs')
    assets[original['path']]=original['sha256']
    for subset,groups in SUBSETS.items():
        run=ADAPTER/subset;p=read(run/'plan.json')
        require(sha(run/'plan.json')==SUBSET_SHAS[subset],'Subset plan changed')
        require(p['groups']==groups and p['effective_expert_count']==len(groups),'Subset membership differs')
        require((p['stages'],p['linear_weights'],p['alpha'],p['rho'],p['lambda'],p['eps'])==(47,418,.3,2.,.05,1e-8),'Method constants changed')
        require(p['workspace_cap_bytes']==2*2**30 and p['regression_rows']==1110300*len(groups),'Workspace/row budget changed')
        require(p['resource_proposal']=={'allocator_fraction':.5,'minimum_free_disk_gib':120,'minimum_free_mib':49152,'minimum_runtime_free_mib':12288},'Original admission differs')
        assets[str(run/'plan.json')]=SUBSET_SHAS[subset]
        assets[str(p['bank'])]=p['bank_sha256']
        for name in ['shard-index.json','quotas.json']:assets[str(run/name)]=sha(run/name)
        assets.update(p['implementations']);assets.update(p['deployment_files_sha256'])
        for x in p['model_files']:
            s=Path(x['path']).stat();require(s.st_size==x['size'] and s.st_mtime_ns==x['mtime_ns'],'Model stat drift')
            assets[x['path']]=x['sha256']
        for entry in p['input_identity']['original_input_bindings'].values():
            for k in ['manifest','observations']:
                if k in entry and isinstance(entry[k],dict):assets[entry[k]['path']]=entry[k]['sha256']
    source_paths=[Path(__file__),CONFIG/'config.yaml',REPO/'experiments/claude-firstpass-cause-20260920/card_flock.py',
        REPO/'scripts/eval_tcr_10k_formal_bounded.py',REPO/'scripts/eval_pi05_policy_with_procedural_bank.py',
        REPO/'src/vla_merge/libero_procedural_bank.py',WORK/'pi05_lora_finetune_v2_20260826/lerobot/src/lerobot/configs/eval.py',
        WORK/'pi05_lora_finetune_v2_20260826/lerobot/src/lerobot/policies/factory.py']
    assets.update({str(p):sha(p) for p in source_paths})
    selection=Path(master['evaluation_selection']['path']);bank=selection.parent.parent
    assets[str(selection)]=master['evaluation_selection']['sha256'];assets[str(bank/'manifest.json')]=sha(bank/'manifest.json')
    for path,digest in assets.items():
        require(Path(path).is_file(),'Missing frozen asset '+path)
        if Path(path).stat().st_size<100*2**20:require(sha(path)==digest,'Small source/input hash differs '+path)
    from vla_merge.libero_procedural_bank import load_selection
    load_selection(bank,selection,verify_source_files=True)
    return master,assets,{'bank':str(bank),'selection':str(selection),'sha256':sha(selection),'eval_seed':274001,
        'bank_manifest_sha256':sha(bank/'manifest.json')}

def prepare():
    require(not RUN.exists(),'Queue already exists')
    master,assets,selection=cpu_preflight();jobs=make_jobs(master)
    for subset in SUBSETS:
        run=ADAPTER/subset
        require(not any((run/n).exists() for n in ['native-smoke.json','complete.json']),'Subset stage already executed')
        require(not list(run.glob('teachers-*.json')) and not list((run/'cache').glob('**/*.safetensors')),'Existing captures may not be duplicated')
        require(not Path(read(run/'plan.json')['checkpoint']).exists(),'Output already exists')
    rows=gpu_rows()
    plan={'schema':'fig4_featcal_single_permit_queue_v2','created_at':now(),'master_plan_sha256':MASTER_SHA,
        'subset_plans':{s:{'path':str(ADAPTER/s/'plan.json'),'sha256':SUBSET_SHAS[s]} for s in SUBSETS},
        'resource':RESOURCE,'host':HOST,'gpus':list(GPUS),'gpu_uuids':{str(g):rows[g]['uuid'] for g in GPUS},
        'jobs':jobs,'formal_jobs':5,'formal_episodes':500,'assets':assets,'selection':selection,
        'native_smoke':'Original API: selected Soup plus all selected teachers; full-joint/explicit outputs bitwise equal, with additional finite guard',
        'export_acceptance':'47 original atomic prefix barriers, snapshot hashes/target sets, finite 813-tensor export, original bitwise reload receipt',
        'teacher_policy':'fresh per-subset captures; no old teacher/student cache reuse','api_source_unchanged':True,
        'child_lease_model':'Parent owns dual leases/build slot; child verifies inherited FDs and parent identity, then calls API methods directly to avoid recursive locking',
        'serialized_config_policy':'Preserve selected Soup lineage; native eval config loader overwrites pretrained_path with explicit --policy.path (source hashes locked)',
        'no_retry':True,'no_training':True,'no_score_based_gate':True,'gpu_launched_during_preparation':False,
        'formal_evaluator':'Unchanged bounded native entry, allocator .20 / duty .25 / runtime reserve12GiB',
        'queue_is_not_execution_authorization':True,
        'backend':{'build_TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':'1','formal_TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':None,
            'reference':master['source_binding']['featcal_plan'],'scope':'smoke/teachers/solve match original Table1 construction; formal keeps its unchanged evaluation contract'},
        'supersedes':'v1 cleared the original construction TF32 override; preserve v1 and use new explicit permit, never hot-edit a running source'}
    RUN.mkdir(parents=True);write(RUN/'plan.json',plan);digest=sha(RUN/'plan.json')
    write(RUN/'PLAN-SHA256.json',{'sha256':digest})
    write(RUN/'PERMIT-TEMPLATE-NOT-AUTHORIZED.json',{'schema':'fig4_featcal_queue_execution_permit_v2','allowed':False,
        'queue_plan_sha256':digest,'master_plan_sha256':MASTER_SHA,'host':HOST,'gpu_uuids':plan['gpu_uuids'],
        'job_ids':[j['id'] for j in jobs],'formal_episodes':500})
    return {'status':'CPU_PREPARED_GPU_NOT_STARTED','plan_sha256':digest,'stage_jobs':14,'formal_jobs':5,'formal_episodes':500}

def plan(large=False):
    p=read(RUN/'plan.json');require(sha(RUN/'plan.json')==read(RUN/'PLAN-SHA256.json')['sha256'],'Queue plan changed')
    require(p['resource']==RESOURCE and p['host']==HOST and p['gpus']==list(GPUS),'Resource policy changed')
    for path,digest in p['assets'].items():
        require(not STOP,'Owner stopped queue during input verification')
        require(Path(path).is_file(),'Frozen asset missing')
        if large or Path(path).stat().st_size<100*2**20:require(sha(path)==digest,'Frozen asset hash drift '+path)
    return p
def validate_permit(permit,p,digest):
    expected={'schema':'fig4_featcal_queue_execution_permit_v2','allowed':True,'queue_plan_sha256':digest,
        'master_plan_sha256':MASTER_SHA,'host':p['host'],'gpu_uuids':p['gpu_uuids'],
        'job_ids':[j['id'] for j in p['jobs']],'formal_episodes':500}
    for k,v in expected.items():require(permit.get(k)==v,'Missing or mismatched execution authorization: '+k)
def consume_permit(path,p,root=RUN):
    validate_permit(read(path),p,sha(root/'plan.json'))
    write(root/'AUTHORIZATION-CONSUMED.json',{'permit_sha256':sha(path),'queue_plan_sha256':sha(root/'plan.json'),'time':now(),'pid':os.getpid()})

def audit_formal(job,p,model):
    out=Path(job['output'])/'eval';info=read(out/'eval_info.json');reset=read(out/'procedural_bank_receipt.json');bounded=read(out/'bounded_resources.json')
    selection=p['selection'];require(reset['eval_seed']==274001 and reset['repeat_id']=='repeat-01','Formal seed/repeat differs')
    require(reset['selection_sha256']==selection['sha256'] and reset['bank_manifest_sha256']==selection['bank_manifest_sha256'],'Reset bank differs')
    require(reset['eval_info_sha256']==sha(out/'eval_info.json') and reset['entrypoint_sha256']==p['assets'][str(REPO/'scripts/eval_pi05_policy_with_procedural_bank.py')],'Result/evaluator changed')
    require(bounded['evaluation_completed'] is True and bounded['changes_policy_rng'] is False,'Native evaluator did not complete')
    bank=read(Path(selection['bank'])/'manifest.json')['tasks'];selected=read(selection['selection'])['tasks'];suite=job['suite']
    keys={f'{suite}/{i:02d}' for i in range(10)};require(set(reset['tasks'])==keys,'Reset task coverage differs')
    for key in keys:
        actual=reset['tasks'][key];indices=selected[key]
        require(actual['suite']==suite and actual['task_id']==int(key[-2:]) and actual['task_name']==bank[key]['task_name'],'Task identity differs')
        require(actual['state_indices']==indices and actual['raw_state_sha256']==[bank[key]['states'][i]['raw_sha256'] for i in indices],'Actual reset differs')
    rows=info['per_task'];require(len(rows)==10 and {r['task_id'] for r in rows}==set(range(10)),'Task denominator differs')
    outcomes=[]
    for row in rows:
        values=row['metrics']['successes'];require(len(values)==10 and all(type(x) is bool for x in values),'Ten bools per task required')
        require(row['task_group']==suite,'Suite differs');outcomes.extend(values)
    require(info['overall']['n_episodes']==100 and abs(info['overall']['pc_success']-sum(outcomes))<1e-6,'Overall denominator differs')
    return {'id':job['id'],'subset':job['subset'],'suite':suite,'episodes':100,'successes':sum(outcomes),
            'model_sha256':model['sha256'],'eval_info_sha256':sha(out/'eval_info.json'),'reset_receipt_sha256':sha(out/'procedural_bank_receipt.json')}

def verify_export(api,run):
    import torch
    from safetensors import safe_open
    complete=read(run/'complete.json');result=complete['checkpoint'];digest=sha(run/'plan.json')
    require(complete['status']=='complete' and complete['reload_bitwise_equal'] is True,'Original reload parity incomplete')
    require(result['plan_sha256']==digest and result['subset']==api.subset and result['linear_weights']==418,'Export identity differs')
    current=api.soup_sha;previous=None;seen=set()
    for step in api.steps:
        snapshot,receipt=api.baseline._snapshot_paths(run/'cache',step.step);s=read(receipt)
        require(s['status']=='complete' and s['step']==step.step and s['input_student_prefix_sha256']==current and s['previous_solve']==previous and s['atomic_step_load'] is True,'Broken atomic prefix chain')
        require(sha(snapshot)==s['snapshot']['sha256'],'Snapshot changed')
        with safe_open(snapshot,framework='pt',device='cpu') as h:
            require(set(h.keys())==set(step.target_module_paths)==set(s['solved_tensor_sha256']),'Stage target set differs')
            for key in h.keys():
                x=h.get_tensor(key);require(torch.isfinite(x).all().item() and api.baseline.tensor_sha256(x)==s['solved_tensor_sha256'][key],'Invalid snapshot tensor')
        require(not seen.intersection(step.target_module_paths),'Duplicate solved target');seen.update(step.target_module_paths)
        current=api.baseline.prefix_chain_sha256(current,step.step,s['solved_tensor_sha256'])
        require(current==s['output_student_prefix_sha256'],'Prefix digest differs')
        previous={'path':str(receipt.relative_to(run/'cache')),'sha256':sha(receipt)}
    require(len(seen)==418 and current==result['final_prefix_sha256'],'Incomplete 47-stage/418-target solve')
    model=Path(result['path'])/'model.safetensors';require(sha(model)==result['model_sha256'],'Export bytes changed')
    with safe_open(model,framework='pt',device='cpu') as h:
        require(len(h.keys())==813,'Export tensor count differs')
        for key in h.keys():require(torch.isfinite(h.get_tensor(key)).all().item(),'Export has nonfinite tensor')
    return {'path':str(model.parent),'sha256':result['model_sha256'],'stat':stat_id(model),'stages':47,'linear_weights':418,'reload_bitwise_equal':True,
        'sidecars':{str(x.relative_to(model.parent)):sha(x) for x in model.parent.rglob('*') if x.is_file() and x.name!='model.safetensors'}}

def worker(job_id,ticket_path):
    p=plan();job=next(j for j in p['jobs'] if j['id']==job_id);ticket=read(ticket_path)
    require(Path(ticket_path).resolve()==(RUN/'tickets'/f'{job_id}.json').resolve(),'Worker ticket path differs')
    require(ticket['job_id']==job_id and ticket['plan_sha256']==sha(RUN/'plan.json'),'Worker ticket differs')
    require(os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='1','Construction backend must match original Table1 override=1')
    require(ticket['gpu'] in GPUS and ticket['uuid']==p['gpu_uuids'][str(ticket['gpu'])],'Unpermitted GPU identity')
    require(ticket['permit_sha256']==read(RUN/'AUTHORIZATION-CONSUMED.json')['permit_sha256'],'Worker lacks consumed permit binding')
    require(os.getppid()==ticket['parent_pid'] and stamp(ticket['parent_pid'])==ticket['parent_start'],'Worker not owned by live supervisor')
    require(socket.gethostname()==HOST and os.environ.get('CUDA_VISIBLE_DEVICES')==str(ticket['gpu']),'Worker GPU/host differs')
    for row in ticket['leases']:
        s=os.fstat(row['fd']);actual=Path(row['path']).stat()
        require((s.st_dev,s.st_ino)==(row['device'],row['inode'])==(actual.st_dev,actual.st_ino),'Inherited lease descriptor changed')
        fcntl.flock(row['fd'],fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=ADAPTER/job['subset']
    with (run/f'{job["stage"]}-{job.get("group","all")}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(shutil.disk_usage(run).free>=120*2**30,'Disk admission fell below 120GiB')
        import torch
        api_module=module(API_SOURCE,'fig4_featcal_original_api')
        api=api_module.API(WORK,FIG/'plan.json',job['subset']);api.active_gpu=ticket['gpu'];torch.set_num_threads(4)
        original_check=api.check_runtime_memory
        def check():
            original_check();require(shutil.disk_usage(run).free>=32*2**30,'Own stage disk reserve below 32GiB')
        api.check_runtime_memory=check
        kernel=api.baseline.featcal_hybrid_linear_weight
        def checked_kernel(*args,**kwargs):check();result=kernel(*args,**kwargs);check();return result
        api.baseline.featcal_hybrid_linear_weight=checked_kernel
        forward=api.forward
        def finite_forward(*args,**kwargs):
            x=forward(*args,**kwargs);require(torch.isfinite(x).all().item(),'Native velocity has nonfinite values');return x
        api.forward=finite_forward;torch.cuda.set_per_process_memory_fraction(.5,0)
        if job['stage']=='smoke':api.smoke(run);result=read(run/'native-smoke.json')
        elif job['stage']=='teachers':
            require(job['group'] in api.groups,'Excluded teacher');api.teachers(run,job['group']);result=read(run/f'teachers-{job["group"]}.json')
        elif job['stage']=='solve':api.solve(run);result=verify_export(api,run)
        else:raise ValueError('Not an API worker stage')
        write(RUN/'worker-results'/f'{job_id}.json',{'job_id':job_id,'stage':job['stage'],'subset':job['subset'],'time':now(),'result':result,
            'subset_plan_sha256':SUBSET_SHAS[job['subset']],'original_api_unchanged':True,'gpu':ticket['gpu']})

def stop_handler(*_):
    global STOP
    STOP=True
def parent_death_guard(expected):
    if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError('Cannot install parent-death guard')
    if os.getppid()!=expected:os._exit(143)
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT})
def spawn_owned(command,env,log,leases):
    old=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT})
    try:
        require(not STOP and not signal.sigpending().intersection({signal.SIGTERM,signal.SIGINT}),'Stop requested before child launch')
        parent_pid=os.getpid()
        return subprocess.Popen(command,cwd=WORK,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=tuple(l.handle for l in leases),preexec_fn=lambda:parent_death_guard(parent_pid))
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,old)
def terminate_own(items):
    for x in items:
        if x['process'].poll() is None:x['process'].terminate()
    for x in items:
        try:x['process'].wait(timeout=30)
        except subprocess.TimeoutExpired:x['process'].kill();x['process'].wait()
        x['log'].close()
        for lease in x['leases']:lease.release()
def environment(gpu,stage='formal'):
    require(stage in ['smoke','teachers','solve','formal'],'Unknown stage backend')
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),ITERATION_PHYSICAL_GPU=str(gpu),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=str(gpu),
        LIBERO_CONFIG_PATH=str(CONFIG),PALIGEMMA_TOKENIZER_PATH=str(WORK/'pi05_lora_finetune_v2_20260826/assets/paligemma-3b-pt-224-tokenizer'),
        PYTHONPATH=':'.join([str(REPO),str(REPO/'src'),str(REPO/'scripts'),str(WORK/'pi05_lora_finetune_v2_20260826/src'),str(WORK/'pi05_lora_finetune_v2_20260826/lerobot/src')]))
    for key in ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','PI05_LIBERO_INIT_STATE_OFFSET','PI05_LIBERO_INIT_STATE_COUNT','LIBERO_PRO_REPO','LIBERO_PRO_ASSET_DIR']:e.pop(key,None)
    if stage!='formal':e['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='1'
    return e

def run(permit_path):
    p=plan();require(socket.gethostname()==HOST,'Wrong host');require(not (RUN/'STARTED.json').exists(),'No restart')
    import card_flock
    with (RUN/'supervisor.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
        consume_permit(permit_path,p)
        plan(large=True)
        write(RUN/'ASSET-GATE.json',{'status':'full_source_and_model_hashes_pass','time':now(),'assets':len(p['assets'])})
        require({str(g):gpu_rows()[g]['uuid'] for g in GPUS}==p['gpu_uuids'],'GPU UUID map changed')
        signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
        write(RUN/'STARTED.json',{'pid':os.getpid(),'start_ticks':stamp(os.getpid()),'time':now(),'plan_sha256':sha(RUN/'plan.json')})
        pending=list(p['jobs']);active={};accepted={};models={}
        try:
            while pending or active:
                require(not STOP,'Owner stopped queue; no retry')
                for jid,x in list(active.items()):
                    code=x['process'].poll()
                    if code is None:continue
                    x['process'].wait();x['log'].close()
                    require(code==0,f'{jid} exited {code}; no retry')
                    transition=wait_context_clear(x['gpu'],x['uuid'])
                    write(RUN/'exits'/f'{jid}.json',{'returncode':code,'pid':x['process'].pid,'time':now(),'context_release_observations':transition})
                    job=x['job']
                    if job['stage']=='formal':result=audit_formal(job,p,models[job['subset']])
                    else:
                        evidence=read(RUN/'worker-results'/f'{jid}.json')
                        require(evidence['job_id']==jid and evidence['subset_plan_sha256']==SUBSET_SHAS[job['subset']],'Worker result binding differs')
                        result=evidence['result']
                        if job['stage']=='solve':models[job['subset']]=result
                    accepted[jid]=result;write(RUN/'accepted'/f'{jid}.json',result)
                    for lease in x['leases']:lease.release()
                    active.pop(jid)
                rows=gpu_rows()
                for gpu in GPUS:
                    require(not STOP,'Owner stopped queue before dispatch')
                    if len(active)>=4:break
                    if any(x['gpu']==gpu for x in active.values()):continue
                    ready=[j for j in pending if set(j['needs'])<=set(accepted)]
                    if not ready:break
                    row=rows[gpu]
                    if not resource_ok(row,shutil.disk_usage(RUN).free):continue
                    job=ready[0];leases=[]
                    legacy=card_flock._try_lock(WORK/'vla-merge-runtime/resource-leases'/HOST/f'gpu-{gpu}.lock',job['id'],{'job':job['id'],'stage':'fig4-featcal','gpu':gpu})
                    if legacy is None:continue
                    leases.append(legacy)
                    card=card_flock.take_card(gpu,row['uuid'],job['id'],'fig4-featcal')
                    if card is None:legacy.release();continue
                    leases.append(card)
                    compatibility=card_flock._try_lock(WORK/'vla-merge-runtime/resource-leases'/f'{HOST}-gpu-{gpu}.lock',job['id'],{'job':job['id'],'stage':'fig4-featcal-legacy-compatibility','gpu':gpu})
                    if compatibility is None:
                        for lease in leases:lease.release()
                        continue
                    leases.append(compatibility)
                    if job['stage']!='formal':
                        slot=card_flock.take_build_slot(job['id'])
                        if slot is None:
                            for lease in leases:lease.release()
                            continue
                        leases.append(slot)
                    again=gpu_rows()[gpu]
                    if again['uuid']!=row['uuid'] or not resource_ok(again,shutil.disk_usage(RUN).free):
                        for lease in leases:lease.release()
                        continue
                    process=None;log=None
                    try:
                        require(not (RUN/'launches'/f'{job["id"]}.json').exists(),'Duplicate stage launch')
                        if job['stage']=='formal':
                            m=models[job['subset']];require(stat_id(Path(m['path'])/'model.safetensors')==m['stat'],'Export model changed')
                            for rel,digest in m['sidecars'].items():require(sha(Path(m['path'])/rel)==digest,'Export sidecar changed')
                            require(not Path(job['output']).exists(),'Formal output exists')
                            command=job['command']
                        else:
                            ticket=RUN/'tickets'/f'{job["id"]}.json'
                            write(ticket,{'job_id':job['id'],'plan_sha256':sha(RUN/'plan.json'),'parent_pid':os.getpid(),'parent_start':stamp(os.getpid()),'gpu':gpu,
                                'uuid':row['uuid'],'permit_sha256':read(RUN/'AUTHORIZATION-CONSUMED.json')['permit_sha256'],
                                'leases':[{'fd':l.handle,'path':str(l.path),'device':os.fstat(l.handle).st_dev,'inode':os.fstat(l.handle).st_ino} for l in leases]})
                            command=[str(PYTHON),'-u',str(Path(__file__)),'worker','--job-id',job['id'],'--ticket',str(ticket)]
                        path=RUN/'logs'/f'{job["id"]}.log';path.parent.mkdir(parents=True,exist_ok=True);log=path.open('x')
                        process=spawn_owned(command,environment(gpu,job['stage']),log,leases)
                        active[job['id']]={'job':job,'process':process,'log':log,'leases':leases,'gpu':gpu,'uuid':row['uuid']}
                        write(RUN/'launches'/f'{job["id"]}.json',{'pid':process.pid,'start_ticks':stamp(process.pid),'time':now(),'gpu':gpu,'uuid':row['uuid'],'admission':again,'command':command,
                            'model_sha256':models[job['subset']]['sha256'] if job['stage']=='formal' else None})
                        pending.remove(job)
                        require(not STOP,'Owner stopped queue during registered child launch')
                    except BaseException:
                        if process is not None:
                            terminate_own([{'process':process,'log':log,'leases':leases}]);active.pop(job['id'],None)
                        else:
                            if log:log.close()
                            for lease in leases:lease.release()
                        raise
                write(RUN/'state.json',{'status':'running_or_waiting_owned_idle_cards','time':now(),'pending':[j['id'] for j in pending],
                    'active':{k:{'pid':v['process'].pid,'gpu':v['gpu'],'stage':v['job']['stage']} for k,v in active.items()},'accepted':accepted},False)
                require(not STOP,'Owner stopped queue')
                if pending or active:time.sleep(20)
            formal={j['id']:accepted[j['id']] for j in p['jobs'] if j['stage']=='formal'}
            require(len(formal)==5 and sum(x['episodes'] for x in formal.values())==500,'Formal matrix incomplete')
            write(RUN/'DONE.json',{'status':'complete','time':now(),'models':models,'formal':formal,'formal_episodes':500,'native_smoke_not_counted':True})
        except BaseException as exc:
            terminate_own(list(active.values()))
            write(RUN/'FAILED.json',{'status':'failed_no_retry','time':now(),'error':repr(exc),'accepted':accepted,'pending':[j['id'] for j in pending],
                'other_processes_signalled':False,'partial_artifacts_preserved':True});raise

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('action',choices=['prepare','preflight','run','worker']);a.add_argument('--permit',type=Path);a.add_argument('--job-id');a.add_argument('--ticket',type=Path)
    args=a.parse_args()
    if args.action!='worker':os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.action=='prepare':print(json.dumps(prepare(),indent=2))
    elif args.action=='preflight':
        p=plan();cpu_preflight();print(json.dumps({'status':'CPU_PASS_GPU_NOT_STARTED','jobs':len(p['jobs']),'formal_episodes':500,'plan_sha256':sha(RUN/'plan.json')}))
    elif args.action=='worker':require(args.job_id and args.ticket,'Worker ticket required');worker(args.job_id,args.ticket)
    else:
        require(args.permit is not None,'Separate explicit execution permit required')
        try:run(args.permit)
        except BaseException as exc:
            if (RUN/'AUTHORIZATION-CONSUMED.json').exists() and not (RUN/'FAILED.json').exists():
                write(RUN/'FAILED.json',{'status':'failed_no_retry_before_dispatch','time':now(),'error':repr(exc),'other_processes_signalled':False})
            raise
