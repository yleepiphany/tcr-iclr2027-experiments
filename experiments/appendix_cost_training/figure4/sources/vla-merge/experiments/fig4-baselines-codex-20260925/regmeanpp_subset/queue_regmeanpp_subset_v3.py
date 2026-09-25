#!/usr/bin/env python3
"""CPU-freezable, single-use queue for two subset builds and five native suites."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from fig4_regmeanpp_contract import WORK,FIG4,FIG4_SHA,digest,read,require,frozen_figure,validate_adapter_plan
from prepare_regmeanpp_subset import OUTPUT as BUILDS,command as build_command,source_parity

HERE=Path(__file__).resolve().parent
VLA=WORK/'vla-merge';RUNTIME=WORK/'vla-merge-runtime';SOURCE=WORK/'pi05_lora_finetune_v2_20260826'
RUN=RUNTIME/'experiments/fig4-baselines-codex-20260925/regmeanpp-queue-v3'
PYTHON=SOURCE/'.venv/bin/python'
FLOCK=VLA/'experiments/claude-firstpass-cause-20260920/card_flock.py'
HOST='dsw-824375-57c745db88-n6tv9'
GPUS={1:'GPU-c18a3bf3-c47c-9a8a-75b0-5a0f37b33a65',4:'GPU-48baba1c-f0e7-93b7-4a30-b3c21d20e61a',5:'GPU-4ed28198-b742-ee3c-2acb-4183dc944c81',6:'GPU-da625da2-f504-7927-4025-907e876718aa'}
MIN_FREE=70*1024;FLOOR=12*1024

def now():return datetime.now(timezone.utc).isoformat()
def write(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    else:
        temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)
def binding(path,sha=None):
    path=Path(path);return {'path':str(path),'sha256':sha or digest(path),'bytes':path.stat().st_size}

def expected_tasks(suite,bank,selection):
    result={}
    for tid in range(10):
        key=f'{suite}/{tid:02d}';task=bank['tasks'][key];indices=selection['tasks'][key]
        require(indices==list(range(10)),'Only fixed repeat-01 indices allowed')
        raw={r['index']:r['raw_sha256'] for r in task['states']}
        result[key]={'suite':suite,'task_id':tid,'task_name':task['task_name'],'state_indices':indices,'raw_state_sha256':[raw[i] for i in indices]}
    return result

def make_plan():
    fig=frozen_figure();source_parity(fig)
    selected=[j for j in fig['new_builds'] if j['method']=='regmeanpp']
    evaluations=[copy.deepcopy(j) for j in fig['new_evaluations'] if j['method']=='regmeanpp']
    require(len(selected)==2 and len(evaluations)==5,'Unexpected RegMean++ subset workload')
    selection_path=Path(fig['evaluation_selection']['path']);require(digest(selection_path)==fig['evaluation_selection']['sha256'],'Selection bytes changed')
    selection=read(selection_path);bank_path=selection_path.parent.parent/'manifest.json';bank=read(bank_path)
    require(selection['repeat_id']=='repeat-01' and selection['eval_seed']==274001 and selection['bank_manifest_sha256']==digest(bank_path),'Bank/repeat/seed differs')
    assets={}
    def add(path,sha=None):
        b=binding(path,sha);assets[b['path']]=b
    builds=[]
    for source in selected:
        path=BUILDS/source['subset']/'plan.json';p=validate_adapter_plan(read(path),False)
        require(p['job_id']==source['id'],'Subset adapter identity differs')
        add(path);add(p['subset_bank']['path'],p['subset_bank']['sha256']);add(p['dense_bank']['path'],p['dense_bank']['sha256'])
        subset=read(p['subset_bank']['path']);add(Path(p['base_model'])/'model.safetensors',subset['base']['model']['actual_sha256'])
        bank_experts={r['name']:r for r in read(p['dense_bank']['path'])['experts']}
        for n in p['experts']:
            inp=p['inputs'][n];add(inp['manifest_path'],inp['manifest_sha256']);add(inp['replay_path'],inp['replay_sha256'])
            add(Path(inp['dense_path'])/'model.safetensors',inp['dense_sha256'])
            adapter=next(x for x in subset['expert_bank']['experts'] if x['name']==n)
            add(Path(inp['adapter_path'])/'adapter_model.safetensors',adapter['adapter']['actual_sha256'])
            for label in ('manifest','verification'):
                x=bank_experts[n]['dense_checkpoint'][label];add(x['path'],x['sha256'])
        for path,sha in p['implementation_sha256'].items():add(path,sha)
        builds.append({'id':p['job_id'],'subset':p['subset'],'experts':p['experts'],'checkpoint':p['output'],'observations':p['observations_total'],'rows':p['expected_realized_rows'],'build_plan':binding(BUILDS/p['subset']/'plan.json'),'command':build_command(p,BUILDS/p['subset']/'plan.json')})
    for job in evaluations:
        require(job['needs_build'] in {b['id'] for b in builds} and job['repeat']=='repeat-01' and job['seed']==274001 and job['episodes']==100,'Evaluation identity differs')
        command=job['command_template'];require('--output_dir='+job['output'] in command,'Frozen evaluation output differs')
        require('--procedural-selection='+str(selection_path) in command and '--seed=274001' in command,'Frozen reset command differs')
        job['expected_tasks']=expected_tasks(job['suite'],bank,selection)
    source=fig['inputs']['featcal']['spatial'];sample=read(source['manifest']['path'])['samples'][0]
    require(sample['observation_index']==0 and sample['timestep']==1 and sample['noise_replica']==0,'Native smoke identity differs')
    request={'observations':source['observations'],'manifest':source['manifest'],'sample':sample,'scope':'One preprocessed static observation for native export acceptance only; not an added regression sample or success gate'}
    add(source['observations']['path'],source['observations']['sha256']);add(source['manifest']['path'],source['manifest']['sha256']);add(bank_path);add(selection_path)
    for path in (Path(__file__),HERE/'accept_subset_checkpoint.py',FLOCK,VLA/'scripts/eval_tcr_10k_formal_bounded.py',VLA/'scripts/eval_pi05_policy_with_procedural_bank.py',VLA/'scripts/run_pi05_generation_path_probe.py',SOURCE/'src/eval_with_local_tokenizer.py'):
        add(path)
    return {'schema':'fig4_regmeanpp_two_build_five_suite_queue_v1','fig4_plan':str(FIG4),'fig4_plan_sha256':FIG4_SHA,'host':HOST,'builds':builds,'evaluations':evaluations,'build_count':2,'evaluation_jobs':5,'episodes':500,'four_expert_endpoint_reused_not_queued':True,'native_request':request,'assets':list(assets.values()),'selection':binding(selection_path),'bank':binding(bank_path),'resource':{'gpu_uuids':{str(g):u for g,u in GPUS.items()},'minimum_free_mib':MIN_FREE,'max_idle_used_mib':64,'require_zero_utilization':True,'require_no_compute_pid':True,'runtime_floor_mib':FLOOR,'dual_leases':'existing host/UUID card_flock plus host/index resource lease','build_slot':'existing host-wide build slot; held through native acceptance','max_active':4},'authorization':'A separate single-use allowed execution permit must bind this exact queue plan SHA before scheduling; permit may wait without GPU while PRO owns every card.','no_training':True,'no_parameter_search':True,'no_score_based_gate':True,'no_retry':True,'gpu_jobs_launched_by_preparation':0}

def validate_plan(plan,large=False):
    require(plan==make_plan(),'Queue reconstruction differs from frozen parent or assets')
    for asset in plan['assets']:
        p=Path(asset['path']);require(p.is_file() and p.stat().st_size==asset['bytes'],'Asset absent/size differs: '+str(p))
        if large or asset['bytes']<=64*(1<<20):require(digest(p)==asset['sha256'],'Asset SHA differs: '+str(p))
    return plan

def validate_permit(permit,plan_sha):
    require(permit.get('schema')=='fig4_regmeanpp_queue_execution_permit_v1' and permit.get('allowed') is True,'Missing allowed execution permit')
    require(permit.get('queue_plan_sha256')==plan_sha and permit.get('fig4_plan_sha256')==FIG4_SHA,'Permit identity differs')
    require(permit.get('host')==HOST and permit.get('gpu_uuids')=={str(g):u for g,u in GPUS.items()},'Permit card allocation differs')
    require(permit.get('build_ids')==['regmeanpp-spatial-goal','regmeanpp-spatial-object-goal'] and permit.get('evaluation_jobs')==5 and permit.get('episodes')==500,'Permit workload differs')

def board(gpu):
    line=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    uuid,free,used,util=[x.strip() for x in line.split(',')]
    pids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    return {'gpu':gpu,'uuid':uuid,'free_mib':int(free),'used_mib':int(used),'utilization':int(util),'compute_pids':pids.splitlines() if pids else []}
def admissible(row):return row['gpu'] in GPUS and row['uuid']==GPUS[row['gpu']] and row['free_mib']>=MIN_FREE and row['used_mib']<=64 and row['utilization']==0 and not row['compute_pids']

def wait_for_idle_transition(gpu,releasing_pid,timeout_seconds=60.,poll_seconds=2.,*,board_fn=board,sleep_fn=time.sleep,clock=time.monotonic,process_exists=None):
    """Wait under the same held leases for the completed CUDA context to drain.

    A stale NVML entry may name only the already-reaped child. Any other PID,
    or reuse of that PID by a live process, is a foreign owner and fails closed.
    This helper never releases locks or signals any process.
    """
    require(timeout_seconds>0 and poll_seconds>0,'Invalid bounded transition wait')
    process_exists=process_exists or (lambda pid: Path(f"/proc/{pid}").exists())
    started=clock();deadline=started+timeout_seconds;history=[]
    while True:
        row=board_fn(gpu)
        require(row['uuid']==GPUS[gpu],'GPU UUID changed during context release')
        foreign=[pid for pid in row['compute_pids'] if str(pid)!=str(releasing_pid) or process_exists(pid)]
        require(not foreign,'Foreign compute PID during context release: '+str(foreign))
        history.append({**row,'elapsed_seconds':clock()-started})
        if admissible(row):
            return {'status':'idle_after_completed_child','releasing_child_pid':releasing_pid,'maximum_wait_seconds':timeout_seconds,'waited_seconds':clock()-started,'checks':history,'same_leases_retained':True,'signals_sent':0}
        current=clock()
        if current>=deadline:raise TimeoutError('CUDA context did not become idle within bounded transition wait')
        sleep_fn(min(poll_seconds,deadline-current))

def environment(gpu):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),ITERATION_PHYSICAL_GPU=str(gpu),REGMEAN_ALLOCATOR_FRACTION='.70',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=str(gpu),LIBERO_CONFIG_PATH=str(WORK/'.datasets/LIBERO/20260919/config-standard'),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',PALIGEMMA_TOKENIZER_PATH=str(SOURCE/'assets/paligemma-3b-pt-224-tokenizer'),PYTHONPATH=':'.join(map(str,[RUNTIME/'python-overlay',SOURCE/'src',SOURCE/'lerobot/src',VLA/'scripts',VLA/'src',HERE])))
    for key in ('PI05_LIBERO_INIT_STATE_OFFSET','PI05_LIBERO_INIT_STATE_COUNT','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','LIBERO_PRO_REPO','LIBERO_PRO_ASSET_DIR'):env.pop(key,None)
    return env

def verify_evaluation(job,model_sha,plan):
    output=Path(job['output']);info=read(output/'eval_info.json');reset=read(output/'procedural_bank_receipt.json')
    require(reset['bank_manifest_sha256']==plan['bank']['sha256'] and reset['selection_sha256']==plan['selection']['sha256'],'Reset bank/selection changed')
    require(reset['repeat_id']=='repeat-01' and reset['eval_seed']==274001 and reset['tasks']==job['expected_tasks'],'Reset task/seed/raw state identity differs')
    expected_entry=next(x['sha256'] for x in plan['assets'] if x['path']==str(VLA/'scripts/eval_pi05_policy_with_procedural_bank.py'))
    require(reset['entrypoint_sha256']==expected_entry and reset['eval_info_sha256']==digest(output/'eval_info.json'),'Evaluator/result SHA differs')
    require(len(info['per_task'])==10 and {x['task_id'] for x in info['per_task']}==set(range(10)),'Task count differs')
    values=[]
    for task in info['per_task']:
        row=task['metrics']['successes'];require(task['task_group']==job['suite'] and len(row)==10 and all(type(x) is bool for x in row),'Expected ten boolean outcomes per task');values.extend(row)
    require(info['overall']['n_episodes']==100 and info['overall']['pc_success']==sum(values),'Aggregate mismatch')
    resources=read(output/'bounded_resources.json');require(resources['evaluation_completed'] is True and resources['changes_policy_rng'] is False and resources['changes_weights'] is False,'Native bounded evaluator failed')
    return {'id':job['id'],'model_sha256':model_sha,'episodes':100,'successes':sum(values),'eval_info_sha256':digest(output/'eval_info.json'),'reset_receipt_sha256':digest(output/'procedural_bank_receipt.json')}

def run_queue(permit_path):
    require(socket.gethostname()==HOST,'Wrong execution host')
    plan=validate_plan(read(RUN/'plan.json'),True);plan_sha=digest(RUN/'plan.json');permit=read(permit_path);validate_permit(permit,plan_sha)
    spec=importlib.util.spec_from_file_location('fig4_subset_flocks',FLOCK);flocks=importlib.util.module_from_spec(spec);spec.loader.exec_module(flocks)
    owner=flocks._try_lock(RUN/'supervisor.lock','fig4-regmeanpp-queue',{'stage':'fig4-regmeanpp-subset'})
    require(owner is not None,'Queue already owned')
    try:
        require(not(RUN/'AUTHORIZATION-CONSUMED.json').exists(),'Single-use execution permit already consumed')
        require(all(not Path(j['checkpoint']).exists() for j in plan['builds']),'Existing model output requires separate recovery, not overwrite')
        require(all(not Path(j['output']).exists() for j in plan['evaluations']),'Existing evaluation output requires separate recovery')
        write(RUN/'AUTHORIZATION-CONSUMED.json',{'queue_plan_sha256':plan_sha,'permit_path':str(Path(permit_path).resolve()),'permit_sha256':digest(permit_path),'pid':os.getpid(),'host':HOST,'consumed_at':now()},True)
        execute(plan,plan_sha,flocks)
    finally:owner.release()

def execute(plan,plan_sha,flocks):
    pending_builds=list(plan['builds']);pending_evals=list(plan['evaluations']);active={};builds={};evaluations={};failed={};stop=False
    def on_signal(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    write(RUN/'STARTED.json',{'pid':os.getpid(),'host':HOST,'plan_sha256':plan_sha,'started_at':now()},True)
    def spawn(job,stage,gpu,held):
        env=environment(gpu);identifier=job['id'];base=RUN/'work'/identifier
        if stage=='build':
            permit=base/'single-use-build-permit.json'
            write(permit,{'schema':'fig4_regmeanpp_execution_permit_v1','allowed':True,'job_id':identifier,'fig4_plan_sha256':FIG4_SHA,'build_plan_sha256':job['build_plan']['sha256'],'queue_plan_sha256':plan_sha,'gpu':gpu,'uuid':GPUS[gpu],'owner_pid':os.getpid()},True)
            env['FIG4_REGMEANPP_EXECUTION_PERMIT']=str(permit);cmd=job['command']
        elif stage=='accept':
            cmd=[str(PYTHON),'-u',str(HERE/'accept_subset_checkpoint.py'),'--queue-plan',str(RUN/'plan.json'),'--queue-plan-sha256',plan_sha,'--build-id',identifier,'--receipt',str(base/'acceptance.json')]
        else:cmd=job['command_template']
        logpath=base/(stage+'.log');logpath.parent.mkdir(parents=True,exist_ok=True);log=logpath.open('x')
        try:child=subprocess.Popen(cmd,cwd=VLA,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(x.handle for x in held if x is not None))
        except BaseException:log.close();raise
        active[gpu]={'job':job,'stage':stage,'child':child,'held':held,'log':log,'floor':False}
        write(base/(stage+'-launch.json'),{'pid':child.pid,'gpu':gpu,'uuid':GPUS[gpu],'command':cmd,'queue_plan_sha256':plan_sha,'started_at':now()},True)
    def release(item):
        for held in reversed(item['held']):
            if held is not None:held.release()
    def finish(gpu,item,allow_next):
        nonlocal stop
        child=item['child'];code=child.wait();item['log'].close();job=item['job'];stage=item['stage'];base=RUN/'work'/job['id'];error=None;artifact=None
        try:
            require(code==0 and not item['floor'],'Own worker failed or crossed resource floor')
            if stage=='build':
                m=read(Path(job['checkpoint'])/'block_regmeanpp_manifest.json');require(m['experiment_manifest']['sha256']==job['build_plan']['sha256'],'Build manifest identity differs')
            elif stage=='accept':
                artifact=read(base/'acceptance.json');require(artifact['status']=='accepted_export_and_one_native_request' and artifact['build_plan_sha256']==job['build_plan']['sha256'] and artifact['selected_experts']==job['experts'],'Native export acceptance differs');builds[job['id']]=artifact
            else:
                artifact=verify_evaluation(job,builds[job['needs_build']]['model_sha256'],plan);evaluations[job['id']]=artifact
        except Exception as exc:error=repr(exc);failed[job['id']+'/'+stage]=error;stop=True
        write(base/(stage+'-exit.json'),{'returncode':code,'accepted':error is None,'artifact':artifact,'error':error,'finished_at':now()},True)
        active.pop(gpu)
        if error is None and stage=='build' and allow_next:
            try:
                transition=wait_for_idle_transition(gpu,child.pid)
                write(base/'build-to-accept-resource-transition.json',transition,True)
                spawn(job,'accept',gpu,item['held']);return
            except Exception as exc:failed[job['id']+'/accept-launch']=repr(exc);stop=True
        release(item)
    try:
        while pending_builds or pending_evals or active:
            for gpu,item in list(active.items()):
                if item['child'].poll() is not None:finish(gpu,item,not stop);continue
                row=board(gpu)
                if (row['uuid']!=GPUS[gpu] or row['free_mib']<FLOOR) and not item['floor']:
                    os.killpg(item['child'].pid,signal.SIGTERM);item['floor']=True
            if not stop:
                for gpu in GPUS:
                    if gpu in active:continue
                    choices=[('build',j) for j in pending_builds]+[('eval',j) for j in pending_evals if j['needs_build'] in builds]
                    if not choices:break
                    stage,job=choices[0];row=board(gpu)
                    if not admissible(row):continue
                    card=flocks.take_card(gpu,GPUS[gpu],job['id'],'fig4-regmeanpp-subset')
                    if card is None:continue
                    legacy=flocks._try_lock(RUNTIME/'resource-leases'/HOST/f'gpu-{gpu}.lock',job['id'],{'stage':'fig4-regmeanpp-subset','gpu':gpu})
                    if legacy is None:card.release();continue
                    slot=flocks.take_build_slot(job['id']) if stage=='build' else None
                    if (stage=='build' and slot is None) or not admissible(board(gpu)):
                        if slot:slot.release()
                        legacy.release();card.release();continue
                    held=(card,legacy,slot)
                    try:
                        require(not Path(job['checkpoint'] if stage=='build' else job['output']).exists(),'Output already exists')
                        if stage=='build':validate_adapter_plan(read(job['build_plan']['path']),True)
                        else:
                            checkpoint=Path(next(b['checkpoint'] for b in plan['builds'] if b['id']==job['needs_build']))
                            expected_sha=builds[job['needs_build']]['model_sha256']
                            require(digest(checkpoint/'model.safetensors')==expected_sha,'Accepted checkpoint changed before evaluation')
                            require(read(checkpoint/'block_regmeanpp_manifest.json')['model_sha256']==expected_sha,'Checkpoint manifest changed')
                        if not admissible(board(gpu)):
                            for lock in reversed(held):
                                if lock:lock.release()
                            continue
                        spawn(job,stage,gpu,held)
                    except BaseException:
                        for lock in reversed(held):
                            if lock:lock.release()
                        raise
                    (pending_builds if stage=='build' else pending_evals).remove(job)
            write(RUN/'state.json',{'accepted_builds':builds,'accepted_evaluations':evaluations,'failed':failed,'active':{str(g):{'id':x['job']['id'],'stage':x['stage'],'pid':x['child'].pid} for g,x in active.items()},'pending_builds':[j['id'] for j in pending_builds],'pending_evaluations':[j['id'] for j in pending_evals],'stopped':stop,'updated_at':now()})
            if stop and not active:break
            if pending_builds or pending_evals or active:time.sleep(15)
    except BaseException as exc:
        failed['supervisor']=repr(exc);stop=True;raise
    finally:
        # Keep leases until only our current children finish; never start a new
        # phase from cleanup, retry, overwrite, or signal another process.
        for gpu,item in list(active.items()):finish(gpu,item,False)
        complete=len(builds)==2 and len(evaluations)==5 and not failed
        write(RUN/'queue-ended.json',{'complete':complete,'accepted_builds':builds,'accepted_evaluations':evaluations,'failed':failed,'pending_builds':[j['id'] for j in pending_builds],'pending_evaluations':[j['id'] for j in pending_evals],'episodes':100*len(evaluations),'stopped':stop,'ended_at':now()})

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','preflight','run'));parser.add_argument('--permit',type=Path);args=parser.parse_args()
    if args.action=='run':require(args.permit is not None,'Explicit single-use permit required');run_queue(args.permit);return
    if args.action=='prepare':
        require(not RUN.exists(),'Queue plan directory already exists');p=make_plan();validate_plan(p,False);write(RUN/'plan.json',p,True)
        template={'schema':'fig4_regmeanpp_queue_execution_permit_v1','allowed':False,'queue_plan_sha256':digest(RUN/'plan.json'),'fig4_plan_sha256':FIG4_SHA,'host':HOST,'gpu_uuids':{str(g):u for g,u in GPUS.items()},'build_ids':[j['id'] for j in p['builds']],'evaluation_jobs':5,'episodes':500,'note':'Review-only template, not execution authorization. A coordinating owner must create an independently approved allowed permit.'}
        write(RUN/'PERMIT-TEMPLATE-NOT-AUTHORIZED.json',template,True)
    p=validate_plan(read(RUN/'plan.json'),False)
    print(json.dumps({'status':'PASS_CPU_QUEUE_PLAN','plan':str(RUN/'plan.json'),'plan_sha256':digest(RUN/'plan.json'),'builds':2,'native_export_checks':2,'formal_suites':5,'episodes':500,'gpu_execution':'unapproved; run requires a fresh single-use permit and waits for PRO to release reserved cards','gpu_jobs_launched':0},indent=2))
if __name__=='__main__':main()
