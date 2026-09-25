"""Frozen one-checkpoint RoboTwin RegMean++ 9-job formal contract."""
from __future__ import annotations
import copy,hashlib,importlib.util,json,math,os,statistics,struct
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
leases=load('_robotwin_formal_lease_helpers',ROOT/'native_smoke_supervisor_v2/supervisor.py')
sha=leases.sha;read=leases.read;write=leases.write;now=leases.now
CORE=leases.CORE;CORE_SHA=leases.CORE_SHA;HOST=leases.HOST;GPUS=leases.GPUS
WORK=leases.v1.WORK;REPO=leases.v1.REPO;RUNTIME=leases.RUNTIME;FLOCK_FILE=leases.FLOCK_FILE
CHECKPOINT=CORE/'checkpoint'
MODEL_SHA='87268d409201894aee6c903dc9ade71f9a5ac591654f7e55dc7d0da4c3b00b5e'
ACCEPTED=CORE.parent/'materialize-supervisor-v1/MODEL-ACCEPTED.json'
ACCEPTED_SHA='ff00d95bbb372376dd1effe9d5bf0f43b33d018235006205c5936d1afe074f81'
PROTOCOL=RUNTIME/'experiments/iclr2027-table5-20260910/evaluation-queues/robotwin-three-expert-formal-v2/protocol.json'
PROTOCOL_SHA='ed47fc5340cfb53a4a1c497a08ef3aca984c0ad50182748ac3df3ca7e060d0de'
PYTHON=RUNTIME/'envs/iclr2027-robotwin2-py312-mplib-curobo-v3/bin/python'
GROUPS=('coordination','receptacle','precision');METHOD='RegMean++ (static M3 spectral smoothing)'
NATIVE=REPO/'scripts/run_iclr2027_robotwin_checkpoint_development.py'
SLICER=REPO/'scripts/run_iclr2027_robotwin_development_slices.py'
ENV_ADAPTER=WORK/'pi05_lora_finetune_v2_20260826/lerobot/src/lerobot/envs/robotwin.py'

def bind(path):
    p=Path(path).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def verify_binding(row):
    if sha(row['path'])!=row['sha256'] or ('bytes' in row and Path(row['path']).stat().st_size!=row['bytes']):
        raise ValueError('Frozen artifact identity changed: '+row['path'])
def sources():
    source=WORK/'pi05_lora_finetune_v2_20260826/lerobot/src'
    paths=list(HERE.glob('*.py'))+[ROOT/'run_regmeanpp_m3.py',ROOT/'native_smoke_supervisor_v2/supervisor.py',
        ROOT/'native_smoke_supervisor_v1/supervisor.py',FLOCK_FILE,NATIVE,SLICER,ENV_ADAPTER,
        REPO/'scripts/run_iclr2027_robotwin_native_pi05_smoke_v2.py',
        REPO/'scripts/watch_robotwin_experts_formal_v2.py',REPO/'scripts/watch_robotwin_model_soups_formal_v1.py',
        source/'lerobot/policies/pi05/modeling_pi05.py',source/'lerobot/policies/pi05/configuration_pi05.py',
        source/'lerobot/policies/pi_gemma.py']
    return {str(p):sha(p) for p in sorted(paths)}

def expected_keys(tasks):
    values=[(int(t['task_index']),t['task'],int(seed)) for t in tasks for seed in t['seeds']]
    if len(tasks)!=10 or len(values)!=60 or len(set(values))!=60 or any(len(t['seeds'])!=6 for t in tasks):
        raise ValueError('Exactly ten tasks and six unique seeds/task required')
    return set(values)

def verify_panel(job,bank):
    expected=expected_keys(job['tasks'])
    actual={(int(e['task_index']),e['task'],int(e['seed'])) for e in bank['entries'] if e['group']==job['group']}
    if actual!=expected or bank['repeat']!=job['repeat']:raise ValueError('Reset bank/task panel mismatch')
    if any(e['action_mode']!='joint' or e['task_config']!='demo_clean' for e in bank['entries']):raise ValueError('Reset protocol changed')
    namespace='iclr2027-table5-robotwin-formal-reset-v2'
    for t in job['tasks']:
        seeds=[]
        for ep in range(6):
            x=int(hashlib.sha256(f'{namespace}\0{job["repeat"]}\0{t["task_index"]}\0{ep}'.encode()).hexdigest()[:8],16)%2**31
            if (job['group'],job['repeat'],t['task_index'],ep)==('receptacle',2,27,0):x=877886121
            seeds.append(x)
        if t['seeds']!=seeds:raise ValueError('Frozen seed derivation or explicit amendment differs')
    return expected

def runtime_for(parent,job,deployment):
    """Exactly the successful dense queue's complete-job native call arguments."""
    value=copy.deepcopy(parent);source={int(t['task_index']):t for t in parent['tasks']}
    value['checkpoint']=str(deployment);value['formal_result']=True
    value['purpose']='robotwin_regmeanpp_m3_formal_evaluation';value['plateau_eligible']=False
    formal={int(t['task_index']):t for t in job['tasks']}
    if set(source)!=set(formal) or any(source[k]['task']!=formal[k]['task'] for k in formal):raise ValueError('Parent/native task identity differs')
    value['tasks']=[dict(source[k],seeds=sorted(formal[k]['seeds'])) for k in sorted(formal)]
    return value

def reset_references(group,repeat,keys):
    top=PROTOCOL.parent/f'runs/{group}/repeat-{repeat:02d}/complete.json'
    done=read(top);verify_binding(done['receipt']);receipt=read(done['receipt']['path']);rows={};bindings=[bind(top),done['receipt']]
    for b in receipt['rows']:
        verify_binding(b);row=read(b['path']);key=(row['task_index'],row['task'],row['seed'])
        if key not in keys or key in rows:raise ValueError('Expert reset-reference coverage differs')
        value=row['initial_observation_sha256']
        if not isinstance(value,str) or len(value)!=64:raise ValueError('Missing expert reset fingerprint')
        rows[key]=value;bindings.append(b)
    if set(rows)!=keys:raise ValueError('Expert reset references incomplete')
    return {f'{i}:{seed}':v for (i,_,seed),v in rows.items()},bindings

def prepare(run):
    run=Path(run).resolve()
    if run.exists():raise FileExistsError('Fresh formal output root required')
    if sha(ACCEPTED)!=ACCEPTED_SHA or sha(PROTOCOL)!=PROTOCOL_SHA:raise ValueError('Accepted model/formal protocol identity changed')
    accepted=read(ACCEPTED)
    if accepted['status']!='PASS' or accepted['model_sha256']!=MODEL_SHA or accepted['saved_and_loaded_tensors_bitwise']!=813:
        raise ValueError('Native accepted model missing')
    if sha(CHECKPOINT/'model.safetensors')!=MODEL_SHA:raise ValueError('Model bytes changed before formal freeze')
    support={str(p):sha(p) for p in CHECKPOINT.rglob('*') if p.is_file() and p.name!='model.safetensors'}
    cfg=read(CHECKPOINT/'config.json')
    if cfg['use_peft'] or cfg['output_features']['action']['shape']!=[14]:raise ValueError('Expected accepted dense 14D RoboTwin deployment')
    protocol=read(PROTOCOL);frozen={str(PROTOCOL):PROTOCOL_SHA,str(ACCEPTED):ACCEPTED_SHA};jobs=[]
    for ref in protocol['jobs']:
        verify_binding(ref);job=read(ref['path']);verify_binding(job['reset_bank']);verify_binding(job['development_parent'])
        keys=verify_panel(job,read(job['reset_bank']['path']));parent=read(job['development_parent']['path'])
        for b in (ref,job['reset_bank'],job['development_parent']):frozen[b['path']]=b['sha256']
        reset_hashes,refs=reset_references(job['group'],job['repeat'],keys)
        for b in refs:frozen[b['path']]=b['sha256']
        horizon=Path(parent['robotwin']['root'])/'task_config/_eval_step_limit.yml';frozen[str(horizon)]=sha(horizon)
        import yaml
        horizons=yaml.safe_load(horizon.read_text())
        runtime=runtime_for(parent,job,run/'deployment')
        identifier=f'{job["group"]}-repeat-{job["repeat"]:02d}'
        jobs.append({'id':identifier,'group':job['group'],'repeat':job['repeat'],'episodes':60,'tasks':job['tasks'],
            'reset_bank':job['reset_bank'],'source_job':ref,'source_parent':job['development_parent'],
            'reference_reset_sha256':reset_hashes,'horizons':{t['task']:int(horizons[t['task']]) for t in job['tasks']},
            'runtime':runtime,'output':str(run/'jobs'/identifier/'attempt-01')})
    if {(j['group'],j['repeat']) for j in jobs}!={(g,r) for g in GROUPS for r in (1,2,3)}:raise ValueError('Require all nine jobs')
    run.mkdir();(run/'deployment').mkdir();(run/'deployment/pretrained_model').symlink_to(CHECKPOINT,target_is_directory=True)
    stat=(CHECKPOINT/'model.safetensors').stat()
    plan={'schema':'robotwin_regmeanpp_m3_formal_540_v1','status':'CPU_FROZEN_REQUIRES_FORMAL_PERMIT','created_at':now(),
        'run':str(run),'checkpoint':str(CHECKPOINT),'deployment':str(run/'deployment'),'model_sha256':MODEL_SHA,
        'model_file':{'path':str(CHECKPOINT/'model.safetensors'),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns},
        'model_accepted':bind(ACCEPTED),'method':METHOD,'groups':list(GROUPS),'repeats':[1,2,3],
        'jobs':jobs,'job_count':9,'episodes':540,'source_protocol_sha256':PROTOCOL_SHA,'full_native_horizon':True,
        'action_mode':'joint','condition':'demo_clean','model_sidecar_sha256':support,'frozen_files_sha256':frozen,
        'sources_sha256':sources(),'host':HOST,'allowed_gpus':GPUS,'python':str(PYTHON),
        'resource_gate':{'free_mib_minimum':40960,'used_mib_maximum':64,'utilization_maximum':0,'compute_pids':[],
            'allocator_fraction':.50,'leases':['host_uuid','host_directory_index','legacy_flat_host_index'],
            'continuous_pass_fds':True,'one_worker_per_gpu':True},
        'poll_seconds':15,'maximum_wait_seconds':172800,'single_use_permit_required':True,'maximum_attempts_per_job':1,
        'automatic_retry':False,'stop_new_jobs_after_any_failure':True,'retain_running_children_on_failure':True,
        'aggregation':'task macro within each 180-episode repeat; mean and sample std over three repeats',
        'gpu_used':False,'formal_jobs_started':0}
    write(run/'plan.json',plan);digest=sha(run/'plan.json')
    for job in jobs:write(run/'manifests'/f'{job["id"]}.json',{'schema':'robotwin_regmeanpp_m3_formal_job_v1','plan_sha256':digest,
        'method':METHOD,'formal_result':True,'model_sha256':MODEL_SHA,'model_accepted_sha256':ACCEPTED_SHA,**job})
    write(run/'PERMIT-TEMPLATE-NOT-AUTHORIZED.json',{'schema':'robotwin_regmeanpp_m3_formal_execution_permit_v1','allowed':False,
        'plan_sha256':digest,'run':str(run),'host':HOST,'model_sha256':MODEL_SHA,'model_accepted_sha256':ACCEPTED_SHA,
        'gpus':list(GPUS),'gpu_uuids':GPUS,'jobs':[j['id'] for j in jobs],'episodes':540,'maximum_attempts_per_job':1})
    write(run/'CPU-PREPARED.json',{'status':'PASS','plan_sha256':digest,'model_sha256':MODEL_SHA,
        'jobs':9,'episodes':540,'reset_fingerprints_frozen':540,'model_bytes_fully_hashed':stat.st_size,
        'deployment_is_symlink_to_accepted_checkpoint':True,'gpu_used':False})
    return plan

def validate(planfile,full_model=False):
    p=read(planfile)
    if Path(planfile).resolve()!=Path(p['run'])/'plan.json' or p['sources_sha256']!=sources():raise ValueError('Plan path/source drift')
    if p['model_sha256']!=MODEL_SHA or p['model_accepted']['sha256']!=ACCEPTED_SHA or p['job_count']!=9 or p['episodes']!=540:
        raise ValueError('Frozen model/panel identity changed')
    if p['automatic_retry'] or p['maximum_attempts_per_job']!=1 or p['python']!=str(PYTHON) or p['host']!=HOST:
        raise ValueError('Execution scope changed')
    if {int(k):v for k,v in p['allowed_gpus'].items()}!=GPUS:raise ValueError('GPU allowlist changed')
    if (Path(p['deployment'])/'pretrained_model').resolve()!=CHECKPOINT:raise ValueError('Deployment target changed')
    for path,digest in {**p['frozen_files_sha256'],**p['model_sidecar_sha256']}.items():
        if sha(path)!=digest:raise ValueError('Frozen model sidecar/native protocol/reference changed: '+path)
    stat=Path(p['model_file']['path']).stat()
    if stat.st_size!=p['model_file']['bytes'] or stat.st_mtime_ns!=p['model_file']['mtime_ns']:raise ValueError('Model stat changed')
    if full_model and sha(p['model_file']['path'])!=MODEL_SHA:raise ValueError('Model SHA mismatch')
    digest=sha(planfile)
    if len(p['jobs'])!=9 or {(j['group'],j['repeat']) for j in p['jobs']}!={(g,n) for g in GROUPS for n in (1,2,3)}:
        raise ValueError('Formal job coverage differs')
    for j in p['jobs']:
        source=read(j['source_job']['path']);bank=read(j['reset_bank']['path']);verify_panel(source,bank)
        if j['tasks']!=source['tasks'] or j['runtime']!=runtime_for(read(j['source_parent']['path']),source,Path(p['deployment'])):
            raise ValueError('Native runtime arguments changed')
        expected={'schema':'robotwin_regmeanpp_m3_formal_job_v1','plan_sha256':digest,'method':METHOD,'formal_result':True,
            'model_sha256':MODEL_SHA,'model_accepted_sha256':ACCEPTED_SHA,**j}
        if read(Path(p['run'])/'manifests'/f'{j["id"]}.json')!=expected:raise ValueError('Per-job manifest changed')
    return p,digest

def validate_permit(permit,p,digest):
    required={'schema':'robotwin_regmeanpp_m3_formal_execution_permit_v1','allowed':True,'plan_sha256':digest,
        'run':p['run'],'host':HOST,'model_sha256':MODEL_SHA,'model_accepted_sha256':ACCEPTED_SHA,
        'jobs':[j['id'] for j in p['jobs']],'episodes':540,'maximum_attempts_per_job':1}
    if any(permit.get(k)!=v for k,v in required.items()):raise ValueError('Formal permit scope differs')
    gpus=permit.get('gpus',[])
    if not gpus or len(gpus)!=len(set(gpus)) or any(type(g)!=int or g not in GPUS for g in gpus):raise ValueError('Invalid permitted GPU subset')
    if {int(k):v for k,v in permit.get('gpu_uuids',{}).items()}!={g:GPUS[g] for g in gpus}:raise ValueError('Permit GPU UUID mismatch')
    return gpus

def aggregate(receipts):
    if len(receipts)!=9 or {(r['group'],r['repeat']) for r in receipts}!={(g,n) for g in GROUPS for n in (1,2,3)}:
        raise ValueError('Exactly nine distinct accepted receipts required')
    if any(r['episodes']!=60 or r['model_sha256']!=MODEL_SHA or r['status']!='PASS' for r in receipts):raise ValueError('Incomplete or wrong-model receipt')
    rates=[sum(r['successes'] for r in receipts if r['repeat']==n)/180*100 for n in (1,2,3)]
    groups={g:[next(r['successes']/60*100 for r in receipts if r['group']==g and r['repeat']==n) for n in (1,2,3)] for g in GROUPS}
    return {'episodes':540,'successes':sum(r['successes'] for r in receipts),'overall':{'repeat_percent':rates,
        'mean_percent':statistics.mean(rates),'sample_std_percent':statistics.stdev(rates)},
        'groups':{g:{'repeat_percent':v,'mean_percent':statistics.mean(v),'sample_std_percent':statistics.stdev(v)} for g,v in groups.items()}}
