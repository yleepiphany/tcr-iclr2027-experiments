"""CPU-only frozen-input and deployment contract for the RoboTwin M3 port."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
import torch
from safetensors import safe_open
from graph_regmeanpp_m3 import GROUPS,INTERFACES,KERNEL_SHA,TOTAL_ROWS,schedule

WORK=Path('/mnt/workspace/Wilson/parameter-fusion')
REPO=WORK/'vla-merge';RUNTIME=WORK/'vla-merge-runtime'
READY=RUNTIME/'experiments/robotwin-static-baselines-readiness-20260925'
KERNEL=REPO/'experiments/static-observation-baselines-20260922/regmeanpp_spectral_smoothing_v1/spectral_smoothing.py'
TABLE1=KERNEL.with_name('materialize_smoothed.py')
TABLE1_SHA='c393ef3bc65a6b881b29b77cdf77eb5c1cea18bf755ffcac9535ee3ad53708b6'
READY_SHA='b925cacea32778d91218fa0af315565a37ecefac2a5336ccdb57ca72131f8a03'
HOST='dsw-824375-57c745db88-n6tv9'
GPUS={1:'GPU-c18a3bf3-c47c-9a8a-75b0-5a0f37b33a65',4:'GPU-48baba1c-f0e7-93b7-4a30-b3c21d20e61a',
      5:'GPU-4ed28198-b742-ee3c-2acb-4183dc944c81',6:'GPU-da625da2-f504-7927-4025-907e876718aa'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while data:=f.read(16*1024**2):h.update(data)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def load_static():
    sys.path.insert(0,str(READY/'source-v1'))
    import static_bank
    return static_bank

def implementations():
    source=WORK/'pi05_lora_finetune_v2_20260826'
    paths=list(Path(__file__).parent.glob('*.py'))+[
        READY/'source-v1/static_bank.py',READY/'source-v1/prepare_readiness.py',KERNEL,TABLE1,
        REPO/'scripts/materialize_pi05_block_regmeanpp.py',REPO/'src/vla_merge/featcal_execution.py',
        REPO/'experiments/claude-firstpass-cause-20260920/card_flock.py',
        source/'lerobot/src/lerobot/policies/pi05/modeling_pi05.py',
        source/'lerobot/src/lerobot/policies/pi_gemma.py']
    return {str(p.resolve()):sha(p) for p in sorted(paths)}

def check_bank(bank_path):
    bank=read(bank_path);selection=read(bank['selection_plan'])
    if sha(bank['selection_plan'])!=READY_SHA or bank['selection_plan_sha256']!=READY_SHA:
        raise ValueError('Accepted static selection identity drift')
    if bank['schema']!='robotwin_static_observation_initial_noise_bank_v1':raise ValueError('Wrong static bank')
    if set(bank['groups'])!=set(GROUPS) or bank['observation_count']!=150:raise ValueError('Wrong M3 observation budget')
    if bank['method_views']['regmeanpp']!={'replicas':[0],'native_calls':150}:raise ValueError('RegMean++ input view changed')
    for field in ('action_labels_used','reward_or_success_fields_read','expert_rollouts_used','tcr_execution_features_used'):
        if bank[field] is not False:raise ValueError('Disallowed calibration source '+field)
    if bank['physical_timestep']!=1. or bank['latent_shape']!=[50,32]:raise ValueError('Static t=1 latent contract changed')
    static=load_static();counts={};files={str(Path(bank_path).resolve()):sha(bank_path),bank['selection_plan']:READY_SHA}
    for g in GROUPS:
        e=bank['groups'][g];file=Path(bank_path).parent/e['tensor_file']
        if e['observations']!=50 or e['noise_states']!=150 or sha(file)!=e['tensor_sha256']:
            raise ValueError('Static tensor identity/budget drift')
        files[str(file)]=e['tensor_sha256'];counts[g]=0
        for task in range(10):
            batch=static.load_batch(bank_path,g,task,replica=0,verify=False)
            static.validate_model_batch(batch)
            if batch['x_t'].shape[0]!=5:raise ValueError('Five observations/task required')
            counts[g]+=len(batch['x_t'])
        records=e['records']
        if {(r['task_slot'],r['request']) for r in records}!={(t,r) for t in range(10) for r in range(5)}:
            raise ValueError('Task/request coverage differs')
    return bank,selection,files,counts

def validate_support(models):
    refs={};reference_config=None;files={}
    normfiles=('policy_preprocessor_step_3_normalizer_processor.safetensors','policy_postprocessor_step_0_unnormalizer_processor.safetensors')
    for g in GROUPS:
        root=Path(models[g]['path']);config=read(root/'config.json')
        normalized={k:v for k,v in config.items() if k!='pretrained_path'}
        if reference_config is None:reference_config=normalized
        elif normalized!=reference_config:raise ValueError('Expert config mismatch beyond pretrained path')
        if config['max_action_dim']!=32 or config['chunk_size']!=50 or config['output_features']['action']['shape']!=[14]:
            raise ValueError('RoboTwin native action interface differs')
        if config['use_visual_memory'] or config['use_proprioceptive_memory']:raise ValueError('Unexpected temporal memory')
        for filename in normfiles:
            with safe_open(str(root/filename),framework='pt',device='cpu') as f:
                values={k:f.get_tensor(k).flatten() for k in f.keys()}
            if filename not in refs:refs[filename]=values
            elif set(values)!=set(refs[filename]) or any(not torch.equal(v,refs[filename][k]) for k,v in values.items()):
                raise ValueError('Semantic normalizer mismatch')
        for file in root.rglob('*'):
            if file.is_file() and file.name!='model.safetensors':files[str(file)]=sha(file)
    return files

def prepare(run,bank_path):
    run=Path(run).resolve()
    if run.exists():raise FileExistsError('Independent new plan directory required')
    if torch.cuda.is_initialized():raise RuntimeError('CPU preparation must not initialize CUDA')
    if sha(KERNEL)!=KERNEL_SHA or sha(TABLE1)!=TABLE1_SHA:raise ValueError('Table1 source changed')
    bank,selection,input_files,counts=check_bank(bank_path)
    models={x['name']:x['dense_checkpoint'] for x in selection['models']}
    if tuple(models)!=GROUPS:raise ValueError('Dense expert order differs')
    support=validate_support(models);model_files=[];shape_by_key={};all_keys=None
    targets={p+'.weight' for s in schedule() for p in s.targets}
    modified=targets|{p+'.bias' for p in INTERFACES}
    for label,entry in [('base',selection['base']),*models.items()]:
        file=Path(entry['path'])/'model.safetensors';before=file.stat();actual=sha(file);after=file.stat()
        if actual!=entry['model_sha256'] or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('Accepted model identity changed')
        print(json.dumps({'event':'model_hash_verified','model':label,'sha256':actual}),flush=True)
        model_files.append({'label':label,'path':str(file),'sha256':actual,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns})
        with safe_open(str(file),framework='pt',device='cpu') as f:
            keys=set(f.keys())
            if len(keys)!=813 or not modified<=keys:raise ValueError('Native tensor scope differs')
            shapes={k:f.get_slice(k).get_shape() for k in keys}
        if all_keys is None:all_keys=keys;shape_by_key=shapes
        elif keys!=all_keys or shapes!=shape_by_key:raise ValueError('Expert tensor shape mismatch')
    quotas=[]
    for s in schedule():
        for p in s.targets:
            shape=shape_by_key[p+'.weight']
            if len(shape)!=2:raise ValueError('Non-matrix adapted weight')
            quotas.append({'module':p,'stage':s.index,'weight_shape':shape,
                'rows_per_expert':50*s.repetitions*s.rows_per_observation,'experts':3})
    plan={'schema':'robotwin_regmeanpp_static_m3_native_graph_spectral_v1','status':'CPU_PREPARED_NATIVE_GPU_PARITY_REQUIRED',
        'created_at':datetime.now(timezone.utc).isoformat(),'groups':list(GROUPS),'bank':str(Path(bank_path).resolve()),
        'input_files_sha256':input_files,'observations_by_group':counts,'noise_replicas':[0],'physical_timestep':1.,
        'native_static_inputs':150,'effective_regression_rows':TOTAL_ROWS,'stages':[s.to_dict() for s in schedule()],
        'quotas':quotas,'target_weights':418,'modified_tensor_count':422,'tensor_count':813,
        'models':models,'base':selection['base'],'model_files':model_files,'model_full_bytes_verified':True,
        'deployment_files_sha256':support,'semantic_normalizers_verified':True,
        'normalizer_export_source':'coordination; semantic equality checked, binary shape/layout may differ',
        'recipe':selection['regmeanpp']['recipe'],'spectral_smoothing':selection['regmeanpp']['spectral_smoothing'],
        'initialization':'common base for unadapted tensors; arithmetic mean of three accepted dense experts for 422 adapted tensors',
        'prefix':'merged; all current-block expert weights applied together; all inputs captured before any solve commits',
        'native_graph':'fresh static raw inputs; full native prefix recomputation, no saved TCR/replay activation tensors',
        'sampler':'original per-observation linspace(round) max16; vision separately per camera; no new padding weighting',
        'implementations':implementations(),'gpu_smoke_passed':False,'formal_evaluation_authorized':False,
        'resource_gate':{'host':HOST,'gpu_uuids':GPUS,'minimum_free_mib':71680,'maximum_used_mib':64,
            'require_no_compute_pids':True,'utilization':0,'allocator_fraction':.70,'uuid_and_legacy_index_flocks':True,
            'single_use_explicit_execution_permit_required':True,'build_slot_required':True},
        'gpu_parity_gate':{'scope':'one static replica0 observation/group at merged mean initialization',
            'native_velocity_and_all_418_sampled_inputs':'bitwise equal to original model.forward with fixed synthetic flow inputs',
            'candidate_blocks':['vision0','language0','action0'],'candidate_capture':'early-stop vs full forward bitwise sampled rows',
            'no_rollouts':True,'formal_episodes':0},
        'claim':'Disclosed Table1 spectral extension, not unmodified official RegMean++; CPU checks do not establish native GPU parity or success.'}
    write(run/'plan.json',plan)
    write(run/'CPU-PREPARED.json',{'status':'PASS','plan_sha256':sha(run/'plan.json'),'observations':150,'regression_rows':TOTAL_ROWS,
        'matrix_shape_checks':418*4,'models_full_bytes_verified':4,'gpu_used':False,'cuda_initialized':torch.cuda.is_initialized()})
    return plan

def validate(run,*,full_models=True):
    run=Path(run);plan=read(run/'plan.json')
    if plan['implementations']!=implementations():raise ValueError('Frozen implementation changed')
    if plan['groups']!=list(GROUPS) or plan['effective_regression_rows']!=TOTAL_ROWS:raise ValueError('M3 recipe changed')
    for file,digest in {**plan['input_files_sha256'],**plan['deployment_files_sha256']}.items():
        if sha(file)!=digest:raise ValueError('Frozen input/support changed: '+file)
    for row in plan['model_files']:
        stat=Path(row['path']).stat()
        if stat.st_size!=row['bytes'] or stat.st_mtime_ns!=row['mtime_ns']:raise ValueError('Model stat changed')
        if full_models and sha(row['path'])!=row['sha256']:raise ValueError('Model SHA changed')
    return plan,sha(run/'plan.json')
