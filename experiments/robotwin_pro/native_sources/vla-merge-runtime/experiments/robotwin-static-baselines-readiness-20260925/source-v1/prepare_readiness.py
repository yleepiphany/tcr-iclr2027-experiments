#!/usr/bin/env python3
"""CPU-only source/selection plan for RoboTwin static RegMean++ and FeatCal.

This does not build a checkpoint or claim a GPU-ready collector. Output explicitly
lists the missing M=3 adapters and processed observation bank. Existing assets are
read only; --output must be new.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

GROUPS=('coordination','receptacle','precision')

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def select_observations(split, formal_tasks):
    """One pre-existing calibration episode per task; five deterministic frames."""
    tasks={(r['group_id'],int(r['task_index'])):r for r in split['tasks']}
    observations=[]
    for group in GROUPS:
        expected=formal_tasks[group]
        if len(expected)!=10 or len({x['task_index'] for x in expected})!=10:
            raise ValueError('Require ten distinct frozen tasks per expert')
        for task_slot, task in enumerate(expected):
            row=tasks[(group,int(task['task_index']))]
            if row['task']!=task['task']:raise ValueError('Split/formal task identity mismatch')
            selected=min(row['splits']['calibration'],key=lambda x:(x['hash_rank'],x['converted_episode']))
            if selected['hash_rank'] not in range(45,50):raise ValueError('Expected calibration split ranks 45..49')
            episode=selected['converted_episode']
            if any(episode==x['converted_episode'] for part in ('train','development') for x in row['splits'][part]):
                raise ValueError('Calibration/train/development episode overlap')
            # Match the existing 50-action-context sampling convention without
            # reading any action values or success labels.
            end=int(selected['frames'])-50
            if end<4:raise ValueError('Episode too short for five valid starts')
            frames=[i*end//4 for i in range(5)]
            if len(set(frames))!=5:raise ValueError('Nonunique frame selection')
            for request,frame in enumerate(frames):
                observations.append({'group':group,'task_slot':task_slot,'task_index':task['task_index'],
                    'task':task['task'],'episode_index':episode,'frame_index':frame,'request':request,
                    'split_hash':selected['split_hash'],'source_hdf5_sha256':selected['source_hdf5_sha256'],
                    'trajectory_signature_sha256':selected['trajectory_signature_sha256']})
    if len(observations)!=150:raise ValueError('Expected 150 observations')
    return observations

def noise_id(group,task_slot,request,replica):
    if group not in GROUPS or task_slot not in range(10) or request not in range(5) or replica not in range(3):
        raise ValueError('Bad noise identity')
    return 202609250000+GROUPS.index(group)*100000+task_slot*1000+request*10+replica

def build(work):
    runtime=work/'vla-merge-runtime';experiments=runtime/'experiments'
    table5=experiments/'iclr2027-table5-20260910'
    bank_path=experiments/'claude-robotwin-tcr-20260923/dense-experts-v2/expert-dense-bank.json'
    bank=read(bank_path)
    if bank['status']!='passed_parameter_exact' or bank['expert_order']!=list(GROUPS):raise ValueError('Dense bank not accepted M=3')
    split_path=table5/'dataset-splits/robotwin-clean50-content-hash-40-5-5-v1.json'
    split=read(split_path)
    formal_path=table5/'evaluation-queues/robotwin-three-expert-formal-v2/protocol.json'
    formal=read(formal_path)
    tasks={}
    for ref in formal['jobs']:
        job=read(ref['path'])
        if sha(ref['path'])!=ref['sha256']:raise ValueError('Frozen formal job drift')
        if job['repeat']==1:tasks[job['group']]=job['tasks']
    if set(tasks)!=set(GROUPS) or formal['episodes_total']!=540:raise ValueError('Formal task protocol differs')
    observations=select_observations(split,tasks)
    from safetensors import safe_open
    import numpy as np
    configs={};normalizers={};models=[];files=[bank_path,split_path,formal_path]
    for expert in bank['experts']:
        name=expert['name'];path=Path(expert['dense_checkpoint']['path']);config=read(path/'config.json')
        if config['max_action_dim']!=32 or config['chunk_size']!=50 or config['output_features']['action']['shape']!=[14]:
            raise ValueError('RoboTwin PI0.5 action interface differs')
        if config['use_visual_memory'] or config['use_proprioceptive_memory']:raise ValueError('Unexpected memory model')
        configs[name]={k:v for k,v in config.items() if k!='pretrained_path'}
        entry={**expert,'actual_bytes':(path/'model.safetensors').stat().st_size}
        with safe_open(str(path/'model.safetensors'),framework='numpy') as f:
            entry['actual_tensor_count']=len(f.keys())
        if entry['actual_tensor_count']!=813:raise ValueError('Model tensor schema differs')
        normalizers[name]={}
        for filename in ['policy_preprocessor_step_3_normalizer_processor.safetensors','policy_postprocessor_step_0_unnormalizer_processor.safetensors']:
            with safe_open(str(path/filename),framework='numpy') as f:
                normalizers[name][filename]={k:f.get_tensor(k).reshape(-1) for k in f.keys()}
            files.append(path/filename)
        files.extend([path/'config.json',Path(expert['dense_checkpoint']['manifest']['path']),Path(expert['dense_checkpoint']['verification']['path'])])
        models.append(entry)
    for name in GROUPS[1:]:
        if configs[name]!=configs[GROUPS[0]]:raise ValueError('Expert configs differ beyond pretrained path')
        for filename,vals in normalizers[name].items():
            baseline=normalizers[GROUPS[0]][filename]
            if set(vals)!=set(baseline) or any(not np.array_equal(vals[k],baseline[k]) for k in vals):
                raise ValueError('Semantic normalizer mismatch')
    soup=experiments/'claude-robotwin-tcr-20260923/dense-soup-v1/pretrained_model'
    soup_manifest=soup/'dense_equivalent_manifest.json';files.append(soup_manifest)
    regmean=experiments/'static-observation-baselines-20260922/regmeanpp_spectral_smoothing/attempt-01/plan.json'
    featcal=experiments/'static-observation-baselines-20260922/featcal/plan.json'
    files.extend([regmean,featcal]);reg=read(regmean);feat=read(featcal)
    dataset=runtime/'datasets/robotwin_unified-1287871839fa-clean50-alltasks-train40-quantiles-chw-v4'
    for leaf in ['meta/info.json','meta/stats.json','meta/tasks.parquet']:
        if not (dataset/leaf).is_file():raise ValueError('Dataset metadata missing')
        files.append(dataset/leaf)
    if not (dataset/'data').is_dir() or not (dataset/'videos').is_dir():raise ValueError('Dataset payload links missing')
    candidates=[]
    for row in observations:
        for replica in range(3):candidates.append({**row,'noise_replica':replica,'noise_seed':noise_id(row['group'],row['task_slot'],row['request'],replica),'timestep':1.0})
    if len({r['noise_seed'] for r in candidates})!=450:raise ValueError('Duplicate noise seeds')
    return {'schema':'robotwin_static_baseline_cpu_readiness_v1','created_at':datetime.now(timezone.utc).isoformat(),
        'status':'CPU_INPUT_SELECTION_AND_ASSET_COMPATIBILITY_PASSED_GPU_BLOCKED',
        'paper_rows_required':['Table5/RoboTwin RegMean++','Table5/RoboTwin FeatCal','Table16 group breakdown'],
        'groups':list(GROUPS),'models':models,'base':bank['base'],'soup':{'path':str(soup),'manifest':read(soup_manifest)},
        'soup_numeric_semantics':'existing Table5 uniform task-delta Soup, rank192 then PEFT-safe dense materialization; not newly asserted bitwise equality to mean of pre-rounded dense experts',
        'normalizer_semantic_parity':True,'model_full_hashes_recomputed':False,
        'model_metadata_checked':True,'dataset_root':str(dataset),'calibration_split':'pre-existing hash ranks45..49; lowest rank per formal task',
        'observation_count':150,'processed_observation_tensors_ready':False,'observations':observations,
        'noise_candidates':candidates,'demonstration_action_values_read':False,'success_values_read':False,
        'regmeanpp':{'reference_plan':str(regmean),'native_calls':150,'noise_replicas_per_observation':1,
                     'recipe':reg['recipe'],'spectral_smoothing':reg['spectral_smoothing'],'bias_rule':reg['bias_rule'],
                     'expected_rows_if_same_per_call_quota':1776300,'is_unmodified_official_solver':False},
        'featcal':{'reference_plan':str(featcal),'native_calls':450,'noise_replicas_per_observation':3,
                   'alpha':feat['alpha'],'rho':feat['rho'],'lambda':feat['lambda'],'covariance_eps':feat['covariance_eps'],
                   'linear_weights':418,'forward_stages':47,'bias_policy':feat['bias_policy'],
                   'expected_rows_if_same_per_call_quota':3330900},
        'evaluation':{'frozen_protocol':str(formal_path),'jobs':9,'episodes':540,'repeats':3,'trials_per_task_repeat':6},
        'frozen_files':{str(p):sha(p) for p in files},'gpu_used':False,'training':False,
        'blocking_gates':['Process and freeze the 150 raw observation images/state/instructions and tensor hashes with native RoboTwin preprocessing; retain no action labels.',
          'Implement and test an M=3 static-input collector instead of reusing TCR native rollout/late-flow or cached-prefix features.',
          'Generalize the LIBERO-only 4-expert FeatCal shard/forward-order driver without altering its exact solver or budgets.',
          'Bridge M=3 static features to the disclosed RegMean++ alpha=.3 spectral-smoothing extension; do not replace it with the RoboTwin TCR ridge/cap solver.',
          'Verify all model bytes and retained deployment sidecars, perform native one-request/schema parity, then build/reload checks before freezing the 540-episode method-specific evaluation.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('New isolated output required')
    report=build(args.workspace_root.resolve());args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:report[k] for k in ['status','observation_count','normalizer_semantic_parity','processed_observation_tensors_ready','gpu_used']}))

if __name__=='__main__':main()
