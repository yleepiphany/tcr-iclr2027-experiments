"""Table11: measure only the updated ordinary-RegMean alpha-.3 checkpoint.

Thin adapter around the existing C146 request/teacher/metric implementation.
No simulator success evaluation, teacher regeneration or threshold tuning.
GPU collection is explicit and was not run as part of release CPU smoke.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

MODEL_SHA='2d7acd682ec0d242eaa8726338125b65fea282787c508c60073135dcea11f54b'
MODEL_NAME='regmean_spectral_smoothing_a03'
MODEL_REL='vla-merge-runtime/experiments/static-observation-baselines-20260922/regmean_spectral_smoothing_a03/attempt-01/checkpoint'
CORE_REL='vla-merge/experiments/main-action-fidelity-20260921/run_candidate_actions.py'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def describe(root,run):
    model=root/MODEL_REL
    required=[root/CORE_REL,model/'model.safetensors',model/'block_regmeanpp_manifest.json',
        root/'vla-merge-runtime/experiments/main-action-fidelity-20260921/heldout-raw-attempt-02/bank-index.json',
        root/'coordination/2026-09-21/c146-teacher-actions-audit.json']
    missing=[str(p) for p in required if not p.is_file()]
    return {'status':'BLOCKED' if missing else 'READY_FOR_ORIGINAL_PROTOCOL_PREFLIGHT',
        'model_name':MODEL_NAME,'model_path':str(model),'model_sha256':MODEL_SHA,'run':str(run),
        'missing_assets':missing,'heldout_requests':400,'native_denoising_steps':10,'executed_action_steps':10,
        'continuous_dimensions':[0,1,2,3,4,5],'coordinate_scale':[1.]*6,
        'aggregation':'40-task macro, ten requests/task; report cosine coverage',
        'success_episodes':0,'gpu_validation_performed_by_release':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','prepare','collect','score'])
    parser.add_argument('--workspace-root',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--gpu',type=int);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();root=args.workspace_root.resolve();run=args.run.resolve()
    report=describe(root,run);print(json.dumps(report,indent=2))
    if args.action=='plan' or not args.execute:return
    if report['missing_assets']:raise SystemExit(2)
    core=load('release_candidate_core',root/CORE_REL)
    model=root/MODEL_REL;identity=model/'block_regmeanpp_manifest.json'
    construction=json.loads(identity.read_text())
    if construction['model_sha256']!=MODEL_SHA or construction['offdiag_scale']!=.3 or construction['replay_prefix']!='expert':
        raise ValueError('This is not the updated ordinary-RegMean alpha-.3 checkpoint')
    core.model_specs=lambda:{MODEL_NAME:{'path':str(model),'model_sha256':MODEL_SHA,
        'identity_source':str(identity),'group':'main'}}
    if args.action=='prepare':
        core.CANDIDATE_GPUS=() if args.gpu is None else (args.gpu,)
        core.MAX_ACTIVE=1
        core.prepare(run)
        return
    if args.action=='score':
        scorer=load('release_action_scorer',root/Path(CORE_REL).parent/'audit_and_score.py')
        scorer.RUN=run;scorer.OUTPUT=run/'strict-audit-and-metrics.json'
        scorer.main();return
    if args.gpu is None or not 0<=args.gpu<=7:raise ValueError('Collection requires one explicitly assigned physical GPU')
    if (run/'queue-ended.json').exists():raise FileExistsError('Preserve completed/failed measurement; choose a fresh run')
    plan=json.loads((run/'plan.json').read_text())
    if set(plan['models'])!={MODEL_NAME} or plan['models'][MODEL_NAME]['model_sha256']!=MODEL_SHA:raise ValueError('Candidate plan identity differs')
    rows={r['index']:r for r in core.gpu_rows()};row=rows[args.gpu]
    if row['free_mib']<40960:raise RuntimeError('Requires at least40GiB free; no process will be stopped')
    lease=core.card_flock.take_card(args.gpu,row['uuid'],'release-regmean-a03-actions','offline-native-forward')
    if lease is None:raise RuntimeError('GPU lease unavailable; no process will be stopped')
    try:
        command=[str(core.PYTHON),'-u',str(root/CORE_REL),'--worker','--model',MODEL_NAME,'--run',str(run)]
        with (run/'collection.log').open('x') as log:
            result=subprocess.run(command,cwd=root/'vla-merge',env=core.environment(args.gpu),stdout=log,stderr=subprocess.STDOUT)
        manifest=run/'actions'/MODEL_NAME/'manifest.json'
        valid=False
        if result.returncode==0 and manifest.exists():
            m=json.loads(manifest.read_text());valid=m['requests']==400 and m['model_sha256']==MODEL_SHA and m['determinism_first_request'] is True
        core.write(run/'queue-ended.json',{'complete':valid,'status':'complete' if valid else 'failed',
            'return_code':result.returncode,'models':[MODEL_NAME],'success_evaluation':False})
        if not valid:raise RuntimeError('Candidate collection failed/incomplete; no metrics published')
    finally:lease.release()

if __name__=='__main__':main()
