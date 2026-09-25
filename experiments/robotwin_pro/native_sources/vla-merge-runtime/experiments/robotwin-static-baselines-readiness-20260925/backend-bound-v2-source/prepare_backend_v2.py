#!/usr/bin/env python3
"""CPU-only fresh core plan plus explicit Table1 TF32 construction contract."""
import argparse,json,os,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone
os.environ['CUDA_VISIBLE_DEVICES']=''
OLD_PLAN_SHA='f9fbc255cf37f5bec14fc2ba36d1b7c147b2d7205e6999f3a6a007df331c7eb5'
TABLE1_PLAN_SHA='89002c4ba14f26e1addccc2928c6a04d950e543f9ddc2cc51ae82430d69fb96d'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True);a=p.parse_args();w=a.workspace_root
 root=w/'vla-merge-runtime/experiments/robotwin-static-baselines-readiness-20260925';old=root/'featcal-m3-cpu-plan-v1';run=root/'featcal-m3-tf32-v2'
 if run.exists():raise FileExistsError('Fresh v2 output required; never modify v1 or reuse its stages')
 assert sha(old/'plan.json')==OLD_PLAN_SHA
 table1=w/'vla-merge-runtime/experiments/static-observation-baselines-20260922/featcal/plan.json'
 assert sha(table1)==TABLE1_PLAN_SHA and read(table1)['backend']=={'tf32_override':'1'}
 sys.path.insert(0,str(root/'source-v1'));from featcal_m3 import API
 import torch
 torch.set_num_threads(4);api=API(w,root/'static-bank-v1/bank.json');api.prepare(run)
 new=read(run/'plan.json');before=read(old/'plan.json')
 for key in ['groups','models','soup','base','bank','bank_sha256','observations','noise_states','physical_timestep','regression_rows','stages','linear_weights','unchanged_biases','alpha','rho','lambda','eps','solve_dtype','workspace_cap_bytes','implementations','deployment_files_sha256']:
  assert new[key]==before[key],('Unexpected method/input/source change',key)
 assert not torch.cuda.is_initialized()
 old_complete=read(old/'complete.json')
 contract={'schema':'robotwin_featcal_tf32_bound_construction_v2','created_at':datetime.now(timezone.utc).isoformat(),
  'core_plan':str(run/'plan.json'),'core_plan_sha256':sha(run/'plan.json'),
  'required_environment':{'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':'1','NVIDIA_TF32_OVERRIDE':None},
  'required_torch_cuda_matmul_allow_tf32':True,
  'applies_to':['smoke','teacher_capture','student_capture','solve','reload'],
  'original_table1_plan':{'path':str(table1),'sha256':TABLE1_PLAN_SHA,'backend':{'tf32_override':'1'}},
  'original_api_unchanged':True,'original_api':{'path':str(root/'source-v1/featcal_m3.py'),'sha256':sha(root/'source-v1/featcal_m3.py')},
  'bank_reuse':'same immutable observation/noise inputs only; no v1 smoke, teacher, student, solved prefix, model or evaluation reused',
  'predecessor':{'plan':str(old/'plan.json'),'plan_sha256':OLD_PLAN_SHA,'complete':str(old/'complete.json'),
   'complete_sha256':sha(old/'complete.json'),'model_sha256':old_complete['checkpoint']['model_sha256'],
   'disposition':'preserve backend-unbound artifact; prohibited as paper or formal-evaluation checkpoint'},
  'no_score_based_selection':True,'formal_episodes':0,'gpu_started':False,
  'preparer_source_sha256':sha(__file__),'primary_backend_reference':'https://docs.pytorch.org/docs/2.7/cuda_environment_variables.html'}
 write(run/'backend-contract.json',contract);write(run/'BACKEND-SHA256.json',{'sha256':sha(run/'backend-contract.json')})
 print(json.dumps({'status':'CPU_PREPARED_GPU_NOT_STARTED','run':str(run),'core_plan_sha256':sha(run/'plan.json'),'backend_contract_sha256':sha(run/'backend-contract.json'),'original_api_unchanged':True,'cuda_initialized':False},indent=2))
if __name__=='__main__':main()
