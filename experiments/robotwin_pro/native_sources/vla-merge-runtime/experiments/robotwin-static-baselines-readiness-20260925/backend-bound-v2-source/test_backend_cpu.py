import os,json,sys,subprocess,tempfile,unittest,types
from pathlib import Path
from unittest.mock import patch
import safe_successor_v2 as m

class Tests(unittest.TestCase):
 def test_construction_environment_overrides_inheritance(self):
  parent={'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':'0','NVIDIA_TF32_OVERRIDE':'0','kept':'yes'};env=m.build_environment(parent,4)
  self.assertEqual(env['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'],'1');self.assertNotIn('NVIDIA_TF32_OVERRIDE',env)
  self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'4');self.assertEqual(env['kept'],'yes');self.assertEqual(parent['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'],'0')
 def test_fresh_cpu_process_effective_tf32_and_negative_gates(self):
  script="""import json,torch,safe_successor_v2 as m
try:
 result=m.require_build_backend(torch);result['accepted']=True
except ValueError as error:
 result={'accepted':False,'error':str(error)}
result['cuda_initialized']=torch.cuda.is_initialized();print(json.dumps(result))
"""
  for override,nvidia,accept in [(None,None,False),('0',None,False),('1','0',False),('1',None,True)]:
   env={**os.environ,'CUDA_VISIBLE_DEVICES':''};env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',None);env.pop('NVIDIA_TF32_OVERRIDE',None)
   if override is not None:env['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']=override
   if nvidia is not None:env['NVIDIA_TF32_OVERRIDE']=nvidia
   result=json.loads(subprocess.check_output([sys.executable,'-c',script],cwd=Path(m.__file__).parent,env=env,text=True))
   self.assertEqual(result['accepted'],accept);self.assertFalse(result['cuda_initialized'])
   if accept:self.assertTrue(result['torch_cuda_matmul_allow_tf32'])
 def test_unbound_or_changed_native_stage_cannot_resume(self):
  fake=types.SimpleNamespace(__version__='synthetic-cpu-fixture',backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True))))
  with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':'1'}):
   os.environ.pop('NVIDIA_TF32_OVERRIDE',None);root=Path(temporary);artifact=root/'native-smoke.json';artifact.write_text('{"status":"synthetic"}')
   with self.assertRaises(FileNotFoundError):m.require_backend_stage(root,'smoke')
   m.record_backend_stage(root,'smoke',fake);self.assertEqual(m.require_backend_stage(root,'smoke')['core_plan_sha256'],m.EXPECTED_PLAN_SHA)
   artifact.write_text('{"status":"changed"}')
   with self.assertRaises(ValueError):m.require_backend_stage(root,'smoke')
 def test_disabled_torch_backend_cannot_be_certified(self):
  fake=types.SimpleNamespace(backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False))))
  with patch.dict(os.environ,{'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':'1'}):
   os.environ.pop('NVIDIA_TF32_OVERRIDE',None)
   with self.assertRaises(ValueError):m.require_build_backend(fake)
 def test_new_plan_same_bank_parameters_api_and_no_old_stages(self):
  w=Path('/mnt/workspace/Wilson/parameter-fusion');root,source,bank,run=m.paths(w)
  old=m.read(root/'featcal-m3-cpu-plan-v1/plan.json');new=m.read(run/'plan.json');b=m.read(run/'backend-contract.json')
  self.assertEqual(m.sha(run/'plan.json'),m.EXPECTED_PLAN_SHA);self.assertEqual(m.sha(run/'backend-contract.json'),m.EXPECTED_BACKEND_SHA)
  for key in ['bank','bank_sha256','groups','models','soup','base','implementations','observations','noise_states','physical_timestep','regression_rows','stages','linear_weights','alpha','rho','lambda','eps','solve_dtype','workspace_cap_bytes']:
   self.assertEqual(old[key],new[key],key)
  self.assertTrue(b['original_api_unchanged']);self.assertEqual(source,root/'source-v1')
  self.assertEqual(b['predecessor']['disposition'],'preserve backend-unbound artifact; prohibited as paper or formal-evaluation checkpoint')
  self.assertFalse((run/'native-smoke.json').exists());self.assertFalse((run/'complete.json').exists());self.assertFalse((run/'cache').exists())
  self.assertFalse(list(run.glob('teachers-*.json')))
if __name__=='__main__':unittest.main()
