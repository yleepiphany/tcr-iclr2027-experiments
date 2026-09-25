import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,copy,fcntl,json,tempfile,unittest
from pathlib import Path
import torch
import contract as c
import run_formal as runner
import worker
from receipt import audit_episode

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.tasks=[{'task_index':i,'task':f'task{i}','seeds':list(range(i*10,i*10+6))} for i in range(10)]
        self.job={'id':'coordination-repeat-01','tasks':self.tasks,'group':'coordination','repeat':1,
            'reference_reset_sha256':{f'{i}:{seed}':'a'*64 for i in range(10) for seed in range(i*10,i*10+6)},
            'horizons':{f'task{i}':10 for i in range(10)}}
        self.row={'task_index':0,'task':'task0','seed':0,'success':False,'steps':10,'horizon':10,'initial_observation_sha256':'a'*64}
        self.initial={'phase':'env_reset','seed':0,'episode_horizon':10,'registered_episode_horizon':10,
            'observation_sha256':'a'*64,'state_shape':[1,14],'camera_keys':['cam_high','cam_left_wrist','cam_right_wrist']}
        self.action={'phase':'step_action_ready','normalized_action':[-.5]*14,'postprocessed_action':[.5]*14,'expected_executed_action':[.5]*14}
    def tearDown(self):self.temp.cleanup()
    def audit(self,row=None,initial=None,action=None):
        return audit_episode(row or self.row,initial or self.initial,action or self.action,self.job,torch.zeros(14),torch.ones(14)*2)
    def test_01_native_runtime_matches_reference_dense_job_sorting(self):
        parent={'tasks':[dict(t,episode_ids=[100+i]) for i,t in enumerate(reversed(self.tasks))],
            'checkpoint':'old','runtime_environment':{'A':'B'},'robotwin':{'root':'/sim'},'lerobot':{'root':'/code'}}
        runtime=c.runtime_for(parent,self.job,Path('/new/deployment'))
        self.assertEqual(runtime['checkpoint'],'/new/deployment');self.assertFalse(runtime['plateau_eligible'])
        self.assertEqual([t['task_index'] for t in runtime['tasks']],list(range(10)))
        self.assertEqual(runtime['runtime_environment'],parent['runtime_environment'])
        self.assertEqual(parent['checkpoint'],'old')
    def test_02_panel_duplicates_rejected(self):
        tasks=copy.deepcopy(self.tasks);tasks[0]['seeds'][1]=tasks[0]['seeds'][0]
        with self.assertRaises(ValueError):c.expected_keys(tasks)
    def test_03_real_frozen_amended_reset_panel(self):
        protocol=c.read(c.PROTOCOL);total=0
        for ref in protocol['jobs']:
            c.verify_binding(ref);job=c.read(ref['path']);bank=c.read(job['reset_bank']['path'])
            total+=len(c.verify_panel(job,bank))
        self.assertEqual(total,540)
    def test_04_resource_empty_floor_and_allowlist(self):
        row={'gpu':6,'uuid':c.GPUS[6],'free_mib':81153,'used_mib':1,'utilization':0,'compute_pids':[]}
        self.assertTrue(runner.admissible(row))
        for patch in ({'free_mib':40959},{'used_mib':65},{'utilization':1},{'compute_pids':[42]},{'uuid':'wrong'},{'gpu':0}):
            self.assertFalse(runner.admissible({**row,**patch}))
    def test_05_three_lock_names_include_both_legacy_schemes(self):
        paths=worker.lease_paths(6)
        self.assertEqual(len(paths),3)
        self.assertIn(str(c.RUNTIME/'resource-leases'/c.HOST/'gpu-6.lock'),paths)
        self.assertIn(str(c.RUNTIME/'resource-leases'/f'{c.HOST}-gpu-6.lock'),paths)
        self.assertTrue(any(c.GPUS[6] in p for p in paths))
    def test_06_native_row_and_quantile_action_pass(self):
        key,error=self.audit();self.assertEqual(key,(0,'task0',0));self.assertEqual(error,0.)
    def test_07_no_truncated_failures_or_nonbool_success(self):
        for patch in ({'steps':9},{'steps':0},{'success':0},{'horizon':11}):
            with self.assertRaises(ValueError):self.audit(row={**self.row,**patch})
    def test_08_wrong_reset_identity_rejected(self):
        for patch in ({'seed':1},{'observation_sha256':'b'*64},{'state_shape':[1,32]},{'camera_keys':['cam_high']}):
            with self.assertRaises(ValueError):self.audit(initial={**self.initial,**patch})
    def test_09_wrong_action_scale_or_nonfinite_rejected(self):
        for patch in ({'postprocessed_action':[.6]*14},{'normalized_action':[float('nan')]*14},{'expected_executed_action':[0.]*13}):
            with self.assertRaises(ValueError):self.audit(action={**self.action,**patch})
    def test_10_no_automatic_second_attempt(self):
        output=self.root/'attempt-01';jobroot=self.root/'job';jobroot.mkdir()
        worker.no_previous_attempt(jobroot,output)
        (jobroot/'FAILED.json').write_text('{}')
        with self.assertRaises(FileExistsError):worker.no_previous_attempt(jobroot,output)
    def test_11_permit_template_and_exact_panel_identity(self):
        jobs=[{'id':f'{g}-{r}'} for g in c.GROUPS for r in (1,2,3)];plan={'run':'/frozen','jobs':jobs}
        permit={'schema':'robotwin_regmeanpp_m3_formal_execution_permit_v1','allowed':True,'plan_sha256':'a'*64,
            'run':'/frozen','host':c.HOST,'model_sha256':c.MODEL_SHA,'model_accepted_sha256':c.ACCEPTED_SHA,
            'jobs':[j['id'] for j in jobs],'episodes':540,'maximum_attempts_per_job':1,'gpus':[6],'gpu_uuids':{'6':c.GPUS[6]}}
        self.assertEqual(c.validate_permit(permit,plan,'a'*64),[6])
        for patch in ({'allowed':False},{'model_sha256':'wrong'},{'gpus':[0]},{'episodes':60},{'maximum_attempts_per_job':2},{'jobs':[]}):
            with self.assertRaises(ValueError):c.validate_permit({**permit,**patch},plan,'a'*64)
    def test_12_aggregate_three_repeat_mean_and_sample_std(self):
        rows=[{'status':'PASS','group':g,'repeat':r,'episodes':60,'successes':r*6,'model_sha256':c.MODEL_SHA} for g in c.GROUPS for r in (1,2,3)]
        result=c.aggregate(rows);self.assertEqual(result['overall']['repeat_percent'],[10.,20.,30.])
        self.assertEqual(result['overall']['mean_percent'],20.);self.assertEqual(result['overall']['sample_std_percent'],10.)
        with self.assertRaises(ValueError):c.aggregate(rows[:-1])
    def test_13_real_flock_handoff_helper_does_not_unlock(self):
        locks=[]
        class Lock:
            def __init__(self,path):
                self.path=path;self.handle=os.open(path,os.O_CREAT|os.O_RDWR,0o600);self._released=False
                fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            def release(self):raise AssertionError('Should close descriptors, never unlock inherited OFDs')
        class Flocks:
            def take(_self,*a,**kw):
                lock=Lock(self.root/f'lock-{len(locks)}');locks.append(lock);return lock
            take_card=take;_try_lock=take
        held=runner.take_leases(Flocks(),6)
        self.assertEqual(len(held),3)
        c.leases.verify_inherited_leases(c.leases.lease_records(held),[str(x.path) for x in held])
        c.leases.close_own(held)
        self.assertTrue(all(x._released for x in held))
    def test_14_native_call_uses_full_unsmoked_engine(self):
        tree=ast.parse((Path(c.__file__).parent/'worker.py').read_text())
        calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='simulator_audit']
        self.assertEqual(len(calls),1);self.assertIs(calls[0].args[-1].value,False)
        self.assertFalse(any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr in ('kill','terminate','send_signal') for x in ast.walk(tree)))
    def test_15_cuda_not_initialized(self):self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'gpu_used':False,'gpu_queries':False,'native_jobs_started':0,'cuda_initialized':torch.cuda.is_initialized(),'signals_sent':0}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
