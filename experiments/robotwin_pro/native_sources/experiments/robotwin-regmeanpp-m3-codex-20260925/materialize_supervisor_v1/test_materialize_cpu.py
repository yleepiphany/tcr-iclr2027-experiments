import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,copy,json,sys,tempfile,unittest
from pathlib import Path
import supervisor as s
import child_guard
from accept_model import validate_manifest
sys.path.insert(0,str(s.ROOT))
from graph_regmeanpp_m3 import schedule
import torch

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.planfile=self.root/'plan.json';self.planfile.write_text('{}')
        self.permit={'schema':'robotwin_regmeanpp_m3_single_use_execution_permit_v1','allowed':True,'stage':'materialize',
            'plan_sha256':s.CORE_SHA,'run':str(s.CORE),'gpu':4,'gpu_uuid':s.GPUS[4],'host':s.HOST,
            'supervisor_plan_sha256':s.sha(self.planfile),'native_smoke_sha256':s.SMOKE_SHA}
        self.quotas=[{'module':p,'weight_shape':[2,2],'rows_per_expert':50*t.repetitions*t.rows_per_observation} for t in schedule() for p in t.targets]
        self.modelsha='f'*64
        self.manifest={'schema':'robotwin_regmeanpp_m3_native_graph_spectral_v1','plan_sha256':s.CORE_SHA,'model_sha256':self.modelsha,
            'groups':['coordination','receptacle','precision'],'total_regression_rows':1776300,'target_weights':418,'modified_tensors':422,'total_tensors':813,
            'replay_prefix':'merged','noise_replica':0,'physical_timestep':1.,'action_labels_used':False,'expert_rollouts_used':False,
            'strict_original_equivalence':False,'method':'original_gram_mean_centered_weak_spectrum_smoothing_v1','modules':{}}
        for q in self.quotas:
            self.manifest['modules'][q['module']]={'method':'original_gram_mean_centered_weak_spectrum_smoothing_v1','strict_original_equivalence':False,
                'filter':'smooth','offdiag_scale':.3,'solve_dtype':'float64/complex128','input_width':2,'output_width':2,
                'rows_by_expert':{g:q['rows_per_expert'] for g in ('coordination','receptacle','precision')},
                'teacher_output_targets_used':False,'success_based_selection':False,'final_weight_clipping':False,
                'nonzero_feature_energy_floor':False,'trust_cap':False,'smooth_normal_equation_relative_residual':1e-12,
                'original_centered_equation_relative_residual':2.5,'bias_semantics':'uniform_expert_mean_not_augmented_regression'}
    def tearDown(self):self.temp.cleanup()
    def check(self,m=None):return validate_manifest(m or self.manifest,{'quotas':self.quotas},self.modelsha)
    def test_01_complete_original_scope_rows_and_residuals(self):
        d=self.check();self.assertEqual(d['modules'],418);self.assertEqual(d['regression_rows'],1776300)
        self.assertEqual(d['max_original_centered_relative_residual_report_only'],2.5)
    def test_02_missing_module_rejected(self):
        m=copy.deepcopy(self.manifest);m['modules'].pop(next(iter(m['modules'])))
        with self.assertRaises(ValueError):self.check(m)
    def test_03_wrong_input_rows_rejected(self):
        m=copy.deepcopy(self.manifest);next(iter(m['modules'].values()))['rows_by_expert']['precision']-=1
        with self.assertRaises(ValueError):self.check(m)
    def test_04_core_model_recipe_identity_rejected(self):
        for patch in ({'model_sha256':'wrong'},{'plan_sha256':'wrong'},{'noise_replica':1},{'replay_prefix':'expert'},{'action_labels_used':True}):
            with self.assertRaises(ValueError):self.check({**self.manifest,**patch})
    def test_05_solver_drift_and_bad_smooth_residual_rejected(self):
        for patch in ({'offdiag_scale':.95},{'trust_cap':True},{'filter':'hard_rank'},{'smooth_normal_equation_relative_residual':1e-4},
                      {'smooth_normal_equation_relative_residual':float('nan')},{'relative_residual':0.}):
            m=copy.deepcopy(self.manifest);next(iter(m['modules'].values())).update(patch)
            with self.assertRaises(ValueError):self.check(m)
    def test_06_new_stage_permit_and_smoke_binding(self):
        s.validate_permit(self.permit,self.planfile,4)
        for patch in ({'stage':'smoke'},{'native_smoke_sha256':'wrong'},{'supervisor_plan_sha256':'wrong'},{'allowed':False}):
            with self.assertRaises(ValueError):s.validate_permit({**self.permit,**patch},self.planfile,4)
    def test_07_existing_build_artifacts_prevent_reentry(self):
        for index,name in enumerate(('materialize-PERMIT-CONSUMED.json','materialize-FAILED.json','BUILD-COMPLETE.json','build-progress.jsonl','checkpoint')):
            core=self.root/f'claimed-{index}';core.mkdir();(core/name).write_text('{}')
            with self.assertRaises(ValueError):child_guard.claim_once(core,{'pid':999})
        fresh=self.root/'fresh';fresh.mkdir();child_guard.claim_once(fresh,{'pid':123})
        with self.assertRaises(ValueError):child_guard.claim_once(fresh,{'pid':999})
        self.assertEqual(json.loads((fresh/'materialize-PERMIT-CONSUMED.json').read_text())['pid'],123)
    def test_08_runtime_floor_throttle_and_forced_check(self):
        calls=[];clock=[0.]
        def board(gpu):calls.append(gpu);return {'uuid':s.GPUS[gpu],'free_mib':20000}
        floor=child_guard.RuntimeFloor(board,4,s.GPUS[4],clock=lambda:clock[0]);floor();floor();self.assertEqual(len(calls),1)
        floor(force=True);self.assertEqual(len(calls),2);clock[0]=6.;floor();self.assertEqual(len(calls),3)
    def test_09_runtime_floor_failure_records_actual_value(self):
        floor=child_guard.RuntimeFloor(lambda g:{'uuid':s.GPUS[g],'free_mib':12287},4,s.GPUS[4])
        with self.assertRaises(RuntimeError):floor()
        self.assertEqual(floor.minimum_seen,12287)
    def test_10_runtime_uuid_change_rejected(self):
        floor=child_guard.RuntimeFloor(lambda g:{'uuid':'wrong','free_mib':81153},4,s.GPUS[4])
        with self.assertRaises(RuntimeError):floor()
    def test_11_formal_frozen_9_jobs_540_reset_amendment(self):
        panel=s.freeze_formal_protocol();self.assertEqual(len(panel['jobs']),9);self.assertEqual(panel['episodes'],540)
        self.assertIsNone(panel['model_sha256']);self.assertFalse(panel['autolaunch'])
        job=next(j for j in panel['jobs'] if j['group']=='receptacle' and j['repeat']==2)
        task=next(t for t in job['tasks'] if t['task_index']==27)
        self.assertIn(877886121,task['seeds']);self.assertNotIn(437698954,task['seeds'])
    def test_12_original_materialize_entrypoint_only_and_no_signals(self):
        tree=ast.parse((Path(s.__file__).parent/'child_guard.py').read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertEqual(calls.count('materialize'),1);self.assertNotIn('smoke',calls)
        self.assertFalse(set(calls)&{'kill','terminate','send_signal'})
    def test_13_cuda_uninitialized(self):self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'gpu_used':False,'gpu_queries':False,
        'cuda_initialized':torch.cuda.is_initialized(),'native_children_launched':0,'signals_sent':0}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
