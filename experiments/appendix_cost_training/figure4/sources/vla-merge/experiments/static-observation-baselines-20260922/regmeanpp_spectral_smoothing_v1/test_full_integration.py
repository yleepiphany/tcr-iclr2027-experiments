"""CPU-only full-adapter tests: metadata honesty, mean bias and unchanged replay."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest
import torch
torch.set_num_threads(4)
from smoothing_adapter import solve_smoothing_adapter
from smoothing_contract import NAMES,SCHEMA,SMOOTHING,validate_realized_rows

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'regmeanpp_conservative_v1'

def functions(path):
    return {n.name:n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef)}

class IntegrationTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(77)
    def fixture(self):
        return ({n:torch.randn(6,7) for n in NAMES},{n:torch.randn(3,7) for n in NAMES})
    def test_adapter_separates_equation_residuals_and_does_not_select_from_metric(self):
        xs,ws=self.fixture();a,m=solve_smoothing_adapter(xs,ws,torch.zeros(3,7),offdiag_scale=.3)
        b,_=solve_smoothing_adapter(xs,ws,torch.randn(3,7)*100,offdiag_scale=.3)
        self.assertTrue(torch.equal(a,b));self.assertNotIn('relative_residual',m)
        self.assertIn('original_centered_equation_relative_residual',m)
        self.assertLess(m['smooth_normal_equation_relative_residual'],1e-7)
        self.assertFalse(m['strict_original_equivalence']);self.assertTrue(m['spectral_regularization_applied'])
        self.assertFalse(m['soup_centered_ridge']);self.assertFalse(m['correction_cap'])
    def test_capture_and_replay_helpers_ast_unchanged(self):
        old=functions(OLD/'materialize_conservative.py');new=functions(HERE/'materialize_smoothed.py')
        changed=[]
        for name in set(old)&set(new):
            if ast.dump(old[name],include_attributes=False)!=ast.dump(new[name],include_attributes=False):changed.append(name)
        self.assertEqual(set(changed),{'main','solve_spec','solve_dense_module','load_replay_states'})
        for name in ['advance_vision','advance_action','capture_inputs','assign_source','restore_merged','grouped_state_rows']:
            self.assertEqual(ast.dump(old[name],include_attributes=False),ast.dump(new[name],include_attributes=False))
    def test_actual_dense_bias_function_remains_mean(self):
        xs,ws=self.fixture();bias={n:torch.full((3,),float(i)) for i,n in enumerate(NAMES)}
        fn=copy.deepcopy(functions(HERE/'materialize_smoothed.py')['solve_dense_module'])
        env={'torch':torch,'Any':object,'dense_sources':{'action_in_proj':('unused',ws,bias,torch.zeros(3,7),torch.zeros(3))},
             'names':list(NAMES),'device':torch.device('cpu'),'args':type('Args',(),{'offdiag_scale':.3})(),
             'solve_smoothing_adapter':solve_smoothing_adapter}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual_dense_bias>','exec'),env)
        weight,merged_bias,m=env['solve_dense_module']('action_in_proj',{n:{n:xs[n]} for n in NAMES})
        self.assertTrue(torch.equal(merged_bias,torch.full((3,),1.5)));self.assertEqual(weight.shape,(3,7))
        self.assertTrue(m['spectral_regularization_applied']);self.assertEqual(m['bias_semantics'],'uniform_expert_mean_not_augmented_regression')
    def test_full418_contract_accepts_only_honest_smoothing_metadata(self):
        xs,ws=self.fixture();_,m=solve_smoothing_adapter(xs,ws,torch.zeros(3,7),offdiag_scale=.3)
        metrics={}
        for name in ['model.time_mlp_in','model.time_mlp_out']:
            metrics[name]={**m,'rows_by_expert':{n:50 for n in NAMES}}
        for i in range(162):metrics[f'model.vision_tower.module_{i}']={**m,'rows_by_expert':{n:2400 for n in NAMES}}
        for i in range(254):metrics[f'model.other_{i}']={**m,'rows_by_expert':{n:800 for n in NAMES}}
        self.assertEqual(validate_realized_rows(metrics),2368400)
        metrics['model.time_mlp_in']={**metrics['model.time_mlp_in'],'relative_residual':0.}
        with self.assertRaises(ValueError):validate_realized_rows(metrics)
    def test_distinct_output_and_real_layer_acceptance_gate(self):
        code=(HERE/'run_build.py').read_text()
        self.assertIn('regmeanpp_spectral_smoothing/attempt-01',code)
        self.assertIn('REAL-FIRST-LAYER-ACCEPTANCE.json',code)
        self.assertIn('spectral_smoothing_sha256',code)
        self.assertFalse(SMOOTHING['success_based_selection'])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IntegrationTests))
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'cuda_initialized':torch.cuda.is_initialized()}
    (HERE/'integration-test-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
