import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
torch.set_num_threads(2)
import fig4_regmeanpp_contract as contract
from prepare_regmeanpp_subset import OUTPUT,source_parity

SOURCE=Path('/mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/static-observation-baselines-20260922/regmeanpp_spectral_smoothing_v1')
sys.path.insert(0,str(SOURCE))
from spectral_smoothing import solve_smoothed_weight

class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.plans=[json.loads((OUTPUT/s/'plan.json').read_text()) for s in ('spatial-goal','spatial-object-goal')]
    def test_parent_identity_and_cpu_preflight(self):
        for p in self.plans:contract.validate_adapter_plan(p)
        self.assertTrue(source_parity(contract.frozen_figure())['all_nested_numerical_and_capture_functions_unchanged'])
    def test_expert_prefix_rejected(self):
        p=copy.deepcopy(self.plans[0]);p['recipe']['replay_prefix']='expert'
        with self.assertRaisesRegex(ValueError,'recipe'):contract.validate_adapter_plan(p)
    def test_excluded_expert_source_rejected(self):
        p=copy.deepcopy(self.plans[0]);p['inputs']['long']=p['inputs']['goal']
        with self.assertRaisesRegex(ValueError,'Excluded'):contract.validate_adapter_plan(p)
    def test_budget_change_rejected(self):
        p=copy.deepcopy(self.plans[1]);p['expected_realized_rows']=2368400
        with self.assertRaisesRegex(ValueError,'budget'):contract.validate_adapter_plan(p)
    def test_selected_dense_bank_projection(self):
        for p in self.plans:
            contract.ACTIVE=p
            experts={n:Path(p['inputs'][n]['adapter_path']) for n in p['experts']}
            result=contract.validate_dense_bank(p['dense_bank']['path'],experts,False)
            self.assertEqual(list(result),p['experts']);self.assertNotIn('long',result)
        contract.ACTIVE=None
    def test_real_schema_row_counts_and_contamination(self):
        names=self.plans[0]['experts']
        template={'strict_original_equivalence':False,'spectral_regularization_applied':True,
          'offdiag_scale':.3,'solve_dtype':'float64/complex128','filter':'smooth',
          'tau_rule':'input_width * eps(float32) * deterministic_power_lambda_max_estimate',
          'soup_centered_ridge':False,'correction_cap':False,
          'smooth_normal_equation_relative_residual':1e-13,'original_centered_equation_relative_residual':.01,
          'expert_objective_weights':{n:.5 for n in names}}
        metrics={}
        for i in range(254):metrics[f'other.{i}']={**template,'rows_by_expert':dict.fromkeys(names,800)}
        for i in range(162):metrics[f'model.vision_tower.layer{i}']={**template,'rows_by_expert':dict.fromkeys(names,2400)}
        for name in ('model.time_mlp_in','model.time_mlp_out'):metrics[name]={**template,'rows_by_expert':dict.fromkeys(names,50)}
        self.assertEqual(contract.validate_rows_for_names(metrics,names),1184200)
        metrics['other.0']['rows_by_expert']['long']=800
        with self.assertRaisesRegex(ValueError,'contamination'):contract.validate_rows_for_names(metrics,names)
    def test_two_three_expert_kernel_mean_and_no_excluded_weights(self):
        generator=torch.Generator().manual_seed(25314)
        pool={n:torch.randn(3,5,generator=generator) for n in ('spatial','object','goal','long')}
        common_x=torch.randn(8,5,generator=generator)
        for p in self.plans:
            names=p['experts'];xs={n:common_x for n in names};ws={n:pool[n] for n in names}
            a,info=solve_smoothed_weight(xs,ws,device='cpu')
            expected=(sum(w.double() for w in ws.values())/len(names)).float()
            torch.testing.assert_close(a,expected,rtol=1e-6,atol=1e-7)
            pool['long']=torch.full((3,5),1e8)
            b,_=solve_smoothed_weight(xs,{n:pool[n] for n in names},device='cpu')
            self.assertTrue(torch.equal(a,b));self.assertEqual(set(info['rows_by_expert']),set(names))
    def test_missing_execution_permit_blocks_before_gpu(self):
        p=self.plans[0];path=OUTPUT/p['subset']/'plan.json'
        args=SimpleNamespace(**p['recipe'],experiment_manifest=path,expected_experiment_sha256=contract.digest(path),
             expert=[n+'='+p['inputs'][n]['adapter_path'] for n in p['experts']],
             calibration=[n+'='+p['inputs'][n]['replay_path'] for n in p['experts']],
             manifest=[n+'='+p['inputs'][n]['manifest_path'] for n in p['experts']],
             output=p['output'],base_model=p['base_model'],dense_expert_bank=p['dense_bank']['path'])
        with patch.object(contract,'validate_adapter_plan',return_value=p),patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'no independent Figure 4 execution permit'):contract.validate_recipe(args)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SubsetTests))
    print(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
                      'cuda_initialized':torch.cuda.is_initialized(),'gpu_jobs_launched':0}))
    raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
