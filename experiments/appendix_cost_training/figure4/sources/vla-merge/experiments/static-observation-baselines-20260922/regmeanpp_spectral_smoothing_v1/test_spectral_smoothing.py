"""CPU-only invariants and independent equation oracles for the frozen smoother."""
import json
from pathlib import Path
import sys
import unittest

import torch
torch.set_num_threads(4)
from spectral_smoothing import prepare_original_equation,solve_smoothed_weight


class SmoothingTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(3407)
    def fixture(self,d=7,n=5,o=3,k=4):
        return ({str(i):torch.randn(n,d,dtype=torch.float64) for i in range(k)},
                {str(i):torch.randn(o,d,dtype=torch.float64) for i in range(k)})
    def test_dense_shift_woodbury_and_eigh_match(self):
        for d,n in [(7,5),(19,2)]:
            xs,ws=self.fixture(d,n)
            results=[solve_smoothed_weight(xs,ws,strategy=s)[0] for s in ('dense','woodbury','eigh_reference')]
            for value in results[1:]:self.assertTrue(torch.allclose(value,results[0],atol=2e-6,rtol=2e-6))
    def test_independent_regularized_normal_equation_oracle(self):
        xs,ws=self.fixture();actual,meta=solve_smoothed_weight(xs,ws,strategy='dense')
        diagonal,z,r,mean,_=prepare_original_equation(xs,ws);a=torch.diag(diagonal)+z.T@z;tau=meta['tau']
        independent=mean+torch.linalg.solve(a@a+tau*tau*torch.eye(a.shape[0],dtype=torch.float64),a@r).T
        self.assertTrue(torch.allclose(actual.double(),independent,atol=2e-6,rtol=2e-6))
        self.assertLess(meta['smooth_normal_equation_relative_residual'],1e-9)
    def test_weak_feature_exact_counterexample_is_smoothed_not_weight_clipped(self):
        eps=1e-7;xs={str(i):torch.tensor([[1.,eps if i<2 else -eps]],dtype=torch.float64) for i in range(4)}
        ws={str(i):torch.tensor([[1. if i<2 else -1.,0.]],dtype=torch.float64) for i in range(4)}
        smooth,meta=solve_smoothed_weight(xs,ws,strategy='woodbury')
        hard,_=solve_smoothed_weight(xs,ws,filter_kind='hard_rank')
        lam=4*eps*eps;analytic=3e6*lam*lam/(lam*lam+meta['tau']**2)
        self.assertAlmostEqual(float(smooth[0,1]),analytic,delta=analytic*1e-5)
        self.assertLess(float(smooth.norm()),1e-7);self.assertEqual(float(hard.norm()),0.)
        self.assertGreater(meta['original_centered_equation_relative_residual'],.99)
        self.assertLess(meta['smooth_normal_equation_relative_residual'],1e-9)
        self.assertFalse(meta['strict_original_equivalence']);self.assertFalse(meta['final_weight_clipping'])
    def test_well_conditioned_close_to_original_inverse(self):
        xs,ws=self.fixture(d=6,n=20)
        actual,meta=solve_smoothed_weight(xs,ws)
        d,z,r,mean,_=prepare_original_equation(xs,ws);a=torch.diag(d)+z.T@z
        original=mean+torch.linalg.solve(a,r).T
        error=float(torch.linalg.vector_norm(actual.double()-original)/torch.linalg.vector_norm(original))
        self.assertLess(error,2e-7)
    def test_single_expert_exact_invariance(self):
        xs,ws=self.fixture(k=1);value,_=solve_smoothed_weight(xs,ws)
        self.assertTrue(torch.equal(value,ws['0'].float()))
    def test_shared_expert_exact_invariance_and_zero_support_mean(self):
        xs,ws=self.fixture();common=torch.randn(3,7,dtype=torch.float64)
        shared={n:common.clone() for n in ws};value,_=solve_smoothed_weight(xs,shared)
        self.assertTrue(torch.equal(value,common.float()))
        for n in xs:xs[n][:,-2:]=0
        value,meta=solve_smoothed_weight(xs,ws,strategy='woodbury')
        mean=sum(ws.values())/len(ws)
        self.assertTrue(torch.equal(value[:,-2:],mean.float()[:,-2:]))
        self.assertEqual(meta['exact_zero_energy_coordinates'],2)
    def test_all_zero_input_is_mean_without_floor(self):
        xs,ws=self.fixture();xs={n:torch.zeros_like(x) for n,x in xs.items()}
        value,meta=solve_smoothed_weight(xs,ws)
        self.assertTrue(torch.equal(value,(sum(ws.values())/len(ws)).float()));self.assertEqual(meta['tau'],0.)
    def test_global_input_scale_invariance(self):
        xs,ws=self.fixture();a,am=solve_smoothed_weight(xs,ws)
        b,bm=solve_smoothed_weight({n:x*7 for n,x in xs.items()},ws)
        self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=2e-6));self.assertAlmostEqual(bm['tau']/am['tau'],49.,places=8)
    def test_no_reward_knobs_and_fixed_dtype_rule(self):
        xs,ws=self.fixture()
        with self.assertRaises(ValueError):solve_smoothed_weight(xs,ws,numerical_rank_dtype=torch.float64)
        _,meta=solve_smoothed_weight(xs,ws)
        self.assertEqual(meta['rank_epsilon'],torch.finfo(torch.float32).eps)
        self.assertAlmostEqual(meta['tau'],7*torch.finfo(torch.float32).eps*meta['spectral_scale_estimate'])
        self.assertFalse(meta['success_based_selection']);self.assertFalse(meta['teacher_output_targets_used']);self.assertFalse(meta['trust_cap'])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SmoothingTests))
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'cuda_initialized':torch.cuda.is_initialized()}
    Path(__file__).with_name('cpu-test-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    raise SystemExit(0 if result.wasSuccessful() and not receipt['cuda_initialized'] else 1)
