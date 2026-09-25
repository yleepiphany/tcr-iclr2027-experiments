#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,hashlib,json,tempfile,unittest
from pathlib import Path
import torch
from torch import nn
from graph_regmeanpp_m3 import *
from contract_regmeanpp_m3 import KERNEL,sha,GPUS
from run_regmeanpp_m3 import admissible,check_permit

class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__();self.first=nn.Linear(2,2,bias=False);self.last=nn.Linear(2,2,bias=False)
    def forward(self,x):return self.last(torch.tanh(self.first(x)))
class TinyGraph(nn.Module):
    def __init__(self):super().__init__();self.blocks=nn.ModuleList([TinyBlock(),TinyBlock()])
    def forward(self,x):return self.blocks[1](self.blocks[0](x))

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.kernel=load_kernel(KERNEL,sha)
    def test_01_scope_and_budget(self):
        steps=schedule();self.assertEqual(len(steps),67)
        self.assertEqual(sum(len(x.targets) for x in steps),418)
        self.assertEqual(sum(len(x.targets)*50*x.repetitions*x.rows_per_observation*3 for x in steps),1776300)
        self.assertEqual([(s.family,len(s.targets)) for s in steps[45:48]],[('interface',1)]*3)
    def test_02_batch_sampler_equals_original_single_observation(self):
        x=torch.arange(5*50*3).reshape(5,50,3).float()
        expected=torch.cat([v.index_select(0,torch.linspace(0,49,16).round().long()) for v in x])
        self.assertTrue(torch.equal(sample_observation_rows(x),expected))
        self.assertEqual(sample_observation_rows(torch.ones(5,1024)).shape,(5,1024))
        self.assertEqual(sample_observation_rows(torch.ones(5,256,1152)).shape,(80,1152))
    def test_03_two_block_merged_prefix_and_candidate_internal(self):
        torch.manual_seed(2);model=TinyGraph();x=torch.randn(5,16,2)
        first=Stage(0,'toy',('blocks.0.first','blocks.0.last'))
        second=Stage(1,'toy',('blocks.1.first','blocks.1.last'))
        prefix={p:torch.eye(2)*(.7+i*.1) for i,p in enumerate(first.targets)}
        commit_block(model,first,prefix);prefix_before={p:model.get_submodule(p).weight.clone() for p in first.targets}
        merged_boundary=model.blocks[0](x).detach()
        for index,group in enumerate(GROUPS):
            w={p:torch.eye(2)*(1.2+index*.3+i*.2) for i,p in enumerate(second.targets)}
            old={p:model.get_submodule(p).weight.clone() for p in second.targets}
            with candidate_block(model,second,w):
                captured=capture_stage(model,second,lambda:model(x))
                oracle_first=sample_observation_rows(merged_boundary)
                oracle_last=sample_observation_rows(torch.tanh(torch.nn.functional.linear(merged_boundary,w[second.targets[0]])))
                self.assertTrue(torch.equal(captured[second.targets[0]],oracle_first))
                self.assertTrue(torch.equal(captured[second.targets[1]],oracle_last))
                full=capture_stage(model,second,lambda:model(x),stop_early=False)
                for p in second.targets:self.assertTrue(torch.equal(captured[p],full[p]))
            for p in second.targets:self.assertTrue(torch.equal(model.get_submodule(p).weight,old[p]))
            for p in first.targets:self.assertTrue(torch.equal(model.get_submodule(p).weight,prefix_before[p]))
    def test_04_atomic_commit_rejects_partial_or_nonfinite(self):
        model=TinyGraph();s=Stage(0,'toy',('blocks.0.first','blocks.0.last'));before=model.blocks[0].first.weight.clone()
        with self.assertRaises(ValueError):commit_block(model,s,{'blocks.0.first':torch.eye(2)})
        with self.assertRaises(ValueError):commit_block(model,s,{'blocks.0.first':torch.eye(2),'blocks.0.last':torch.full((2,2),float('nan'))})
        self.assertTrue(torch.equal(before,model.blocks[0].first.weight))
    def fixture(self,rows,width,weak=False):
        torch.manual_seed(20260925);xs={g:torch.randn(rows,width,dtype=torch.float64) for g in GROUPS}
        if weak:
            for x in xs.values():x[:,-1]=0;x[:,-2]*=1e-10
        ws={g:torch.randn(4,width,dtype=torch.float64) for g in GROUPS}
        return xs,ws
    def oracle(self,xs,ws,tau,kind):
        grams={g:.3*(x.T@x)+.7*torch.diag(x.square().sum(0)) for g,x in xs.items()}
        a=sum(grams.values());mean=sum(ws.values())/3;r=sum(grams[g]@(ws[g]-mean).T for g in GROUPS)
        if kind=='complex':delta=torch.linalg.solve(a.to(torch.complex128)+1j*tau*torch.eye(a.shape[0]),r.to(torch.complex128)).real
        else:
            e,u=torch.linalg.eigh(a);delta=u@((e/(e*e+tau*tau)).unsqueeze(1)*(u.T@r))
        return (mean+delta.T).float()
    def test_05_dense_two_independent_oracles(self):
        for weak in (False,True):
            x,w=self.fixture(11,7,weak)
            result,m=self.kernel.solve_smoothed_weight(x,w,strategy='dense',device='cpu')
            for kind in ('complex','eigen'):torch.testing.assert_close(result,self.oracle(x,w,m['tau'],kind),rtol=2e-5,atol=2e-6)
            self.assertLessEqual(m['smooth_normal_equation_relative_residual'],1e-7)
            self.assertNotIn('relative_residual',m)
    def test_06_woodbury_two_independent_oracles(self):
        for weak in (False,True):
            x,w=self.fixture(3,19,weak)
            result,m=self.kernel.solve_smoothed_weight(x,w,strategy='woodbury',device='cpu')
            for kind in ('complex','eigen'):torch.testing.assert_close(result,self.oracle(x,w,m['tau'],kind),rtol=2e-5,atol=2e-6)
            self.assertLessEqual(m['smooth_normal_equation_relative_residual'],1e-7)
    def test_07_constant_weights_and_mean_bias(self):
        x,w=self.fixture(5,9,True);same={g:w[GROUPS[0]].clone() for g in GROUPS}
        result,_=self.kernel.solve_smoothed_weight(x,same,device='cpu')
        self.assertTrue(torch.equal(result,same[GROUPS[0]].float()))
        self.assertEqual(set(INTERFACES),{'model.action_in_proj','model.action_out_proj','model.time_mlp_in','model.time_mlp_out'})
    def test_08_resource_guard(self):
        row={'gpu':1,'uuid':GPUS[1],'free_mib':81153,'used_mib':1,'utilization':0,'compute_pids':[]}
        self.assertTrue(admissible(row,GPUS))
        for patch in ({'free_mib':71679},{'used_mib':65},{'compute_pids':[44]},{'utilization':1},{'uuid':'wrong'},{'gpu':0}):
            self.assertFalse(admissible({**row,**patch},GPUS))
    def test_09_single_use_permit_identity(self):
        p={'schema':'robotwin_regmeanpp_m3_single_use_execution_permit_v1','allowed':True,'stage':'smoke',
            'plan_sha256':'abc','run':'/tmp/frozen','gpu':1,'gpu_uuid':GPUS[1],'host':'host'}
        check_permit(p,'smoke','abc','/tmp/frozen',1,GPUS[1],'host')
        for patch in ({'allowed':False},{'stage':'materialize'},{'plan_sha256':'wrong'},{'run':'/tmp/other'},{'gpu':4},{'host':'other'}):
            with self.assertRaises(ValueError):check_permit({**p,**patch},'smoke','abc','/tmp/frozen',1,GPUS[1],'host')
    def test_10_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'gpu_used':False,'cuda_initialized':torch.cuda.is_initialized(),
        'oracles':['independent complex shifted direct solve','independent Hermitian eigendecomposition'],
        'cases':['dense regular','dense rank-deficient/weak','Woodbury regular','Woodbury rank-deficient/weak']}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
