import unittest
import torch
from linear_calibration import affine_parameters,capture_rows,expert_masses,row_indices,solve_module,expert_block_state,capture_expert_block

class Policy:
    def __init__(self,calls=2):
        self.layer=torch.nn.Linear(3,2,bias=True);self.calls=calls
    def linear_modules(self):return [('layer',self.layer)]
    def replay(self,request):
        return [self.layer(request['x']+i) for i in range(self.calls)]

class Tests(unittest.TestCase):
    def test_expert_block_restores_after_exception(self):
        block=torch.nn.Sequential(torch.nn.LayerNorm(3),torch.nn.Linear(3,2))
        before={k:v.clone() for k,v in block.state_dict().items()}
        expert={k:v+1 for k,v in before.items()}
        with self.assertRaisesRegex(RuntimeError,'capture failed'):
            with expert_block_state(block,expert):
                for k,v in block.state_dict().items():self.assertTrue(torch.equal(v,expert[k]))
                raise RuntimeError('capture failed')
        for k,v in block.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
    def test_merged_prefix_expert_inside_block(self):
        class BlockPolicy:
            def __init__(self):
                self.prefix=torch.nn.Linear(2,2,bias=False)
                self.block=torch.nn.Sequential(torch.nn.Linear(2,2,bias=False),torch.nn.Linear(2,2,bias=False))
                with torch.no_grad():
                    self.prefix.weight.copy_(torch.eye(2)*2)
                    for m in self.block:m.weight.copy_(torch.eye(2)*3)
                self.replays=0
            def linear_modules(self):return [('first',self.block[0]),('second',self.block[1]),('prefix',self.prefix)]
            def replay(self,request):
                self.replays+=1
                return self.block(self.prefix(request['x']))
        p=BlockPolicy();req={'x':torch.ones(1,2),'attention_mask':torch.ones(1,2)}
        expert={k:torch.eye(2)*5 for k in p.block.state_dict()}
        xs,_=capture_expert_block(p,p.block,expert,['first','second'],req,request_id='q',cap=8,seed=2,expected_calls={'first':1,'second':1})
        self.assertTrue(torch.equal(xs['first'],torch.full((1,2),2.)))
        self.assertTrue(torch.equal(xs['second'],torch.full((1,2),10.)))
        self.assertEqual(p.replays,1)
        self.assertTrue(torch.equal(p.block[0].weight,torch.eye(2)*3))
        self.assertTrue(torch.equal(p.prefix.weight,torch.eye(2)*2))
        with self.assertRaises(ValueError):capture_expert_block(p,p.block,expert,['prefix'],req,request_id='q',cap=8,seed=2,expected_calls={'prefix':1})
    def test_call_budget_identity_and_bias(self):
        policy=Policy();x=torch.arange(30).reshape(10,3).float()
        request={'x':x,'attention_mask':torch.ones((1,5))}
        kwargs=dict(request_id='q',cap=8,seed=7,expected_calls=2)
        a,receipt=capture_rows(policy,'layer',request,**kwargs)
        b,_=capture_rows(policy,'layer',request,**kwargs)
        self.assertTrue(torch.equal(a,b));self.assertEqual(a.shape,(8,4))
        self.assertTrue(torch.equal(a[:,-1],torch.ones(8)))
        self.assertEqual([len(c['selected_rows']) for c in receipt['calls']],[4,4])
        self.assertEqual(len(policy.layer._forward_pre_hooks),0)
    def test_reject_padding_and_extra_calls_cleanup(self):
        p=Policy();req={'x':torch.ones(4,3),'attention_mask':torch.zeros(1,2)}
        with self.assertRaises(ValueError):capture_rows(p,'layer',req,request_id='x',cap=8,seed=1,expected_calls=2)
        req['attention_mask'].fill_(1)
        with self.assertRaises(ValueError):capture_rows(p,'layer',req,request_id='x',cap=8,seed=1,expected_calls=1)
        self.assertEqual(len(p.layer._forward_pre_hooks),0)
    def test_small_module_capacity(self):
        p=Policy(1);req={'x':torch.ones(1,3),'attention_mask':torch.ones(1,2)}
        x,r=capture_rows(p,'layer',req,request_id='x',cap=8,seed=1,expected_calls=1)
        self.assertEqual(len(x),1);self.assertEqual(r['rows'],1)
    def test_masses_and_noop(self):
        p=Policy();prior=affine_parameters(p.layer);xs=[torch.ones(5,4)]*4
        masses,d=expert_masses(xs,[prior]*4,prior,'relative')
        self.assertEqual(masses,[.25]*4)
        result=solve_module(p.layer,xs,[prior]*4,prior,mass_rule='relative',ridge_multiplier=.05,max_correction_ratio=.3)
        self.assertTrue(torch.equal(affine_parameters(p.layer),prior));self.assertEqual(result['unclipped_correction_norm'],0.)
    def test_affine_solve_and_trust(self):
        p=torch.nn.Linear(3,2);prior=affine_parameters(p);g=torch.Generator().manual_seed(5)
        xs=[torch.cat((torch.randn(20,3,generator=g),torch.ones(20,1)),1)]*4
        experts=[prior+2]*4
        result=solve_module(p,xs,experts,prior,mass_rule='uniform',ridge_multiplier=.001,max_correction_ratio=.1)
        change=affine_parameters(p)-prior
        self.assertLessEqual(float(change.norm()),float(prior.norm())*.100001)
        self.assertLess(result['trust_scale'],1.)
        with self.assertRaises(ValueError):solve_module(p,xs,experts,prior,mass_rule='uniform',ridge_multiplier=.001,max_correction_ratio=.1)
    def test_uniform_mean_solution(self):
        p=torch.nn.Linear(3,2,bias=False);prior=affine_parameters(p)
        xs=[torch.eye(3)]*4;experts=[prior+i/10 for i in range(4)]
        result=solve_module(p,xs,experts,prior,mass_rule='uniform',ridge_multiplier=.1,max_correction_ratio=1e6)
        expected=prior+torch.full_like(prior,.15)/(1+.1)
        torch.testing.assert_close(p.weight.detach(),expected)
        self.assertAlmostEqual(result['ridge'],.1/3)

if __name__=='__main__':unittest.main()
