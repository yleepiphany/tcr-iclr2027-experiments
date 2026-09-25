"""Small CPU graph tests of both subset adapters's real hook/index/prefix components.

The graph has all 418 names and the actual 47-stage hook order, but width=2.
It is not a native PI0.5 forward or a substitute for the required GPU smoke.
"""
import os
from pathlib import Path
import unittest
import tempfile
from contextlib import ExitStack
from unittest.mock import patch
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from featcal_subset import API
from subset_inputs import SUBSETS

WORK=Path(os.environ.get('FIG4_TEST_WORK','/mnt/workspace/Wilson/parameter-fusion'))
MASTER=WORK/'vla-merge-runtime/experiments/fig4-baselines-codex-20260925/plan.json'

class TinyGraph(torch.nn.Module):
    def __init__(self,steps):
        super().__init__();self.steps=steps
        for step in steps:
            for path in step.target_module_paths:
                node=self
                for part in path.split('.')[:-1]:
                    if part not in node._modules:node.add_module(part,torch.nn.Module())
                    node=node._modules[part]
                layer=torch.nn.Linear(2,2,bias=False)
                with torch.no_grad():layer.weight.copy_(torch.eye(2))
                node.add_module(path.split('.')[-1],layer)
        self.eval()
    def forward(self):
        seed=torch.arange(5*12*2,dtype=torch.float32).reshape(5,12,2)/100
        for _ in range(3):
            value=seed
            for step in self.steps:
                if step.stage!='vision_prefix':continue
                outputs=[self.get_submodule(p)(value) for p in step.target_module_paths]
                value=sum(outputs)/len(outputs)
        for step in self.steps:
            if step.stage=='vision_prefix':continue
            for path in step.target_module_paths:
                x=value[:,0,:] if path in {'model.time_mlp_in','model.time_mlp_out'} else value
                output=self.get_submodule(path)(x)
                if output.ndim==3:value=output
        return value

def masks(path,occurrence,value):return torch.ones(value.shape[:-1],dtype=torch.bool)

class GraphChecks:
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.api=API(WORK,MASTER,cls.SUBSET);cls.steps=cls.api.steps
        logical=[{'base_key':p+'.weight','base_shape':[2,2]} for s in cls.steps for p in s.target_module_paths]
        logical += [{'base_key':p+'.bias','base_shape':[2]} for p in ('model.action_in_proj','model.action_out_proj','model.time_mlp_in','model.time_mlp_out')]
        cls.quotas=cls.api.cache.derive_module_quotas({'adaptation_domain':{'logical_tensors':logical}},cls.steps)
        cls.index=cls.api.cache.build_shard_index(cls.quotas,cls.steps,cls.api.cache.build_call_slots(),contract_sha256='test-plan',
            expert_model_sha256={g:'test-'+g for g in cls.api.groups},initial_student_sha256='a'*64)
        cls.specs=cls.api.spec_map(cls.index)
    def teacher(self,model):
        captures={q.module_path:[] for q in self.quotas};keys={q.module_path:[] for q in self.quotas}
        for replica in range(3):
            c=self.api.cache.MultiLayerStreamingRowCollector(model,self.steps,self.quotas,
                call_slot_ids=self.api.slots(self.api.groups[0],0,replica),call_ordinals_within_task=[r*3+replica for r in range(5)],
                row_mask_provider=masks,selection_seed=self.api.baseline.teacher_runner.SELECTION_SEED)
            result=c.capture(model)
            self.assertEqual(result.trace,self.api.cache.expected_full_forward_trace(self.steps))
            for path in captures:captures[path].append(result.inputs_by_module[path]);keys[path].append(result.row_keys_by_module[path])
        return captures,keys
    def student_step(self,model,step,teacher_keys):
        qs=[q for q in self.quotas if q.module_path in step.target_module_paths];values={p:[] for p in step.target_module_paths};keys={p:[] for p in step.target_module_paths}
        for replica in range(3):
            c=self.api.SingleStepSelectedRowCollector(model,step,qs,teacher_row_keys=teacher_keys,
                call_slot_ids=self.api.slots(self.api.groups[0],0,replica),call_ordinals_within_task=[r*3+replica for r in range(5)],row_mask_provider=masks)
            result=c.capture(model)
            for path in values:values[path].append(result.inputs_by_module[path]);keys[path].append(result.row_keys_by_module[path])
        return {p:torch.cat(values[p]) for p in values},{p:torch.cat(keys[p]) for p in keys}
    def test_actual_stage_weight_call_shard_identity(self):
        self.assertEqual(len(self.steps),47);self.assertEqual(len(self.quotas),418)
        self.assertEqual(len(self.api.cache.build_call_slots()),150*len(self.api.groups));self.assertEqual(self.index['shard_count'],940*len(self.api.groups))
        self.assertEqual(sum(q.rows_per_expert for q in self.quotas)*len(self.api.groups),self.api.total_rows)
        import vla_merge.featcal_cache_plan as original
        self.assertEqual(original.EXPERTS,('spatial','object','goal','long'))  # historical module untouched
    def test_real_collectors_pair_rows_and_propagate_updated_prefix(self):
        teacher=TinyGraph(self.steps);student=TinyGraph(self.steps);captures,keys=self.teacher(teacher)
        step=self.steps[1];tk={p:torch.cat(keys[p]) for p in step.target_module_paths}
        before,paired=self.student_step(student,step,tk)
        for path in step.target_module_paths:
            self.assertTrue(torch.equal(before[path],torch.cat(captures[path])))
            self.assertTrue(torch.equal(paired[path],tk[path]))
        replacements={p:2*torch.eye(2) for p in self.steps[0].target_module_paths}
        self.api.baseline.rollback_safe_step_load({p:student.get_submodule(p) for p in replacements},replacements)
        after,paired=self.student_step(student,step,tk)
        for path in step.target_module_paths:
            self.assertFalse(torch.equal(before[path],after[path]))
            self.assertTrue(torch.equal(paired[path],tk[path]))
        self.assertTrue(all(not m._forward_pre_hooks for m in student.modules()))
    def test_corrupt_teacher_row_indices_are_rejected(self):
        teacher=TinyGraph(self.steps);captures,keys=self.teacher(teacher);step=self.steps[0]
        tk={p:torch.cat(keys[p]) for p in step.target_module_paths};tk[step.target_module_paths[0]][0,2]=999
        with self.assertRaises(ValueError):self.student_step(TinyGraph(self.steps),step,tk)
    def test_original_exact_solver_consumes_selected_expert_shards_and_commits_atomically(self):
        model=TinyGraph(self.steps);captures,keys=self.teacher(model);step=self.steps[0];b=self.api.baseline
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cache=root/'cache';cache.mkdir()
            for role in ('teacher','student'):
                for group in self.api.groups:
                    for task in range(10):
                        spec=self.specs[(role,group,task,0)]
                        tensors=b.teacher_runner.assemble_step_tensors(spec,captures,keys)
                        self.api.cache.write_completed_shard_atomic(cache,spec,tensors,plan_sha256='test-plan',
                            call_state_manifest_sha256=self.api.input_sha,
                            source_checkpoint_sha256='test-'+group if role=='teacher' else 'a'*64,
                            student_prefix_input_sha256='a'*64 if role=='student' else None)
            soup={p+'.weight':torch.eye(2) for p in step.target_module_paths}
            save_file(soup,str(root/'soup.safetensors'))
            save_file({k:torch.zeros_like(v) for k,v in soup.items()},str(root/'base.safetensors'))
            save_file({k:2*v for k,v in soup.items()},str(root/'expert.safetensors'))
            original=b.featcal_hybrid_linear_weight;calls=[]
            def cpu_kernel(weights,factors,**kwargs):
                calls.append(len(weights));kwargs['solve_device']='cpu';return original(weights,factors,**kwargs)
            with ExitStack() as stack:
                sh=stack.enter_context(safe_open(str(root/'soup.safetensors'),framework='pt',device='cpu'))
                bh=stack.enter_context(safe_open(str(root/'base.safetensors'),framework='pt',device='cpu'))
                eh=stack.enter_context(safe_open(str(root/'expert.safetensors'),framework='pt',device='cpu'))
                with patch.object(b,'featcal_hybrid_linear_weight',side_effect=cpu_kernel):
                    prefix,previous,receipt=b._solve_formal_step(cache_root=cache,student=model,step=step,
                        specs=self.specs,plan_sha256='test-plan',current_prefix='a'*64,previous_solve=None,
                        soup_handle=sh,base_handle=bh,expert_handles={g:eh for g in self.api.groups})
            self.assertEqual(calls,[len(self.api.groups)]*6)
            self.assertTrue(receipt['atomic_step_load']);self.assertNotEqual(prefix,'a'*64)
            self.assertTrue((cache/previous['path']).is_file())
            self.assertTrue((cache/'prefix-snapshots/step_00.safetensors').is_file())
            for path in step.target_module_paths:self.assertTrue(torch.allclose(model.get_submodule(path).weight,2*torch.eye(2),atol=1e-5,rtol=1e-5))
    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_initialized():raise AssertionError('CPU graph tests initialized CUDA')

class M2(GraphChecks,unittest.TestCase):SUBSET='spatial-goal'
class M3(GraphChecks,unittest.TestCase):SUBSET='spatial-object-goal'

if __name__=='__main__':unittest.main()
