import copy
import importlib.util
import os
from pathlib import Path
import unittest
import torch
from subset_inputs import SUBSETS,SubsetInputs,read,initial_noise,validate_model_batch,tensor_sha

WORK=Path(os.environ.get('FIG4_TEST_WORK','/mnt/workspace/Wilson/parameter-fusion'))
MASTER=WORK/'vla-merge-runtime/experiments/fig4-baselines-codex-20260925/plan.json'

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.master=read(MASTER)
        source=WORK/'vla-merge/experiments/static-observation-baselines-20260922/featcal/featcal_observation_only.py'
        spec=importlib.util.spec_from_file_location('original_table1_observations_readonly',source)
        cls.original=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.original)
    def test_full_two_three_observation_and_noise_counts_and_original_byte_identity(self):
        for subset,groups in SUBSETS.items():
            raw=SubsetInputs(self.master,subset);receipt=raw.verify();seen_obs=set();seen_noise=set()
            self.assertEqual(receipt['observations'],len(groups)*50);self.assertEqual(receipt['noise_calls'],len(groups)*150)
            for group in groups:
                for task in range(10):
                    for replica in range(3):
                        batch=raw.batch(group,task,replica);self.assertEqual(batch['x_t'].shape,(5,50,32))
                        for request in range(5):
                            original,seed=self.original.initial_noise(group,task,request,(0,5,9)[replica])
                            self.assertTrue(torch.equal(batch['x_t'][request:request+1],original))
                            seen_obs.add((group,task,request));seen_noise.add(tensor_sha(original))
                            self.assertEqual(initial_noise(group,task,request,replica)[1],seed)
            self.assertEqual(len(seen_obs),len(groups)*50);self.assertEqual(len(seen_noise),len(groups)*150)
    def test_goal_seed_is_never_reindexed_for_m2(self):
        noise,seed=initial_noise('goal',0,0,0);self.assertEqual(seed,202609420000)
        raw=SubsetInputs(self.master,'spatial-goal');self.assertTrue(torch.equal(raw.batch('goal',0,0)['x_t'][:1],noise))
        self.assertFalse(torch.equal(noise,initial_noise('object',0,0,0)[0]))
    def test_outside_subset_and_four_expert_rebuild_are_rejected(self):
        raw=SubsetInputs(self.master,'spatial-goal')
        for group in ('object','long'):
            with self.assertRaises(ValueError):raw.batch(group,0,0)
        with self.assertRaises(ValueError):SubsetInputs(self.master,'spatial-object-goal-long')
    def test_labels_nonunit_time_and_false_camera_claim_rejected(self):
        raw=SubsetInputs(self.master,'spatial-goal');batch=raw.batch('spatial',0,0)
        for key in ('actions','action','targets','reward'):
            bad={**batch,key:torch.zeros(5)}
            with self.assertRaises(ValueError):validate_model_batch(bad)
        with self.assertRaises(ValueError):validate_model_batch({**batch,'time':torch.zeros(5)})
        with self.assertRaises(ValueError):validate_model_batch({**batch,'image_mask_2':torch.ones(5,dtype=torch.bool)})
    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_initialized():raise AssertionError('CPU tests initialized CUDA')
if __name__=='__main__':unittest.main()
