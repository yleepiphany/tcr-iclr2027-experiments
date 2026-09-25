import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('pro_transfer',Path(__file__).with_name('run_transfer.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class TransferTests(unittest.TestCase):
    def job(self):
        return {'id':'repeat-01-object-libero_spatial-task00','output':'/old/output',
                'command':['python','eval.py','--policy-sha256=abc','--seed=274001','--output_dir=/old/output','--eval.n_episodes=10'],
                'state_ids':list(range(10)),'state_raw_sha256':['a']*10,'repeat':'repeat-01','eval_seed':274001}
    def test_only_output_token_changes(self):
        job=self.job();before=copy.deepcopy(job);new=module.translate_job('regmean_pp',job,Path('/new'))
        self.assertEqual(job,before)
        self.assertEqual([i for i,(a,b) in enumerate(zip(job['command'],new['command'])) if a!=b],[4])
        self.assertEqual(new['state_ids'],job['state_ids']);self.assertEqual(new['state_raw_sha256'],job['state_raw_sha256'])
    def test_missing_or_duplicate_output_rejected(self):
        for command in ([],['--output_dir=/old/output']*2):
            job=self.job();job['command']=command
            with self.assertRaises(ValueError):module.translate_job('featcal',job,Path('/new'))
    def test_any_started_or_result_marker_blocks_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);module.require_unstarted(root)
            for name in ('started.json','state.json','queue-ended.json','results','launches','exits'):
                (root/name).touch()
                with self.assertRaises(ValueError):module.require_unstarted(root)
                (root/name).unlink()
    def test_idle_gate_rejects_training_and_other_compute(self):
        good={'gpu':1,'uuid':module.GPUS[1],'free_mib':81153,'used_mib':1,'utilization':0,'compute_pids':[]}
        self.assertTrue(module.idle(good))
        for change in ({'compute_pids':['123']},{'used_mib':65},{'utilization':1},{'free_mib':40959},{'uuid':'different'}):
            self.assertFalse(module.idle({**good,**change}))
    def test_separate_method_outputs(self):
        a=module.translate_job('regmean_pp',self.job(),Path('/new'))
        b=module.translate_job('featcal',self.job(),Path('/new'))
        self.assertNotEqual(a['output'],b['output']);self.assertNotEqual(a['transfer_id'],b['transfer_id'])

if __name__=='__main__':unittest.main()
