"""Synthetic checks of the reused per-episode/reset audit; never evaluates a policy."""
import json
from pathlib import Path
import tempfile
import unittest
import run_ties_formal_v1 as runner

class AuditTests(unittest.TestCase):
    def test_valid_missing_episode_and_reset_mismatch(self):
        plan=runner.template();p=plan['native_reset_preflight']
        selection=runner.read(p['selection'])['tasks'];bank=runner.read(Path(p['bank'])/'manifest.json')['tasks']
        audit=runner.load_module(runner.AUDIT_SOURCE,'test_original_episode_audit')
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'eval';output.mkdir()
            job={'id':'synthetic-only','full_suite':'libero_spatial','output':tmp}
            info={'per_task':[{'task_id':i,'metrics':{'successes':[i*10+j<37 for j in range(10)]}} for i in range(10)],
                  'overall':{'n_episodes':100,'pc_success':37.}}
            def write_info():
                (output/'eval_info.json').write_text(json.dumps(info))
            write_info()
            receipt={'selection_sha256':p['selection_sha256'],'eval_seed':274001,'repeat_id':'repeat-01',
                     'bank_manifest_sha256':p['bank_manifest_sha256'],
                     'entrypoint_sha256':plan['source_sha256'][str(runner.REPO/'scripts/eval_pi05_policy_with_procedural_bank.py')],
                     'eval_info_sha256':runner.sha(output/'eval_info.json'),'tasks':{}}
            for i in range(10):
                key=f'libero_spatial/{i:02d}';ix=selection[key]
                receipt['tasks'][key]={'suite':'libero_spatial','task_id':i,'state_indices':ix,
                    'raw_state_sha256':[bank[key]['states'][n]['raw_sha256'] for n in ix]}
            def write_receipt():
                (output/'procedural_bank_receipt.json').write_text(json.dumps(receipt))
            write_receipt();(output/'bounded_resources.json').write_text('{"evaluation_completed":true}')
            ap={'selection':{'sha256':p['selection_sha256'],'eval_seed':274001,'selection':p['selection'],'bank':p['bank']},
                'bank_manifest_sha256':p['bank_manifest_sha256'],'procedural_entrypoint_sha256':receipt['entrypoint_sha256']}
            accepted=audit.audit_result(job,'synthetic-model',ap)
            self.assertEqual((accepted['episodes'],accepted['successes']),(100,37))
            info['per_task'][0]['metrics']['successes'].pop();write_info()
            receipt['eval_info_sha256']=runner.sha(output/'eval_info.json');write_receipt()
            with self.assertRaises(ValueError):audit.audit_result(job,'synthetic-model',ap)
            info['per_task'][0]['metrics']['successes'].append(True);write_info()
            receipt['eval_info_sha256']=runner.sha(output/'eval_info.json')
            receipt['tasks']['libero_spatial/00']['raw_state_sha256'][0]='0'*64;write_receipt()
            with self.assertRaises(ValueError):audit.audit_result(job,'synthetic-model',ap)

if __name__=='__main__':unittest.main()
