import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file, load_file
import pass_engine as engine
from native_block_plan import make_plan
from materialize_soup import ORDER, SHARED
from test_block_calibration import Policy, Bank


class PassTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = Policy()
        self.bank = Bank(self.policy)
        self.requests = {expert: [{'id': expert + '/A/0',
            'inputs': {'x': torch.eye(2), 'attention_mask': torch.ones(1, 2)}}] for expert in ORDER}
        trace = [{'name': 'action_head.' + str(i), 'input_shape': [1, 2, 2], 'weight_shape': [2, 2]} for i in (0, 1)]
        self.plan = make_plan([trace] * 4, cap=2, requests_per_expert=1)
        self.recipe = dict(pass_id='A', pool='A', mass_rule='relative', row_seed=123, cap=2,
                           ridge_multiplier=.01, max_correction_ratio=10., input_provenance={'test': True})
        self.template = self.root / 'prior'
        self.template.mkdir()
        state = self.policy.model.state_dict()
        save_file(state, str(self.template / 'part.safetensors'))
        (self.template / 'model.safetensors.index.json').write_text(json.dumps(
            {'weight_map': {key: 'part.safetensors' for key in state}, 'metadata': {}}))
        for name in (*SHARED, 'dataset_statistics.json'):
            (self.template / name).write_text('{}')
        for name, module in (('action_head', self.policy.head), ('proprio_projector', self.policy.proprio)):
            torch.save({'module.' + key: value for key, value in module.state_dict().items()},
                       str(self.template / (name + '--prior_checkpoint.pt')))

    def test_real_small_two_pass_regression_and_complete_export(self):
        a = engine.run_pass(self.policy, self.bank, self.requests, self.plan, self.recipe,
                            template=self.template, run=self.root / 'A')
        self.assertEqual(a['rows'], 16)
        self.assertEqual(a['modules'], 2)
        checkpoint = Path(a['checkpoint'])
        saved = torch.load(str(checkpoint / 'action_head--tcr_checkpoint.pt'), weights_only=True)
        for key, value in self.policy.head.state_dict().items():
            self.assertTrue(torch.equal(value, saved['module.' + key]))
        requests_b = copy.deepcopy(self.requests)
        for records in requests_b.values():
            records[0]['id'] = records[0]['id'].replace('/A/', '/B/')
            records[0]['inputs']['x'] *= 1.5
        b = engine.run_pass(self.policy, self.bank, requests_b, self.plan,
                            {**self.recipe, 'pass_id': 'B', 'pool': 'B', 'mass_rule': 'uniform'},
                            template=checkpoint, run=self.root / 'B')
        self.assertTrue(b['complete'])
        self.assertFalse(b['native_reload_verified'])
        self.assertFalse(b['success_evaluated'])
        self.assertTrue(torch.equal(load_file(str(Path(b['checkpoint']) / 'part.safetensors'))['weight'],
                                    self.policy.model.weight))
        with self.assertRaises(FileExistsError):
            engine.run_pass(self.policy, self.bank, self.requests, self.plan, self.recipe,
                            template=self.template, run=self.root / 'A')

    def test_mass_schedule_and_row_contract_rejected(self):
        with self.assertRaises(ValueError):
            engine.validate_recipe({**self.recipe, 'mass_rule': 'uniform'}, self.plan, self.requests)
        plan = copy.deepcopy(self.plan)
        plan['planned_rows_per_pass_if_all_linears_solved'] += 1
        with self.assertRaises(ValueError): engine.validate_recipe(self.recipe, plan, self.requests)

    def test_failure_does_not_export_or_mark_complete(self):
        with patch.object(engine, 'calibrate_block', side_effect=RuntimeError('intentional test failure')):
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                engine.run_pass(self.policy, self.bank, self.requests, self.plan, self.recipe,
                                template=self.template, run=self.root / 'failed')
        self.assertTrue((self.root / 'failed/failed.json').exists())
        self.assertFalse((self.root / 'failed/manifest.json').exists())
        self.assertFalse((self.root / 'failed/export').exists())

    def test_stale_loaded_prior_rejected_before_capture(self):
        with torch.no_grad():
            self.policy.head[0].weight.add_(1)
        with self.assertRaisesRegex(ValueError, 'fixed pass prior'):
            engine.run_pass(self.policy, self.bank, self.requests, self.plan, self.recipe,
                            template=self.template, run=self.root / 'wrong-prior')
        self.assertEqual(self.policy.calls, [])
        self.assertFalse((self.root / 'wrong-prior').exists())


if __name__ == '__main__':
    unittest.main()
