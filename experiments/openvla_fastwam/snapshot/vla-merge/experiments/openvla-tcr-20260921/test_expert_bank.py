import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import save_file

from expert_bank import ExpertBank, resolve_block
from materialize_soup import ORDER, sha


class BankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        experts, files = {}, {}
        for i, name in enumerate(ORDER):
            root = self.root / name
            root.mkdir()
            experts[name] = {'local_path': str(root)}
            state = {'layer.weight': torch.full((2, 3), float(i)),
                     'layer.bias': torch.full((2,), float(i))}
            save_file(state, str(root / 'part.safetensors'))
            (root / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {key: 'part.safetensors' for key in state}}))
            for component in ('action_head', 'proprio_projector'):
                torch.save({'module.weight': torch.full((2, 3), float(i)),
                            'module.bias': torch.full((2,), float(i))},
                           root / (component + '--1_checkpoint.pt'))
            for path in root.iterdir():
                files[str(path)] = {'sha256': sha(path), 'size': path.stat().st_size,
                                    'mtime_ns': path.stat().st_mtime_ns}
        self.records = files
        self.ledger = self.root / 'ledger.json'
        self.ledger.write_text(json.dumps({'complete': True, 'experts': experts, 'files': files}))

    def test_lookup_and_no_source_mutation(self):
        with ExpertBank(self.ledger) as bank:
            for i, name in enumerate(ORDER):
                for component in ('backbone.layer', 'action_head', 'proprio_projector'):
                    weight, conversions = bank.affine(name, component, torch.nn.Linear(3, 2))
                    self.assertEqual(conversions, [])
                    self.assertTrue(torch.equal(weight, torch.full((2, 4), float(i))))
                    weight.fill_(100)
                    self.assertEqual(bank.tensor(name, component + '.weight')[0, 0].item(), i)
        self.assertTrue(all(sha(Path(path)) == record['sha256'] for path, record in self.records.items()))

    def test_explicit_native_cast_and_shape_guard(self):
        module = torch.nn.Linear(3, 2).to(torch.bfloat16)
        with ExpertBank(self.ledger) as bank:
            state, conversions = bank.block_state(ORDER[1], 'backbone.layer', module)
            self.assertEqual(set(state), {'weight', 'bias'})
            self.assertEqual(state['weight'].dtype, torch.bfloat16)
            self.assertEqual(len(conversions), 2)
            self.assertEqual(conversions[0]['source_dtype'], 'torch.float32')
            with self.assertRaises(ValueError):
                bank.block_state(ORDER[0], 'backbone.layer', torch.nn.Linear(4, 2))
            with self.assertRaises(KeyError):
                bank.tensor(ORDER[0], 'backbone.unknown.weight')

    def test_nonfinite_rejected_without_mutating_native(self):
        module = torch.nn.Linear(3, 2)
        before = {key: value.clone() for key, value in module.state_dict().items()}
        with ExpertBank(self.ledger) as bank:
            bank.components[ORDER[0]]['action_head']['weight'] = torch.full((2, 3), float('nan'))
            with self.assertRaises(ValueError):
                bank.block_state(ORDER[0], 'action_head', module)
        self.assertTrue(all(torch.equal(value, module.state_dict()[key]) for key, value in before.items()))

    def test_changed_source_rejected(self):
        path = self.root / ORDER[0] / 'model.safetensors.index.json'
        path.write_text('{}')
        with self.assertRaises(ValueError):
            ExpertBank(self.ledger)

    def test_native_resolution(self):
        policy = SimpleNamespace(model=torch.nn.Sequential(torch.nn.Linear(3, 2)),
                                 head=torch.nn.Linear(2, 1), proprio=torch.nn.Linear(2, 2))
        self.assertIs(resolve_block(policy, 'backbone.0'), policy.model[0])
        self.assertIs(resolve_block(policy, 'action_head'), policy.head)
        with self.assertRaises(ValueError):
            resolve_block(policy, 'unknown')


if __name__ == '__main__':
    unittest.main()
