import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file, load_file
import export_native as exporter
from materialize_soup import SHARED, sha


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.template = self.root / 'prior'
        self.template.mkdir()
        self.backbone = {'linear.weight': torch.ones(2, 3), 'unused_pool.weight': torch.zeros(2, 3)}
        save_file({'linear.weight': self.backbone['linear.weight']}, str(self.template / 'part1.safetensors'))
        save_file({'unused_pool.weight': self.backbone['unused_pool.weight']}, str(self.template / 'part2.safetensors'))
        index = {'metadata': {'total_size': 48}, 'weight_map': {'linear.weight': 'part1.safetensors',
                                                              'unused_pool.weight': 'part2.safetensors'}}
        (self.template / 'model.safetensors.index.json').write_text(json.dumps(index))
        for name in (*SHARED, 'dataset_statistics.json'):
            (self.template / name).write_text(json.dumps({'preserved': name}))
        self.states = {'backbone': {key: value.to(torch.bfloat16).clone() for key, value in self.backbone.items()}}
        for name in ('action_head', 'proprio_projector'):
            torch.save({'module.weight': torch.ones(2, 3), 'module.bias': torch.zeros(2)},
                       str(self.template / (name + '--1_checkpoint.pt')))
            self.states[name] = {'weight': torch.full((2, 3), 2., dtype=torch.bfloat16),
                                 'bias': torch.ones(2, dtype=torch.bfloat16)}
        self.source_hashes = {path.name: sha(path) for path in self.template.iterdir()}

    def export(self):
        return exporter.export_native(self.states, self.template, self.root / 'out', provenance={'test': True})

    def test_complete_tensor_roundtrip_and_metadata(self):
        result = self.export()
        self.assertTrue(result['complete'])
        self.assertTrue(result['serialization_only'])
        self.assertFalse(result['gpu_reload_verified'])
        output = self.root / 'out/checkpoint'
        index = json.loads((output / 'model.safetensors.index.json').read_text())
        self.assertEqual(index['metadata']['total_size'], 24)
        for key, shard in index['weight_map'].items():
            actual = load_file(str(output / shard))[key]
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(actual, self.states['backbone'][key]))
        for name in ('action_head', 'proprio_projector'):
            actual = torch.load(str(output / (name + '--tcr_checkpoint.pt')), weights_only=True)
            self.assertEqual(set(actual), {'module.weight', 'module.bias'})
            for key, value in self.states[name].items():
                self.assertTrue(torch.equal(value, actual['module.' + key]))
        for name in (*SHARED, 'dataset_statistics.json'):
            self.assertEqual(sha(output / name), self.source_hashes[name])
        self.assertEqual({path.name: sha(path) for path in self.template.iterdir()}, self.source_hashes)
        with self.assertRaises(FileExistsError): self.export()

    def test_partial_state_rejected_before_output(self):
        del self.states['proprio_projector']['bias']
        with self.assertRaises(ValueError): self.export()
        self.assertFalse((self.root / 'out').exists())

    def test_nonfinite_export_retains_failure_without_manifest(self):
        self.states['action_head']['weight'].fill_(float('nan'))
        with self.assertRaises(ValueError): self.export()
        self.assertTrue((self.root / 'out/started.json').exists())
        self.assertFalse((self.root / 'out/manifest.json').exists())

    def test_weight_mutation_during_export_rejected(self):
        original = exporter._snapshot
        calls = 0
        def snapshot(value):
            nonlocal calls
            calls += 1
            result = original(value)
            if calls == 1:
                self.states['backbone']['linear.weight'].add_(1)
            return result
        with patch.object(exporter, '_snapshot', side_effect=snapshot):
            with self.assertRaisesRegex(ValueError, 'mutated'):
                self.export()
        self.assertFalse((self.root / 'out/manifest.json').exists())

    def test_source_nested_output_forbidden(self):
        with self.assertRaises(FileExistsError):
            exporter.export_native(self.states, self.template, self.template / 'nested', provenance={})


if __name__ == '__main__':
    unittest.main()
