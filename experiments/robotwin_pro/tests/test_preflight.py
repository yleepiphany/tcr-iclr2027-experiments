import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_smoke', ROOT/'smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((ROOT/'config/catalog.json').read_text())

    def test_exact_source_provenance(self):
        self.assertGreater(len(smoke.package_preflight()['files']), 50)

    def test_stale_regmeanpp_is_rejected(self):
        key = 'libero_pro/regmean_pp'
        path = ROOT/'evidence'/self.catalog['methods'][key]['selection_manifest']
        with self.assertRaisesRegex(smoke.Blocked, 'Stale'):
            smoke.validate_current_selection(key, json.loads(path.read_text()), self.catalog)

    def test_stale_featcal_is_rejected(self):
        key = 'libero_pro/featcal'
        path = ROOT/'evidence'/self.catalog['methods'][key]['selection_manifest']
        with self.assertRaisesRegex(smoke.Blocked, 'Stale'):
            smoke.validate_current_selection(key, json.loads(path.read_text()), self.catalog)

    def test_current_hash_without_full_reset_manifest_is_rejected(self):
        key = 'libero_pro/regmean_pp'
        row = {'model': {'sha256': self.catalog['methods'][key]['current_checkpoint_sha256']}}
        with self.assertRaisesRegex(smoke.Blocked, 'three-repeat'):
            smoke.validate_current_selection(key, row, self.catalog)

    def test_new_model_hash_cannot_unblock_archived_runner(self):
        key = 'libero_pro/regmean_pp'
        result = smoke.method_preflight(key, self.catalog['methods'][key], self.catalog, None)
        self.assertEqual(result['status'], 'BLOCKED_STALE_SELECTION')

    def test_robotwin_cannot_substitute_libero_model(self):
        for key in ('robotwin/regmean_pp', 'robotwin/featcal'):
            with self.assertRaisesRegex(smoke.Blocked, 'No accepted checkpoint'):
                smoke.validate_current_selection(key, {}, self.catalog)

if __name__ == '__main__':
    unittest.main()
