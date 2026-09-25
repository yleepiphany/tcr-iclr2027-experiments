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

    def test_regmean_smoke_v2_binds_original_failure_and_sources(self):
        records={x['source_path']:ROOT/x['path'] for x in smoke.package_preflight()['files']}
        base=ROOT/'evidence/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925'
        path=base/'native-smoke-supervisor-v2/plan.json'
        plan=json.loads(path.read_text())
        self.assertEqual(smoke.digest(records[plan['core_run']+'/plan.json']),plan['core_plan_sha256'])
        for source,digest in {**plan['sources_sha256'],**plan['prior_pre_cuda_failure_files_sha256']}.items():
            self.assertEqual(smoke.digest(records[source]),digest)
        launch=json.loads((base/'native-smoke-supervisor-v2/launch-receipt.json').read_text())
        started=json.loads((base/'native-smoke-supervisor-v2/STARTED.json').read_text())
        self.assertEqual(launch['plan_sha256'],smoke.digest(path))
        self.assertEqual(started['supervisor_plan_sha256'],smoke.digest(path))
        self.assertEqual(launch['pid'],started['pid'])
        self.assertEqual(launch['formal_episodes'],0)

    def test_regmean_smoke_recovery_is_not_a_result_or_materializer(self):
        base=ROOT/'evidence/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925'
        plan=json.loads((base/'native-smoke-supervisor-v2/plan.json').read_text())
        self.assertEqual((plan['stage'],plan['maximum_child_launches']),('smoke',1))
        self.assertFalse(plan['materialize']);self.assertFalse(plan['automatic_retry'])
        self.assertFalse(plan['retry_after_gpu_entry']);self.assertEqual(plan['signals_sent'],0)
        race=json.loads((ROOT/'evidence/coordination/2026-09-25/robotwin-regmeanpp-smoke-admission-race-20260925.json').read_text())
        self.assertFalse(race['core_smoke_consumed']);self.assertFalse(race['native_smoke_result_exists'])
        self.assertEqual(race['formal_episodes'],0)
        self.assertIn('before CUDA',race['child_failure'])

if __name__ == '__main__':
    unittest.main()
