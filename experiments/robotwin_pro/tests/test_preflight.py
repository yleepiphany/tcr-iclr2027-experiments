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

    def test_regmean_native_smoke_pass_is_bound_and_not_formal(self):
        base=ROOT/'evidence/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925'
        path=base/'cpu-plan-v1/native-smoke.json'
        native=json.loads(path.read_text())
        complete=json.loads((base/'native-smoke-supervisor-v2/COMPLETE.json').read_text())
        self.assertEqual(native['status'],'PASS')
        self.assertEqual(smoke.digest(path),complete['native_smoke_receipt_sha256'])
        self.assertEqual(smoke.digest(path),'454a48b6e639020a0ec93604831cc2bb1d001cb076f29800c168197b2ca85e91')
        self.assertEqual({r['group'] for r in native['reports']},{'coordination','receptacle','precision'})
        for row in native['reports']:
            self.assertEqual(row['sampled_inputs_bitwise'],418)
            self.assertTrue(row['native_velocity_bitwise'])
            self.assertEqual(row['candidate_blocks_bitwise'],['vision0','language0','action0'])
            self.assertEqual((row['replica'],row['physical_timestep']),(0,1.))
        self.assertEqual(native['formal_episodes'],0)
        self.assertFalse(complete['materializer_started'])

    def test_regmean_materialize_sources_permit_and_launch_exact(self):
        records={x['source_path']:ROOT/x['path'] for x in smoke.package_preflight()['files']}
        base=ROOT/'evidence/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/materialize-supervisor-v1'
        path=base/'plan.json';plan=json.loads(path.read_text())
        for source,digest in {**plan['sources_sha256'],**plan['formal_binding']['source_files_sha256']}.items():
            self.assertEqual(smoke.digest(records[source]),digest)
        self.assertEqual(smoke.digest(records[plan['core_run']+'/plan.json']),plan['core_plan_sha256'])
        self.assertEqual(smoke.digest(records[plan['core_run']+'/native-smoke.json']),plan['native_smoke_sha256'])
        permit_path=base/'MATERIALIZE-EXECUTION-PERMIT.json'
        permit=json.loads(permit_path.read_text());launch=json.loads((base/'launch-receipt.json').read_text())
        started=json.loads((base/'STARTED.json').read_text())
        self.assertEqual(permit['stage'],'materialize')
        self.assertEqual(permit['supervisor_plan_sha256'],smoke.digest(path))
        self.assertEqual(permit['native_smoke_sha256'],plan['native_smoke_sha256'])
        self.assertEqual(launch['permit_sha256'],smoke.digest(permit_path))
        self.assertEqual(started['permit_sha256'],smoke.digest(permit_path))
        self.assertEqual(launch['plan_sha256'],smoke.digest(path))
        self.assertEqual(started['supervisor_plan_sha256'],smoke.digest(path))
        self.assertEqual(launch['pid'],started['pid'])
        self.assertEqual(launch['pid'],3270866)

    def test_regmean_materialize_snapshot_stays_unfinished_and_formal_bound(self):
        base=ROOT/'evidence/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/materialize-supervisor-v1'
        plan=json.loads((base/'plan.json').read_text())
        self.assertEqual((plan['stage'],plan['maximum_child_launches']),('materialize',1))
        self.assertFalse(plan['automatic_retry']);self.assertFalse(plan['automatic_formal_evaluation'])
        self.assertTrue(plan['original_kernel_and_67_stages_unchanged'])
        self.assertTrue(plan['resource_gate']['continuous_pass_fds'])
        self.assertEqual(plan['resource_gate']['minimum_free_mib'],71680)
        self.assertEqual(plan['acceptance']['saved_and_loaded_tensor_bitwise_checks'],813)
        panel=plan['formal_binding'];self.assertEqual((len(panel['jobs']),panel['episodes']),(9,540))
        self.assertIsNone(panel['model_sha256']);self.assertFalse(panel['autolaunch'])
        job=next(x for x in panel['jobs'] if x['group']=='receptacle' and x['repeat']==2)
        seeds=next(x['seeds'] for x in job['tasks'] if x['task_index']==27)
        self.assertIn(877886121,seeds);self.assertNotIn(437698954,seeds)
        snapshot=json.loads((ROOT/'evidence/ROBOTWIN-REGMEANPP-MATERIALIZE-SNAPSHOT.json').read_text())
        self.assertEqual(snapshot['snapshot']['native_smoke_status'],'PASS')
        self.assertFalse(snapshot['snapshot']['model_materialization_complete'])
        self.assertFalse(snapshot['snapshot']['model_accepted'])
        self.assertFalse(snapshot['snapshot']['formal_evaluation_started_by_materialize_queue'])
        self.assertFalse(snapshot['model_weights_included'])

if __name__ == '__main__':
    unittest.main()
