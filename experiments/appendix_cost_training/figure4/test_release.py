"""Portable CPU checks for the exact Figure 4 release snapshots."""
import unittest
import check

class ReleaseTests(unittest.TestCase):
    def test_snapshot_and_frozen_plans(self):
        result=check.check_release()
        self.assertEqual(result['status'],'PASS')
        self.assertFalse(result['gpu_or_training_started'])

    def test_regmean_37_function_parity(self):
        self.assertEqual(check.source_parity()['unchanged_functions'],37)

    def test_ties_two_three_expert_oracle(self):
        self.assertEqual(check.test_ties_oracle()['independent_numpy_oracle'],'bitwise_equal')

    def test_regmean_subset_quota_and_exclusion(self):
        self.assertEqual(check.test_regmean_contract()['rows'],{'2':1184200,'3':1776300})

    def test_featcal_backend_permit_and_stage_matrix(self):
        result=check.test_featcal_queue()
        self.assertEqual((result['stage_jobs'],result['formal_episodes']),(14,500))
        self.assertEqual(result['build_tf32_override'],'1')
        self.assertIsNone(result['formal_tf32_override'])

    def test_featcal_v3_exact_recovery_and_consumed_launch_proof(self):
        result=check.test_featcal_recovery()
        self.assertEqual((result['new_stage_jobs'],result['new_formal_episodes'],result['reused_spatial_episodes']),(9,400,100))
        self.assertEqual(result['reused_spatial_successes'],88)
        self.assertFalse(result['gpu_or_queue_started_by_portable_check'])

if __name__=='__main__':unittest.main()
