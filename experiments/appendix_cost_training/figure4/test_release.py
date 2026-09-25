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

if __name__=='__main__':unittest.main()
