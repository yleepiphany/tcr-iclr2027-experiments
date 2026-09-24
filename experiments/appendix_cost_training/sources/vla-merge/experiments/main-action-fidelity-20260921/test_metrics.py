import unittest
import torch
from metrics import continuous_metrics as metrics, task_macro


class ActionMetrics(unittest.TestCase):
    def test_identity(self):
        x = torch.ones(1, 50, 32)
        m = metrics(x, x, coordinate_scale=[1.] * 6)
        self.assertEqual(m['mse'], 0.)
        self.assertAlmostEqual(m['cosine'], 1.)
        self.assertEqual(m['continuous_coordinates'], 60)

    def test_unused_and_gripper_ignored(self):
        x, y = torch.ones(50, 32), torch.ones(50, 32)
        y[:, 6:] = 10
        y[10:, :6] = 20
        self.assertEqual(metrics(x, y, coordinate_scale=[1.] * 6)['mse'], 0.)

    def test_fixed_scale(self):
        x, y = torch.zeros(10, 7), torch.zeros(10, 7)
        x[:, 0] = 1
        self.assertAlmostEqual(metrics(x, y, coordinate_scale=[2., 1., 1., 1., 1., 1.])['mse'], 4/6)

    def test_zero_kept_for_mse_not_cos(self):
        x, y = torch.zeros(10, 7), torch.ones(10, 7)
        m = metrics(x, y, coordinate_scale=[1.] * 6)
        self.assertEqual(m['mse'], 1.)
        self.assertIsNone(m['cosine'])
        self.assertFalse(m['cosine_valid'])

    def test_fail_closed(self):
        x = torch.ones(10, 7)
        for scale in ([1.], [0.] * 6, [float('nan')] * 6):
            with self.assertRaises(ValueError): metrics(x, x, coordinate_scale=scale)
        for y in (torch.zeros(9, 7), torch.zeros(10, 8), x * float('inf')):
            with self.assertRaises(ValueError): metrics(x, y, coordinate_scale=[1.] * 6)

    def test_aggregation_and_missing(self):
        rows = [dict(suite='spatial', task_id=t, episode_id=0, request_id=i,
                     mse=float(t), cosine=1. if i == 0 else None, cosine_valid=i == 0)
                for t in (0, 1) for i in (0, 1)]
        expected = [('spatial', 0), ('spatial', 1)]
        out = task_macro(rows, expected_tasks=expected, requests_per_task=2)
        self.assertEqual(out['mse'], .5)
        self.assertEqual(out['cosine_valid_fraction'], .5)
        self.assertEqual(out['cosine'], 1.)
        for bad in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError): task_macro(bad, expected_tasks=expected, requests_per_task=2)
        rows[2]['cosine'], rows[2]['cosine_valid'] = None, False
        out = task_macro(rows, expected_tasks=expected, requests_per_task=2)
        self.assertIsNone(out['cosine'])
        self.assertFalse(out['cosine_tasks_complete'])


if __name__ == '__main__': unittest.main()
