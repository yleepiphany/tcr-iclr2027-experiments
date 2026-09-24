import unittest
import torch
from ridge import solve


class Regression(unittest.TestCase):
    def test_primal_dual_equivalence(self):
        generator = torch.Generator().manual_seed(71)
        xs = [torch.randn(n, 12, generator=generator, dtype=torch.float64) for n in (3, 4)]
        ds = [torch.randn(5, 12, generator=generator, dtype=torch.float64) for _ in xs]
        a, _ = solve(xs, ds, [.25, .75], .17, branch="primal")
        b, _ = solve(xs, ds, [.25, .75], .17, branch="dual")
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)

    def test_per_expert_mass_not_raw_row_count(self):
        x = torch.eye(3, dtype=torch.float64)
        ds = [torch.eye(3), torch.eye(3) * 2]
        a, _ = solve([x, x], ds, [.5, .5], .1)
        b, _ = solve([x.repeat(3, 1), x], ds, [.5, .5], .1)
        torch.testing.assert_close(a, b)

    def test_wide_module_identity(self):
        x = torch.ones(4, 28672)
        c, receipt = solve([x], [torch.zeros(7, 28672)], [1.], .01)
        self.assertEqual(receipt["system_dim"], 4)
        self.assertEqual(receipt["branch"], "dual")
        self.assertEqual(torch.count_nonzero(c).item(), 0)

    def test_reject_invalid(self):
        x, d = torch.ones(3, 4), torch.zeros(2, 4)
        for ridge in (0, -1, float("nan")):
            with self.assertRaises(ValueError): solve([x], [d], [1.], ridge)
        for masses in ([0], [-1], [float("nan")], [.7]):
            with self.assertRaises(ValueError): solve([x], [d], masses, .1)
        with self.assertRaises(ValueError): solve([x], [d], [1.], .1, max_system=2)
        with self.assertRaises(ValueError): solve([x * float("nan")], [d], [1.], .1)


if __name__ == "__main__":
    unittest.main()
