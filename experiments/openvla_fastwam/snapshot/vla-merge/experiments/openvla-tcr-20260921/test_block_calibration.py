import unittest
from unittest.mock import patch
import torch

import block_calibration as bc
from materialize_soup import ORDER


class Policy:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2, bias=False)
        self.head = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        self.proprio = torch.nn.Identity()
        self.head.register_buffer('constant', torch.tensor([13]))
        with torch.no_grad():
            self.model.weight.copy_(torch.eye(2) * 2)
            for module in self.head:
                module.weight.copy_(torch.eye(2))
                module.bias.zero_()
        self.calls = []

    def linear_modules(self):
        return [('backbone', self.model), ('action_head.0', self.head[0]), ('action_head.1', self.head[1])]

    def replay(self, request):
        self.calls.append((self.head[0].weight.detach().clone(), self.head[1].weight.detach().clone()))
        with torch.no_grad():
            return self.head(self.model(request['x']))


class Bank:
    def __init__(self, policy):
        self.states = {}
        for i, expert in enumerate(ORDER):
            state = {key: value.clone() for key, value in policy.head.state_dict().items()}
            state['0.weight'] = torch.eye(2) * (2 + i)
            state['1.weight'] = torch.eye(2) * (3 + i)
            self.states[expert] = state

    def block_state(self, expert, name, block):
        return self.states[expert], []


class BlockTests(unittest.TestCase):
    def setUp(self):
        self.policy = Policy()
        self.bank = Bank(self.policy)
        self.saved = {key: value.clone() for key, value in self.policy.head.state_dict().items()}
        self.requests = {expert: [{'id': expert + '/request-0',
                                  'inputs': {'x': torch.eye(2), 'attention_mask': torch.ones(1, 2)}}]
                         for expert in ORDER}
        self.kwargs = dict(module_names=['action_head.0', 'action_head.1'],
                           expected_calls={'action_head.0': 1, 'action_head.1': 1},
                           expected_rows_per_expert={'action_head.0': 2, 'action_head.1': 2},
                           cap=2, seed=100, mass_rule='uniform', ridge_multiplier=.01,
                           max_correction_ratio=10.)

    def run_block(self):
        return bc.calibrate_block(self.policy, self.bank, 'action_head', self.requests, **self.kwargs)

    def assert_restored(self):
        for key, value in self.saved.items():
            self.assertTrue(torch.equal(value, self.policy.head.state_dict()[key]), key)

    def test_success_collects_all_before_any_solve(self):
        original = bc.solve_module
        inputs = []
        def solve(*args, **kwargs):
            self.assertEqual(len(self.policy.calls), 4)
            inputs.append([x.clone() for x in args[1]])
            return original(*args, **kwargs)
        with patch.object(bc, 'solve_module', side_effect=solve):
            result = self.run_block()
        self.assertTrue(result['complete'])
        self.assertFalse(result['is_complete_model'])
        self.assertEqual(result['collected_rows'], 16)
        self.assertTrue(torch.equal(self.policy.model.weight, torch.eye(2) * 2))
        self.assertEqual(self.policy.head.constant.item(), 13)
        for i in range(4):
            torch.testing.assert_close(inputs[0][i][:, :2], torch.eye(2) * 2)
            torch.testing.assert_close(inputs[1][i][:, :2], torch.eye(2) * 2 * (2 + i))
        self.assertFalse(torch.equal(self.saved['0.weight'], self.policy.head[0].weight))
        self.assertFalse(torch.equal(self.saved['1.weight'], self.policy.head[1].weight))

    def test_second_solve_failure_rolls_back_first(self):
        original = bc.solve_module
        count = 0
        def solve(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                self.assertFalse(torch.equal(self.policy.head[0].weight, self.saved['0.weight']))
                raise RuntimeError('injected second solve failure')
            return original(*args, **kwargs)
        with patch.object(bc, 'solve_module', side_effect=solve):
            with self.assertRaisesRegex(RuntimeError, 'second solve failure'):
                self.run_block()
        self.assert_restored()

    def test_capture_failure_restores_state_and_hooks(self):
        with patch.object(self.policy, 'replay', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_block()
        self.assert_restored()
        self.assertTrue(all(not module._forward_pre_hooks for module in self.policy.head))

    def test_wrong_rows_reject_before_solve(self):
        self.kwargs['expected_rows_per_expert']['action_head.1'] = 3
        with patch.object(bc, 'solve_module') as solve:
            with self.assertRaisesRegex(ValueError, 'Realized rows'):
                self.run_block()
            solve.assert_not_called()
        self.assert_restored()

    def test_duplicate_request_rejected(self):
        self.requests[ORDER[0]] *= 2
        with self.assertRaisesRegex(ValueError, 'Duplicate request'):
            self.run_block()
        self.assertEqual(len(self.policy.calls), 0)
        self.assert_restored()


if __name__ == '__main__':
    unittest.main()
