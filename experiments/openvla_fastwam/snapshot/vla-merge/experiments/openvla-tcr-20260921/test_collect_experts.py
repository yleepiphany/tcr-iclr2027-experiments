from types import SimpleNamespace
import unittest
import numpy as np
from collect_experts import capture_rollout, request_indices


class Env:
    def __init__(self, terminal_step=None, success=False):
        self.terminal_step, self.success = terminal_step, success
        self.steps, self.actions, self.seeds = 0, [], []
    def seed(self, seed): self.seeds.append(seed)
    def reset(self): self.steps = 0
    def set_init_state(self, state): self.state = state.copy(); return 'observation'
    def step(self, action):
        self.steps += 1
        self.actions.append(action)
        done = self.terminal_step is not None and self.steps == self.terminal_step + 10
        return 'observation', 0, done, {}
    def check_success(self):
        return self.success and self.terminal_step is not None and self.steps >= self.terminal_step + 10


class Policy:
    def __init__(self):
        self.resize_size, self.calls, self.seeds = 224, [], []
        self.native = SimpleNamespace(set_seed_everywhere=self.seeds.append,
            get_libero_dummy_action=lambda _: [0] * 7,
            prepare_observation=lambda obs, size: (obs, None),
            process_action=lambda action, _: action)
    def request(self, observation, instruction, capture):
        self.calls.append((instruction, capture))
        actions = np.full((8, 7), len(self.calls), dtype=float)
        return actions, {'request_id': len(self.calls)}


class CaptureTests(unittest.TestCase):
    def test_quantiles_unique_and_endpoints(self):
        for count in range(5, 200):
            selected = request_indices(count)
            self.assertEqual(len(set(selected)), 5)
            self.assertEqual(selected[0], 0)
            self.assertEqual(selected[-1], count - 1)
            self.assertEqual(selected, sorted(selected))
        for count in (-1, 0, 1, 4):
            with self.assertRaises(ValueError): request_indices(count)

    def test_native_chunk_seed_warmup_and_failure_kept(self):
        policy, env = Policy(), Env()
        state = np.array([.1, .2])
        records, result = capture_rollout(policy, env, SimpleNamespace(language='task'), state, 55, 42)
        self.assertFalse(result['success_diagnostic_only'])
        self.assertEqual(result['action_steps'], 42)
        self.assertEqual([r['action_step'] for r in records], [0, 8, 16, 24, 32, 40])
        self.assertEqual(request_indices(len(records)), [0, 1, 2, 3, 5])
        self.assertEqual(policy.seeds, [55]); self.assertEqual(env.seeds, [55])
        self.assertEqual(env.actions[:10], [[0] * 7] * 10)
        self.assertEqual(env.actions[10:18], [[1.] * 7] * 8)
        np.testing.assert_array_equal(env.state, state)

    def test_success_not_used_for_selection(self):
        selections = []
        for success in (False, True):
            records, result = capture_rollout(Policy(), Env(terminal_step=35, success=success),
                SimpleNamespace(language='task'), np.ones(3), 9, 100)
            self.assertEqual(result['success_diagnostic_only'], success)
            self.assertEqual(result['action_steps'], 35)
            selections.append(request_indices(len(records)))
        self.assertEqual(selections[0], selections[1])

    def test_short_episode_cannot_fabricate_five_requests(self):
        records, result = capture_rollout(Policy(), Env(terminal_step=8, success=True),
            SimpleNamespace(language='task'), np.ones(3), 9, 100)
        self.assertEqual(result['requests'], 1)
        with self.assertRaises(ValueError): request_indices(len(records))


if __name__ == '__main__':
    unittest.main()
