"""CPU contracts; no model import or CUDA allocation."""
import copy
from types import SimpleNamespace as NS
import unittest
import numpy as np
from evaluate import state_for_episode, task_contract, validate_rows, rollout, sha256_state
from native_oft import validate_actions


def fixture():
    tasks = {}
    for i in range(10):
        states = np.arange(50, dtype=np.float64).reshape(10, 5) + i * 100
        tasks[("libero_spatial", i)] = NS(
            states=states, indices=tuple(range(10)), task_name=str(i),
            problem_folder="spatial", bddl_file=f"{i}.bddl",
            raw_sha256=tuple(sha256_state(s) for s in states))
    selection = NS(tasks=tasks, eval_seed=274001)
    rows = [dict(suite="libero_spatial", task_id=t, episode_index=e,
                 state_index=e, raw_state_sha256=tasks[("libero_spatial", t)].raw_sha256[e],
                 rollout_seed=274001 + e, valid=True, success=(e % 2 == 0))
            for t in range(10) for e in range(10)]
    return selection, rows


class Contracts(unittest.TestCase):
    def test_valid_rows(self):
        selection, rows = fixture()
        self.assertEqual(validate_rows(rows, selection, "libero_spatial"), 50)

    def test_reject_mutations(self):
        selection, rows = fixture()
        for key, value in (("suite", "libero_goal"), ("state_index", 2),
                           ("raw_state_sha256", "bad"), ("rollout_seed", 1),
                           ("valid", False), ("success", 1)):
            with self.subTest(key=key):
                mutated = copy.deepcopy(rows)
                mutated[0][key] = value
                with self.assertRaises(ValueError):
                    validate_rows(mutated, selection, "libero_spatial")
        for bad in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                validate_rows(bad, selection, "libero_spatial")

    def test_exact_reset_and_bounds(self):
        selection, _ = fixture()
        selected = selection.tasks[("libero_spatial", 0)]
        original = selected.states.copy()
        state, digest = state_for_episode(selected, 2)
        self.assertEqual(digest, selected.raw_sha256[2])
        state[0] += 1
        np.testing.assert_array_equal(original, selected.states)
        for invalid in (-1, 10, True, 1.0):
            with self.assertRaises(ValueError):
                state_for_episode(selected, invalid)
        selected.states[0, 0] += 1
        with self.assertRaises(ValueError):
            state_for_episode(selected, 0)

    def test_task_metadata(self):
        selection, _ = fixture()
        task = NS(name="0", problem_folder="spatial", bddl_file="0.bddl")
        task_contract(selection, "libero_spatial", 0, task)
        task.bddl_file = "wrong.bddl"
        with self.assertRaises(ValueError):
            task_contract(selection, "libero_spatial", 0, task)

    def test_action_shape_finite(self):
        validate_actions(np.zeros((8, 7)))
        for values in (np.zeros((10, 7)), np.full((8, 7), np.nan), np.full((8, 7), np.inf)):
            with self.assertRaises(ValueError):
                validate_actions(values)

    def test_rollout_uses_success_not_done_and_fails_closed(self):
        class Env:
            def __init__(self): self.steps = 0
            def seed(self, value): self.seed_value = value
            def reset(self): pass
            def set_init_state(self, state): return {}
            def step(self, action):
                self.steps += 1
                return {}, 0, self.steps == 11, {}
            def check_success(self): return False
        native = NS(set_seed_everywhere=lambda x: None,
                    get_libero_dummy_action=lambda _: [0] * 7,
                    prepare_observation=lambda obs, size: (obs, None),
                    process_action=lambda action, _: action)
        policy = NS(native=native, resize_size=224, request=lambda *args: np.zeros((8, 7)))
        env = Env()
        result = rollout(policy, env, NS(language="test"), np.zeros(5), 274001, 220)
        self.assertIs(result["success"], False)
        self.assertEqual(result["action_steps"], 1)
        self.assertEqual(env.seed_value, 274001)
        def fail(*args): raise RuntimeError("inference failed")
        policy.request = fail
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            rollout(policy, Env(), NS(language="test"), np.zeros(5), 274001, 220)


if __name__ == "__main__":
    unittest.main()
