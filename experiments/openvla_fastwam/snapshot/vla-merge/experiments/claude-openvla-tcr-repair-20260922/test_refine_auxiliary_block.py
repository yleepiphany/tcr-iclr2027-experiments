import copy

import pytest
import torch

import refine_auxiliary_block as refined
from materialize_soup import ORDER


class ResidualPolicy:
    def __init__(self):
        self.model = torch.nn.Identity()
        self.proprio = torch.nn.Identity()
        self.head = torch.nn.ModuleDict({
            "fc1": torch.nn.Linear(2, 2, bias=False),
            "residual": torch.nn.Linear(2, 2, bias=False),
            "fc2": torch.nn.Linear(2, 2, bias=False),
        })
        with torch.no_grad():
            for layer in self.head.values():
                layer.weight.copy_(torch.eye(2))

    def linear_modules(self):
        return [("action_head." + name, layer) for name, layer in self.head.items()]

    def replay(self, request):
        x = torch.relu(self.head["fc1"](request["x"]))
        x = x + torch.relu(self.head["residual"](x))
        return self.head["fc2"](x)


class Bank:
    def __init__(self, policy):
        self.states = {}
        for index, expert in enumerate(ORDER):
            state = copy.deepcopy(policy.head.state_dict())
            state["fc1.weight"] = torch.eye(2) * (2 + index)
            self.states[expert] = state

    def block_state(self, expert, block_name, block):
        return self.states[expert], []


def _requests():
    return {expert: [{"id": expert + "/A/0", "inputs": {
        "x": torch.tensor([[1.0, 2.0], [2.0, 1.0]]),
        "attention_mask": torch.ones(1, 2)}}] for expert in ORDER}


def test_ordered_replay_uses_solved_prefix_inside_residual(monkeypatch):
    policy = ResidualPolicy()
    seen = {}

    def fake_solve(module, xs, experts, prior, **kwargs):
        name = next(name for name, layer in policy.head.items() if layer is module)
        seen[name] = [x.clone() for x in xs]
        if name == "fc1":
            with torch.no_grad():
                module.weight.copy_(torch.eye(2) * 5)
        return {"ridge": 0.1}

    monkeypatch.setattr(refined, "solve_module", fake_solve)
    names = ["action_head.fc1", "action_head.residual", "action_head.fc2"]
    result = refined.calibrate_auxiliary_block_ordered(
        policy, Bank(policy), "action_head", _requests(),
        module_names=list(reversed(names)),
        expected_calls=dict.fromkeys(names, 1),
        expected_rows_per_expert=dict.fromkeys(names, 2),
        cap=2, seed=3407, mass_rule="relative",
        ridge_multiplier=0.05, max_correction_ratio=3.0)
    assert result["actual_linear_order"] == names
    assert result["candidate_only"] is True
    # The second layer sees the already merged fc1=5, not any expert fc1=2..5.
    expected = torch.tensor([[5.0, 10.0], [10.0, 5.0]])
    for x in seen["residual"]:
        assert torch.equal(x, expected)
    # The last layer sees the live residual sum, including its nonlinear path.
    for x in seen["fc2"]:
        assert torch.equal(x, expected * 2)


def test_failed_solve_rolls_back_complete_block(monkeypatch):
    policy = ResidualPolicy()
    before = {key: value.clone() for key, value in policy.head.state_dict().items()}

    def fail_after_first(module, xs, experts, prior, **kwargs):
        with torch.no_grad():
            module.weight.fill_(9)
        raise RuntimeError("solver failure")

    monkeypatch.setattr(refined, "solve_module", fail_after_first)
    names = ["action_head.fc1", "action_head.residual", "action_head.fc2"]
    with pytest.raises(RuntimeError, match="solver failure"):
        refined.calibrate_auxiliary_block_ordered(
            policy, Bank(policy), "action_head", _requests(),
            module_names=names, expected_calls=dict.fromkeys(names, 1),
            expected_rows_per_expert=dict.fromkeys(names, 2),
            cap=2, seed=3407, mass_rule="relative",
            ridge_multiplier=0.05, max_correction_ratio=3.0)
    assert all(torch.equal(before[key], value) for key, value in policy.head.state_dict().items())


def test_actual_small_ridge_solver_commits_each_layer_in_native_order():
    policy = ResidualPolicy()
    names = ["action_head.fc1", "action_head.residual", "action_head.fc2"]
    result = refined.calibrate_auxiliary_block_ordered(
        policy, Bank(policy), "action_head", _requests(),
        module_names=names, expected_calls=dict.fromkeys(names, 1),
        expected_rows_per_expert=dict.fromkeys(names, 2),
        cap=2, seed=3407, mass_rule="relative",
        ridge_multiplier=0.05, max_correction_ratio=3.0)
    assert result["complete"] is True
    assert result["actual_linear_order"] == names
    assert all(result["modules"][name]["ridge"] > 0 for name in names)
    assert not torch.equal(policy.head["fc1"].weight, torch.eye(2))


def test_repeated_calls_are_counted_and_interleaved_feedback_rejected():
    class Repeated(ResidualPolicy):
        def replay(self, request):
            x = self.head["fc1"](request["x"])
            x = self.head["fc1"](x)
            return self.head["fc2"](x)

    policy = Repeated()
    names = ["action_head.fc1", "action_head.fc2"]
    assert refined.native_linear_order(policy, [row[0]["inputs"] for row in _requests().values()],
                                       policy.head, names,
                                       {"action_head.fc1": 2, "action_head.fc2": 1}) == names

    class Interleaved(ResidualPolicy):
        def replay(self, request):
            x = self.head["fc1"](request["x"])
            x = self.head["fc2"](x)
            return self.head["fc1"](x)

    policy = Interleaved()
    with pytest.raises(ValueError, match="Interleaved"):
        refined.native_linear_order(policy, [row[0]["inputs"] for row in _requests().values()],
                                    policy.head, names,
                                    {"action_head.fc1": 2, "action_head.fc2": 1})
