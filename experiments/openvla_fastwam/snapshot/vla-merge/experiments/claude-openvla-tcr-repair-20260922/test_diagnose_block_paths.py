import copy

import pytest
import torch

from diagnose_block_paths import compare_paths


class Policy:
    def __init__(self):
        self.prefix = torch.nn.Linear(2, 2, bias=False)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Sequential(
                torch.nn.Linear(2, 2, bias=False),
                torch.nn.ReLU(),
            ),
            torch.nn.Linear(2, 2, bias=False),
        )
        with torch.no_grad():
            self.prefix.weight.copy_(torch.eye(2))
            for layer in (self.head[0], self.head[2][0], self.head[3]):
                layer.weight.copy_(torch.eye(2))

    def linear_modules(self):
        return [("action_head.fc1", self.head[0]),
                ("action_head.residual", self.head[2][0]),
                ("action_head.fc2", self.head[3])]

    def replay(self, request):
        x = self.prefix(request["x"])
        x = self.head[1](self.head[0](x))
        x = x + self.head[2](x)
        return self.head[3](x)


def test_whole_expert_block_changes_later_solve_inputs_and_restores_state():
    live = Policy()
    expert = copy.deepcopy(live)
    with torch.no_grad():
        expert.prefix.weight.copy_(torch.eye(2) * 3)
        expert.head[0].weight.copy_(torch.eye(2) * 2)
    before = {name: value.clone() for name, value in live.head.state_dict().items()}
    result = compare_paths(live, expert, live.head, expert.head.state_dict(),
                           {"x": torch.tensor([[1.0, 2.0]])},
                           ["action_head.fc1", "action_head.residual", "action_head.fc2"])
    assert result["action_head.fc1"]["solve_vs_live"][0]["relative_mse"] == 0
    assert result["action_head.residual"]["solve_vs_live"][0]["relative_mse"] > 0
    assert result["action_head.fc2"]["solve_vs_live"][0]["relative_mse"] > 0
    assert result["action_head.fc1"]["expert_vs_solve"][0]["relative_mse"] > 0
    assert all(torch.equal(before[name], value) for name, value in live.head.state_dict().items())


def test_failed_replay_cleans_hooks_and_restores_state():
    live = Policy()
    expert = copy.deepcopy(live)
    before = {name: value.clone() for name, value in live.head.state_dict().items()}
    with pytest.raises(ValueError, match="Missing named Linear"):
        compare_paths(live, expert, live.head, expert.head.state_dict(),
                      {"x": torch.ones(1, 2)}, ["missing"])
    assert all(torch.equal(before[name], value) for name, value in live.head.state_dict().items())
    assert not live.head[0]._forward_pre_hooks
