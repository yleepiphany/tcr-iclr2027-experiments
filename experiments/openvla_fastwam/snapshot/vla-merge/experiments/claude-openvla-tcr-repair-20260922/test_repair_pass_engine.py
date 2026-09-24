from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import build_candidate
import pass_engine as engine
from materialize_soup import ORDER, SHARED, sha
from native_block_plan import make_plan


class Policy:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2, bias=False)
        self.head = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        self.proprio = torch.nn.Identity()
        self.head.register_buffer("constant", torch.tensor([13]))
        with torch.no_grad():
            self.model.weight.copy_(torch.eye(2) * 2)
            for module in self.head:
                module.weight.copy_(torch.eye(2)); module.bias.zero_()
        self.calls = []

    def linear_modules(self):
        return [("backbone", self.model), ("action_head.0", self.head[0]),
                ("action_head.1", self.head[1])]

    def replay(self, request):
        self.calls.append(1)
        with torch.no_grad():
            return self.head(self.model(request["x"]))


class Bank:
    def __init__(self, policy):
        self.states = {}
        for index, expert in enumerate(ORDER):
            state = {key: value.clone() for key, value in policy.head.state_dict().items()}
            state["0.weight"] = torch.eye(2) * (2 + index)
            state["1.weight"] = torch.eye(2) * (3 + index)
            self.states[expert] = state

    def block_state(self, expert, name, block):
        return self.states[expert], []


def _template(path: Path, policy: Policy):
    path.mkdir()
    state = policy.model.state_dict()
    save_file(state, str(path / "part.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {key: "part.safetensors" for key in state}, "metadata": {}}))
    for name in (*SHARED, "dataset_statistics.json"):
        (path / name).write_text("{}")
    for name, module in (("action_head", policy.head), ("proprio_projector", policy.proprio)):
        torch.save({"module." + key: value for key, value in module.state_dict().items()},
                   str(path / (name + "--prior_checkpoint.pt")))


def _recipe(candidate, pass_id):
    return build_candidate.recipe(candidate, pass_id, {"test": True}) | {
        "cap": 2,
    }


def _fixture(tmp_path):
    policy = Policy(); bank = Bank(policy)
    requests = {expert: [{"id": expert + "/A/0",
                          "inputs": {"x": torch.eye(2),
                                     "attention_mask": torch.ones(1, 2)}}]
                for expert in ORDER}
    trace = [{"name": "action_head." + str(index), "input_shape": [1, 2, 2],
              "weight_shape": [2, 2]} for index in (0, 1)]
    plan = make_plan([trace] * 4, cap=2, requests_per_expert=1)
    template = tmp_path / "prior"; _template(template, policy)
    return policy, bank, requests, plan, template


def test_two_pass_reuses_exact_same_candidate_a_ridges(tmp_path):
    policy, bank, requests, plan, template = _fixture(tmp_path)
    pass_a = engine.run_pass(policy, bank, requests, plan, _recipe("R1", "A"),
                             template=template, run=tmp_path / "A")
    requests_b = copy.deepcopy(requests)
    for rows in requests_b.values():
        rows[0]["id"] = rows[0]["id"].replace("/A/", "/B/")
        rows[0]["inputs"]["x"] *= 10
    source_path = tmp_path / "A/manifest.json"
    source = {"candidate": "R1", "pass_id": "A", "manifest": str(source_path),
              "manifest_sha256": sha(source_path)}
    pass_b = engine.run_pass(
        policy, bank, requests_b, plan, _recipe("R1", "B"),
        template=Path(pass_a["checkpoint"]), run=tmp_path / "B",
        fixed_ridges=pass_a["ridge_map"], fixed_ridge_source=source)
    assert pass_b["ridge_map"] == pass_a["ridge_map"]
    assert pass_b["ridge_source"] == "fixed_from_same_candidate_pass_A"
    assert pass_b["fixed_ridge_source"] == source
    assert pass_b["numerical_contract"]["solve_dtype"] == "float64"


def test_pass_b_rejects_incomplete_or_cross_candidate_ridge_source(tmp_path):
    policy, bank, requests, plan, template = _fixture(tmp_path)
    pass_a = engine.run_pass(policy, bank, requests, plan, _recipe("R1", "A"),
                             template=template, run=tmp_path / "A")
    source_path = tmp_path / "A/manifest.json"
    source = {"candidate": "R1", "pass_id": "A", "manifest": str(source_path),
              "manifest_sha256": sha(source_path)}
    with pytest.raises(ValueError, match="full-module ridge map"):
        engine.run_pass(policy, bank, requests, plan, _recipe("R1", "B"),
                        template=Path(pass_a["checkpoint"]), run=tmp_path / "bad-map",
                        fixed_ridges={}, fixed_ridge_source=source)
    with pytest.raises(ValueError, match="same-candidate"):
        engine.run_pass(policy, bank, requests, plan, _recipe("R2", "B"),
                        template=Path(pass_a["checkpoint"]), run=tmp_path / "cross",
                        fixed_ridges=pass_a["ridge_map"], fixed_ridge_source=source)


def test_candidate_recipe_freezes_only_row_mode_difference():
    r1 = build_candidate.recipe("R1", "A", {"x": 1})
    r2 = build_candidate.recipe("R2", "A", {"x": 1})
    changed = {key for key in r1 if r1[key] != r2[key]}
    assert changed == {"candidate", "row_mode"}
    assert r1["row_mode"] == "uniform"
    assert r2["row_mode"] == "llm_action_4_context_4"

