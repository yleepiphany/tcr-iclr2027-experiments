from pathlib import Path
from types import SimpleNamespace

import pytest

import robotwin_solver_contract as contract


def args(**updates):
    values = {**contract.COMMON, "max_rows_per_sample": 10,
              "expert_loss_normalization": "prior", "prior_model": Path("prior")}
    values.update(updates)
    return SimpleNamespace(**values)


def test_m3_pass_recipes_are_distinct_and_strict(monkeypatch):
    monkeypatch.setenv("ROBOTWIN_TCR_PASS", "A")
    contract.validate_recipe(args())
    with pytest.raises(ValueError, match="max_rows_per_sample"):
        contract.validate_recipe(args(max_rows_per_sample=16))
    monkeypatch.setenv("ROBOTWIN_TCR_PASS", "B")
    contract.validate_recipe(args(max_rows_per_sample=16, expert_loss_normalization="none"))
    with pytest.raises(ValueError, match="expert_loss_normalization"):
        contract.validate_recipe(args(max_rows_per_sample=16))


@pytest.mark.parametrize("which,cap,total", [
    ("A", 10, 3_330_900),
    ("B", 16, 5_328_900),
])
def test_realized_rows_use_three_experts(monkeypatch, which, cap, total):
    monkeypatch.setenv("ROBOTWIN_TCR_PASS", which)
    metrics = {}
    for suffix in ("in", "out"):
        metrics[f"model.time_mlp_{suffix}"] = {
            "rows_by_expert": {name: 150 for name in contract.NAMES}}
    for index in range(162):
        metrics[f"model.vision_tower.layer.{index}"] = {
            "rows_by_expert": {name: 450 * cap for name in contract.NAMES}}
    for index in range(254):
        metrics[f"model.other.{index}"] = {
            "rows_by_expert": {name: 150 * cap for name in contract.NAMES}}
    assert contract.validate_realized_rows(metrics) == total


def test_pass_b_requires_frozen_ridge_start_point():
    config = {
        "schema": "claude_second_round_v1", "arm": "robotwin_b", "repeat": 1,
        "row_cap_per_request_module": 16, "expert_masses": "uniform_three",
        "merged_slots": [], "start_point": "a", "start_point_manifest": "b",
        "start_point_sha256": "c",
    }
    contract.validate_second_round_config(config)
    broken = dict(config, row_cap_per_request_module=10)
    with pytest.raises(ValueError, match="row_cap"):
        contract.validate_second_round_config(broken)


def test_pass_b_accepts_exactly_three_independent_repeat_ids():
    config = {
        "schema": "claude_second_round_v1", "arm": "robotwin_b",
        "row_cap_per_request_module": 16, "expert_masses": "uniform_three",
        "merged_slots": [], "start_point": "a", "start_point_manifest": "b",
        "start_point_sha256": "c",
    }
    for repeat in (1, 2, 3):
        contract.validate_second_round_config({**config, "repeat": repeat})
    for repeat in (0, 4, True):
        with pytest.raises(ValueError, match="three frozen repeats"):
            contract.validate_second_round_config({**config, "repeat": repeat})
