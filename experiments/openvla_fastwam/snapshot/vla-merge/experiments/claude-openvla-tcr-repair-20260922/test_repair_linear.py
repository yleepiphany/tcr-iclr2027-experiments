from __future__ import annotations

import importlib.util
from pathlib import Path
import types

import pytest
import torch

import linear_calibration as repair


def _old_module():
    path = Path(__file__).resolve().parent.parent / "openvla-tcr-20260921/linear_calibration.py"
    spec = importlib.util.spec_from_file_location("old_openvla_linear_calibration", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_r1_uniform_selector_is_byte_for_byte_old_behavior():
    old = _old_module()
    for count, cap in ((1, 8), (7, 8), (31, 8), (100, 3)):
        kwargs = dict(seed=274001, request_id="request/7",
                      module_name="backbone.language_model.layer", call_index=0)
        assert torch.equal(repair.row_indices(count, cap, **kwargs),
                           old.row_indices(count, cap, **kwargs))


def test_bias_ridge_formula_and_zero_delta_are_exact():
    module = torch.nn.Linear(3, 2, bias=True)
    prior = repair.affine_parameters(module)
    xs = [torch.cat((torch.arange(15).reshape(5, 3).float(), torch.ones(5, 1)), 1)] * 4
    result = repair.solve_module(
        module, xs, [prior.clone() for _ in range(4)], prior,
        mass_rule="uniform", ridge_multiplier=.05, max_correction_ratio=3.)
    trace = float(xs[0].double().square().sum() / len(xs[0]))
    assert result["ridge"] == pytest.approx(.05 * max(trace / 4, 1e-12))
    assert result["bias_augmented"] is True
    assert result["unclipped_correction_norm"] == 0
    assert torch.equal(repair.affine_parameters(module), prior)


def _fixed_ridge_report(feature_scale: float):
    module = torch.nn.Linear(3, 2, bias=False)
    prior = repair.affine_parameters(module)
    xs = [torch.eye(3) * feature_scale for _ in range(4)]
    experts = [prior + (index + 1) / 10 for index in range(4)]
    report = repair.solve_module(
        module, xs, experts, prior, mass_rule="uniform", ridge_multiplier=.05,
        max_correction_ratio=100., fixed_ridge=.123,
        ridge_source="fixed_from_same_candidate_pass_A")
    assert torch.isfinite(module.weight).all()
    return report


def test_pass_b_fixed_ridge_does_not_change_with_b_feature_scale():
    small = _fixed_ridge_report(1.)
    large = _fixed_ridge_report(10.)
    assert small["ridge"] == large["ridge"] == .123
    assert small["computed_ridge_for_this_feature_bank"] != \
        large["computed_ridge_for_this_feature_bank"]
    assert small["ridge_source"] == "fixed_from_same_candidate_pass_A"


def _trust_report(delta: float):
    module = torch.nn.Linear(2, 2, bias=False)
    prior = repair.affine_parameters(module)
    return repair.solve_module(
        module, [torch.eye(2)] * 4, [prior + delta] * 4, prior,
        mass_rule="uniform", ridge_multiplier=.01, max_correction_ratio=3.)


def test_trust_radius_scales_with_expert_delta_not_prior_norm():
    one = _trust_report(1.)
    two = _trust_report(2.)
    assert two["mean_expert_delta_norm"] == pytest.approx(
        2 * one["mean_expert_delta_norm"])
    assert two["trust_limit"] == pytest.approx(2 * one["trust_limit"])


class _Vision:
    def get_num_patches(self): return 2
    def get_num_images_in_input(self): return 2


class _Policy:
    model = types.SimpleNamespace(vision_backbone=_Vision())
    head = types.SimpleNamespace(action_dim=7)
    native = types.SimpleNamespace(NUM_ACTIONS_CHUNK=8)


def _native_request(mask_value=1):
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 29871]]),
        "attention_mask": torch.full((1, 5), mask_value),
        "proprio": torch.ones(1, 8),
        "use_film": False,
    }


def test_r2_is_deterministic_four_action_four_context_without_duplicates():
    request = _native_request()
    bounds = repair.native_action_bounds(_Policy(), request, 67)
    kwargs = dict(seed=9, request_id="q", module_name="backbone.language_model.x",
                  call_index=0)
    first = repair.action_context_row_indices(67, 8, bounds, **kwargs)
    second = repair.action_context_row_indices(67, 8, bounds, **kwargs)
    assert torch.equal(first, second)
    assert len(first) == len(set(first.tolist())) == 8
    start, stop = bounds
    assert sum(start <= int(row) < stop for row in first) == 4
    assert sum(not start <= int(row) < stop for row in first) == 4


def test_r2_matches_observed_native_605_row_layout():
    class Vision:
        def get_num_patches(self): return 256
        def get_num_images_in_input(self): return 2
    policy = _Policy()
    policy.model = types.SimpleNamespace(vision_backbone=Vision())
    request = {
        "input_ids": torch.cat((torch.arange(33), torch.tensor([29901]))).view(1, 34),
        "attention_mask": torch.ones(1, 34, dtype=torch.long),
        "proprio": torch.ones(8), "use_film": False,
    }
    # Actual accepted A/B receipts report 605 rows for this frozen native path:
    # 512 image patches + 1 proprio + 35 prompt + 56 actions + 1 stop.
    assert repair.native_action_bounds(policy, request, 605) == (547, 603)


def test_r2_rejects_padding_wrong_layout_and_insufficient_strata():
    with pytest.raises(ValueError, match="unpadded"):
        repair.native_action_bounds(_Policy(), _native_request(0), 67)
    with pytest.raises(ValueError, match="row count"):
        repair.native_action_bounds(_Policy(), _native_request(), 66)
    with pytest.raises(ValueError, match="Insufficient"):
        repair.action_context_row_indices(
            59, 8, (0, 56), seed=1, request_id="q",
            module_name="backbone.language_model.x", call_index=0)
