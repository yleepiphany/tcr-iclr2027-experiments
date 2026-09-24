import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import appendix_union_contract as union  # noqa: E402
from pi05_table3_contract import row_indices  # noqa: E402


def sample_rows(value, cap):
    return value.reshape(-1, value.shape[-1])[:cap]


def samples():
    return [{"flow_index": (0, 5, 9)[index % 3],
             "selected_request_slot": (index // 3) % 5}
            for index in range(300)]


def test_union_preserves_both_source_row_rules():
    metadata = samples()
    action = torch.arange(64, dtype=torch.float32).reshape(32, 2)
    vision = torch.arange(64, dtype=torch.float32).reshape(32, 2)
    select = lambda value, index, module: union.select_rows(
        value, index, module, metadata,
        sample_rows=sample_rows, row_indices=row_indices)
    assert sum(len(select(action, index, "model.action_in_proj"))
               for index in range(3)) == 30  # A: 10 per flow state
    assert sum(len(select(action, index, "model.action_in_proj"))
               for index in range(150, 153)) == 16  # B: 16 per request
    assert sum(len(select(vision, index, "model.vision_tower.layers.0.q"))
               for index in range(450, 459)) == 16  # B: 16 across 3 flows × 3 cameras
    time = torch.ones(1, 2)
    assert sum(len(select(time, index, "model.time_mlp_in"))
               for index in range(150, 153)) == 1


def test_union_rejects_incomplete_or_mixed_source_identity():
    with pytest.raises(ValueError, match="150 states from each source"):
        union.select_rows(torch.ones(1, 2), 150, "model.time_mlp_in", samples()[:-1],
                          sample_rows=sample_rows, row_indices=row_indices)
    assert union.source_for_state(149, is_vision=False)[0] == "A_new"
    assert union.source_for_state(150, is_vision=False)[0] == "B_new"
    assert union.source_for_state(449, is_vision=True)[0] == "A_new"
    assert union.source_for_state(450, is_vision=True)[0] == "B_new"
    with pytest.raises(ValueError, match="exceeds"):
        union.source_for_state(300, is_vision=False)


def test_union_source_paths_match_exact_a_then_b():
    cal = {}
    man = {}
    for suite in union.base.SUITES:
        roots = [union.base.INPUTS / pool / "inputs" / suite / "across"
                 for pool in union.POOLS]
        cal[suite] = [root / "replay.safetensors" for root in roots]
        man[suite] = [root / "replay.json" for root in roots]
    union.validate_sources(cal, man)
    cal["spatial"].reverse()
    with pytest.raises(ValueError, match="tensor sources differ"):
        union.validate_sources(cal, man)


def test_union_realized_budget_is_the_sum_of_a_and_b():
    import json

    fixed = json.loads(union.base.RIDGE.read_text())["modules"]
    names = list(union.base.SUITES)
    metrics = {}
    for module, source in fixed.items():
        first = (150 if module in union.TIME_MODULES else
                 4500 if ".vision_tower." in module else 1500)
        second = 50 if module in union.TIME_MODULES else 800
        metrics[module] = {"ridge": source["ridge"],
                           "rows_by_expert": {name: first + second for name in names}}
    assert union.validate_rows(metrics, names) == 5_772_800
    metrics[next(iter(metrics))]["rows_by_expert"][names[0]] -= 1
    with pytest.raises(ValueError, match="row budget differs"):
        union.validate_rows(metrics, names)
