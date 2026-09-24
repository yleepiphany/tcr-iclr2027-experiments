import copy
from pathlib import Path

import build_ordered_candidate as builder
import pass_engine_ordered as engine
from materialize_soup import sha
from test_repair_pass_engine import _fixture


def test_ordered_two_pass_keeps_full_scope_and_reuses_its_own_a_ridge(tmp_path):
    policy, bank, requests_a, plan, template = _fixture(tmp_path)
    recipe_a = builder.recipe("R2_aux_ordered", "A", {"test": True}) | {"cap": 2}
    result_a = engine.run_pass(policy, bank, requests_a, plan, recipe_a,
                               template=template, run=tmp_path / "A")
    assert result_a["auxiliary_replay"] == "ordered_current_linear_expert_only"
    assert result_a["modules"] == plan["linear_count"]
    assert result_a["rows"] == plan["planned_rows_per_pass_if_all_linears_solved"]
    assert result_a["ridge_map"]

    requests_b = copy.deepcopy(requests_a)
    for rows in requests_b.values():
        rows[0]["id"] = rows[0]["id"].replace("/A/", "/B/")
    source_path = tmp_path / "A/manifest.json"
    source = {"candidate": "R2_aux_ordered", "pass_id": "A",
              "manifest": str(source_path), "manifest_sha256": sha(source_path)}
    recipe_b = builder.recipe("R2_aux_ordered", "B", {"test": True}) | {"cap": 2}
    result_b = engine.run_pass(policy, bank, requests_b, plan, recipe_b,
                               template=Path(result_a["checkpoint"]), run=tmp_path / "B",
                               fixed_ridges=result_a["ridge_map"], fixed_ridge_source=source)
    assert result_b["ridge_map"] == result_a["ridge_map"]
    assert result_b["auxiliary_replay"] == "ordered_current_linear_expert_only"
    assert result_b["rows"] == result_a["rows"]
    assert result_b["modules"] == result_a["modules"]
