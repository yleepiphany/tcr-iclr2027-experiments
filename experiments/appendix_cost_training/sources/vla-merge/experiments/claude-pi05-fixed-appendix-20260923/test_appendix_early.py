import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tcr-mainline-local-20260919"))
import appendix_early_contract as early  # noqa: E402


def trace(policy: Path):
    samples = [{"prompt_signature": prompt, "selected_request_slot": slot,
                "request_index": slot, "flow_index": flow,
                "task_episode_index": 0, "vision_count": 3}
               for prompt in range(10) for slot in range(5) for flow in (0, 5, 9)]
    return {"task": "spatial", "repeat_id": "A_early",
            "calibration_policy": str(policy), "source_kind": "expert_execution",
            "table3_capture_version": 1, "table3_request_selection": "early",
            "sample_count": 150, "samples": samples, "request_mode": "initial",
            "max_calls_per_prompt": 1,
            "prompt_seen_call_counts": {str(prompt): 400 for prompt in range(10)},
            "selected_requests": {str(prompt): list(range(5)) for prompt in range(10)}}


def test_early_contract_accepts_true_first_five_and_rejects_across_labels(tmp_path):
    policy = tmp_path / "expert"
    good = trace(policy)
    early.validate_trace(good, policy, "spatial", 1)
    bad = json.loads(json.dumps(good))
    bad["samples"][10]["request_index"] = 13
    with pytest.raises(ValueError, match="native selection"):
        early.validate_trace(bad, policy, "spatial", 1)
    bad = trace(policy)
    bad["table3_request_selection"] = "across"
    with pytest.raises(ValueError, match="identity differs"):
        early.validate_trace(bad, policy, "spatial", 1)


def test_early_fixed_ridge_and_full_row_budget():
    fixed = json.loads(early.base.RIDGE.read_text())["modules"]
    names = list(early.base.SUITES)
    metrics = {}
    for module, source in fixed.items():
        first = (150 if module in early.base.TIME_MODULES else
                 4500 if ".vision_tower." in module else 1500)
        metrics[module] = {"ridge": source["ridge"],
                           "rows_by_expert": {name: first for name in names}}
    assert early.validate_rows(metrics, names, 1) == 4_441_200
    module = next(iter(metrics))
    metrics[module]["ridge"] *= 1.01
    with pytest.raises(ValueError, match="ridge map differs"):
        early.validate_rows(metrics, names, 1)
