from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit_eval_job import audit, sha


def test_strict_eval_audit_rejects_missing_or_wrong_reset(tmp_path: Path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"schema": "fastwam_tcr_eval_bank_v1", "jobs": [{
        "id": "formal-spatial-task00", "stage": "formal", "suite": "libero_spatial",
        "task_id": 0, "stock_indices": [1, 2], "reset_sha256": ["a", "b"],
        "seed": 7, "stats_sha256": "stats"}]}))
    output = tmp_path / "job"
    output.mkdir()
    (output / "started.json").write_text(json.dumps({
        "job_id": "formal-spatial-task00", "bank_sha256": sha(bank),
        "checkpoint_sha256": "model", "stats_sha256": "stats"}))
    rows = [{"job_id": "formal-spatial-task00", "stage": "formal",
             "suite": "libero_spatial", "task_id": 0, "stock_state_index": i,
             "state_sha256": h, "seed": 7, "success": bool(i == 1)}
            for i, h in ((1, "a"), (2, "b"))]
    episodes = output / "episodes.jsonl"
    episodes.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (output / "complete.json").write_text(json.dumps({
        "complete": True, "job_id": "formal-spatial-task00", "episodes": 2,
        "episodes_sha256": sha(episodes), "checkpoint_sha256": "model"}))
    assert audit(bank, output, "formal-spatial-task00", "model")["episode_count"] == 2

    episodes.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="digest"):
        audit(bank, output, "formal-spatial-task00", "model")

    rows[1]["state_sha256"] = "wrong"
    episodes.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    complete = json.loads((output / "complete.json").read_text())
    complete["episodes_sha256"] = sha(episodes)
    (output / "complete.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="identity"):
        audit(bank, output, "formal-spatial-task00", "model")


def test_frozen_bank_has_disjoint_development_formal_and_calibration_resets():
    root = Path(__file__).resolve().parents[3]
    base = root / "vla-merge-runtime/experiments/claude-fastwam-tcr-20260923"
    bank = json.loads((base / "evaluation-bank-v1.json").read_text())
    calibration = json.loads((base / "calibration-bank-v1.json").read_text())
    assert len(bank["jobs"]) == 80
    development = [h for j in bank["jobs"] if j["stage"] == "development"
                   for h in j["reset_sha256"]]
    formal = [h for j in bank["jobs"] if j["stage"] == "formal"
              for h in j["reset_sha256"]]
    calibration_hashes = [j["reset_sha256"] for j in calibration["jobs"]]
    assert len(development) == 40
    assert len(formal) == 400
    assert len(set(development + formal + calibration_hashes)) == 520
    assert {tuple(j["stock_indices"]) for j in bank["jobs"]
            if j["stage"] == "formal"} == {tuple(range(1, 11))}
