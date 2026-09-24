from unittest.mock import patch

import run_candidates as runner


def test_resource_rule_uses_measured_peak_plus_eight_gib():
    assert runner.RESERVE_MIB == 8 * 1024
    assert runner.MEASURED_REQUIREMENT_MIB == 30_228
    assert runner.ADMISSION_MIB >= runner.MEASURED_REQUIREMENT_MIB + runner.RESERVE_MIB
    assert runner.RUNTIME_FLOOR_MIB == 0


def test_dynamic_cards_rescan_and_rank_all_candidates():
    rows = {0: {"index": 0, "free_mib": 50_000}, 1: {"index": 1, "free_mib": 70_000}}
    with patch.object(runner.formal, "gpu_row", side_effect=lambda gpu: rows[gpu]):
        cards = runner.DynamicCards((0, 1))
        assert list(cards) == [1, 0]
        rows[0]["free_mib"] = 75_000
        assert list(cards) == [0, 1]


def test_environment_binds_selected_card():
    with patch.object(runner.base, "environment", return_value={"PYTHONPATH": "base"}):
        for gpu in runner.GPUS:
            env = runner.environment(gpu)
            assert env["CUDA_VISIBLE_DEVICES"] == str(gpu)
            assert env["ITERATION_PHYSICAL_GPU"] == str(gpu)
            assert str(runner.HERE) in env["PYTHONPATH"]


def test_plan_uses_shared_queue_evaluator_identity_field():
    source = runner.Path(runner.__file__).read_text()
    assert '"evaluator_sha256": formal.sha(HERE / "build_candidate.py")' in source
    assert '"worker_sha256"' not in source
