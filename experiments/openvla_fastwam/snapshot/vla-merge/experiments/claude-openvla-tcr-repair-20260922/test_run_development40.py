import run_development40 as runner


def test_job_matrix_is_exact_and_fixed():
    jobs = runner.jobs_for(runner.Path("/tmp/development40-test"))
    assert len(jobs) == 20
    assert {(job["model"], job["suite"]) for job in jobs} == {
        (model, suite) for model in runner.MODELS for suite in runner.SUITES}
    assert all(any(arg.endswith("selection-40.json") for arg in job["command"])
               for job in jobs)
    assert all(job["checkpoint"] == str(runner.MODELS[job["model"]].resolve())
               for job in jobs)


def test_dynamic_cards_prefers_more_free_memory(monkeypatch):
    rows = {0: {"index": 0, "free_mib": 40_000},
            1: {"index": 1, "free_mib": 70_000}}
    monkeypatch.setattr(runner.formal, "gpu_row", lambda gpu: rows[gpu])
    assert list(runner.DynamicCards((0, 1))) == [1, 0]


def test_admission_preserves_requested_reserve():
    assert runner.ADMISSION_MIB >= (
        runner.MEASURED_EVAL_REQUIREMENT_MIB + runner.RESERVE_MIB)
    assert runner.RUNTIME_FLOOR_MIB == 0


def test_scientific_environment_contains_library_under_test():
    paths = runner.environment(0)["PYTHONPATH"].split(":")
    assert str(runner.VLA_MERGE_SRC) in paths
