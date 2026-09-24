import run_development400 as runner


def test_confirmation_matrix_is_exact():
    jobs = runner.jobs_for(runner.Path("/tmp/development400-test"))
    assert len(jobs) == 8
    assert {(job["model"], job["suite"]) for job in jobs} == {
        (model, suite) for model in ("old_B", "R2") for suite in runner.SUITES}
    assert all(any(arg.endswith("selection-400.json") for arg in job["command"])
               for job in jobs)


def test_confirmation_is_development_only_and_keeps_reserve():
    assert runner.ADMISSION_MIB >= (
        runner.MEASURED_EVAL_REQUIREMENT_MIB + runner.RESERVE_MIB)
    assert runner.RUNTIME_FLOOR_MIB == 0
