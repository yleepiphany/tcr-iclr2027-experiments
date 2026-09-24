import run_repair_formal1200 as runner


def test_formal_matrix_is_exact():
    jobs = runner.jobs_for(runner.Path("/tmp/R2-formal-test"))
    assert len(jobs) == 12
    assert {(job["repeat"], job["suite"]) for job in jobs} == {
        (repeat, suite) for repeat in runner.REPEATS for suite in runner.SUITES}
    assert all(job["checkpoint"] if "checkpoint" in job else
               str(runner.CHECKPOINT) in job["command"] for job in jobs)


def test_repeat_selections_and_seeds_are_distinct():
    contracts = [runner.repeat_contract(repeat) for repeat in runner.REPEATS]
    assert [row[0] for row in contracts] == ["repeat-01", "repeat-02", "repeat-03"]
    assert [row[1] for row in contracts] == [274001, 274002, 274003]
    assert len({row[2] for row in contracts}) == 3


def test_formal_resource_contract_preserves_reserve():
    assert runner.ADMISSION_MIB >= (
        runner.MEASURED_EVAL_REQUIREMENT_MIB + runner.RESERVE_MIB)
    assert runner.RUNTIME_FLOOR_MIB == 0
