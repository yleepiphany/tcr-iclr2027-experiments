import evaluate_development as target


def selection(name: str):
    root = target.dev_bank.WORK / ("vla-merge-runtime/experiments/"
        "claude-openvla-tcr-repair-20260922/dev-bank-v1")
    suffix = "40" if name == "development-40" else "400"
    return target.load_development_selection(root, root / f"selection-{suffix}.json")


def test_development_40_and_400_are_nested_and_exact():
    small, large = selection("development-40"), selection("development-400")
    assert len(small.tasks) == len(large.tasks) == 40
    for key in small.tasks:
        assert small.tasks[key].indices == (36,)
        assert large.tasks[key].indices == tuple(range(36, 46))
        assert small.tasks[key].raw_sha256 == large.tasks[key].raw_sha256[:1]


def test_row_audit_rejects_wrong_reset_identity():
    selected = selection("development-40")
    suite = "libero_spatial"
    rows = []
    for task_id in range(10):
        task = selected.tasks[(suite, task_id)]
        rows.append({"suite": suite, "task_id": task_id, "episode_index": 0,
                     "state_index": 36, "raw_state_sha256": task.raw_sha256[0],
                     "rollout_seed": selected.eval_seed, "valid": True,
                     "success": False})
    assert target.validate_rows(rows, selected, suite) == 0
    rows[0]["raw_state_sha256"] = "0" * 64
    try:
        target.validate_rows(rows, selected, suite)
    except ValueError as exc:
        assert "receipt differs" in str(exc)
    else:
        raise AssertionError("wrong development reset was accepted")
