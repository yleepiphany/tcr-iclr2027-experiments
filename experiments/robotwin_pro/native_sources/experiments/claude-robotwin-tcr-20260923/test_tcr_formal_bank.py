"""The native runner receives the formal six-seed bank, not the expert parent."""

from pathlib import Path

import run_tcr_formal_v2 as formal


def test_all_nine_native_jobs_use_exact_frozen_formal_resets():
    source = formal.experts_module()
    assert formal.EXPERT_PROTOCOL.is_file()
    keys = set()
    for repeat in formal.REPEATS:
        assert formal.checkpoint(repeat).name == "pretrained_model"
        for group in formal.GROUPS:
            manifest, parent, expected = source.validate_job(group, repeat)
            assert sum(len(task["seeds"]) for task in parent["tasks"]) == 200
            runtime, actual = formal.native_formal_runtime(source, group, repeat)
            expert_hashes, binding = formal.expert_reset_hashes(source, group, repeat)
            assert runtime["tasks"] == manifest["tasks"]
            assert actual == expected and len(actual) == 60
            assert set(expert_hashes) == expected
            assert binding["terminal_sha256"] and binding["receipt_sha256"]
            assert formal.sha(Path(binding["terminal_path"])) == binding["terminal_sha256"]
            assert formal.sha(Path(binding["receipt_path"])) == binding["receipt_sha256"]
            keys.update((repeat, task_index, seed) for task_index, seed in actual)
    assert len(keys) == 540
