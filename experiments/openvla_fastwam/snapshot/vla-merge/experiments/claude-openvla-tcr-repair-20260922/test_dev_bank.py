import json
from pathlib import Path
import shutil

import pytest

import dev_bank


def test_frozen_development_bank_is_exact_and_outcome_blind():
    root = (dev_bank.WORK / "vla-merge-runtime/experiments/"
            "claude-openvla-tcr-repair-20260922/dev-bank-v1")
    result = dev_bank.verify(root)
    assert result == {
        "accepted": True,
        "tasks": 40,
        "states": 400,
        "selection_40_episodes": 40,
        "selection_400_episodes": 400,
        "manifest_sha256": result["manifest_sha256"],
        "policy_outcomes_read": False,
    }


def test_selection_tampering_fails_closed(tmp_path: Path):
    source = (dev_bank.WORK / "vla-merge-runtime/experiments/"
              "claude-openvla-tcr-repair-20260922/dev-bank-v1")
    # Copy the small reset bank so path-containment checks remain meaningful.
    for name in ("manifest.json", "manifest.sha256", "selection-40.json",
                 "selection-400.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    shutil.copytree(source / "states", tmp_path / "states")
    selection = json.loads((tmp_path / "selection-40.json").read_text())
    first = sorted(selection["tasks"])[0]
    selection["tasks"][first]["stock_offsets"] = [37]
    (tmp_path / "selection-40.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="offsets differ"):
        dev_bank.verify(tmp_path)
