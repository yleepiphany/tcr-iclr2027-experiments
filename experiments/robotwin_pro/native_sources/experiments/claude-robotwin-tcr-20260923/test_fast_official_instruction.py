from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path[:0] = [str(HERE), str(WORK / "vla-merge-runtime/environments/robotwin2-0aee-open3d-optional-v2")]
import fast_official_instruction as fast
from description.utils.generate_episode_instructions import generate_episode_descriptions


RT = SimpleNamespace(
    _robotwin_blocks_episode_info=lambda task, info: info,
    OFFICIAL_INSTRUCTION_TYPE_ENV="LEROBOT_ROBOTWIN_INSTRUCTION_TYPE",
    OFFICIAL_INSTRUCTION_MAX_ENV="LEROBOT_ROBOTWIN_INSTRUCTION_MAX",
)


@pytest.mark.parametrize("task,labels", [
    ("blocks_ranking_rgb", ("red block", "green block", "blue block")),
    ("blocks_ranking_size", ("large block", "medium block", "small block")),
])
@pytest.mark.parametrize("instruction_type", ["seen", "unseen"])
@pytest.mark.parametrize("count", [1, 7, 100, 1003])
def test_exact_text_and_rng_state(monkeypatch, task, labels, instruction_type, count):
    info = {"{A}": labels[0], "{B}": labels[1], "{C}": labels[2],
            "{a}": "left", "{b}": "right", "{c}": "left"}
    monkeypatch.setenv(RT.OFFICIAL_INSTRUCTION_TYPE_ENV, instruction_type)
    monkeypatch.setenv(RT.OFFICIAL_INSTRUCTION_MAX_ENV, str(count))
    random.seed(517)
    np.random.seed(907)
    rows = generate_episode_descriptions(task, [info], count)
    reference = str(np.random.choice(rows[0][instruction_type]))
    python_after_reference = random.getstate()
    numpy_after_reference = np.random.get_state()

    random.seed(517)
    np.random.seed(907)
    actual = fast.fast_instruction(task, info, RT)
    python_after_fast = random.getstate()
    numpy_after_fast = np.random.get_state()
    assert actual == reference
    assert python_after_fast == python_after_reference
    assert numpy_after_fast[0] == numpy_after_reference[0]
    assert np.array_equal(numpy_after_fast[1], numpy_after_reference[1])
    assert numpy_after_fast[2:] == numpy_after_reference[2:]


def test_rejects_unproven_object_description_path(monkeypatch):
    monkeypatch.setenv(RT.OFFICIAL_INSTRUCTION_MAX_ENV, "100")
    info = {"{A}": "objects/red", "{B}": "green block", "{C}": "blue block",
            "{a}": "left", "{b}": "right", "{c}": "left"}
    with pytest.raises(ValueError, match="unsupported description value"):
        fast.fast_instruction("blocks_ranking_rgb", info, RT)
