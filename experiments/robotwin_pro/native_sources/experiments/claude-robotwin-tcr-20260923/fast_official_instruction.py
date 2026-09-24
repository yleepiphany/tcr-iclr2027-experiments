"""Exact, lazy official-language selection for RoboTwin's two block tasks.

The upstream generator repeats a short, deterministic list until it contains
one million strings and then uses NumPy choice once. For these two tasks the
episode values are literal strings, never object-description JSON references.
This implementation performs the same two Python shuffles and one NumPy index
draw, then materializes only the chosen string. It fails closed if upstream
templates or episode values acquire a random replacement path.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


TASKS = frozenset(("blocks_ranking_rgb", "blocks_ranking_size"))
DEFAULT_MAX_DESCRIPTIONS = 1_000_000


def fast_instruction(task_name: str, env: Any, robotwin_module: Any) -> str:
    if task_name not in TASKS:
        raise ValueError(f"Fast official instruction is not registered for {task_name}")
    info = robotwin_module._robotwin_blocks_episode_info(task_name, env)
    if not isinstance(info, dict) or not info:
        raise ValueError("Block episode info was unavailable")
    from description.utils import generate_episode_instructions as official

    data = official.load_task_instructions(task_name)
    seen = official.filter_instructions(data.get("seen", []), info)
    unseen = official.filter_instructions(data.get("unseen", []), info)
    if not seen and not unseen:
        raise ValueError("No official block instruction matches the episode")

    # The official replacement function would call Python random.choice for a
    # value whose object-description JSON exists. Such an input cannot use the
    # lazy equivalence and must be reviewed rather than silently approximated.
    object_root = (Path(official.__file__).resolve().parent / "../objects_description").resolve()
    for value in info.values():
        if not isinstance(value, str) or "\\" in value or "/" in value:
            raise ValueError("Official episode info has an unsupported description value")
        if (object_root / f"{value}.json").exists():
            raise ValueError("Official instruction replacement became stochastic")

    instruction_type = os.environ.get(robotwin_module.OFFICIAL_INSTRUCTION_TYPE_ENV, "seen")
    choices = {"seen": seen, "unseen": unseen}
    selected = choices.get(instruction_type) or seen or unseen
    try:
        count = int(os.environ.get(robotwin_module.OFFICIAL_INSTRUCTION_MAX_ENV,
                                   str(DEFAULT_MAX_DESCRIPTIONS)))
    except ValueError:
        count = DEFAULT_MAX_DESCRIPTIONS
    if count <= 0:
        raise ValueError("Official instruction count must be positive")
    # Upstream builds [template_0, ..., template_N] in cyclic order, truncates
    # at count, and calls np.random.choice on the resulting list. NumPy choice
    # on an integer draws the same index and advances the RNG identically.
    index = int(np.random.choice(count))
    template = selected[index % len(selected)]
    if instruction_type == "unseen" and unseen:
        return official.replace_placeholders_unseen(template, info)
    return official.replace_placeholders(template, info)


def install() -> None:
    from lerobot.envs import robotwin as robotwin_module

    if getattr(robotwin_module, "_tcr_fast_instruction_installed", False):
        return
    original = robotwin_module._generate_robotwin_official_instruction

    def replacement(task_name: str, env: Any) -> str:
        if task_name in TASKS:
            return fast_instruction(task_name, env, robotwin_module)
        return original(task_name, env)

    robotwin_module._generate_robotwin_official_instruction = replacement
    robotwin_module._tcr_fast_instruction_installed = True
