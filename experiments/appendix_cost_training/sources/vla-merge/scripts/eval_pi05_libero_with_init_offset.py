#!/usr/bin/env python3
"""Evaluate PI0.5 with an explicit offset into LIBERO's fixed init-state bank."""

from __future__ import annotations

import os
from typing import Any

import eval_with_local_tokenizer  # noqa: F401  (installs runtime compatibility overrides)


def install_init_state_offset() -> Any:
    from lerobot.envs import libero

    offset = int(os.environ.get("PI05_LIBERO_INIT_STATE_OFFSET", "0"))
    count = int(os.environ.get("PI05_LIBERO_INIT_STATE_COUNT", "1"))
    original_init = libero.LiberoEnv.__init__

    def init_with_offset(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if self.init_states and self._init_states is not None:
            available = len(self._init_states)
            if offset < 0 or count <= 0 or offset + count > available:
                raise ValueError(
                    f"Requested LIBERO init-state range [{offset}, {offset + count}) "
                    f"but task {self.task!r} only has {available} states"
                )
            self.init_state_id += offset
            print(
                f"Applied LIBERO init-state range [{offset}, {offset + count}) "
                f"to task={self.task!r}; first_init_state_id={self.init_state_id}",
                flush=True,
            )

    libero.LiberoEnv.__init__ = init_with_offset
    return original_init


def install_task_text_mode() -> Any | None:
    """Optionally replace LIBERO task text for language-conditioning ablations."""
    mode = os.environ.get("PI05_TASK_TEXT_MODE", "correct").strip().lower()
    if mode == "correct":
        return None
    if mode not in {"generic", "shift"}:
        raise ValueError(f"Unsupported PI05_TASK_TEXT_MODE={mode!r}")

    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep

    suite = os.environ.get("PI05_LIBERO_SUITE", "").strip()
    shifted_prompts: dict[str, str] = {}
    if mode == "shift":
        if not suite:
            raise RuntimeError("PI05_LIBERO_SUITE is required for shifted prompt evaluation")
        from libero.libero.benchmark.libero_suite_task_map import libero_task_map

        tasks = libero_task_map[suite]
        normalized = [task.replace("_", " ").strip().lower() for task in tasks]
        shifted_prompts = {
            task: normalized[(index + 1) % len(normalized)]
            for index, task in enumerate(normalized)
        }

    original_call = Pi05PrepareStateTokenizerProcessorStep.__call__

    def replace_one(value: str) -> str:
        if mode == "generic":
            return "perform the manipulation task"
        normalized = value.replace("_", " ").strip().lower()
        if normalized not in shifted_prompts:
            raise KeyError(f"Task text is not part of {suite}: {value!r}")
        return shifted_prompts[normalized]

    def call_with_task_text_mode(
        step: Pi05PrepareStateTokenizerProcessorStep,
        transition: Any,
    ) -> Any:
        copied = dict(transition)
        complementary = dict(copied.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        task_value = complementary.get(step.task_key)
        if isinstance(task_value, str):
            complementary[step.task_key] = replace_one(task_value)
        elif isinstance(task_value, list | tuple) and all(
            isinstance(item, str) for item in task_value
        ):
            complementary[step.task_key] = [replace_one(item) for item in task_value]
        else:
            raise TypeError(f"Unexpected LIBERO task text value: {task_value!r}")
        copied[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return original_call(step, copied)

    Pi05PrepareStateTokenizerProcessorStep.__call__ = call_with_task_text_mode
    print(f"Installed PI0.5 task-text mode={mode} suite={suite}", flush=True)
    return original_call


def main() -> None:
    from lerobot.scripts import lerobot_eval

    install_init_state_offset()
    install_task_text_mode()
    lerobot_eval.main()


if __name__ == "__main__":
    main()
