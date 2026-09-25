#!/usr/bin/env python3
"""Opt-in PI0.5/LIBERO evaluation using an immutable procedural reset selection.

This entrypoint deliberately does not alter the legacy offset evaluator.  It consumes
``--procedural-bank`` plus a separate selection JSON whose task bindings may contain
arbitrary, non-contiguous state indices.  All bank/source hashes are checked before
LeRobot constructs an environment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_merge.libero_procedural_bank import (  # noqa: E402
    LoadedSelection,
    ProceduralBankError,
    load_selection,
    sha256_file,
)


def parse_entry_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--procedural-bank", type=Path, required=True)
    parser.add_argument("--procedural-selection", type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def _value_from_cli(arguments: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, value in enumerate(arguments):
        if value.startswith(prefix):
            return value[len(prefix) :]
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def _validate_cli_contract(
    selection: LoadedSelection, remaining: list[str]
) -> tuple[Path, str]:
    env_type = _value_from_cli(remaining, "--env.type")
    if env_type != "libero":
        raise ProceduralBankError("Procedural bank entry requires --env.type=libero")
    suite = _value_from_cli(remaining, "--env.task")
    if suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
        raise ProceduralBankError(
            "Procedural bank entry requires one standard source suite"
        )
    suite_tasks = {
        task_id: task
        for (task_suite, task_id), task in selection.tasks.items()
        if task_suite == suite
    }
    if set(suite_tasks) != set(range(10)):
        raise ProceduralBankError(
            f"Selection must bind all ten tasks for evaluated suite {suite}"
        )
    episodes_raw = _value_from_cli(remaining, "--eval.n_episodes")
    if episodes_raw is None or not re.fullmatch(r"\d+", episodes_raw):
        raise ProceduralBankError(
            "Procedural bank entry requires explicit --eval.n_episodes"
        )
    episodes = int(episodes_raw)
    if any(len(task.indices) != episodes for task in suite_tasks.values()):
        raise ProceduralBankError(
            f"eval.n_episodes={episodes} differs from selected state count for {suite}"
        )
    output_raw = _value_from_cli(remaining, "--output_dir")
    if not output_raw:
        raise ProceduralBankError(
            "Procedural bank entry requires an explicit --output_dir"
        )
    output = Path(output_raw).expanduser().absolute()
    seed_raw = _value_from_cli(remaining, "--seed")
    if selection.eval_seed is not None:
        if seed_raw is None or not re.fullmatch(r"-?\d+", seed_raw):
            raise ProceduralBankError(
                "Formal procedural selection requires an explicit integer --seed"
            )
        if int(seed_raw) != selection.eval_seed:
            raise ProceduralBankError(
                f"CLI eval seed {seed_raw} differs from selection seed {selection.eval_seed}"
            )
    return output, suite


def install_procedural_bank(selection: LoadedSelection) -> tuple[Any, Any]:
    """Patch only this process's LiberoEnv to use the selected immutable arrays."""
    from lerobot.envs import libero

    original_init = libero.LiberoEnv.__init__
    original_reset = libero.LiberoEnv.reset
    signature = inspect.signature(original_init)

    def init_with_procedural_bank(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        suite = str(bound.arguments["task_suite_name"])
        task_id = int(bound.arguments["task_id"])
        identity = (suite, task_id)
        if identity not in selection.tasks:
            raise ProceduralBankError(
                f"No procedural state selection for constructed LIBERO task {suite}/{task_id}"
            )
        if not bool(bound.arguments["hard_reset"]):
            raise ProceduralBankError(
                "Procedural final-bank evaluation requires hard_reset=True"
            )
        task_suite = bound.arguments["task_suite"]
        task = task_suite.get_task(task_id)
        selected = selection.tasks[identity]
        if (
            task.name != selected.task_name
            or task.problem_folder != selected.problem_folder
            or task.bddl_file != selected.bddl_file
        ):
            raise ProceduralBankError(
                f"Current task metadata differs from bank for {suite}/{task_id}"
            )

        # Prevent the stock loader from reading a canonical .pruned_init file.  Restore
        # the selected immutable rows immediately after normal environment construction.
        bound.arguments["init_states"] = False
        original_init(*bound.args, **bound.kwargs)
        self.init_states = True
        self._init_states = selected.states.copy()
        self.init_state_id = self.episode_index
        self._procedural_bank_task = selected
        print(
            json.dumps(
                {
                    "procedural_bank": selection.bank_manifest_sha256,
                    "selection": selection.selection_sha256,
                    "task": selected.task_key,
                    "state_indices": list(selected.indices),
                    "raw_state_sha256": list(selected.raw_sha256),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def reset_without_wrap(self: Any, *args: Any, **kwargs: Any) -> Any:
        selected = getattr(self, "_procedural_bank_task", None)
        if selected is not None and self.init_state_id >= len(self._init_states):
            raise ProceduralBankError(
                f"Procedural selection exhausted for {selected.task_key}; refusing modulo reuse"
            )
        return original_reset(self, *args, **kwargs)

    libero.LiberoEnv.__init__ = init_with_procedural_bank
    libero.LiberoEnv.reset = reset_without_wrap
    return original_init, original_reset


def _write_eval_receipt(output: Path, selection: LoadedSelection, suite: str) -> Path:
    result = output / "eval_info.json"
    if not result.is_file():
        raise FileNotFoundError(f"LeRobot evaluation did not produce {result}")
    receipt = output / "procedural_bank_receipt.json"
    if receipt.exists():
        raise FileExistsError(f"Refusing existing procedural bank receipt: {receipt}")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "bank_root": str(selection.bank_root),
        "bank_manifest_sha256": selection.bank_manifest_sha256,
        "selection_path": str(selection.selection_path),
        "selection_sha256": selection.selection_sha256,
        "repeat_id": selection.repeat_id,
        "eval_seed": selection.eval_seed,
        "eval_info_sha256": sha256_file(result),
        "tasks": {
            task.task_key: {
                "suite": task.suite,
                "task_id": task.task_id,
                "task_name": task.task_name,
                "state_indices": list(task.indices),
                "raw_state_sha256": list(task.raw_sha256),
            }
            for task in selection.tasks.values()
            if task.suite == suite
        },
    }
    receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    import eval_with_local_tokenizer  # noqa: F401  (installs runtime compatibility overrides)

    entry_args, remaining = parse_entry_args(sys.argv[1:])
    selection = load_selection(
        entry_args.procedural_bank,
        entry_args.procedural_selection,
        verify_source_files=True,
    )
    output, suite = _validate_cli_contract(selection, remaining)
    install_procedural_bank(selection)
    sys.argv = [sys.argv[0], *remaining]
    from lerobot.scripts import lerobot_eval

    lerobot_eval.main()
    receipt = _write_eval_receipt(output, selection, suite)
    print(json.dumps({"procedural_bank_receipt": str(receipt)}), flush=True)


if __name__ == "__main__":
    main()
