#!/usr/bin/env python3
"""Opt-in LIBERO extension evaluation with explicit task and reset-state receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_merge.libero_extension import (  # noqa: E402
    ExtensionManifestError,
    load_extension_selection,
    primary_policy_file,
)
from vla_merge.libero_procedural_bank import sha256_file, sha256_state  # noqa: E402


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--extension-selection", type=Path, required=True)
    parser.add_argument("--extension-selection-sha256", required=True)
    parser.add_argument("--extension-job-id", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--repeat-id", required=True)
    return parser.parse_known_args(argv)


def _cli_value(arguments: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, value in enumerate(arguments):
        if value.startswith(prefix):
            return value[len(prefix) :]
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def _load_selected_states(job: Any) -> np.ndarray:
    raw = torch.load(job.init_path, weights_only=False)
    states = np.asarray(raw)
    if states.ndim != 2 or not np.issubdtype(states.dtype, np.floating):
        raise ExtensionManifestError(
            f"Unexpected init-state array for {job.job_id}: {states.shape}/{states.dtype}"
        )
    if any(index >= states.shape[0] for index in job.state_ids):
        raise ExtensionManifestError(f"State ID out of range for {job.job_id}")
    selected = np.ascontiguousarray(states[list(job.state_ids)])
    if not bool(np.isfinite(selected).all()):
        raise ExtensionManifestError(f"Selected non-finite state for {job.job_id}")
    for state, expected in zip(selected, job.state_raw_sha256, strict=True):
        if sha256_state(state) != expected:
            raise ExtensionManifestError(
                f"Selected raw-state hash mismatch for {job.job_id}"
            )
    return selected


def _validate_cli(
    args: argparse.Namespace, remaining: list[str], selection: Any, job: Any
) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", args.extension_selection_sha256):
        raise ExtensionManifestError("Invalid expected selection SHA256")
    if selection.sha256 != args.extension_selection_sha256:
        raise ExtensionManifestError("Selection SHA256 changed after runner planning")
    if args.repeat_id != selection.repeat_id:
        raise ExtensionManifestError("repeat-id differs from selection")
    if args.policy_sha256 != selection.policy_checkpoint_sha256:
        raise ExtensionManifestError(
            "Policy hash differs from the checkpoint bound by selection"
        )
    if _cli_value(remaining, "--env.task") != job.runtime_suite:
        raise ExtensionManifestError("CLI runtime suite differs from extension job")
    raw_task_ids = _cli_value(remaining, "--env.task_ids")
    try:
        task_ids = json.loads(raw_task_ids or "null")
    except json.JSONDecodeError as exc:
        raise ExtensionManifestError("env.task_ids must be a JSON list") from exc
    if task_ids != [job.task_id]:
        raise ExtensionManifestError("CLI task_ids differs from extension job")
    if _cli_value(remaining, "--eval.n_episodes") != str(len(job.state_ids)):
        raise ExtensionManifestError(
            "Episode count differs from explicit state selection"
        )
    if _cli_value(remaining, "--seed") != str(selection.eval_seed):
        raise ExtensionManifestError("CLI seed differs from selection")
    policy_raw = _cli_value(remaining, "--policy.path")
    if not policy_raw:
        raise ExtensionManifestError("Explicit --policy.path is required")
    policy_file = primary_policy_file(Path(policy_raw))
    if sha256_file(policy_file) != args.policy_sha256:
        raise ExtensionManifestError("Policy primary weight SHA256 mismatch")
    output_raw = _cli_value(remaining, "--output_dir")
    if not output_raw:
        raise ExtensionManifestError("Explicit --output_dir is required")
    return Path(output_raw).expanduser().absolute()


def install_extension_selection(
    job: Any, selected_states: np.ndarray
) -> tuple[Any, Any]:
    from libero.libero import get_libero_path
    from lerobot.envs import libero

    original_init = libero.LiberoEnv.__init__
    original_reset = libero.LiberoEnv.reset
    signature = inspect.signature(original_init)

    def init_with_selection(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        suite = str(bound.arguments["task_suite_name"])
        task_id = int(bound.arguments["task_id"])
        if suite != job.runtime_suite or task_id != job.task_id:
            raise ExtensionManifestError(
                f"Evaluator constructed unexpected task {suite}/{task_id}; expected "
                f"{job.runtime_suite}/{job.task_id}"
            )
        if not bool(bound.arguments["hard_reset"]):
            raise ExtensionManifestError(
                "Extension evaluation requires hard_reset=True"
            )
        task = bound.arguments["task_suite"].get_task(task_id)
        if task.name != job.task_name:
            raise ExtensionManifestError(
                f"Runtime task name differs for {job.job_id}: {task.name!r}"
            )
        current_bddl = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ).resolve()
        if (
            current_bddl.resolve() != job.bddl_path.resolve()
            or sha256_file(current_bddl) != sha256_file(job.bddl_path)
        ):
            raise ExtensionManifestError(f"Runtime BDDL differs for {job.job_id}")

        bound.arguments["init_states"] = False
        original_init(*bound.args, **bound.kwargs)
        self.init_states = True
        self._init_states = selected_states.copy()
        self.init_state_id = self.episode_index
        self._extension_job = job
        print(
            json.dumps(
                {
                    "extension_job": job.job_id,
                    "runtime_suite": job.runtime_suite,
                    "task_id": job.task_id,
                    "state_ids": list(job.state_ids),
                    "state_raw_sha256": list(job.state_raw_sha256),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def reset_without_wrap(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound_job = getattr(self, "_extension_job", None)
        if bound_job is not None and self.init_state_id >= len(self._init_states):
            raise ExtensionManifestError(
                f"Explicit state selection exhausted for {bound_job.job_id}; refusing modulo reuse"
            )
        return original_reset(self, *args, **kwargs)

    libero.LiberoEnv.__init__ = init_with_selection
    libero.LiberoEnv.reset = reset_without_wrap
    return original_init, original_reset


def _write_receipt(
    output: Path,
    *,
    args: argparse.Namespace,
    selection: Any,
    job: Any,
) -> Path:
    eval_info = output / "eval_info.json"
    if not eval_info.is_file():
        raise FileNotFoundError(eval_info)
    receipt = output / "extension_eval_receipt.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": selection.benchmark,
        "method_id": args.method_id,
        "repeat_id": selection.repeat_id,
        "eval_seed": selection.eval_seed,
        "policy_sha256": args.policy_sha256,
        "policy_checkpoint_sha256": selection.policy_checkpoint_sha256,
        "checkpoint_contract": selection.checkpoint_contract,
        "selection_path": str(selection.path),
        "selection_sha256": selection.sha256,
        "job_id": job.job_id,
        "dimension": job.dimension,
        "source_suite": job.source_suite,
        "runtime_suite": job.runtime_suite,
        "task_id": job.task_id,
        "task_name": job.task_name,
        "state_ids": list(job.state_ids),
        "state_raw_sha256": list(job.state_raw_sha256),
        "bddl_sha256": sha256_file(job.bddl_path),
        "init_states_sha256": sha256_file(job.init_path),
        "eval_info_sha256": sha256_file(eval_info),
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
    }
    receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    args, remaining = _parse_args(sys.argv[1:])
    selection = load_extension_selection(args.extension_selection, verify_files=True)
    matches = [job for job in selection.jobs if job.job_id == args.extension_job_id]
    if len(matches) != 1:
        raise ExtensionManifestError(f"Unknown extension job: {args.extension_job_id}")
    job = matches[0]
    selected_states = _load_selected_states(job)
    output = _validate_cli(args, remaining, selection, job)

    # This import applies the existing tokenizer and optional LIBERO-PRO namespace
    # compatibility hooks without changing the legacy evaluator source.
    import eval_with_local_tokenizer  # noqa: F401
    from lerobot.scripts import lerobot_eval

    install_extension_selection(job, selected_states)
    sys.argv = [sys.argv[0], *remaining]
    lerobot_eval.main()
    receipt = _write_receipt(
        output,
        args=args,
        selection=selection,
        job=job,
    )
    print(json.dumps({"extension_eval_receipt": str(receipt)}), flush=True)


if __name__ == "__main__":
    main()
