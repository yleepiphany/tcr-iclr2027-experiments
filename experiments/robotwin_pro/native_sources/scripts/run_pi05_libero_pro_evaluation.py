#!/usr/bin/env python3
"""Plan or run manifest-bound PI0.5 evaluation on static LIBERO-PRO suites.

Every policy, method, repeat, perturbation dimension, task, reset state, and output
directory is explicit.  The runner never moves, replaces, or silently reuses an
incomplete output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_merge.libero_extension import (  # noqa: E402
    PRO_DIMENSIONS,
    SAFE_ID,
    SOURCE_SUITES,
    ExtensionManifestError,
    load_extension_selection,
    primary_policy_file,
    require_four_suite_coverage,
)
from vla_merge.libero_procedural_bank import sha256_file  # noqa: E402


WORKSPACE = ROOT.parent
SOURCE = WORKSPACE / "pi05_lora_finetune_v2_20260826"
RUNTIME = WORKSPACE / "vla-merge-runtime"
LIBERO_DATA_ROOT = WORKSPACE / ".datasets/LIBERO/20260919"
PRO_REPO = LIBERO_DATA_ROOT / "pro/repo"
PRO_COMMIT = "eafdb809426b13153aa1e4c42d6601844217dfec"
PRO_ASSETS = LIBERO_DATA_ROOT / "pro/assets"
PRO_ASSETS_TREE_SHA256 = (
    "ad5a423f687bb6260d0a01d2f38e2aa97923ee8cceaf27869e1d9e9fd4260114"
)
PRO_CONFIG = LIBERO_DATA_ROOT / "pro/config"
PRO_PYTHON = RUNTIME / "envs/iclr2027-libero-pro-py312-v1/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument(
        "--repeat-id", choices=("repeat-01", "repeat-02", "repeat-03"), required=True
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--source-suite",
        choices=SOURCE_SUITES,
        help="Route a suite-specific expert only to its own source suite.",
    )
    parser.add_argument(
        "--dimension", action="append", choices=PRO_DIMENSIONS, required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume a hash-identical plan and skip only verified completed jobs.",
    )
    return parser.parse_args()


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _policy_sidecars(policy: Path) -> dict[str, str]:
    return {
        path.relative_to(policy).as_posix(): sha256_file(path)
        for path in sorted(policy.rglob("*"))
        if path.is_file()
        and (
            path.suffix == ".json"
            or path.suffix == ".safetensors"
            and "model.safetensors" not in path.name
        )
    }


def build_plan(args: argparse.Namespace) -> dict:
    if not SAFE_ID.fullmatch(args.method_id):
        raise ExtensionManifestError("method-id contains unsafe characters")
    if not re.fullmatch(r"[0-9a-f]{64}", args.policy_sha256):
        raise ExtensionManifestError(
            "policy-sha256 must contain 64 lowercase hex characters"
        )
    dimensions = tuple(dict.fromkeys(args.dimension))
    if len(dimensions) != len(args.dimension):
        raise ExtensionManifestError("Duplicate --dimension")
    if args.launch and (args.gpu is None or not args.gpu.isdigit()):
        raise ExtensionManifestError("--launch requires an explicit numeric --gpu")
    if args.gpu is not None and not args.gpu.isdigit():
        raise ExtensionManifestError("gpu must be numeric")
    if (
        _git_commit(PRO_REPO) != PRO_COMMIT
        or subprocess.check_output(
            ["git", "-C", str(PRO_REPO), "status", "--short"], text=True
        ).strip()
    ):
        raise ExtensionManifestError(
            "LIBERO-PRO source checkout identity is not clean/frozen"
        )
    if not PRO_PYTHON.is_file() or not (PRO_CONFIG / "config.yaml").is_file():
        raise FileNotFoundError(
            "Project-specific LIBERO-PRO prefix/config is incomplete"
        )

    selection = load_extension_selection(
        args.selection,
        expected_benchmark="libero_pro",
        verify_files=True,
    )
    if selection.repeat_id != args.repeat_id:
        raise ExtensionManifestError("Runner repeat differs from selection repeat")
    if selection.source_identity.get("repo_commit") != PRO_COMMIT:
        raise ExtensionManifestError(
            "Selection LIBERO-PRO commit differs from frozen commit"
        )
    if selection.source_identity.get("assets_tree_sha256") != PRO_ASSETS_TREE_SHA256:
        raise ExtensionManifestError(
            "Selection LIBERO-PRO asset tree differs from frozen identity"
        )
    config_sha256 = sha256_file(PRO_CONFIG / "config.yaml")
    selected_config_sha256 = selection.source_identity.get("config_sha256")
    if selected_config_sha256 is not None and selected_config_sha256 != config_sha256:
        raise ExtensionManifestError(
            "Selection LIBERO-PRO config differs from current compatibility config"
        )
    if (
        selection.checkpoint_contract == "pi05-libero-expert-15k-newroot-v2"
        and selected_config_sha256 != config_sha256
    ):
        raise ExtensionManifestError(
            "New-root expert selection must bind the compatibility config SHA256"
        )
    jobs = tuple(
        job
        for job in selection.jobs
        if job.dimension in dimensions
        and (args.source_suite is None or job.source_suite == args.source_suite)
    )
    if {job.dimension for job in jobs} != set(dimensions):
        raise ExtensionManifestError("Selection is missing a requested dimension")
    if args.source_suite is None:
        require_four_suite_coverage(jobs)
    else:
        for dimension in dimensions:
            task_ids = {
                job.task_id
                for job in jobs
                if job.dimension == dimension
                and job.source_suite == args.source_suite
            }
            if task_ids != set(range(10)):
                raise ExtensionManifestError(
                    f"Suite-specific dimension {dimension!r} must bind task IDs 0..9; "
                    f"observed={sorted(task_ids)}"
                )

    policy = args.policy.expanduser().resolve()
    policy_primary = primary_policy_file(policy)
    observed_policy_sha = sha256_file(policy_primary)
    if observed_policy_sha != args.policy_sha256:
        raise ExtensionManifestError(
            f"Policy hash mismatch: expected={args.policy_sha256} observed={observed_policy_sha}"
        )
    if observed_policy_sha != selection.policy_checkpoint_sha256:
        raise ExtensionManifestError(
            "Policy differs from the checkpoint bound by selection"
        )
    run_root = (
        args.output_root.expanduser().absolute() / args.method_id / args.repeat_id
    )
    entry = ROOT / "scripts/eval_pi05_policy_with_extension_selection.py"
    commands = []
    for job in jobs:
        output = run_root / job.job_id
        command = [
            str(PRO_PYTHON),
            str(entry),
            f"--extension-selection={selection.path}",
            f"--extension-selection-sha256={selection.sha256}",
            f"--extension-job-id={job.job_id}",
            f"--policy-sha256={observed_policy_sha}",
            f"--method-id={args.method_id}",
            f"--repeat-id={args.repeat_id}",
            f"--output_dir={output}",
            "--env.type=libero",
            f"--env.task={job.runtime_suite}",
            f"--env.task_ids=[{job.task_id}]",
            "--env.hard_reset=true",
            "--eval.batch_size=1",
            "--eval.use_async_envs=false",
            f"--eval.n_episodes={len(job.state_ids)}",
            f"--seed={selection.eval_seed}",
            f"--policy.path={policy}",
            "--policy.device=cuda",
            "--policy.compile_model=false",
            "--policy.gradient_checkpointing=false",
            "--policy.n_action_steps=10",
        ]
        commands.append(
            {
                "job_id": job.job_id,
                "dimension": job.dimension,
                "source_suite": job.source_suite,
                "runtime_suite": job.runtime_suite,
                "task_id": job.task_id,
                "state_ids": list(job.state_ids),
                "output": str(output),
                "command": command,
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "LIBERO-PRO",
        "no_training_merge_calibration_or_tuning": True,
        "checkpoint_reuse_contract": selection.checkpoint_contract,
        "method_id": args.method_id,
        "repeat_id": args.repeat_id,
        "eval_seed": selection.eval_seed,
        "dimensions": list(dimensions),
        "four_suite_coverage": args.source_suite is None,
        "source_suite": args.source_suite,
        "policy": str(policy),
        "policy_primary_file": str(policy_primary),
        "policy_sha256": observed_policy_sha,
        "policy_sidecars": _policy_sidecars(policy),
        "selection": str(selection.path),
        "selection_sha256": selection.sha256,
        "source": {
            "data_root": str(LIBERO_DATA_ROOT),
            "repo": str(PRO_REPO),
            "repo_commit": PRO_COMMIT,
            "assets": str(PRO_ASSETS),
            "config": str(PRO_CONFIG / "config.yaml"),
            "config_sha256": config_sha256,
        },
        "python": str(PRO_PYTHON),
        "entrypoint": str(entry),
        "entrypoint_sha256": sha256_file(entry),
        "output_root": str(run_root),
        "commands": commands,
    }


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.launch:
        return
    run_root = Path(plan["output_root"])
    plan_path = run_root / "plan.json"
    if run_root.exists():
        if not args.resume_existing:
            raise FileExistsError(
                f"Refusing existing LIBERO-PRO run directory: {run_root}"
            )
        if not plan_path.is_file():
            raise FileNotFoundError(f"Existing run lacks plan: {plan_path}")
        frozen = json.loads(plan_path.read_text(encoding="utf-8"))
        for key in (
            "benchmark",
            "method_id",
            "repeat_id",
            "eval_seed",
            "dimensions",
            "source_suite",
            "policy",
            "policy_sha256",
            "selection",
            "selection_sha256",
            "entrypoint_sha256",
            "commands",
        ):
            if frozen.get(key) != plan.get(key):
                raise ExtensionManifestError(
                    f"Existing plan differs from requested resume at field {key}"
                )
        plan = frozen
    else:
        for row in plan["commands"]:
            if Path(row["output"]).exists():
                raise FileExistsError(f"Refusing existing job output: {row['output']}")
        run_root.mkdir(parents=True, exist_ok=False)
        plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "MUJOCO_GL": "egl",
        # robosuite 1.4 validates this value against the physical IDs listed
        # in CUDA_VISIBLE_DEVICES before CUDA performs logical remapping.
        "MUJOCO_EGL_DEVICE_ID": str(args.gpu),
        "TOKENIZERS_PARALLELISM": "false",
        "PALIGEMMA_TOKENIZER_PATH": str(
            SOURCE / "assets/paligemma-3b-pt-224-tokenizer"
        ),
        "LIBERO_PRO_REPO": str(PRO_REPO),
        "LIBERO_PRO_ASSET_DIR": str(PRO_ASSETS),
        "LIBERO_CONFIG_PATH": str(PRO_CONFIG),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": ":".join(
            (
                str(PRO_REPO),
                str(SOURCE / "lerobot/src"),
                str(SOURCE / "src"),
                str(ROOT / "scripts"),
                str(ROOT / "src"),
                os.environ.get("PYTHONPATH", ""),
            )
        ),
    }
    for row in plan["commands"]:
        output = Path(row["output"])
        receipt = output / "extension_eval_receipt.json"
        if receipt.is_file():
            completed = json.loads(receipt.read_text(encoding="utf-8"))
            expected = {
                "job_id": row["job_id"],
                "policy_sha256": plan["policy_sha256"],
                "selection_sha256": plan["selection_sha256"],
                "repeat_id": plan["repeat_id"],
            }
            if any(completed.get(key) != value for key, value in expected.items()):
                raise ExtensionManifestError(
                    f"Existing receipt identity mismatch: {receipt}"
                )
            eval_info = output / "eval_info.json"
            if (
                not eval_info.is_file()
                or completed.get("eval_info_sha256") != sha256_file(eval_info)
            ):
                raise ExtensionManifestError(
                    f"Existing receipt result hash mismatch: {receipt}"
                )
            continue
        if output.exists():
            raise FileExistsError(
                f"Refusing incomplete existing job output without receipt: {output}"
            )
        status = run_root / f"{row['job_id']}.status.json"
        status.write_text(
            json.dumps({"status": "running", **row}, indent=2, sort_keys=True) + "\n"
        )
        try:
            subprocess.run(row["command"], env=env, check=True)
            receipt = output / "extension_eval_receipt.json"
            if not receipt.is_file():
                raise RuntimeError(f"Missing extension receipt: {receipt}")
            value = {
                "status": "complete",
                **row,
                "receipt": str(receipt),
                "receipt_sha256": sha256_file(receipt),
            }
        except Exception as exc:
            value = {"status": "failed", **row, "error": repr(exc)}
            status.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            raise
        status.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    (run_root / "complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "jobs": len(plan["commands"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
