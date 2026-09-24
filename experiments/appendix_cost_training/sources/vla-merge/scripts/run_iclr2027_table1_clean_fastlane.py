#!/usr/bin/env python3
"""Execute one immutable clean-LIBERO Table 1 fastlane job.

One invocation owns exactly one method/repeat/suite job (100 episodes).  Jobs
may be distributed across hosts and GPUs.  GPU leases are host-qualified, and
each completed job emits 100 episode receipts plus a bound run receipt.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METHODS = ("experts", "model_soups")
REPEATS = ("repeat-01", "repeat-02", "repeat-03")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MIN_ALLOWED_FREE_MIB = 12_000


class FastlaneRunError(ValueError):
    """Raised when a fastlane launch or result violates its manifest."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--repeat", choices=REPEATS)
    parser.add_argument("--suite", choices=SUITES)
    parser.add_argument("--attempt-id")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--min-free-mib", type=int)
    parser.add_argument("--expected-hostname")
    parser.add_argument("--lease-dir", type=Path)
    parser.add_argument("--allow-compute-sharing", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: object, label: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise FastlaneRunError(f"{label} is not an explicit lowercase SHA256")
    return text


def read_bound_json(path: Path, expected: str, label: str) -> tuple[Path, dict[str, Any]]:
    expected = require_sha(expected, f"expected {label} SHA256")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = sha256_file(resolved)
    if observed != expected:
        raise FastlaneRunError(
            f"{label} SHA256 differs: expected={expected}, observed={observed}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FastlaneRunError(f"{label} must contain one JSON object")
    return resolved, payload


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    expected = {
        "benchmark": "clean_LIBERO_only",
        "libero_plus": False,
        "task_text_mode": "correct",
        "hard_reset": True,
        "init_states": True,
        "batch_size": 1,
        "action_steps": 10,
        "episodes_per_task": 10,
        "tasks_per_suite": 10,
        "episodes_per_job": 100,
        "repeat_count": 3,
    }
    if dict(protocol) != expected:
        raise FastlaneRunError("Fastlane clean-LIBERO protocol differs")


def validate_manifest(path: Path, expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    path, manifest = read_bound_json(path, expected_sha256, "fastlane manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "iclr2027_table1_clean_formal_fastlane"
        or manifest.get("fastlane_id") != "experts-model-soups-r6-v1"
        or manifest.get("status") != "ready_for_clean_formal_execution"
        or manifest.get("methods") != list(METHODS)
        or manifest.get("repeats") != list(REPEATS)
        or manifest.get("suites") != list(SUITES)
        or manifest.get("job_count") != 24
        or manifest.get("episode_count") != 2400
    ):
        raise FastlaneRunError("Fastlane manifest identity or budget differs")
    _validate_protocol(manifest.get("protocol", {}))
    separation = manifest.get("development_formal_separation", {})
    if (
        separation.get("formal_bank_used_for_recipe_selection") is not False
        or separation.get("fastlane_methods_are_fixed_before_formal_access") is not True
        or separation.get("pending_method_development_must_not_read_fastlane_outputs")
        is not True
        or separation.get("development_screen_receipts_are_not_formal_results") is not True
    ):
        raise FastlaneRunError("Development/formal separation contract differs")
    evaluator = manifest.get("evaluator_sha_audit", {})
    expected_evaluator = require_sha(
        evaluator.get("revision6_expected_sha256"), "revision6 evaluator"
    )
    if (
        evaluator.get("drift_detected") is not False
        or evaluator.get("observed_sha256") != expected_evaluator
        or sha256_file(Path(str(evaluator.get("path"))).resolve()) != expected_evaluator
    ):
        raise FastlaneRunError("Evaluator SHA binding drifted after manifest freeze")
    runner = manifest.get("runner", {})
    runner_sha = require_sha(runner.get("sha256"), "fastlane runner")
    if Path(str(runner.get("path"))).resolve() != Path(__file__).resolve():
        raise FastlaneRunError("Executing runner path differs from frozen fastlane runner")
    if sha256_file(Path(__file__).resolve()) != runner_sha:
        raise FastlaneRunError("Executing runner SHA256 differs from fastlane manifest")
    for field, label in (
        ("source_revision6", "source revision6"),
        ("scope_override", "scope override"),
    ):
        row = manifest.get(field, {})
        read_bound_json(Path(str(row.get("path"))), str(row.get("sha256", "")), label)
    bank = manifest.get("procedural_bank", {})
    read_bound_json(
        Path(str(bank.get("manifest", {}).get("path"))),
        str(bank.get("manifest", {}).get("sha256", "")),
        "procedural bank manifest",
    )
    read_bound_json(
        Path(str(bank.get("selection_index", {}).get("path"))),
        str(bank.get("selection_index", {}).get("sha256", "")),
        "selection index",
    )
    jobs = manifest.get("jobs")
    episodes = manifest.get("episodes")
    if not isinstance(jobs, list) or not isinstance(episodes, list):
        raise FastlaneRunError("Fastlane jobs/episodes must be lists")
    job_keys = [row.get("job_key") for row in jobs]
    if len(set(job_keys)) != 24:
        raise FastlaneRunError("Fastlane job keys differ or duplicate")
    episode_keys = [row.get("episode_key") for row in episodes]
    if len(set(episode_keys)) != 2400:
        raise FastlaneRunError("Fastlane episode keys differ or duplicate")
    return path, manifest


def select_job(
    manifest: Mapping[str, Any], method: str, repeat: str, suite: str
) -> dict[str, Any]:
    rows = [
        row
        for row in manifest["jobs"]
        if row.get("method") == method
        and row.get("repeat") == repeat
        and row.get("suite") == suite
    ]
    if len(rows) != 1:
        raise FastlaneRunError("Requested fastlane job is missing or duplicate")
    job = rows[0]
    if job.get("episodes") != 100 or job.get("eval_seed") not in (274001, 274002, 274003):
        raise FastlaneRunError("Requested fastlane job protocol differs")
    selection = job.get("selection", {})
    read_bound_json(
        Path(str(selection.get("path"))),
        str(selection.get("sha256", "")),
        f"{repeat} formal selection",
    )
    job_episodes = [
        row
        for row in manifest["episodes"]
        if row.get("method") == method
        and row.get("repeat") == repeat
        and row.get("suite") == suite
    ]
    if len(job_episodes) != 100:
        raise FastlaneRunError("Requested job does not bind exactly 100 episode identities")
    expected_pairs = {(task_id, episode) for task_id in range(10) for episode in range(10)}
    observed_pairs = {(row.get("task_id"), row.get("episode_index")) for row in job_episodes}
    if observed_pairs != expected_pairs:
        raise FastlaneRunError("Requested job episode task/index matrix differs")
    return job


def verify_checkpoint(job: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = dict(job.get("checkpoint", {}))
    root = Path(str(checkpoint.get("path"))).resolve()
    model = root / "model.safetensors"
    expected_model_sha = require_sha(checkpoint.get("model_sha256"), "checkpoint model")
    if not model.is_file() or sha256_file(model) != expected_model_sha:
        raise FastlaneRunError("Checkpoint model SHA256 differs")
    if checkpoint.get("kind") not in {
        "suite_routed_peft_safe_dense_expert",
        "single_merged_checkpoint",
    }:
        raise FastlaneRunError("Checkpoint kind differs")
    for field in ("manifest", "verification"):
        row = checkpoint.get(field, {})
        read_bound_json(
            Path(str(row.get("path"))),
            str(row.get("sha256", "")),
            f"checkpoint {field}",
        )
    return {**checkpoint, "weights_file": str(model)}


def _parse_csv_row(raw: str, expected_fields: int, label: str) -> list[str]:
    rows = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(rows) != 1:
        raise FastlaneRunError(f"Unexpected nvidia-smi {label} row count")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != expected_fields:
        raise FastlaneRunError(f"Unexpected nvidia-smi {label} field count")
    return values


def inspect_gpu(gpu: int, min_free_mib: int, allow_compute_sharing: bool) -> dict[str, Any]:
    if gpu < 0:
        raise FastlaneRunError("GPU index must be non-negative")
    if min_free_mib < MIN_ALLOWED_FREE_MIB:
        raise FastlaneRunError(
            f"min-free-mib must be explicit and >= {MIN_ALLOWED_FREE_MIB}"
        )
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=uuid,name,memory.total,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    uuid, name, total, free, used, utilization = _parse_csv_row(query, 6, "GPU")
    free_mib = int(free)
    if free_mib < min_free_mib:
        raise FastlaneRunError(
            f"GPU {gpu} has {free_mib} MiB free; requires {min_free_mib} MiB"
        )
    try:
        process_raw = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except subprocess.CalledProcessError:
        process_raw = ""
    processes = []
    for line in process_raw.splitlines():
        values = [value.strip() for value in line.split(",", 2)]
        if len(values) == 3 and values[0].isdigit():
            processes.append(
                {"pid": int(values[0]), "process_name": values[1], "used_memory_mib": values[2]}
            )
    if processes and not allow_compute_sharing:
        raise FastlaneRunError(
            f"GPU {gpu} already has compute processes; use explicit --allow-compute-sharing"
        )
    return {
        "host": socket.gethostname(),
        "physical_gpu": gpu,
        "uuid": uuid,
        "name": name,
        "memory_total_mib": int(total),
        "memory_free_mib": free_mib,
        "memory_used_mib": int(used),
        "utilization_percent": int(utilization),
        "minimum_free_mib": min_free_mib,
        "compute_sharing_allowed": allow_compute_sharing,
        "preexisting_compute_processes": processes,
    }


def safe_gpu_snapshot(gpu: int, min_free_mib: int) -> dict[str, Any]:
    try:
        return inspect_gpu(gpu, min_free_mib, True)
    except Exception as exc:  # noqa: BLE001 - telemetry cannot invalidate completed rollouts
        return {"status": "telemetry_failed", "error": f"{type(exc).__name__}: {exc}"}


def _safe_hostname(hostname: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", hostname)
    if not safe:
        raise FastlaneRunError("Hostname cannot be normalized for GPU lease")
    return safe


@contextmanager
def gpu_lease(lease_dir: Path, hostname: str, gpu: int) -> Iterator[dict[str, Any]]:
    root = lease_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{_safe_hostname(hostname)}.gpu-{gpu}.lock"
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FastlaneRunError(
                f"Host-qualified GPU lease is held: {hostname}/gpu-{gpu}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "hostname": hostname,
                    "physical_gpu": gpu,
                    "pid": os.getpid(),
                    "acquired_at": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield {
            "path": str(path),
            "hostname": hostname,
            "physical_gpu": gpu,
            "pid": os.getpid(),
            "real_flock_held": True,
        }
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def build_eval_command(
    manifest: Mapping[str, Any], job: Mapping[str, Any], output_dir: Path
) -> list[str]:
    runtime = manifest["runtime"]
    evaluator = manifest["evaluator_sha_audit"]["path"]
    command = [
        runtime["python_bin"],
        evaluator,
        f"--procedural-bank={manifest['procedural_bank']['root']}",
        f"--procedural-selection={job['selection']['path']}",
        f"--output_dir={output_dir}",
        "--env.type=libero",
        f"--env.task={job['suite']}",
        "--env.init_states=true",
        "--env.hard_reset=true",
        "--eval.batch_size=1",
        "--eval.n_episodes=10",
        f"--seed={job['eval_seed']}",
        f"--policy.path={job['checkpoint']['path']}",
        "--policy.device=cuda",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=false",
        "--policy.n_action_steps=10",
    ]
    required = {
        "--env.init_states=true",
        "--env.hard_reset=true",
        "--eval.batch_size=1",
        "--eval.n_episodes=10",
        "--policy.n_action_steps=10",
    }
    if not required.issubset(command) or sum(token.startswith("--seed=") for token in command) != 1:
        raise RuntimeError("Internal evaluator command protocol differs")
    return command


def build_environment(manifest: Mapping[str, Any], job: Mapping[str, Any], gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    source_root = Path(manifest["runtime"]["source_root"])
    overlay = source_root.parent / "vla-merge-runtime/python-overlay"
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MUJOCO_GL": "egl",
            "TOKENIZERS_PARALLELISM": "false",
            "PALIGEMMA_TOKENIZER_PATH": str(
                source_root / "assets/paligemma-3b-pt-224-tokenizer"
            ),
            "PYTHONPATH": ":".join(
                [
                    str(overlay),
                    str(source_root / "src"),
                    str(source_root / "scripts"),
                    environment.get("PYTHONPATH", ""),
                ]
            ).rstrip(":"),
            "PI05_TASK_TEXT_MODE": "correct",
            "PI05_LIBERO_SUITE": str(job["suite"]),
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "TORCHINDUCTOR_COMPILE_THREADS": "4",
        }
    )
    return environment


def validate_eval_info(path: Path, suite: str) -> tuple[dict[str, Any], dict[int, list[bool]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not {"overall", "per_group", "per_task"}.issubset(payload):
        raise FastlaneRunError("eval_info lacks required aggregate keys")
    per_task = payload["per_task"]
    if not isinstance(per_task, list) or len(per_task) != 10:
        raise FastlaneRunError("eval_info must contain ten per-task rows")
    outcomes: dict[int, list[bool]] = {}
    for row in per_task:
        task_id = row.get("task_id")
        values = row.get("metrics", {}).get("successes")
        if (
            row.get("task_group") != suite
            or type(task_id) is not int
            or not isinstance(values, list)
            or len(values) != 10
            or any(type(value) is not bool for value in values)
            or task_id in outcomes
        ):
            raise FastlaneRunError("eval_info per-task episode outcomes differ")
        outcomes[task_id] = values
    if set(outcomes) != set(range(10)):
        raise FastlaneRunError("eval_info task IDs differ")
    successes = sum(sum(values) for values in outcomes.values())
    if (
        payload["overall"].get("n_episodes") != 100
        or not math.isclose(
            float(payload["overall"].get("pc_success")), float(successes), abs_tol=1e-9
        )
        or suite not in payload["per_group"]
    ):
        raise FastlaneRunError("eval_info aggregate differs from episode outcomes")
    return {
        "episodes": 100,
        "successes": successes,
        "success_percent": float(successes),
    }, outcomes


def build_episode_receipts(
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    job: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    outcomes: Mapping[int, list[bool]],
    host: str,
    gpu: int,
    attempt_id: str,
) -> list[dict[str, Any]]:
    identities = [
        row
        for row in manifest["episodes"]
        if row["method"] == job["method"]
        and row["repeat"] == job["repeat"]
        and row["suite"] == job["suite"]
    ]
    identities.sort(key=lambda row: (row["task_id"], row["episode_index"]))
    if len(identities) != 100:
        raise FastlaneRunError("Manifest does not contain exactly 100 job episodes")
    receipts = []
    for row in identities:
        task_id = row["task_id"]
        episode_index = row["episode_index"]
        receipts.append(
            {
                "schema_version": 1,
                "status": "completed",
                "episode_key": row["episode_key"],
                "method": job["method"],
                "repeat": job["repeat"],
                "suite": job["suite"],
                "task_id": task_id,
                "episode_index": episode_index,
                "state_index": row["state_index"],
                "reset_sha256": row["reset_sha256"],
                "eval_seed": row["eval_seed"],
                "success": outcomes[task_id][episode_index],
                "attempt_id": attempt_id,
                "host": host,
                "physical_gpu": gpu,
                "fastlane_manifest_sha256": manifest_sha256,
                "selection_sha256": job["selection"]["sha256"],
                "checkpoint_sha256": checkpoint["model_sha256"],
            }
        )
    if len({row["episode_key"] for row in receipts}) != 100:
        raise FastlaneRunError("Episode receipt keys differ or duplicate")
    return receipts


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_episode_jsonl(path: Path, receipts: list[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FastlaneRunError(f"Refusing to overwrite episode receipts: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FastlaneRunError(f"Stale episode receipt temporary file exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in receipts:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _required_execution_args(args: argparse.Namespace) -> None:
    required = ("method", "repeat", "suite", "attempt_id", "gpu", "min_free_mib", "expected_hostname", "lease_dir")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise FastlaneRunError(f"Execution rejects implicit defaults; missing {missing}")
    if not re.fullmatch(r"attempt-[0-9]{2}", args.attempt_id):
        raise FastlaneRunError("attempt-id must be explicit, e.g. attempt-01")
    if socket.gethostname() != args.expected_hostname:
        raise FastlaneRunError(
            f"Host differs: expected={args.expected_hostname}, actual={socket.gethostname()}"
        )


def execute_one(
    args: argparse.Namespace,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _required_execution_args(args)
    assert args.method and args.repeat and args.suite and args.attempt_id
    assert args.gpu is not None and args.min_free_mib is not None and args.lease_dir
    job = select_job(manifest, args.method, args.repeat, args.suite)
    checkpoint = verify_checkpoint(job)
    host = socket.gethostname()
    manifest_sha = args.expected_manifest_sha256
    output_root = Path(str(manifest["output_root"])).resolve()
    root = output_root / job["output_relpath"] / args.attempt_id
    output_dir = root / "eval"
    if root.exists():
        raise FastlaneRunError(
            f"Refusing output collision at {root}; choose a new explicit attempt-id"
        )
    command = build_eval_command(manifest, job, output_dir)
    environment = build_environment(manifest, job, args.gpu)
    with gpu_lease(args.lease_dir, host, args.gpu) as lease:
        initial = inspect_gpu(args.gpu, args.min_free_mib, args.allow_compute_sharing)
        root.mkdir(parents=True, exist_ok=False)
        started_at = datetime.now(timezone.utc).isoformat()
        identities = {
            "fastlane_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
            "source_revision6": manifest["source_revision6"],
            "scope_override": manifest["scope_override"],
            "runner_sha256": manifest["runner"]["sha256"],
            "evaluator_sha256": manifest["evaluator_sha_audit"]["observed_sha256"],
            "procedural_bank_sha256": manifest["procedural_bank"]["manifest"]["sha256"],
            "selection_sha256": job["selection"]["sha256"],
            "checkpoint_sha256": checkpoint["model_sha256"],
        }
        prelaunch = {
            "schema_version": 1,
            "status": "launching",
            "job_key": job["job_key"],
            "method": args.method,
            "repeat": args.repeat,
            "suite": args.suite,
            "attempt_id": args.attempt_id,
            "host": host,
            "physical_gpu": args.gpu,
            "started_at": started_at,
            "command": command,
            "environment": {
                key: environment[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "MUJOCO_GL",
                    "PALIGEMMA_TOKENIZER_PATH",
                    "PI05_TASK_TEXT_MODE",
                    "PI05_LIBERO_SUITE",
                )
            },
            "identities": identities,
            "checkpoint": checkpoint,
            "resource": {"lease": lease, "initial": initial},
        }
        write_json(root / "prelaunch_receipt.json", prelaunch)
        log_path = root / "run.log"
        try:
            recheck = inspect_gpu(args.gpu, args.min_free_mib, args.allow_compute_sharing)
            with log_path.open("xb") as log:
                completed = subprocess.run(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            ended_at = datetime.now(timezone.utc).isoformat()
            if completed.returncode != 0:
                raise RuntimeError(f"Evaluator returned {completed.returncode}")
            eval_info = output_dir / "eval_info.json"
            procedural_receipt = output_dir / "procedural_bank_receipt.json"
            if not eval_info.is_file() or not procedural_receipt.is_file():
                raise FastlaneRunError("Evaluator succeeded without required result receipts")
            result, outcomes = validate_eval_info(eval_info, args.suite)
            procedural = json.loads(procedural_receipt.read_text(encoding="utf-8"))
            if (
                procedural.get("bank_manifest_sha256")
                != manifest["procedural_bank"]["manifest"]["sha256"]
                or procedural.get("selection_sha256") != job["selection"]["sha256"]
                or procedural.get("eval_info_sha256") != sha256_file(eval_info)
                or procedural.get("repeat_id") != args.repeat
                or procedural.get("eval_seed") != job["eval_seed"]
                or len(procedural.get("tasks", {})) != 10
            ):
                raise FastlaneRunError("Procedural evaluator receipt differs")
            episode_rows = build_episode_receipts(
                manifest,
                manifest_sha,
                job,
                checkpoint,
                outcomes,
                host,
                args.gpu,
                args.attempt_id,
            )
            episode_path = root / "episode_receipts.jsonl"
            write_episode_jsonl(episode_path, episode_rows)
            final = {
                **prelaunch,
                "status": "completed",
                "ended_at": ended_at,
                "protocol": manifest["protocol"],
                "development_formal_separation": manifest[
                    "development_formal_separation"
                ],
                "resource": {
                    "lease": lease,
                    "initial": initial,
                    "pre_subprocess_recheck": recheck,
                    "final": safe_gpu_snapshot(args.gpu, args.min_free_mib),
                },
                "artifacts": {
                    "eval_info": {"path": str(eval_info), "sha256": sha256_file(eval_info)},
                    "procedural_bank_receipt": {
                        "path": str(procedural_receipt),
                        "sha256": sha256_file(procedural_receipt),
                    },
                    "episode_receipts": {
                        "path": str(episode_path),
                        "sha256": sha256_file(episode_path),
                        "count": len(episode_rows),
                    },
                    "run_log": {"path": str(log_path), "sha256": sha256_file(log_path)},
                },
                "result": result,
                "claim_boundary": "One completed clean-LIBERO formal job; no pending-method recipe selection.",
            }
            run_receipt = root / "run_receipt.json"
            write_json(run_receipt, final)
            return {
                "status": "completed",
                "job_key": job["job_key"],
                "run_receipt": str(run_receipt),
                "run_receipt_sha256": sha256_file(run_receipt),
                "episode_receipt_count": len(episode_rows),
                "result": result,
            }
        except BaseException as exc:
            failure = {
                **prelaunch,
                "status": "failed",
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "resource": {
                    "lease": lease,
                    "initial": initial,
                    "final": safe_gpu_snapshot(args.gpu, args.min_free_mib),
                },
                "artifacts": (
                    {"run_log": {"path": str(log_path), "sha256": sha256_file(log_path)}}
                    if log_path.is_file()
                    else {}
                ),
            }
            failure_path = root / "failure_receipt.json"
            if not failure_path.exists():
                write_json(failure_path, failure)
            raise


def main() -> None:
    args = parse_args()
    manifest_path, manifest = validate_manifest(
        args.manifest, args.expected_manifest_sha256
    )
    if args.plan_only:
        run_values = (
            args.method,
            args.repeat,
            args.suite,
            args.attempt_id,
            args.gpu,
            args.min_free_mib,
            args.expected_hostname,
            args.lease_dir,
            args.allow_compute_sharing,
        )
        if any(value not in (None, False) for value in run_values):
            raise FastlaneRunError("plan-only rejects execution arguments")
        print(
            json.dumps(
                {
                    "status": "fastlane_plan_valid",
                    "manifest": str(manifest_path),
                    "manifest_sha256": args.expected_manifest_sha256,
                    "jobs": manifest["job_count"],
                    "episodes": manifest["episode_count"],
                    "methods": manifest["methods"],
                },
                sort_keys=True,
            )
        )
        return
    result = execute_one(args, manifest_path, manifest)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
