"""Fail-closed manifests for LIBERO benchmark extensions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from vla_merge.libero_procedural_bank import sha256_file


SOURCE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PRO_DIMENSIONS = ("object", "swap", "semantic", "task", "environment")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class ExtensionManifestError(ValueError):
    """Raised when an extension selection or artifact identity is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtensionManifestError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExtensionManifestError(f"Expected a JSON object in {path}")
    return value


def _verified_file(row: object, field: str) -> Path:
    if not isinstance(row, dict):
        raise ExtensionManifestError(f"Missing {field} file receipt")
    # Keep the manifest-visible compatibility path instead of collapsing it to
    # the legacy symlink target.  Callers that need inode equivalence compare
    # resolved paths explicitly.
    path = Path(str(row.get("path", ""))).expanduser().absolute()
    expected = row.get("sha256")
    if (
        not path.is_file()
        or not isinstance(expected, str)
        or sha256_file(path) != expected
    ):
        raise ExtensionManifestError(f"{field} file identity mismatch: {path}")
    return path


@dataclass(frozen=True)
class ExtensionEvalJob:
    job_id: str
    dimension: str
    source_suite: str
    runtime_suite: str
    task_id: int
    task_name: str
    state_ids: tuple[int, ...]
    state_raw_sha256: tuple[str, ...]
    bddl_path: Path
    init_path: Path


@dataclass(frozen=True)
class ExtensionSelection:
    path: Path
    sha256: str
    benchmark: str
    repeat_id: str
    eval_seed: int
    policy_checkpoint_sha256: str
    checkpoint_contract: str
    source_identity: dict[str, Any]
    jobs: tuple[ExtensionEvalJob, ...]

    @property
    def table1_checkpoint_sha256(self) -> str:
        """Backward-compatible alias for selections created for Table 1 policies."""
        return self.policy_checkpoint_sha256


def load_extension_selection(
    path: Path,
    *,
    expected_benchmark: str | None = None,
    verify_files: bool = True,
) -> ExtensionSelection:
    selection_path = path.expanduser().resolve()
    payload = _read_json(selection_path)
    if payload.get("schema_version") != 1:
        raise ExtensionManifestError("Unsupported extension selection schema")
    benchmark = payload.get("benchmark")
    if benchmark not in {"libero_plus", "libero_pro"}:
        raise ExtensionManifestError(f"Unsupported extension benchmark: {benchmark!r}")
    if expected_benchmark is not None and benchmark != expected_benchmark:
        raise ExtensionManifestError(
            f"Selection benchmark {benchmark!r} differs from expected {expected_benchmark!r}"
        )
    repeat_id = payload.get("repeat_id")
    if repeat_id not in {"repeat-01", "repeat-02", "repeat-03"}:
        raise ExtensionManifestError(
            "repeat_id must be repeat-01, repeat-02, or repeat-03"
        )
    eval_seed = payload.get("eval_seed")
    if type(eval_seed) is not int:
        raise ExtensionManifestError("eval_seed must be an integer")
    table1_checkpoint_sha256 = payload.get("table1_checkpoint_sha256")
    policy_checkpoint_sha256 = payload.get("policy_checkpoint_sha256")
    if (
        table1_checkpoint_sha256 is not None
        and policy_checkpoint_sha256 is not None
        and table1_checkpoint_sha256 != policy_checkpoint_sha256
    ):
        raise ExtensionManifestError(
            "Conflicting table1_checkpoint_sha256 and policy_checkpoint_sha256"
        )
    checkpoint_sha256 = policy_checkpoint_sha256 or table1_checkpoint_sha256
    if not isinstance(checkpoint_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", checkpoint_sha256
    ):
        raise ExtensionManifestError(
            "Selection must bind one policy checkpoint SHA256"
        )
    checkpoint_contract = payload.get("checkpoint_contract")
    if checkpoint_contract is None and table1_checkpoint_sha256 is not None:
        checkpoint_contract = "original-corresponding-table1-checkpoint"
    if not isinstance(checkpoint_contract, str) or not SAFE_ID.fullmatch(
        checkpoint_contract
    ):
        raise ExtensionManifestError("Selection lacks a safe checkpoint_contract")
    source_identity = payload.get("source_identity")
    if not isinstance(source_identity, dict):
        raise ExtensionManifestError("Selection lacks source_identity")
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ExtensionManifestError("Selection contains no jobs")

    jobs: list[ExtensionEvalJob] = []
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str, int]] = set()
    for raw in raw_jobs:
        if not isinstance(raw, dict):
            raise ExtensionManifestError("Malformed extension job")
        job_id = raw.get("job_id")
        if (
            not isinstance(job_id, str)
            or not SAFE_ID.fullmatch(job_id)
            or job_id in seen_ids
        ):
            raise ExtensionManifestError(f"Invalid or duplicate job_id: {job_id!r}")
        dimension = raw.get("dimension")
        allowed_dimensions = (
            PRO_DIMENSIONS
            if benchmark == "libero_pro"
            else (
                "Background Textures",
                "Camera Viewpoints",
                "Language Instructions",
                "Light Conditions",
                "Objects Layout",
                "Robot Initial States",
                "Sensor Noise",
            )
        )
        if dimension not in allowed_dimensions:
            raise ExtensionManifestError(
                f"Invalid dimension for {job_id}: {dimension!r}"
            )
        source_suite = raw.get("source_suite")
        runtime_suite = raw.get("runtime_suite")
        if source_suite not in SOURCE_SUITES or not isinstance(runtime_suite, str):
            raise ExtensionManifestError(f"Invalid suite mapping for {job_id}")
        task_id = raw.get("task_id")
        task_name = raw.get("task_name")
        if (
            type(task_id) is not int
            or task_id < 0
            or not isinstance(task_name, str)
            or not task_name
        ):
            raise ExtensionManifestError(f"Invalid task identity for {job_id}")
        state_ids = raw.get("state_ids")
        state_hashes = raw.get("state_raw_sha256")
        if (
            not isinstance(state_ids, list)
            or not state_ids
            or any(type(index) is not int or index < 0 for index in state_ids)
            or len(set(state_ids)) != len(state_ids)
        ):
            raise ExtensionManifestError(f"Invalid state_ids for {job_id}")
        if (
            not isinstance(state_hashes, list)
            or len(state_hashes) != len(state_ids)
            or any(
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in state_hashes
            )
        ):
            raise ExtensionManifestError(f"Invalid state hashes for {job_id}")
        key = (dimension, source_suite, task_id)
        if key in seen_keys:
            raise ExtensionManifestError(
                f"Duplicate dimension/suite/task binding: {key}"
            )
        bddl = raw.get("bddl")
        init = raw.get("init_states")
        if verify_files:
            bddl_path = _verified_file(bddl, f"{job_id}.bddl")
            init_path = _verified_file(init, f"{job_id}.init_states")
        else:
            bddl_path = Path(str((bddl or {}).get("path", ""))).expanduser().resolve()
            init_path = Path(str((init or {}).get("path", ""))).expanduser().resolve()
        jobs.append(
            ExtensionEvalJob(
                job_id=job_id,
                dimension=str(dimension),
                source_suite=str(source_suite),
                runtime_suite=runtime_suite,
                task_id=task_id,
                task_name=task_name,
                state_ids=tuple(state_ids),
                state_raw_sha256=tuple(state_hashes),
                bddl_path=bddl_path,
                init_path=init_path,
            )
        )
        seen_ids.add(job_id)
        seen_keys.add(key)
    return ExtensionSelection(
        path=selection_path,
        sha256=sha256_file(selection_path),
        benchmark=benchmark,
        repeat_id=repeat_id,
        eval_seed=eval_seed,
        policy_checkpoint_sha256=checkpoint_sha256,
        checkpoint_contract=checkpoint_contract,
        source_identity=source_identity,
        jobs=tuple(jobs),
    )


def require_four_suite_coverage(
    jobs: tuple[ExtensionEvalJob, ...] | list[ExtensionEvalJob],
) -> None:
    dimensions = sorted({job.dimension for job in jobs})
    if not dimensions:
        raise ExtensionManifestError("No selected dimensions")
    for dimension in dimensions:
        observed = {job.source_suite for job in jobs if job.dimension == dimension}
        if observed != set(SOURCE_SUITES):
            raise ExtensionManifestError(
                f"Dimension {dimension!r} does not cover all four source suites: {sorted(observed)}"
            )


def primary_policy_file(policy: Path) -> Path:
    root = policy.expanduser().resolve()
    dense = root / "model.safetensors"
    adapter = root / "adapter_model.safetensors"
    present = [path for path in (dense, adapter) if path.is_file()]
    if len(present) != 1:
        raise ExtensionManifestError(
            f"Policy must contain exactly one primary dense or adapter weight file: {root}"
        )
    return present[0]
