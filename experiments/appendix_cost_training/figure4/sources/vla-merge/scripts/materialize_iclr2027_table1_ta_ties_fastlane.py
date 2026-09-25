#!/usr/bin/env python3
"""Materialize the revision-7 Task Arithmetic and TIES candidate grids once.

This is intentionally a grid materializer rather than a one-candidate wrapper:

* the four PEFT-safe dense 10k experts are read into task-vector cache once;
* one unscaled Task Arithmetic direction is reused by all 21 alphas;
* each TIES density performs one global trim/sign/mean merge, then its direction
  is reused by all 23 alphas;
* full PI0.5 checkpoints are exported by reflink-cloning the base and patching
  only the 422 adapted tensors.  A safe full-copy fallback is recorded;
* completed checkpoint hashes are verified before resume/skip.

No evaluation outcome, procedural-final bank, LIBERO-PRO, or LIBERO-Plus input
is opened by this program.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import struct
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from ta_ties_core import (
    BASE_MODEL_SHA256,
    CANONICAL_CACHE_ROOT,
    CANONICAL_DEVELOPMENT_ROOT,
    CANONICAL_EVAL_ENTRY,
    CANONICAL_EVAL_WRAPPER,
    CANONICAL_JOB_LOCK_ROOT,
    CANONICAL_LEASE_ROOT,
    CANONICAL_LEGACY_EVAL_ROOT,
    CANONICAL_LEGACY_LOG_ROOT,
    CANONICAL_LEGACY_LOG_ROOT,
    CANONICAL_OUTPUT_ROOT,
    CANONICAL_PYTHON,
    CANONICAL_REPO_SCRIPTS,
    CANONICAL_V1_TA_ROOT,
    DENSE_BANK_SHA256,
    EVAL_ENTRY_SHA256,
    EVAL_WRAPPER_SHA256,
    EXPERT_BANK_SHA256,
    EXPERT_MODEL_SHA256,
    EXPERT_ORDER,
    FASTLANE_VERSION,
    PYTHON_RUNTIME_SHA256,
    REVISION7_SHA256,
    RUNTIME_POLICY_SIDECARS,
    V1_TA_ALPHA_0P40_SHA256,
    V1_TA_CORE_SHA256,
    V1_TA_DIRECTION_SHA256,
    V1_TA_EXECUTION_SHA256,
    V1_TA_INDEX_SHA256,
    V1_TA_MATERIALIZER_SHA256,
    CandidateSpec,
    ContractError,
    atomic_create_json,
    atomic_write_json,
    canonical_json_sha256,
    full_policy_identity,
    policy_sidecar_tree,
    python_environment_identity,
    read_bound_json,
    require_no_symlink_components,
    resolve_under_root,
    sha256_file,
    task_arithmetic_direction,
    task_arithmetic_specs,
    ties_direction,
    ties_specs,
    verify_complete_checkpoint,
)


EXPECTED_EXPERIMENT_ID = "iclr2027-table1-20260910"
EXPECTED_TENSOR_COUNT = 422
EXPECTED_COORDINATE_COUNT = 2_706_471_968
SOUP_ALPHA = "0.25"
SOUP_MODEL_SHA256 = "a92aacc43146dc41663c0057f2999bd90252fb3fb316ef963f165be67e1be01a"
SOUP_CANONICAL_SIDECAR_TREE_SHA256 = (
    "019e6c6b4ef09da0270ebd0027efb1549827daf2f38cbea4735ed5bb185edde0"
)
FICLONE = 0x40049409
SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision7-manifest", type=Path, required=True)
    parser.add_argument("--expert-dense-bank", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=CANONICAL_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CANONICAL_CACHE_ROOT)
    parser.add_argument(
        "--family",
        choices=("task_arithmetic", "ties_merging", "all"),
        default="all",
    )
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--adopt-v1-task-arithmetic",
        action="store_true",
        help=(
            "Independently re-hash and adopt the frozen complete v1 TA bank by "
            "read-only reference into immutable v2 manifests; never modifies v1."
        ),
    )
    parser.add_argument(
        "--stop-after-cache",
        action="store_true",
        help="Build/verify the one-pass vector and reusable direction caches only.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _model_file(root: Path) -> Path:
    path = root.expanduser().resolve() / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _sidecar_stat_snapshot(policy_root: Path) -> dict[str, dict[str, int]]:
    result = {}
    for path in sorted(policy_root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"policy sidecar snapshot contains symlink: {path}")
        if path.is_file() and path.name != "model.safetensors":
            result[path.relative_to(policy_root).as_posix()] = _stat_identity(path)
    return result


def verify_large_input_stats(contract: Mapping[str, Any]) -> None:
    for raw_path, expected in contract["large_file_stats"].items():
        path = Path(raw_path)
        if _stat_identity(path) != expected:
            raise ContractError(f"frozen large input changed during materialization: {path}")
    for raw_path, expected_sha256 in contract["small_file_hashes"].items():
        path = Path(raw_path)
        if sha256_file(path) != expected_sha256:
            raise ContractError(f"frozen JSON input changed during materialization: {path}")


def _file_identity(
    path: Path,
    label: str,
    expected_sha256: str | None = None,
    *,
    allow_terminal_symlink: bool = False,
) -> dict[str, Any]:
    logical = path.expanduser().absolute()
    require_no_symlink_components(
        logical.parent if allow_terminal_symlink else logical, label
    )
    resolved = logical.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = sha256_file(resolved)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ContractError(
            f"{label} SHA256 differs: expected={expected_sha256}, observed={observed}"
        )
    return {
        "path": str(logical),
        "resolved_path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": observed,
    }


def runtime_component_identities() -> dict[str, Any]:
    """Bind every program that can create or evaluate a v2 candidate."""
    here = Path(__file__).resolve()
    core = here.with_name("ta_ties_core.py")
    runner = here.with_name("run_iclr2027_table1_ta_ties_development.py")
    if here.parent == CANONICAL_REPO_SCRIPTS:
        expected_paths = {
            "core": CANONICAL_REPO_SCRIPTS / core.name,
            "materializer": CANONICAL_REPO_SCRIPTS / here.name,
            "development_runner": CANONICAL_REPO_SCRIPTS / runner.name,
        }
        for label, expected in expected_paths.items():
            actual = {"core": core, "materializer": here, "development_runner": runner}[label]
            if actual != expected:
                raise ContractError(f"{label} is not deployed at its canonical path")
    components = {
        "core": _file_identity(core, "TA/TIES core"),
        "materializer": _file_identity(here, "TA/TIES materializer"),
        "development_runner": _file_identity(runner, "TA/TIES development runner"),
        "eval_wrapper": _file_identity(
            CANONICAL_EVAL_WRAPPER, "development evaluator wrapper", EVAL_WRAPPER_SHA256
        ),
        "eval_entry": _file_identity(
            CANONICAL_EVAL_ENTRY, "development evaluator entry", EVAL_ENTRY_SHA256
        ),
        "python_runtime": _file_identity(
            CANONICAL_PYTHON,
            "PI0.5 Python runtime",
            PYTHON_RUNTIME_SHA256,
            allow_terminal_symlink=True,
        ),
        "python_environment": python_environment_identity(),
    }
    components["identity_sha256"] = canonical_json_sha256(components)
    return components


class _GlobalMaterializerLock:
    def __init__(self) -> None:
        self.handle: Any | None = None
        self.path = CANONICAL_LEASE_ROOT / "materializer-output-cache.global.lock"

    def __enter__(self) -> dict[str, Any]:
        resolve_under_root(
            CANONICAL_LEASE_ROOT,
            CANONICAL_LEASE_ROOT,
            "canonical lease root",
            reject_reserved_names=False,
        ).mkdir(parents=True, exist_ok=True)
        require_no_symlink_components(CANONICAL_LEASE_ROOT, "canonical lease root")
        require_no_symlink_components(self.path, "materializer global lock")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.handle = os.fdopen(os.open(self.path, flags, 0o600), "r+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise ContractError(
                f"TA/TIES output/cache global lock is held: {self.path}"
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "fastlane_version": FASTLANE_VERSION,
            "pid": os.getpid(),
            "acquired_at": utc_now(),
            "canonical_output_root": str(CANONICAL_OUTPUT_ROOT),
            "canonical_cache_root": str(CANONICAL_CACHE_ROOT),
        }
        self.handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return {"path": str(self.path), "real_flock_held": True, **payload}

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        assert self.handle is not None
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def validate_inputs(
    revision_path: Path, dense_bank_path: Path, *, verify_large_files: bool
) -> dict[str, Any]:
    revision_path = revision_path.expanduser().resolve()
    dense_bank_path = dense_bank_path.expanduser().resolve()
    revision = read_bound_json(revision_path, REVISION7_SHA256, "Table 1 revision 7")
    _require(
        revision.get("schema_version") == 1
        and revision.get("experiment_id") == EXPECTED_EXPERIMENT_ID
        and revision.get("immutable_revision") == 7,
        "Only immutable Table 1 revision 7 is accepted",
    )
    frozen = revision.get("frozen_inputs", {})
    _require(
        frozen.get("expert_dense_bank_sha256") == DENSE_BANK_SHA256
        and frozen.get("expert_bank_manifest_sha256") == EXPERT_BANK_SHA256
        and frozen.get("formal_bank_use_during_development") == "forbidden",
        "Revision 7 frozen-input/development separation contract differs",
    )
    searches = revision.get("method_searches", {})
    task_recipe = searches.get("task_arithmetic", {})
    ties_recipe = searches.get("ties_merging", {})
    _require(
        task_recipe.get("mode") == "successive_halving"
        and task_recipe.get("candidates", {}).get("candidate_count") == 21
        and task_recipe.get("candidates", {}).get("global_alpha_start") == 0.0
        and task_recipe.get("candidates", {}).get("global_alpha_step") == 0.05
        and task_recipe.get("candidates", {}).get("global_alpha_stop_inclusive") == 1.0
        and task_recipe.get("per_suite_or_per_expert_scale") == "forbidden",
        "Revision 7 Task Arithmetic grid differs",
    )
    _require(
        ties_recipe.get("mode") == "successive_halving"
        and ties_recipe.get("scope")
        == "one global trim per expert over the full 422-tensor adapted domain"
        and ties_recipe.get("candidates", {}).get("candidate_count") == 69
        and ties_recipe.get("candidates", {}).get("density") == [0.1, 0.2, 0.3]
        and ties_recipe.get("candidates", {}).get("disjoint") == "mean"
        and ties_recipe.get("candidates", {}).get("sign_election") == "mass"
        and ties_recipe.get("candidates", {}).get("global_alpha_start") == 0.8
        and ties_recipe.get("candidates", {}).get("global_alpha_step") == 0.1
        and ties_recipe.get("candidates", {}).get("global_alpha_stop_inclusive") == 3.0,
        "Revision 7 TIES grid differs",
    )

    dense_bank = read_bound_json(
        dense_bank_path, DENSE_BANK_SHA256, "PEFT-safe dense expert bank"
    )
    _require(
        dense_bank.get("schema_version") == 1
        and dense_bank.get("kind") == "iclr2027_table1_peft_safe_dense_expert_bank"
        and dense_bank.get("status") == "passed_parameter_exact"
        and dense_bank.get("deployment_policy")
        == "dense_only_no_unmerged_adapter_fallback"
        and dense_bank.get("source_expert_bank", {}).get("sha256")
        == EXPERT_BANK_SHA256,
        "Dense expert bank identity or deployment policy differs",
    )
    _require(
        dense_bank.get("semantics", {}).get("expert_weight")
        == "peft_cpu_safe_merge_delta_cast_to_base_dtype_then_base_dtype_add",
        "Dense expert rounding semantics differ",
    )
    tensor_contract = dense_bank.get("tensor_contract", {})
    _require(
        tensor_contract.get("tensor_count") == 813
        and tensor_contract.get("logical_adapted_tensor_count") == EXPECTED_TENSOR_COUNT
        and tensor_contract.get("required_max_abs_error") == 0.0,
        "Dense expert tensor contract differs",
    )
    base = dense_bank.get("base", {})
    base_root = Path(str(base.get("path", ""))).resolve()
    _require(
        base.get("model_sha256") == BASE_MODEL_SHA256
        and base.get("tensor_count") == 813,
        "Dense bank base model contract differs",
    )
    base_model = _model_file(base_root)

    expert_rows = dense_bank.get("experts")
    _require(
        isinstance(expert_rows, list)
        and tuple(row.get("name") for row in expert_rows) == EXPERT_ORDER,
        "Dense expert order differs",
    )
    expert_roots: dict[str, Path] = {}
    for row in expert_rows:
        name = str(row["name"])
        source = row.get("source_adapter", {})
        dense = row.get("dense_checkpoint", {})
        dense_root = Path(str(dense.get("path", ""))).resolve()
        _require(
            dense.get("model_sha256") == EXPERT_MODEL_SHA256[name],
            f"{name}: dense expert model SHA contract differs",
        )
        # This rejects the historical mixed 15k/10k bank even if a caller tries
        # to substitute paths while retaining valid-looking JSON fields.
        _require(
            "/checkpoints/010000/pretrained_model" in str(source.get("path", ""))
            and source.get("role") == "provenance_only"
            and "10k-full-dense-peft-v2" in str(dense_root),
            f"{name}: source is not the frozen 10k PEFT-safe-v2 expert",
        )
        _model_file(dense_root)
        expert_roots[name] = dense_root

    source_bank_path = Path(str(dense_bank["source_expert_bank"]["path"])).resolve()
    expert_bank = read_bound_json(
        source_bank_path, EXPERT_BANK_SHA256, "frozen 10k expert bank"
    )
    _require(
        expert_bank.get("kind") == "iclr2027_table1_frozen_expert_bank"
        and expert_bank.get("validation", {}).get("status") == "valid"
        and expert_bank.get("expert_bank", {}).get("frozen") is True
        and expert_bank.get("expert_bank", {}).get("shared_training_step") == 10_000,
        "Frozen expert bank is not the shared 10k bank",
    )
    logical = expert_bank.get("adaptation_domain", {}).get("logical_tensors")
    _require(
        isinstance(logical, list)
        and len(logical) == EXPECTED_TENSOR_COUNT
        and len({row.get("base_key") for row in logical}) == EXPECTED_TENSOR_COUNT,
        "Frozen 422-tensor domain differs",
    )

    soup = searches.get("model_soups", {})
    soup_root = Path(str(soup.get("checkpoint_path", ""))).resolve()
    soup_model = _model_file(soup_root)
    _require(
        soup.get("mode") == "fixed_no_search"
        and soup.get("recipe", {}).get("kind") == "uniform_soup"
        and soup.get("recipe", {}).get("expert_weights") == [0.25] * 4
        and soup.get("checkpoint_sha256") == SOUP_MODEL_SHA256,
        "Uniform-soup parity target differs",
    )
    read_bound_json(
        Path(str(soup.get("checkpoint_manifest_path", ""))),
        str(soup.get("checkpoint_manifest_sha256", "")),
        "uniform-soup checkpoint manifest",
    )

    if verify_large_files:
        checks = [(base_model, BASE_MODEL_SHA256)] + [
            (_model_file(expert_roots[name]), EXPERT_MODEL_SHA256[name])
            for name in EXPERT_ORDER
        ] + [(soup_model, str(soup["checkpoint_sha256"]))]
        for path, expected in checks:
            observed = sha256_file(path)
            if observed != expected:
                raise ContractError(
                    f"model SHA256 differs: path={path}, expected={expected}, observed={observed}"
                )

    return {
        "revision_path": revision_path,
        "revision": revision,
        "dense_bank_path": dense_bank_path,
        "dense_bank": dense_bank,
        "expert_bank_path": source_bank_path,
        "expert_bank": expert_bank,
        "base_root": base_root,
        "base_model": base_model,
        "expert_roots": expert_roots,
        "soup_root": soup_root,
        "soup_model": soup_model,
        "logical_tensors": logical,
        "large_file_stats": {
            str(path): _stat_identity(path)
            for path in [
                base_model,
                *[_model_file(expert_roots[name]) for name in EXPERT_ORDER],
                soup_model,
            ]
        },
        "small_file_hashes": {
            str(revision_path): REVISION7_SHA256,
            str(dense_bank_path): DENSE_BANK_SHA256,
            str(source_bank_path): EXPERT_BANK_SHA256,
            str(Path(str(soup["checkpoint_manifest_path"])).resolve()): str(
                soup["checkpoint_manifest_sha256"]
            ),
        },
    }


def build_segments(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - exercised in PI0.5 environment
        raise RuntimeError("safetensors must be installed in the PI0.5 environment") from exc
    segments = []
    offset = 0
    with safe_open(contract["base_model"], framework="pt", device="cpu") as base:
        base_keys = set(base.keys())
        for row in sorted(contract["logical_tensors"], key=lambda item: item["base_key"]):
            key = str(row["base_key"])
            if key not in base_keys:
                raise ContractError(f"adapted key is absent from base: {key}")
            shape = list(base.get_slice(key).get_shape())
            if shape != row.get("base_shape"):
                raise ContractError(f"adapted shape differs for {key}")
            count = math.prod(shape)
            segments.append(
                {
                    "key": key,
                    "shape": shape,
                    "start": offset,
                    "end": offset + count,
                    "kind": row.get("kind"),
                }
            )
            offset += count
    if len(segments) != EXPECTED_TENSOR_COUNT or offset != EXPECTED_COORDINATE_COUNT:
        raise ContractError(
            f"adapted domain differs: tensors={len(segments)}, coordinates={offset}"
        )
    return segments


def _cache_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "fastlane_version": FASTLANE_VERSION,
        "revision7_sha256": REVISION7_SHA256,
        "dense_bank_sha256": DENSE_BANK_SHA256,
        "expert_bank_sha256": EXPERT_BANK_SHA256,
        "base_model_sha256": BASE_MODEL_SHA256,
        "expert_model_sha256": EXPERT_MODEL_SHA256,
        "expert_order": list(EXPERT_ORDER),
        "tensor_count": EXPECTED_TENSOR_COUNT,
        "coordinate_count": EXPECTED_COORDINATE_COUNT,
        "task_vector_dtype": "float32",
        "fusion_dtype": "float64",
        "task_vector_semantics": "dense_peft_safe_expert_minus_common_base",
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise ContractError(f"refusing to fsync a symlinked artifact: {path}")
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif path.is_dir():
            _fsync_directory(path)
    _fsync_directory(root)


def _cleanup_owned_partials(root: Path, prefix: str) -> list[str]:
    """Remove only v2-owned incomplete directories under an exact root."""
    recovered: list[str] = []
    if not root.exists():
        return recovered
    require_no_symlink_components(root, "partial cleanup root")
    for path in sorted(root.glob(f"{prefix}*")):
        if path.is_symlink() or not path.is_dir() or path.parent.resolve() != root.resolve():
            raise ContractError(f"unsafe partial artifact during resume: {path}")
        recovered.append(path.name)
        shutil.rmtree(path)
    if recovered:
        _fsync_directory(root)
    return recovered


def _verify_cache_file(path: Path, row: Mapping[str, Any], expected_bytes: int) -> None:
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ContractError(f"cache file size differs: {path}")
    observed = sha256_file(path)
    if observed != row.get("sha256"):
        raise ContractError(f"cache file hash differs: {path}")


def load_or_build_vectors(
    contract: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    cache_root: Path,
    components: Mapping[str, Any],
) -> tuple[list[np.memmap], dict[str, Any]]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch and safetensors are required for materialization") from exc
    cache_root = cache_root.expanduser().absolute()
    final = cache_root / "task-vectors-v2"
    manifest_path = final / "cache-manifest.json"
    expected_bytes = EXPECTED_COORDINATE_COUNT * np.dtype(np.float32).itemsize
    identity = _cache_identity(contract)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("fastlane_version") != FASTLANE_VERSION
            or manifest.get("status") != "complete"
            or manifest.get("identity") != identity
            or manifest.get("runtime_components") != components
        ):
            raise ContractError("existing task-vector cache identity differs")
        vectors = []
        for name in EXPERT_ORDER:
            path = final / f"{name}.f32.bin"
            _verify_cache_file(path, manifest["vectors"][name], expected_bytes)
            vectors.append(np.memmap(path, mode="r", dtype=np.float32))
        return vectors, manifest

    cache_root.mkdir(parents=True, exist_ok=True)
    recovered = _cleanup_owned_partials(cache_root, ".task-vectors-v2.partial-")
    temporary = cache_root / f".task-vectors-v2.partial-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir()
    vector_paths = {name: temporary / f"{name}.f32.bin" for name in EXPERT_ORDER}
    started = time.monotonic()
    try:
        vectors = {
            name: np.memmap(
                path, mode="w+", dtype=np.float32, shape=(EXPECTED_COORDINATE_COUNT,)
            )
            for name, path in vector_paths.items()
        }
        with ExitStack() as stack:
            base = stack.enter_context(
                safe_open(contract["base_model"], framework="pt", device="cpu")
            )
            experts = {
                name: stack.enter_context(
                    safe_open(
                        _model_file(contract["expert_roots"][name]),
                        framework="pt",
                        device="cpu",
                    )
                )
                for name in EXPERT_ORDER
            }
            for index, segment in enumerate(segments, start=1):
                key = str(segment["key"])
                start, end = int(segment["start"]), int(segment["end"])
                base_value = base.get_tensor(key).detach().cpu().float()
                for name in EXPERT_ORDER:
                    expert_value = experts[name].get_tensor(key).detach().cpu().float()
                    if expert_value.shape != base_value.shape:
                        raise ContractError(f"{name}/{key}: shape differs from base")
                    delta = (expert_value - base_value).reshape(-1)
                    if not torch.isfinite(delta).all():
                        raise ContractError(f"{name}/{key}: task vector is non-finite")
                    vectors[name][start:end] = delta.numpy()
                if index % 10 == 0 or index == len(segments):
                    print(
                        json.dumps(
                            {
                                "phase": "task_vectors",
                                "materialized_tensors": index,
                                "total_tensors": len(segments),
                            }
                        ),
                        flush=True,
                    )
        for value in vectors.values():
            value.flush()
        for path in vector_paths.values():
            _fsync_file(path)
        vector_rows = {}
        for name, path in vector_paths.items():
            vector_rows[name] = {
                "path": str((final / path.name).resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "fastlane_version": FASTLANE_VERSION,
            "kind": "iclr2027_table1_dense_task_vector_cache",
            "status": "complete",
            "created_at": utc_now(),
            "identity": identity,
            "runtime_components": components,
            "vectors": vector_rows,
            "single_read_contract": {
                "expert_tensor_reads": EXPECTED_TENSOR_COUNT,
                "reads_per_expert_adapted_tensor": 1,
                "candidate_generation_reads_expert_checkpoints": 0,
            },
            "resume_recovered_partials": recovered,
            "wall_seconds": time.monotonic() - started,
        }
        atomic_write_json(temporary / "cache-manifest.json", manifest)
        # Close writable maps before publishing this directory as immutable cache.
        del vectors
        _fsync_directory(temporary)
        temporary.rename(final)
        _fsync_directory(cache_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return (
        [np.memmap(final / f"{name}.f32.bin", mode="r", dtype=np.float32) for name in EXPERT_ORDER],
        manifest,
    )


def load_or_build_direction(
    *,
    cache_root: Path,
    name: str,
    vectors: Sequence[np.ndarray],
    chunk_size: int,
    density: float | None,
    components: Mapping[str, Any],
) -> tuple[np.memmap, dict[str, Any]]:
    directions = cache_root.expanduser().absolute() / "directions-v2"
    directions.mkdir(parents=True, exist_ok=True)
    final = directions / name
    path = final / "direction.f64.bin"
    manifest_path = final / "direction-manifest.json"
    expected_bytes = EXPECTED_COORDINATE_COUNT * np.dtype(np.float64).itemsize
    identity = {
        **_cache_identity({}),
        "direction": name,
        "density": density,
        "operator": (
            "unscaled_sum_task_vectors"
            if density is None
            else "global_trim_mass_sign_mean_disjoint"
        ),
    }
    if final.exists():
        if not final.is_dir() or final.is_symlink() or not manifest_path.is_file():
            raise ContractError(f"direction cache is not an atomic complete directory: {final}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("fastlane_version") != FASTLANE_VERSION
            or manifest.get("status") != "complete"
            or manifest.get("identity") != identity
            or manifest.get("runtime_components") != components
        ):
            raise ContractError(f"existing direction cache identity differs: {name}")
        _verify_cache_file(path, manifest["direction"], expected_bytes)
        return np.memmap(path, mode="r", dtype=np.float64), manifest
    recovered = _cleanup_owned_partials(directions, f".{name}.v2-partial-")
    temporary = directions / f".{name}.v2-partial-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir()
    temporary_path = temporary / "direction.f64.bin"
    started = time.monotonic()
    try:
        output = np.memmap(
            temporary_path,
            mode="w+",
            dtype=np.float64,
            shape=(EXPECTED_COORDINATE_COUNT,),
        )
        if density is None:
            task_arithmetic_direction(vectors, output=output, chunk_size=chunk_size)
            metadata: dict[str, Any] = {
                "operator": "unscaled_sum_task_vectors",
                "alpha_applied_during_checkpoint_export": True,
            }
        else:
            _, metadata = ties_direction(
                vectors,
                keep_fraction=density,
                output=output,
                chunk_size=chunk_size,
            )
        output.flush()
        del output
        _fsync_file(temporary_path)
        row = {
            "path": str(path.resolve()),
            "bytes": temporary_path.stat().st_size,
            "sha256": sha256_file(temporary_path),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "fastlane_version": FASTLANE_VERSION,
            "kind": "iclr2027_table1_reusable_fusion_direction",
            "status": "complete",
            "created_at": utc_now(),
            "identity": identity,
            "runtime_components": components,
            "direction": row,
            "fusion_metadata": metadata,
            "resume_recovered_partials": recovered,
            "wall_seconds": time.monotonic() - started,
        }
        atomic_write_json(temporary / "direction-manifest.json", manifest)
        _fsync_directory(temporary)
        temporary.rename(final)
        _fsync_directory(directions)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return np.memmap(path, mode="r", dtype=np.float64), manifest


def parse_safetensors_layout(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ContractError(f"invalid safetensors header: {path}")
        header_length = struct.unpack("<Q", raw)[0]
        header = json.loads(stream.read(header_length))
    data_start = 8 + header_length
    result = {}
    for key, row in header.items():
        if key == "__metadata__":
            continue
        offsets = row.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ContractError(f"invalid safetensors offsets for {key}")
        result[key] = {
            "dtype": row["dtype"],
            "shape": row["shape"],
            "start": data_start + int(offsets[0]),
            "end": data_start + int(offsets[1]),
        }
    return result


def clone_file(source: Path, destination: Path) -> str:
    """Try copy-on-write reflink, then fall back to a complete byte copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_fd = os.open(source, os.O_RDONLY)
    destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        try:
            fcntl.ioctl(destination_fd, FICLONE, source_fd)
            method = "ficlone_reflink"
        except OSError:
            with os.fdopen(os.dup(source_fd), "rb") as reader, os.fdopen(
                os.dup(destination_fd), "wb"
            ) as writer:
                shutil.copyfileobj(reader, writer, 16 * 1024 * 1024)
            method = "full_copy_fallback"
        os.fsync(destination_fd)
    finally:
        os.close(destination_fd)
        os.close(source_fd)
    return method


def hardlink_model_no_copy(source: Path, destination: Path) -> str:
    """Create a same-filesystem read-only reference without model-byte copying."""
    require_no_symlink_components(source, "hardlink source model")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)
    source_stat, destination_stat = source.stat(), destination.stat()
    if (
        source_stat.st_dev != destination_stat.st_dev
        or source_stat.st_ino != destination_stat.st_ino
        or source_stat.st_size != destination_stat.st_size
    ):
        raise ContractError("model hardlink identity differs")
    return "same_inode_hardlink_zero_model_bytes"


def pwrite_all(descriptor: int, payload: bytes, offset: int) -> None:
    """Write a complete tensor payload even if the kernel returns a short write."""
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.pwrite(descriptor, view[written:], offset + written)
        if count <= 0:
            raise OSError("pwrite made no progress")
        written += count


def copy_policy_sidecars(source: Path, output: Path, final_policy_path: Path) -> None:
    for name in (
        "README.md",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)
    if (source / "tokenizer").is_dir():
        shutil.copytree(source / "tokenizer", output / "tokenizer")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["use_peft"] = False
    config["pretrained_path"] = str(final_policy_path.resolve())
    atomic_write_json(output / "config.json", config)


def _prepare_base_policy_reference(
    temporary_policy: Path,
    final_policy: Path,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    copy_policy_sidecars(contract["soup_root"], temporary_policy, final_policy)
    strategy = hardlink_model_no_copy(
        contract["base_model"], temporary_policy / "model.safetensors"
    )
    identity = full_policy_identity(
        temporary_policy,
        BASE_MODEL_SHA256,
        expected_sidecars=RUNTIME_POLICY_SIDECARS,
        model_sha256_already_verified=BASE_MODEL_SHA256,
        reported_root=final_policy,
    )
    checkpoint = {
        "kind": "common_base_hardlink_reference",
        "path": str(final_policy),
        "model_path": str(final_policy / "model.safetensors"),
        "model_sha256": BASE_MODEL_SHA256,
        "model_bytes": contract["base_model"].stat().st_size,
        "copy_strategy": strategy,
        "modified_tensor_count": 0,
    }
    return checkpoint, identity


def _reference_manifest(
    output_root: Path,
    spec: CandidateSpec,
    contract: Mapping[str, Any],
    components: Mapping[str, Any],
) -> dict[str, Any]:
    root = output_root / "candidates" / spec.candidate_id
    manifest_path = root / "candidate-manifest.json"
    if manifest_path.is_file():
        return verify_complete_checkpoint(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = root.parent / (
        f".{spec.candidate_id}.v2-partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    policy = temporary / "pretrained_model"
    policy.mkdir(parents=True)
    try:
        checkpoint, policy_identity = _prepare_base_policy_reference(
            policy, root / "pretrained_model", contract
        )
        value = {
            "schema_version": SCHEMA_VERSION,
            "fastlane_version": FASTLANE_VERSION,
            "kind": "iclr2027_table1_ta_ties_candidate",
            "status": "complete_full_checkpoint",
            "created_at": utc_now(),
            "candidate": spec.as_dict(),
            "checkpoint": checkpoint,
            "full_policy_identity": policy_identity,
            "runtime_components": components,
            "finite": True,
            "inputs": {
                "revision7_sha256": REVISION7_SHA256,
                "dense_bank_sha256": DENSE_BANK_SHA256,
                "base_model_sha256": BASE_MODEL_SHA256,
            },
            "semantics": "alpha_zero_exact_common_base_hardlink_reference",
            "uniform_soup_full_policy_parity": None,
            "formal_outcomes_read": False,
        }
        atomic_write_json(temporary / "candidate-manifest.json", value)
        _fsync_tree(temporary)
        temporary.rename(root)
        _fsync_directory(root.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return value


def _compare_checkpoints_exact(
    left: Path,
    right: Path,
    *,
    left_sha256: str,
    right_sha256: str,
    left_policy_identity: Mapping[str, Any],
    right_policy_identity: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch and safetensors are required for parity") from exc
    count = 0
    with safe_open(left, framework="pt", device="cpu") as first, safe_open(
        right, framework="pt", device="cpu"
    ) as second:
        first_keys = sorted(first.keys())
        if set(first_keys) != set(second.keys()):
            raise ContractError("TA alpha=0.25 and uniform soup keysets differ")
        for key in first_keys:
            lhs, rhs = first.get_tensor(key), second.get_tensor(key)
            if lhs.dtype != rhs.dtype or lhs.shape != rhs.shape or not torch.equal(lhs, rhs):
                raise ContractError(f"TA alpha=0.25 differs from uniform soup at {key}")
            count += 1
    left_sidecars = left_policy_identity["sidecar_tree"]
    right_sidecars = right_policy_identity["sidecar_tree"]
    left_canonical = [
        (row["path"], row["canonical_sha256"], row["canonicalization"])
        for row in left_sidecars["sidecars"]
    ]
    right_canonical = [
        (row["path"], row["canonical_sha256"], row["canonicalization"])
        for row in right_sidecars["sidecars"]
    ]
    if left_canonical != right_canonical:
        raise ContractError("TA alpha=0.25 and uniform soup runtime sidecar trees differ")
    return {
        "schema_version": SCHEMA_VERSION,
        "fastlane_version": FASTLANE_VERSION,
        "kind": "task_arithmetic_alpha_0p25_uniform_soup_full_policy_parity",
        "status": "passed_full_runtime_policy_parity",
        "created_at": utc_now(),
        # Both full-file hashes were already verified by the caller.  Reusing
        # them avoids two redundant 9.35 GB reads after tensor-exact parity.
        "left_model_sha256": left_sha256,
        "right_model_sha256": right_sha256,
        "tensor_count": count,
        "equal_tensor_count": count,
        "max_abs_error": 0.0,
        "runtime_sidecar_count": len(left_canonical),
        "runtime_sidecar_canonical_tree_sha256": left_sidecars[
            "canonical_tree_sha256"
        ],
        "left_full_policy_identity": left_policy_identity,
        "right_full_policy_identity": right_policy_identity,
        "nonruntime_provenance_excluded": ["parameter_baseline_manifest.json"],
        "prediction_parity_claimed": False,
    }


def _validate_alpha_quarter_parity(
    manifest: Mapping[str, Any], spec: CandidateSpec, candidate_root: Path
) -> None:
    row = manifest.get("uniform_soup_full_policy_parity")
    if spec.family != "task_arithmetic" or spec.alpha_text != SOUP_ALPHA:
        if row is not None:
            raise ContractError(f"unexpected uniform-soup parity receipt: {spec.candidate_id}")
        return
    if not isinstance(row, dict) or set(row) != {
        "status",
        "path",
        "sha256",
        "max_abs_error",
        "runtime_sidecar_count",
    }:
        raise ContractError("TA alpha=0.25 full-policy parity receipt is missing")
    expected_path = candidate_root / "uniform-soup-full-policy-parity.json"
    if Path(row["path"]).resolve(strict=True) != expected_path.resolve(strict=True):
        raise ContractError("TA alpha=0.25 parity receipt path differs")
    receipt = read_bound_json(expected_path, row["sha256"], "TA/soup full-policy parity")
    if (
        row["status"] != "passed_full_runtime_policy_parity"
        or row["max_abs_error"] != 0.0
        or row["runtime_sidecar_count"] != len(RUNTIME_POLICY_SIDECARS)
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("fastlane_version") != FASTLANE_VERSION
        or receipt.get("status") != "passed_full_runtime_policy_parity"
        or receipt.get("max_abs_error") != 0.0
        or receipt.get("runtime_sidecar_count") != len(RUNTIME_POLICY_SIDECARS)
        or receipt.get("left_full_policy_identity")
        != manifest.get("full_policy_identity")
        or receipt.get("right_model_sha256") != SOUP_MODEL_SHA256
        or receipt.get("right_full_policy_identity", {}).get("model", {}).get("sha256")
        != SOUP_MODEL_SHA256
        or receipt.get("right_full_policy_identity", {})
        .get("sidecar_tree", {})
        .get("canonical_tree_sha256")
        != SOUP_CANONICAL_SIDECAR_TREE_SHA256
        or receipt.get("runtime_sidecar_canonical_tree_sha256")
        != SOUP_CANONICAL_SIDECAR_TREE_SHA256
    ):
        raise ContractError("TA alpha=0.25 full-policy parity contract differs")


def export_direction_grid(
    *,
    output_root: Path,
    specs: Sequence[CandidateSpec],
    direction: np.ndarray,
    direction_manifest: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    components: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch and safetensors are required for export") from exc
    candidates_root = output_root / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[CandidateSpec, Path, Path, int, str]] = []
    temporary_paths: set[Path] = set()
    completed: list[dict[str, Any]] = []
    recovered: dict[str, list[str]] = {}
    descriptors: set[int] = set()
    soup_sha = contract["revision"]["method_searches"]["model_soups"][
        "checkpoint_sha256"
    ]
    soup_policy_identity = full_policy_identity(
        contract["soup_root"],
        soup_sha,
        expected_sidecars=RUNTIME_POLICY_SIDECARS,
        model_sha256_already_verified=soup_sha,
    )
    if (
        soup_sha != SOUP_MODEL_SHA256
        or soup_policy_identity["sidecar_tree"]["canonical_tree_sha256"]
        != SOUP_CANONICAL_SIDECAR_TREE_SHA256
    ):
        raise ContractError("frozen uniform-soup full policy identity differs")
    direction_manifest_path = Path(direction_manifest["direction"]["path"]).parent / (
        "direction-manifest.json"
    )
    direction_manifest_receipt = {
        "path": str(direction_manifest_path.resolve()),
        "sha256": sha256_file(direction_manifest_path),
    }
    started = time.monotonic()
    try:
        for spec in specs:
            prefix = f".{spec.candidate_id}.v2-partial-"
            recovered[spec.candidate_id] = _cleanup_owned_partials(candidates_root, prefix)
            if spec.family == "task_arithmetic" and spec.alpha_text == "0.00":
                completed.append(_reference_manifest(output_root, spec, contract, components))
                continue
            final = candidates_root / spec.candidate_id
            if final.exists():
                manifest = verify_complete_checkpoint(final)
                if (
                    manifest.get("candidate") != spec.as_dict()
                    or manifest.get("runtime_components") != components
                    or manifest.get("direction_manifest") != direction_manifest_receipt
                ):
                    raise ContractError(f"completed candidate contract differs: {spec.candidate_id}")
                _validate_alpha_quarter_parity(manifest, spec, final)
                completed.append(manifest)
                continue
            temporary = candidates_root / (
                f".{spec.candidate_id}.v2-partial-{os.getpid()}-{uuid.uuid4().hex}"
            )
            policy = temporary / "pretrained_model"
            policy.mkdir(parents=True)
            temporary_paths.add(temporary)
            copy_policy_sidecars(
                contract["soup_root"], policy, final / "pretrained_model"
            )
            clone_method = clone_file(contract["base_model"], policy / "model.safetensors")
            descriptor = os.open(policy / "model.safetensors", os.O_RDWR)
            descriptors.add(descriptor)
            pending.append((spec, final, temporary, descriptor, clone_method))
        if not pending:
            return completed

        layout = parse_safetensors_layout(contract["base_model"])
        export_ok = False
        try:
            with safe_open(contract["base_model"], framework="pt", device="cpu") as base:
                for index, segment in enumerate(segments, start=1):
                    key = str(segment["key"])
                    row = layout.get(key)
                    if row is None or row["shape"] != segment["shape"]:
                        raise ContractError(f"base safetensors layout differs for {key}")
                    base_value = base.get_tensor(key).detach().cpu()
                    base64 = base_value.double()
                    start, end = int(segment["start"]), int(segment["end"])
                    direction_value = torch.from_numpy(
                        np.asarray(direction[start:end])
                    ).reshape(base_value.shape)
                    expected_bytes = int(row["end"]) - int(row["start"])
                    for spec, _, _, descriptor, _ in pending:
                        value = (
                            base64 + float(spec.alpha) * direction_value
                        ).to(base_value.dtype)
                        if not torch.isfinite(value).all():
                            raise ContractError(
                                f"{spec.candidate_id}/{key}: non-finite output"
                            )
                        raw = value.contiguous().view(torch.uint8).numpy().tobytes()
                        if len(raw) != expected_bytes:
                            raise ContractError(
                                f"{spec.candidate_id}/{key}: serialized size differs"
                            )
                        pwrite_all(descriptor, raw, int(row["start"]))
                    if index % 10 == 0 or index == len(segments):
                        print(
                            json.dumps(
                                {
                                    "phase": "checkpoint_export",
                                    "direction": direction_manifest["identity"]["direction"],
                                    "patched_tensors": index,
                                    "total_tensors": len(segments),
                                    "checkpoint_count": len(pending),
                                }
                            ),
                            flush=True,
                        )
            export_ok = True
        finally:
            for descriptor in list(descriptors):
                try:
                    if export_ok:
                        os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                    descriptors.remove(descriptor)

        for spec, final, temporary, _, clone_method in pending:
            policy = temporary / "pretrained_model"
            model = policy / "model.safetensors"
            model_sha = sha256_file(model)
            policy_identity = full_policy_identity(
                policy,
                model_sha,
                expected_sidecars=RUNTIME_POLICY_SIDECARS,
                model_sha256_already_verified=model_sha,
                reported_root=final / "pretrained_model",
            )
            parity = None
            parity_receipt = None
            if spec.family == "task_arithmetic" and spec.alpha_text == SOUP_ALPHA:
                parity = _compare_checkpoints_exact(
                    model,
                    contract["soup_model"],
                    left_sha256=model_sha,
                    right_sha256=soup_sha,
                    left_policy_identity=policy_identity,
                    right_policy_identity=soup_policy_identity,
                )
                parity_path = temporary / "uniform-soup-full-policy-parity.json"
                atomic_write_json(parity_path, parity)
                parity_receipt = {
                    "status": parity["status"],
                    "path": str(
                        (final / "uniform-soup-full-policy-parity.json").resolve()
                    ),
                    "sha256": sha256_file(parity_path),
                    "max_abs_error": parity["max_abs_error"],
                    "runtime_sidecar_count": parity["runtime_sidecar_count"],
                }
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "fastlane_version": FASTLANE_VERSION,
                "kind": "iclr2027_table1_ta_ties_candidate",
                "status": "complete_full_checkpoint",
                "created_at": utc_now(),
                "candidate": spec.as_dict(),
                "inputs": {
                    "revision7": {
                        "path": str(contract["revision_path"]),
                        "sha256": REVISION7_SHA256,
                    },
                    "dense_expert_bank": {
                        "path": str(contract["dense_bank_path"]),
                        "sha256": DENSE_BANK_SHA256,
                    },
                    "base_model_sha256": BASE_MODEL_SHA256,
                    "expert_model_sha256": EXPERT_MODEL_SHA256,
                    "uniform_soup_nonruntime_manifest": {
                        "path": contract["revision"]["method_searches"][
                            "model_soups"
                        ]["checkpoint_manifest_path"],
                        "sha256": contract["revision"]["method_searches"][
                            "model_soups"
                        ]["checkpoint_manifest_sha256"],
                    },
                },
                "runtime_components": components,
                "direction": direction_manifest["direction"],
                "direction_manifest": direction_manifest_receipt,
                "fusion_metadata": direction_manifest["fusion_metadata"],
                "checkpoint": {
                    "kind": "single_merged_dense_checkpoint",
                    "path": str((final / "pretrained_model").resolve()),
                    "model_path": str(
                        (final / "pretrained_model/model.safetensors").resolve()
                    ),
                    "model_sha256": model_sha,
                    "model_bytes": model.stat().st_size,
                    "copy_strategy": clone_method,
                    "modified_tensor_count": EXPECTED_TENSOR_COUNT,
                },
                "full_policy_identity": policy_identity,
                "finite": True,
                "uniform_soup_full_policy_parity": parity_receipt,
                "resume_recovered_partials": recovered[spec.candidate_id],
                "formal_outcomes_read": False,
                "wall_seconds_grid_export_so_far": time.monotonic() - started,
            }
            atomic_write_json(temporary / "candidate-manifest.json", manifest)
            _fsync_tree(temporary)
            temporary.rename(final)
            temporary_paths.discard(temporary)
            _fsync_directory(candidates_root)
            completed.append(manifest)
            print(
                json.dumps(
                    {
                        "phase": "candidate_complete",
                        "candidate": spec.candidate_id,
                        "model_sha256": model_sha,
                        "policy_identity_sha256": policy_identity["identity_sha256"],
                        "copy_strategy": clone_method,
                    }
                ),
                flush=True,
            )
        return completed
    finally:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        for temporary in temporary_paths:
            if temporary.exists():
                shutil.rmtree(temporary)


def _v1_input_contract() -> dict[str, Any]:
    return {
        "revision7_sha256": REVISION7_SHA256,
        "dense_bank_sha256": DENSE_BANK_SHA256,
        "expert_bank_sha256": EXPERT_BANK_SHA256,
        "base_model_sha256": BASE_MODEL_SHA256,
        "expert_model_sha256": EXPERT_MODEL_SHA256,
        "expert_order": list(EXPERT_ORDER),
        "tensor_count": EXPECTED_TENSOR_COUNT,
        "coordinate_count": EXPECTED_COORDINATE_COUNT,
        "task_vector_dtype": "float32",
        "fusion_dtype": "float64",
        "task_vector_semantics": "dense_peft_safe_expert_minus_common_base",
    }


def _load_verified_v1_ta_source() -> tuple[dict[str, Any], dict[str, Any]]:
    require_no_symlink_components(CANONICAL_V1_TA_ROOT, "v1 TA source root")
    if not CANONICAL_V1_TA_ROOT.is_dir():
        raise FileNotFoundError(CANONICAL_V1_TA_ROOT)
    index_path = CANONICAL_V1_TA_ROOT / "candidate-index.json"
    execution_path = CANONICAL_V1_TA_ROOT / "execution-receipt.json"
    index = read_bound_json(index_path, V1_TA_INDEX_SHA256, "frozen v1 TA index")
    execution = read_bound_json(
        execution_path, V1_TA_EXECUTION_SHA256, "frozen v1 TA execution receipt"
    )
    expected_index_keys = {
        "candidates",
        "complete_by_family",
        "complete_candidate_count",
        "created_at",
        "expected_candidate_count",
        "family_status",
        "inputs",
        "kind",
        "missing_candidates",
        "schema_version",
        "status",
    }
    expected_execution_keys = {
        "cache_root",
        "candidate_counts",
        "candidate_index_sha256",
        "created_at",
        "family",
        "finished_at",
        "formal_outcomes_read",
        "implementation",
        "input_contract",
        "kind",
        "output_root",
        "resource_floor",
        "schema_version",
        "status",
    }
    if (
        set(index) != expected_index_keys
        or index.get("schema_version") != 1
        or index.get("kind") != "iclr2027_table1_ta_ties_candidate_index"
        or index.get("status") != "partial"
        or index.get("expected_candidate_count") != 90
        or index.get("complete_candidate_count") != 21
        or index.get("complete_by_family")
        != {"task_arithmetic": 21, "ties_merging": 0}
        or index.get("family_status")
        != {"task_arithmetic": "complete", "ties_merging": "partial"}
        or index.get("inputs") != _v1_input_contract()
        or [row.get("candidate") for row in index.get("candidates", [])]
        != [spec.as_dict() for spec in task_arithmetic_specs()]
        or index.get("missing_candidates")
        != [spec.candidate_id for spec in ties_specs()]
    ):
        raise ContractError("frozen v1 TA candidate index contract differs")
    if (
        set(execution) != expected_execution_keys
        or execution.get("schema_version") != 1
        or execution.get("kind") != "iclr2027_table1_ta_ties_fastlane_plan"
        or execution.get("status") != "requested_families_complete"
        or execution.get("family") != "task_arithmetic"
        or execution.get("output_root") != str(CANONICAL_V1_TA_ROOT)
        or execution.get("candidate_index_sha256") != V1_TA_INDEX_SHA256
        or execution.get("candidate_counts")
        != {"task_arithmetic": 21, "ties_merging": 69, "total": 90}
        or execution.get("input_contract") != _v1_input_contract()
        or execution.get("formal_outcomes_read") is not False
    ):
        raise ContractError("frozen v1 TA execution receipt contract differs")
    return index, execution


def _validate_v1_candidate_row(
    row: Mapping[str, Any], spec: CandidateSpec
) -> tuple[dict[str, Any], Path]:
    if set(row) != {"candidate", "checkpoint", "manifest", "manifest_sha256"}:
        raise ContractError(f"v1 index row schema differs: {spec.candidate_id}")
    if row.get("candidate") != spec.as_dict():
        raise ContractError(f"v1 candidate recipe differs: {spec.candidate_id}")
    source_root = CANONICAL_V1_TA_ROOT / "candidates" / spec.candidate_id
    manifest_path = source_root / "candidate-manifest.json"
    require_no_symlink_components(source_root, "v1 candidate root")
    require_no_symlink_components(manifest_path, "v1 candidate manifest")
    if Path(str(row["manifest"])).resolve(strict=True) != manifest_path:
        raise ContractError(f"v1 candidate manifest path differs: {spec.candidate_id}")
    manifest = read_bound_json(
        manifest_path, str(row["manifest_sha256"]), f"v1 {spec.candidate_id} manifest"
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "iclr2027_table1_ta_ties_candidate"
        or manifest.get("candidate") != spec.as_dict()
        or manifest.get("checkpoint") != row["checkpoint"]
        or manifest.get("finite") is not True
    ):
        raise ContractError(f"v1 candidate manifest differs: {spec.candidate_id}")
    return manifest, manifest_path


def _v2_adoption_manifest(
    *,
    spec: CandidateSpec,
    source_manifest: Mapping[str, Any],
    source_manifest_path: Path,
    checkpoint: Mapping[str, Any],
    policy_identity: Mapping[str, Any],
    components: Mapping[str, Any],
    parity_receipt: Mapping[str, Any] | None,
    revision_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "fastlane_version": FASTLANE_VERSION,
        "kind": "iclr2027_table1_ta_ties_candidate",
        "status": "complete_full_checkpoint",
        "created_at": utc_now(),
        "candidate": spec.as_dict(),
        "inputs": {
            "revision7": {"path": str(revision_path), "sha256": REVISION7_SHA256},
            "dense_expert_bank_sha256": DENSE_BANK_SHA256,
            "base_model_sha256": BASE_MODEL_SHA256,
            "expert_model_sha256": EXPERT_MODEL_SHA256,
        },
        "runtime_components": components,
        "checkpoint": dict(checkpoint),
        "full_policy_identity": dict(policy_identity),
        "finite": True,
        "source_adoption": {
            "mode": "verified_v1_read_only_reference",
            "source_root": str(CANONICAL_V1_TA_ROOT),
            "source_index": {
                "path": str(CANONICAL_V1_TA_ROOT / "candidate-index.json"),
                "sha256": V1_TA_INDEX_SHA256,
            },
            "source_execution_receipt": {
                "path": str(CANONICAL_V1_TA_ROOT / "execution-receipt.json"),
                "sha256": V1_TA_EXECUTION_SHA256,
            },
            "source_candidate_manifest": {
                "path": str(source_manifest_path),
                "sha256": sha256_file(source_manifest_path),
            },
            "source_v1_code_observed_before_upgrade": {
                "core_sha256": V1_TA_CORE_SHA256,
                "materializer_sha256": V1_TA_MATERIALIZER_SHA256,
            },
            "v1_files_modified": False,
            "model_recomputed": "full_sha256_and_safetensors_header",
            "sidecars_recomputed": "complete_runtime_tree_sha256",
        },
        "fusion_metadata": source_manifest.get("fusion_metadata"),
        "direction": source_manifest.get("direction"),
        "uniform_soup_full_policy_parity": parity_receipt,
        "formal_outcomes_read": False,
    }


def adopt_v1_task_arithmetic(
    contract: Mapping[str, Any],
    components: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for v1 adoption") from exc
    index, execution = _load_verified_v1_ta_source()
    direction_path = (
        CANONICAL_V1_TA_ROOT
        / "cache/directions-v1/task_arithmetic_unscaled_sum.f64.bin"
    )
    require_no_symlink_components(direction_path, "v1 TA direction")
    if (
        not direction_path.is_file()
        or direction_path.stat().st_size != EXPECTED_COORDINATE_COUNT * 8
        or sha256_file(direction_path) != V1_TA_DIRECTION_SHA256
    ):
        raise ContractError("v1 TA reusable direction differs")
    direction_stat = _stat_identity(direction_path)
    soup_identity = full_policy_identity(
        contract["soup_root"],
        SOUP_MODEL_SHA256,
        expected_sidecars=RUNTIME_POLICY_SIDECARS,
        model_sha256_already_verified=SOUP_MODEL_SHA256,
    )
    if (
        soup_identity["sidecar_tree"]["canonical_tree_sha256"]
        != SOUP_CANONICAL_SIDECAR_TREE_SHA256
    ):
        raise ContractError("uniform-soup sidecar identity differs during adoption")

    candidates_root = output_root / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    adoption_rows = []
    observed_model_hashes: set[str] = set()
    source_model_stats: dict[str, dict[str, int]] = {}
    sidecar_rechecks: dict[str, dict[str, Any]] = {}
    for spec, source_row in zip(task_arithmetic_specs(), index["candidates"], strict=True):
        source_manifest, source_manifest_path = _validate_v1_candidate_row(source_row, spec)
        source_checkpoint = source_manifest["checkpoint"]
        final = candidates_root / spec.candidate_id
        recovered = _cleanup_owned_partials(
            candidates_root, f".{spec.candidate_id}.v2-adopt-partial-"
        )
        if spec.alpha_text == "0.00":
            if (
                source_manifest.get("status") != "complete_base_reference"
                or source_checkpoint.get("kind") != "common_base_reference_no_duplicate"
                or Path(source_checkpoint["path"]).resolve(strict=True)
                != contract["base_root"]
                or source_checkpoint.get("model_sha256") != BASE_MODEL_SHA256
            ):
                raise ContractError("v1 alpha-zero base reference differs")
            if final.exists():
                checkpoint = {
                    "kind": "common_base_hardlink_reference",
                    "path": str(final / "pretrained_model"),
                    "model_path": str(final / "pretrained_model/model.safetensors"),
                    "model_sha256": BASE_MODEL_SHA256,
                    "model_bytes": contract["base_model"].stat().st_size,
                    "copy_strategy": "same_inode_hardlink_zero_model_bytes",
                    "modified_tensor_count": 0,
                }
                policy_identity = full_policy_identity(
                    final / "pretrained_model",
                    BASE_MODEL_SHA256,
                    expected_sidecars=RUNTIME_POLICY_SIDECARS,
                )
            else:
                checkpoint = None
                policy_identity = None
            parity_row = None
        else:
            if (
                source_manifest.get("status") != "complete_full_checkpoint"
                or source_manifest.get("formal_outcomes_read") is not False
                or source_checkpoint.get("kind") != "single_merged_dense_checkpoint"
                or source_checkpoint.get("modified_tensor_count") != EXPECTED_TENSOR_COUNT
                or source_checkpoint.get("model_bytes") != contract["base_model"].stat().st_size
            ):
                raise ContractError(f"v1 checkpoint contract differs: {spec.candidate_id}")
            source_policy = (
                CANONICAL_V1_TA_ROOT
                / "candidates"
                / spec.candidate_id
                / "pretrained_model"
            )
            source_model = source_policy / "model.safetensors"
            if (
                Path(source_checkpoint["path"]).resolve(strict=True) != source_policy
                or Path(source_checkpoint["model_path"]).resolve(strict=True) != source_model
            ):
                raise ContractError(f"v1 checkpoint path differs: {spec.candidate_id}")
            expected_model_sha = str(source_checkpoint.get("model_sha256"))
            observed_model_sha = sha256_file(source_model)
            if observed_model_sha != expected_model_sha:
                raise ContractError(f"v1 model SHA256 differs: {spec.candidate_id}")
            with safe_open(source_model, framework="pt", device="cpu") as handle:
                if len(list(handle.keys())) != 813:
                    raise ContractError(f"v1 safetensors key count differs: {spec.candidate_id}")
            policy_identity = full_policy_identity(
                source_policy,
                expected_model_sha,
                expected_sidecars=RUNTIME_POLICY_SIDECARS,
                model_sha256_already_verified=observed_model_sha,
            )
            if (
                policy_identity["sidecar_tree"]["canonical_tree_sha256"]
                != SOUP_CANONICAL_SIDECAR_TREE_SHA256
                or policy_identity["sidecar_tree"]["ignored_nonruntime_provenance"]
                != [
                    {
                        "path": "parameter_baseline_manifest.json",
                        "bytes": source_manifest_path.stat().st_size,
                        "sha256": sha256_file(source_manifest_path),
                    }
                ]
            ):
                raise ContractError(f"v1 full policy sidecar provenance differs: {spec.candidate_id}")
            if source_manifest.get("direction") != {
                "path": str(direction_path),
                "bytes": EXPECTED_COORDINATE_COUNT * 8,
                "sha256": V1_TA_DIRECTION_SHA256,
            }:
                raise ContractError(f"v1 direction binding differs: {spec.candidate_id}")
            if spec.alpha_text == "0.25" and expected_model_sha != SOUP_MODEL_SHA256:
                raise ContractError("v1 alpha=0.25 checkpoint is not the frozen soup model")
            if spec.alpha_text == "0.40" and expected_model_sha != V1_TA_ALPHA_0P40_SHA256:
                raise ContractError("v1 alpha=0.40 independent checkpoint anchor differs")
            observed_model_hashes.add(observed_model_sha)
            source_model_stats[str(source_model)] = _stat_identity(source_model)
            checkpoint = {
                **source_checkpoint,
                "kind": "adopted_v1_read_only_dense_checkpoint",
                "copy_strategy": "read_only_v1_reference_no_copy",
            }
            parity_row = None
            if spec.alpha_text == SOUP_ALPHA:
                temporary_parity_identity = policy_identity
                parity = _compare_checkpoints_exact(
                    source_model,
                    contract["soup_model"],
                    left_sha256=observed_model_sha,
                    right_sha256=SOUP_MODEL_SHA256,
                    left_policy_identity=temporary_parity_identity,
                    right_policy_identity=soup_identity,
                )
                parity_row = {
                    "status": parity["status"],
                    "path": str(final / "uniform-soup-full-policy-parity.json"),
                    "sha256": "PENDING_ATOMIC_ADOPTION",
                    "max_abs_error": 0.0,
                    "runtime_sidecar_count": len(RUNTIME_POLICY_SIDECARS),
                }

        temporary = candidates_root / (
            f".{spec.candidate_id}.v2-adopt-partial-{os.getpid()}-{uuid.uuid4().hex}"
        )
        if final.exists():
            require_no_symlink_components(final, "existing v2 adoption candidate")
            expected_files = {"candidate-manifest.json"}
            if spec.alpha_text == "0.00":
                expected_files |= {
                    "pretrained_model/model.safetensors",
                    *{
                        f"pretrained_model/{relative}"
                        for relative in RUNTIME_POLICY_SIDECARS
                    },
                }
            if spec.alpha_text == SOUP_ALPHA:
                expected_files.add("uniform-soup-full-policy-parity.json")
            observed_files = {
                path.relative_to(final).as_posix()
                for path in final.rglob("*")
                if path.is_file()
            }
            if observed_files != expected_files or any(path.is_symlink() for path in final.rglob("*")):
                raise ContractError(f"existing v2 adoption file tree differs: {spec.candidate_id}")
            existing = json.loads((final / "candidate-manifest.json").read_text(encoding="utf-8"))
            expected_existing = _v2_adoption_manifest(
                spec=spec,
                source_manifest=source_manifest,
                source_manifest_path=source_manifest_path,
                checkpoint=checkpoint,
                policy_identity=policy_identity,
                components=components,
                parity_receipt=existing.get("uniform_soup_full_policy_parity"),
                revision_path=contract["revision_path"],
            )
            stable_fields = set(expected_existing) - {"created_at"}
            if (
                set(existing) != set(expected_existing) | {"resume_recovered_partials"}
                or any(existing.get(field) != expected_existing[field] for field in stable_fields)
                or not isinstance(existing.get("resume_recovered_partials"), list)
            ):
                raise ContractError(f"existing v2 adoption differs: {spec.candidate_id}")
            _validate_alpha_quarter_parity(existing, spec, final)
            v2_manifest = existing
        else:
            temporary.mkdir()
            try:
                if spec.alpha_text == "0.00":
                    (temporary / "pretrained_model").mkdir()
                    checkpoint, policy_identity = _prepare_base_policy_reference(
                        temporary / "pretrained_model",
                        final / "pretrained_model",
                        contract,
                    )
                if spec.alpha_text == SOUP_ALPHA:
                    parity_path = temporary / "uniform-soup-full-policy-parity.json"
                    atomic_write_json(parity_path, parity)
                    parity_row = {
                        **parity_row,
                        "sha256": sha256_file(parity_path),
                    }
                v2_manifest = _v2_adoption_manifest(
                    spec=spec,
                    source_manifest=source_manifest,
                    source_manifest_path=source_manifest_path,
                    checkpoint=checkpoint,
                    policy_identity=policy_identity,
                    components=components,
                    parity_receipt=parity_row,
                    revision_path=contract["revision_path"],
                )
                v2_manifest["resume_recovered_partials"] = recovered
                atomic_write_json(temporary / "candidate-manifest.json", v2_manifest)
                _fsync_tree(temporary)
                temporary.rename(final)
                _fsync_directory(candidates_root)
            except BaseException:
                if temporary.exists():
                    shutil.rmtree(temporary)
                raise
        adoption_rows.append(
            {
                "candidate": spec.as_dict(),
                "source_manifest": {
                    "path": str(source_manifest_path),
                    "sha256": sha256_file(source_manifest_path),
                },
                "source_checkpoint": source_checkpoint,
                "full_policy_identity": policy_identity,
                "v2_manifest": {
                    "path": str(final / "candidate-manifest.json"),
                    "sha256": sha256_file(final / "candidate-manifest.json"),
                },
            }
        )
        effective_policy_root = Path(checkpoint["path"])
        sidecar_rechecks[str(effective_policy_root)] = {
            "tree": policy_identity["sidecar_tree"],
            "stats": _sidecar_stat_snapshot(effective_policy_root),
        }
    if len(observed_model_hashes) != 20:
        raise ContractError("v1 nonzero TA candidate model hashes are not all distinct")
    for raw_path, expected_stat in source_model_stats.items():
        if _stat_identity(Path(raw_path)) != expected_stat:
            raise ContractError(f"v1 source model changed during adoption: {raw_path}")
    for raw_root, expected in sidecar_rechecks.items():
        policy_root = Path(raw_root)
        if _sidecar_stat_snapshot(policy_root) != expected["stats"]:
            raise ContractError(f"policy sidecar stats changed during adoption: {policy_root}")
        observed_tree = policy_sidecar_tree(
            policy_root,
            expected_sidecars=RUNTIME_POLICY_SIDECARS,
            expected_config_pretrained_path=policy_root,
        )
        if observed_tree != expected["tree"]:
            raise ContractError(f"policy sidecar rehash changed during adoption: {policy_root}")
    if (
        sha256_file(CANONICAL_V1_TA_ROOT / "candidate-index.json")
        != V1_TA_INDEX_SHA256
        or sha256_file(CANONICAL_V1_TA_ROOT / "execution-receipt.json")
        != V1_TA_EXECUTION_SHA256
    ):
        raise ContractError("v1 source index/receipt changed during adoption")
    if _stat_identity(direction_path) != direction_stat:
        raise ContractError("v1 source direction changed during adoption")
    adoption_root = output_root / "adoptions/task-arithmetic-v1-read-only-v2"
    adoption_root.mkdir(parents=True, exist_ok=True)
    receipt_path = adoption_root / "adoption-receipt.json"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "fastlane_version": FASTLANE_VERSION,
        "kind": "iclr2027_table1_task_arithmetic_v1_to_v2_adoption",
        "status": "complete_immutable_read_only_adoption",
        "created_at": utc_now(),
        "source": {
            "root": str(CANONICAL_V1_TA_ROOT),
            "candidate_index": {
                "path": str(CANONICAL_V1_TA_ROOT / "candidate-index.json"),
                "sha256": V1_TA_INDEX_SHA256,
            },
            "execution_receipt": {
                "path": str(CANONICAL_V1_TA_ROOT / "execution-receipt.json"),
                "sha256": V1_TA_EXECUTION_SHA256,
            },
            "direction": {
                "path": str(direction_path),
                "sha256": V1_TA_DIRECTION_SHA256,
                "bytes": EXPECTED_COORDINATE_COUNT * 8,
            },
            "pre_upgrade_observed_code_sha256": {
                "core": V1_TA_CORE_SHA256,
                "materializer": V1_TA_MATERIALIZER_SHA256,
            },
        },
        "verification": {
            "candidate_count": len(adoption_rows),
            "full_model_sha256_recomputed_count": 21,
            "safetensors_header_key_count_checked": 20,
            "full_policy_sidecar_tree_recomputed_count": 21,
            "alpha_0p25_full_policy_parity": "passed",
            "alpha_0p40_anchor_sha256": V1_TA_ALPHA_0P40_SHA256,
            "v1_files_modified": False,
            "v2_model_bytes_written": 0,
        },
        "runtime_components": components,
        "candidates": adoption_rows,
        "formal_outcomes_read": False,
        "source_execution_summary": {
            "finished_at": execution.get("finished_at"),
            "candidate_index_sha256": execution.get("candidate_index_sha256"),
        },
    }
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if set(existing) != set(receipt):
            raise ContractError("immutable v1 adoption receipt schema differs")
        stable = set(receipt) - {"created_at"}
        if any(existing.get(field) != receipt[field] for field in stable):
            raise ContractError("immutable v1 adoption receipt differs")
        receipt = existing
    else:
        atomic_create_json(receipt_path, receipt)
    receipt_row = {"path": str(receipt_path), "sha256": sha256_file(receipt_path)}
    index_receipt = write_family_indexes(
        output_root,
        "task_arithmetic",
        components,
        {
            "mode": "verified_v1_read_only_adoption",
            "adoption_receipt": receipt_row,
        },
    )
    return {"adoption_receipt": receipt_row, "family_index": index_receipt}


def _family_specs(family: str) -> list[CandidateSpec]:
    if family == "task_arithmetic":
        return task_arithmetic_specs()
    if family == "ties_merging":
        return ties_specs()
    raise ValueError(f"unsupported family: {family}")


def write_family_indexes(
    output_root: Path,
    family: str,
    components: Mapping[str, Any],
    family_build: Mapping[str, Any],
) -> dict[str, Any]:
    specs = _family_specs(family)
    rows = []
    for spec in specs:
        path = output_root / "candidates" / spec.candidate_id / "candidate-manifest.json"
        if not path.is_file():
            raise ContractError(f"cannot freeze {family} index; candidate is missing: {spec.candidate_id}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("fastlane_version") != FASTLANE_VERSION
            or manifest.get("candidate") != spec.as_dict()
            or manifest.get("runtime_components") != components
            or manifest.get("status")
            not in {"complete_full_checkpoint", "complete_base_reference"}
        ):
            raise ContractError(f"candidate manifest differs before index freeze: {spec.candidate_id}")
        _validate_alpha_quarter_parity(manifest, spec, path.parent)
        rows.append(
            {
                "candidate": spec.as_dict(),
                "manifest": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
                "checkpoint": manifest["checkpoint"],
                "full_policy_identity": manifest["full_policy_identity"],
            }
        )
    slug = family.replace("_", "-")
    index_path = output_root / f"candidate-index-{slug}-v2.json"
    index = {
        "schema_version": SCHEMA_VERSION,
        "fastlane_version": FASTLANE_VERSION,
        "kind": "iclr2027_table1_ta_ties_family_candidate_index",
        "status": "complete_immutable",
        "created_at": utc_now(),
        "family": family,
        "candidate_count": len(rows),
        "inputs": _cache_identity({}),
        "runtime_components": components,
        "family_build": family_build,
        "candidates": rows,
        "overwrite_forbidden": True,
    }
    if index_path.exists():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if set(existing) != set(index):
            raise ContractError(f"immutable {family} index schema differs")
        stable_fields = set(index) - {"created_at"}
        if any(existing.get(field) != index[field] for field in stable_fields):
            raise ContractError(f"immutable {family} index differs from current candidates")
        index = existing
    else:
        atomic_create_json(index_path, index)
    index_sha = sha256_file(index_path)

    survivor_first = 4 if family == "task_arithmetic" else 6
    default = (
        {"global_alpha": 0.4}
        if family == "task_arithmetic"
        else {"density": 0.2, "global_alpha": 1.0}
    )
    selection_path = output_root / f"development-selection-{slug}-v2.json"
    selection = {
        "schema_version": SCHEMA_VERSION,
        "fastlane_version": FASTLANE_VERSION,
        "kind": "iclr2027_table1_ta_ties_development_selection_contract",
        "status": "ready_for_development_successive_halving",
        "created_at": utc_now(),
        "family": family,
        "candidate_index": {"path": str(index_path.resolve()), "sha256": index_sha},
        "runtime_components": components,
        "canonical_allowlist": {
            "candidate_root": str((output_root / "candidates").resolve()),
            "development_attempt_root": str(CANONICAL_DEVELOPMENT_ROOT),
            "legacy_eval_root": str(CANONICAL_LEGACY_EVAL_ROOT),
            "legacy_log_root": str(CANONICAL_LEGACY_LOG_ROOT),
            "lease_root": str(CANONICAL_LEASE_ROOT),
            "job_lock_root": str(CANONICAL_JOB_LOCK_ROOT),
            "quarantine_root": str(CANONICAL_DEVELOPMENT_ROOT / "quarantine"),
            "eval_wrapper": str(CANONICAL_EVAL_WRAPPER),
            "eval_entry": str(CANONICAL_EVAL_ENTRY),
            "python_runtime": str(CANONICAL_PYTHON),
        },
        "recipe": {
            "candidate_count": len(rows),
            "survivors_after_screen_01": survivor_first,
            "survivors_after_screen_03": 2,
            "primary_source_default": default,
        },
        "stages": [
            {
                "stage": "screen-01",
                "new_init_state_ids": [30],
                "episodes_per_task_cumulative": 1,
            },
            {
                "stage": "screen-03",
                "new_init_state_ids": [31, 32],
                "episodes_per_task_cumulative": 3,
            },
            {
                "stage": "select-10",
                "new_init_state_ids": list(range(33, 40)),
                "episodes_per_task_cumulative": 10,
            },
        ],
        "ranking": [
            "highest four-suite task-macro success rate",
            "highest worst-suite success rate",
            "smallest mean task-level failure rate",
            "closest to the primary-source default",
            "lexicographically smallest serialized recipe id",
        ],
        "selection_boundary": {
            "development_only": True,
            "allowlist_enforced": True,
            "resolved_paths_and_no_symlink_components_required": True,
            "exact_job_schema_required": True,
            "each_job_pinned_to_exact_host_index_uuid": True,
            "formal_outcomes_read": False,
            "procedural_final_outcomes_read": False,
            "libero_plus_used": False,
            "libero_pro_used": False,
        },
        "overwrite_forbidden": True,
    }
    if selection_path.exists():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if set(existing) != set(selection):
            raise ContractError(f"immutable {family} selection contract schema differs")
        stable_fields = set(selection) - {"created_at"}
        if any(existing.get(field) != selection[field] for field in stable_fields):
            raise ContractError(f"immutable {family} selection contract differs")
        selection = existing
    else:
        atomic_create_json(selection_path, selection)
    return {
        "family": family,
        "candidate_index": {"path": str(index_path), "sha256": sha256_file(index_path)},
        "selection_contract": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
    }


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    output_root = args.output_root.expanduser().absolute().resolve(strict=False)
    cache_root = args.cache_root.expanduser().absolute().resolve(strict=False)
    require_no_symlink_components(args.output_root, "TA/TIES output root")
    require_no_symlink_components(args.cache_root, "TA/TIES cache root")
    if output_root != CANONICAL_OUTPUT_ROOT or cache_root != CANONICAL_CACHE_ROOT:
        raise ContractError(
            "v2 output/cache roots must equal the canonical isolated roots: "
            f"output={CANONICAL_OUTPUT_ROOT}, cache={CANONICAL_CACHE_ROOT}"
        )
    if args.adopt_v1_task_arithmetic and (
        args.family != "task_arithmetic" or args.stop_after_cache
    ):
        raise ContractError(
            "--adopt-v1-task-arithmetic requires --family task_arithmetic and cannot "
            "be combined with --stop-after-cache"
        )
    components = runtime_component_identities()
    contract = validate_inputs(
        args.revision7_manifest,
        args.expert_dense_bank,
        verify_large_files=not args.plan_only,
    )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "fastlane_version": FASTLANE_VERSION,
        "kind": "iclr2027_table1_ta_ties_fastlane_plan",
        "created_at": utc_now(),
        "family": args.family,
        "execution_mode": (
            "verified_v1_read_only_adoption"
            if args.adopt_v1_task_arithmetic
            else "native_v2_materialization"
        ),
        "output_root": str(output_root),
        "cache_root": str(cache_root),
        "candidate_counts": {"task_arithmetic": 21, "ties_merging": 69, "total": 90},
        "input_contract": _cache_identity(contract),
        "runtime_components": components,
        "canonical_roots": {
            "output": str(CANONICAL_OUTPUT_ROOT),
            "cache": str(CANONICAL_CACHE_ROOT),
            "development": str(CANONICAL_DEVELOPMENT_ROOT),
            "legacy_eval": str(CANONICAL_LEGACY_EVAL_ROOT),
            "legacy_log": str(CANONICAL_LEGACY_LOG_ROOT),
            "lease": str(CANONICAL_LEASE_ROOT),
            "job_lock": str(CANONICAL_JOB_LOCK_ROOT),
        },
        "implementation": {
            "single_expert_vector_read": True,
            "task_arithmetic_direction_build_count": 1,
            "ties_direction_build_count_per_density": 1,
            "alpha_export_reuses_direction": True,
            "atomic_candidate_directories": True,
            "atomic_direction_directory_with_manifest": True,
            "family_indexes_are_versioned_immutable_and_non_overwriting": True,
            "resume_verifies_full_checkpoint_sha256": True,
        },
        "resource_floor": {
            "task_vector_cache_bytes": EXPECTED_COORDINATE_COUNT * 4 * 4,
            "one_float64_direction_bytes": EXPECTED_COORDINATE_COUNT * 8,
            "persistent_direction_count_all": 4,
            "minimum_ram_gib": 64,
            "recommended_ram_gib": 128,
            "gpu_required_for_materialization": False,
            "rollout_gpu_memory_mib_per_worker_estimate": 12_000,
        },
        "formal_outcomes_read": False,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.plan_only:
        return
    families = (
        ("task_arithmetic", "ties_merging")
        if args.family == "all"
        else (args.family,)
    )
    index_receipts: list[dict[str, Any]] = []
    with _GlobalMaterializerLock() as global_lock:
        output_root.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        if args.adopt_v1_task_arithmetic:
            verify_large_input_stats(contract)
            adoption = adopt_v1_task_arithmetic(contract, components, output_root)
            if runtime_component_identities() != components:
                raise ContractError("runtime components changed during v1 adoption")
            verify_large_input_stats(contract)
            receipt_root = output_root / "receipts"
            receipt_root.mkdir(exist_ok=True)
            receipt_path = receipt_root / (
                "materialization-task_arithmetic-adopt-v1-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
                f"pid{os.getpid()}.json"
            )
            atomic_create_json(
                receipt_path,
                {
                    **plan,
                    "status": "verified_v1_read_only_adoption_complete",
                    "finished_at": utc_now(),
                    "global_lock": global_lock,
                    "adoption": adoption,
                    "formal_outcomes_read": False,
                },
            )
            print(
                json.dumps(
                    {
                        "execution_receipt": str(receipt_path),
                        "sha256": sha256_file(receipt_path),
                    }
                ),
                flush=True,
            )
            return
        segments = build_segments(contract)
        vectors, vector_manifest = load_or_build_vectors(
            contract, segments, cache_root, components
        )
        verify_large_input_stats(contract)
        if "task_arithmetic" in families:
            direction, manifest = load_or_build_direction(
                cache_root=cache_root,
                name="task_arithmetic_unscaled_sum",
                vectors=vectors,
                chunk_size=args.chunk_size,
                density=None,
                components=components,
            )
            if not args.stop_after_cache:
                verify_large_input_stats(contract)
                export_direction_grid(
                    output_root=output_root,
                    specs=task_arithmetic_specs(),
                    direction=direction,
                    direction_manifest=manifest,
                    segments=segments,
                    contract=contract,
                    components=components,
                )
                if runtime_component_identities() != components:
                    raise ContractError("runtime components changed during TA materialization")
                direction_manifest_path = (
                    Path(manifest["direction"]["path"]).parent
                    / "direction-manifest.json"
                )
                index_receipts.append(
                    write_family_indexes(
                        output_root,
                        "task_arithmetic",
                        components,
                        {
                            "mode": "native_v2_materialization",
                            "direction_manifests": [
                                {
                                    "path": str(direction_manifest_path),
                                    "sha256": sha256_file(direction_manifest_path),
                                }
                            ],
                        },
                    )
                )
            del direction
        if "ties_merging" in families:
            ties_direction_receipts = []
            for density in (0.1, 0.2, 0.3):
                name = f"ties_density_{str(density).replace('.', 'p')}_direction"
                direction, manifest = load_or_build_direction(
                    cache_root=cache_root,
                    name=name,
                    vectors=vectors,
                    chunk_size=args.chunk_size,
                    density=density,
                    components=components,
                )
                if not args.stop_after_cache:
                    verify_large_input_stats(contract)
                    export_direction_grid(
                        output_root=output_root,
                        specs=[row for row in ties_specs() if row.density == density],
                        direction=direction,
                        direction_manifest=manifest,
                        segments=segments,
                        contract=contract,
                        components=components,
                    )
                    direction_manifest_path = (
                        Path(manifest["direction"]["path"]).parent
                        / "direction-manifest.json"
                    )
                    ties_direction_receipts.append(
                        {
                            "path": str(direction_manifest_path),
                            "sha256": sha256_file(direction_manifest_path),
                        }
                    )
                del direction
            if not args.stop_after_cache:
                if runtime_component_identities() != components:
                    raise ContractError("runtime components changed during TIES materialization")
                index_receipts.append(
                    write_family_indexes(
                        output_root,
                        "ties_merging",
                        components,
                        {
                            "mode": "native_v2_materialization",
                            "direction_manifests": ties_direction_receipts,
                        },
                    )
                )
        receipt_root = output_root / "receipts"
        verify_large_input_stats(contract)
        if runtime_component_identities() != components:
            raise ContractError("runtime components changed before execution receipt")
        receipt_root.mkdir(exist_ok=True)
        receipt_path = receipt_root / (
            f"materialization-{args.family}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"pid{os.getpid()}.json"
        )
        atomic_create_json(
            receipt_path,
            {
                **plan,
                "status": (
                    "cache_complete"
                    if args.stop_after_cache
                    else "requested_families_complete"
                ),
                "finished_at": utc_now(),
                "global_lock": global_lock,
                "task_vector_cache_manifest": {
                    "path": str(
                        (cache_root / "task-vectors-v2/cache-manifest.json").resolve()
                    ),
                    "sha256": sha256_file(
                        cache_root / "task-vectors-v2/cache-manifest.json"
                    ),
                    "identity": vector_manifest["identity"],
                },
                "family_indexes": index_receipts,
                "formal_outcomes_read": False,
            },
        )
        print(
            json.dumps(
                {"execution_receipt": str(receipt_path), "sha256": sha256_file(receipt_path)}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
