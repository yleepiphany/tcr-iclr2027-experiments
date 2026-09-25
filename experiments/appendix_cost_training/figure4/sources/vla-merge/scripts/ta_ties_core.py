#!/usr/bin/env python3
"""Pure-NumPy contracts and kernels for the Table-1 TA/TIES fastlane.

The production materializer imports this module, while the test suite can run
without the very large PI0.5/PyTorch environment.  The numerical definition
matches the repository's audited kernels: task vectors are float32, fusion is
float64, TIES trims each expert globally, elects signs by mass, and disjoint
aggregation is a mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REVISION7_SHA256 = "329bd77011e3ff9d19cca08c71a0d3e481edf78f1f5e44ec3e0b7d9f8f26ac86"
DENSE_BANK_SHA256 = "d61a5f9e56bb0f76d8186a33dc26fe283cf8cfab220e903a460f92e4507f209b"
EXPERT_BANK_SHA256 = "295efdfca18b0c83497bfe499e0ad771a5a8b33aaf86ba424dab7607094d53c1"

BASE_MODEL_SHA256 = "f2165fad32a4b431ca1488a5b886f6738be9e8b480298ecc9e0b55d87d26f365"
EXPERT_MODEL_SHA256 = {
    "spatial": "008cb4a76898edbb70edb83bd54cb472722ba49b2760acfde24381e6d14fc75f",
    "object": "a74c9349a78e122ecf9799e404ad748ffebba0b814b95e3dae4a7b32837f1510",
    "goal": "af9e258354d759710d04f13d40ac6e2e4377d5063f56fa8e0ef574f8c96e2732",
    "long": "85ad87a5d9fbdcdee5d6ff9270a59355326b376f0d50fbc27ae606cbc9b7f8fc",
}
EXPERT_ORDER = tuple(EXPERT_MODEL_SHA256)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

FASTLANE_VERSION = "ta-ties-fastlane-v2"
CANONICAL_OUTPUT_ROOT = Path(
    "/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/"
    "iclr2027-table1-20260910/libero/ta-ties-fastlane-v2"
)
CANONICAL_V1_TA_ROOT = Path(
    "/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/"
    "iclr2027-table1-20260910/libero/ta-ties-fastlane-v1"
)
CANONICAL_CACHE_ROOT = CANONICAL_OUTPUT_ROOT / "cache-v2"
CANONICAL_DEVELOPMENT_ROOT = CANONICAL_OUTPUT_ROOT / "development-v2"
CANONICAL_LEGACY_EVAL_ROOT = Path(
    "/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/"
    "pi05-spatial-object-lora-fusion-20260830-v1/eval"
)
CANONICAL_LEGACY_LOG_ROOT = Path(
    "/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/"
    "pi05-spatial-object-lora-fusion-20260830-v1/logs"
)
CANONICAL_LEASE_ROOT = Path(
    "/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/resource-leases"
)
CANONICAL_JOB_LOCK_ROOT = (
    CANONICAL_LEASE_ROOT / "iclr2027-table1-ta-ties-development-v2/jobs"
)
CANONICAL_REPO_SCRIPTS = Path(
    "/mnt/workspace/Wilson/parameter-fusion/vla-merge/scripts"
)
CANONICAL_EVAL_WRAPPER = CANONICAL_REPO_SCRIPTS / "eval_pi05_policy_suite.sh"
CANONICAL_EVAL_ENTRY = CANONICAL_REPO_SCRIPTS / "eval_pi05_libero_with_init_offset.py"
CANONICAL_PYTHON = Path(
    "/mnt/workspace/Wilson/parameter-fusion/pi05_lora_finetune_v2_20260826/"
    ".venv/bin/python"
)
CANONICAL_BASE_POLICY = Path(
    "/mnt/workspace/Wilson/parameter-fusion/pi05_lora_finetune_v2_20260826/"
    "ckpt/theta0/checkpoints/000200/pretrained_model"
)
EVAL_WRAPPER_SHA256 = "5b1dc67cd512f82e4ed5aa5103d38916e170242b6d011d8774709a9a583c4e77"
EVAL_ENTRY_SHA256 = "109ef6c26f78911688e6941ae72e19a9ae3b6acfed4b58642dffc4f96349ed60"
PYTHON_RUNTIME_SHA256 = "6bcd758cce71c048f2a5f69cf36a6af6f3038211807f2d5509bad71abaa5dbfe"
V1_TA_INDEX_SHA256 = "863cae71a243dcbbc3c36217a392ac12560578325758a7b26c41ad7058eee489"
V1_TA_EXECUTION_SHA256 = "cc3fdd50ba2e49125abc7d253c7f633038be986cb8e6d3ab8a1a7287a2181941"
V1_TA_CORE_SHA256 = "a38ec739c4033d1336fbc841bb9714425b60f9fd447a5af3743be6adcee0b17f"
V1_TA_MATERIALIZER_SHA256 = "e0008244375a9ef21b3b98d36c7fca116e5300632c26457ba56e3426c8fe995a"
V1_TA_DIRECTION_SHA256 = "8641d93301d9fff898dacf791f8d5a116eece51700b6fc1be701fefece1d0dbf"
V1_TA_ALPHA_0P40_SHA256 = "0df4baf0bebeae06c23dc2a4543af64afd0e711cd9cd2b6006d0227b95a99300"

HOST_GPU_UUIDS = {
    "dsw-824375-57c745db88-n6tv9": (
        "GPU-2c67b6c1-0cc2-9dbe-0e5c-00dfea7be3e2",
        "GPU-c18a3bf3-c47c-9a8a-75b0-5a0f37b33a65",
        "GPU-4466064b-48ab-bc29-c275-ce172788e2a6",
        "GPU-11e3fd57-cee9-f647-c043-ec44f9c798da",
        "GPU-48baba1c-f0e7-93b7-4a30-b3c21d20e61a",
        "GPU-4ed28198-b742-ee3c-2acb-4183dc944c81",
        "GPU-da625da2-f504-7927-4025-907e876718aa",
        "GPU-dbecf220-5094-c2fc-6e18-52483a9b2c50",
    ),
    "dsw-967394-56ffd4897d-42wft": (
        "GPU-f2a721cf-9182-5898-1da1-1be12e8b0fb1",
        "GPU-356564e4-6c34-94c2-fd25-0b485ebeaf03",
        "GPU-05093da4-f46b-ea6f-d964-d0a393deb6d8",
        "GPU-21ccffbd-eb6c-73a9-d63a-a5d200607a84",
        "GPU-792667d7-f14a-995e-fc18-c341be95c007",
        "GPU-059c0ed7-bcd0-3a71-69c1-2282316cddc8",
        "GPU-caceb295-b851-2bbf-c127-670161969774",
        "GPU-e027de95-4c80-d8ec-d650-30da9b6eb135",
    ),
}
CANONICAL_SOURCE_ROOT = Path(
    "/mnt/workspace/Wilson/parameter-fusion/pi05_lora_finetune_v2_20260826"
)
CANONICAL_LEROBOT_TREE = (
    CANONICAL_SOURCE_ROOT / "lerobot/src/lerobot/policies/pi05"
)
CANONICAL_SOURCE_OVERRIDE_TREE = CANONICAL_SOURCE_ROOT / "src"
CANONICAL_PYTHON_OVERLAY = Path(
    "/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/python-overlay"
)

RUNTIME_POLICY_SIDECARS = frozenset(
    {
        "README.md",
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    }
)

SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ContractError(ValueError):
    """Raised when an immutable experiment or artifact contract differs."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    alpha_text: str
    alpha: float
    density_text: str | None = None
    density: float | None = None

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": self.candidate_id,
            "family": self.family,
            "global_alpha": self.alpha,
            "global_alpha_decimal": self.alpha_text,
        }
        if self.density is not None:
            row.update(
                {
                    "density": self.density,
                    "density_decimal": self.density_text,
                    "disjoint": "mean",
                    "sign_election": "mass",
                }
            )
        return row


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: object, label: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        raise ContractError(f"{label} must be an explicit lowercase SHA256")
    return text


def read_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    expected_sha256 = require_sha(expected_sha256, f"expected {label} SHA256")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ContractError(
            f"{label} SHA256 differs: expected={expected_sha256}, observed={observed}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain a JSON object")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace one JSON receipt without exposing a partial document."""
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create an immutable JSON document atomically, never replacing a path.

    The fully fsynced temporary inode is hard-linked into place.  Linking fails
    atomically if another process already published the versioned receipt.
    """
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".create", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reject_development_name(path: Path, label: str) -> None:
    """Reject spelling/separator/case variants of non-development namespaces."""
    for component in path.parts:
        normalized = re.sub(r"[^a-z0-9]", "", component.lower())
        if (
            "formal" in normalized
            or "proceduralfinal" in normalized
            or "liberopro" in normalized
            or "liberoplus" in normalized
            or normalized.startswith("pro")
            or normalized.startswith("plus")
            or re.fullmatch(r"repeat0*[123][a-z0-9]*", normalized) is not None
            or re.fullmatch(r"repeat(one|two|three)[a-z0-9]*", normalized) is not None
        ):
            raise ContractError(
                f"{label} contains a forbidden formal/PRO/Plus/repeat namespace: {component}"
            )


def require_no_symlink_components(path: Path, label: str) -> None:
    """Reject symlinks in every currently existing path component."""
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            if current.is_symlink():
                raise ContractError(f"{label} contains a symlink component: {current}")
            current.lstat()
        except FileNotFoundError:
            # Descendants cannot exist once one component is absent.
            break


def resolve_under_root(
    path: Path,
    root: Path,
    label: str,
    *,
    must_exist: bool = False,
    reject_reserved_names: bool = True,
) -> Path:
    """Resolve one path, prove it stays in an allowlisted non-symlink root."""
    path = path.expanduser().absolute()
    root = root.expanduser().absolute()
    if reject_reserved_names:
        reject_development_name(path, label)
    require_no_symlink_components(root, f"{label} allowlist root")
    require_no_symlink_components(path, label)
    resolved_root = root.resolve(strict=root.exists())
    resolved = path.resolve(strict=must_exist)
    if resolved != resolved_root:
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ContractError(
                f"{label} escapes allowlist root: path={resolved}, root={resolved_root}"
            ) from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def policy_sidecar_tree(
    policy_root: Path,
    *,
    expected_sidecars: set[str] | frozenset[str] | None = None,
    ignored_provenance: set[str] | frozenset[str] = frozenset(
        {"parameter_baseline_manifest.json"}
    ),
    expected_config_pretrained_path: Path | None = None,
) -> dict[str, Any]:
    """Hash every runtime policy sidecar and a path-normalized config view."""
    requested_root = policy_root.expanduser().absolute()
    require_no_symlink_components(requested_root, "policy root")
    policy_root = requested_root.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    raw_config_pretrained_path: str | None = None
    for path in sorted(policy_root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"policy tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ContractError(f"policy tree contains a non-regular entry: {path}")
        relative = path.relative_to(policy_root).as_posix()
        if relative == "model.safetensors":
            continue
        raw_sha = sha256_file(path)
        if relative in ignored_provenance:
            ignored.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": raw_sha}
            )
            continue
        canonical_sha = raw_sha
        canonicalization = "byte_exact"
        if relative == "config.json":
            config = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ContractError("policy config must be a JSON object")
            config = dict(config)
            raw_config_pretrained_path = str(config.get("pretrained_path", ""))
            if (
                expected_config_pretrained_path is not None
                and raw_config_pretrained_path
                != str(expected_config_pretrained_path.expanduser().absolute().resolve(strict=False))
            ):
                raise ContractError(
                    "policy config pretrained_path differs from actual candidate policy root: "
                    f"observed={raw_config_pretrained_path}, "
                    f"expected={expected_config_pretrained_path}"
                )
            config["pretrained_path"] = "<POLICY_ROOT>"
            canonical_sha = canonical_json_sha256(config)
            canonicalization = "json_sorted_pretrained_path_normalized"
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": raw_sha,
                "canonical_sha256": canonical_sha,
                "canonicalization": canonicalization,
            }
        )
    observed = {row["path"] for row in rows}
    if expected_sidecars is not None and observed != set(expected_sidecars):
        raise ContractError(
            "policy runtime sidecar set differs: "
            f"missing={sorted(set(expected_sidecars) - observed)}, "
            f"extra={sorted(observed - set(expected_sidecars))}"
        )
    return {
        "root": str(policy_root),
        "sidecar_count": len(rows),
        "sidecars": rows,
        "raw_tree_sha256": canonical_json_sha256(
            [{key: row[key] for key in ("path", "bytes", "sha256")} for row in rows]
        ),
        "canonical_tree_sha256": canonical_json_sha256(
            [
                {
                    key: row[key]
                    for key in ("path", "canonical_sha256", "canonicalization")
                }
                for row in rows
            ]
        ),
        "ignored_nonruntime_provenance": ignored,
        "raw_config_pretrained_path": raw_config_pretrained_path,
        "raw_config_pretrained_path_matches_root": (
            expected_config_pretrained_path is None
            or raw_config_pretrained_path
            == str(expected_config_pretrained_path.expanduser().absolute().resolve(strict=False))
        ),
    }


def full_policy_identity(
    policy_root: Path,
    expected_model_sha256: str,
    *,
    expected_sidecars: set[str] | frozenset[str] | None = None,
    model_sha256_already_verified: str | None = None,
    reported_root: Path | None = None,
) -> dict[str, Any]:
    requested_root = policy_root.expanduser().absolute()
    require_no_symlink_components(requested_root, "policy root")
    root = requested_root.resolve(strict=True)
    model = root / "model.safetensors"
    require_no_symlink_components(model, "policy model")
    expected_model_sha256 = require_sha(expected_model_sha256, "policy model")
    observed = (
        require_sha(model_sha256_already_verified, "already-verified policy model")
        if model_sha256_already_verified is not None
        else sha256_file(model)
    )
    if observed != expected_model_sha256:
        raise ContractError(
            f"policy model SHA256 differs: expected={expected_model_sha256}, observed={observed}"
        )
    identity_root = (
        reported_root.expanduser().absolute().resolve(strict=False)
        if reported_root is not None
        else root
    )
    sidecars = policy_sidecar_tree(
        root,
        expected_sidecars=expected_sidecars,
        expected_config_pretrained_path=identity_root,
    )
    if reported_root is not None:
        require_no_symlink_components(reported_root, "reported policy root")
        sidecars = {**sidecars, "root": str(identity_root)}
    identity = {
        "root": str(identity_root),
        "model": {
            "path": str(identity_root / "model.safetensors"),
            "bytes": model.stat().st_size,
            "sha256": observed,
        },
        "sidecar_tree": sidecars,
    }
    identity["identity_sha256"] = canonical_json_sha256(identity)
    return identity


def source_tree_identity(
    root: Path, *, suffixes: tuple[str, ...] | None = None
) -> dict[str, Any]:
    requested = root.expanduser().absolute()
    require_no_symlink_components(requested, "environment source tree")
    resolved = requested.resolve(strict=True)
    rows = []
    for path in sorted(resolved.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if suffixes is not None and path.suffix not in suffixes:
            continue
        if path.is_symlink():
            raise ContractError(f"selected environment tree contains a symlink: {path}")
        if not path.is_file():
            continue
        rows.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ContractError(f"environment source tree is empty: {resolved}")
    return {
        "root": str(resolved),
        "file_count": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "tree_sha256": canonical_json_sha256(rows),
        "files": rows,
    }


def python_environment_identity() -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ":".join(
        [
            str(CANONICAL_PYTHON_OVERLAY),
            str(CANONICAL_SOURCE_OVERRIDE_TREE),
            str(CANONICAL_REPO_SCRIPTS),
            environment.get("PYTHONPATH", ""),
        ]
    ).rstrip(":")
    code = (
        "import importlib.metadata as md,json,draccus,lerobot,libero,numpy,"
        "robosuite,safetensors,torch,transformers;"
        "mods={'draccus':draccus,'lerobot':lerobot,'libero':libero,'numpy':numpy,"
        "'robosuite':robosuite,'safetensors':safetensors,'torch':torch,"
        "'transformers':transformers};"
        "out={};"
        "exec(\"for n,m in mods.items():\\n try: d=md.version(n)\\n except md.PackageNotFoundError: d=None\\n out[n]={'version':getattr(m,'__version__',None),'distribution_version':d,'file':m.__file__}\");"
        "out['torch']['cuda']=torch.version.cuda;print(json.dumps(out,sort_keys=True))"
    )
    raw = subprocess.check_output(
        [str(CANONICAL_PYTHON), "-c", code], env=environment, text=True
    )
    packages = json.loads(raw)
    for row in packages.values():
        path = Path(row["file"]).resolve(strict=True)
        row["file"] = str(path)
        row["file_sha256"] = sha256_file(path)
    package_roots = {
        name: Path(packages[name]["file"]).parent
        for name in ("libero", "robosuite", "draccus")
    }
    transformers_root = Path(packages["transformers"]["file"]).parent
    package_trees = {
        name: source_tree_identity(root, suffixes=(".py",))
        for name, root in package_roots.items()
    }
    for model_name in ("gemma", "paligemma", "siglip"):
        package_trees[f"transformers_models_{model_name}"] = source_tree_identity(
            transformers_root / "models" / model_name, suffixes=(".py",)
        )
    value = {
        "packages": packages,
        "lerobot_python_tree": source_tree_identity(
            CANONICAL_LEROBOT_TREE, suffixes=(".py",)
        ),
        "source_override_python_tree": source_tree_identity(
            CANONICAL_SOURCE_OVERRIDE_TREE, suffixes=(".py",)
        ),
        "python_overlay_tree": source_tree_identity(CANONICAL_PYTHON_OVERLAY),
        "benchmark_and_runtime_key_trees": package_trees,
    }
    value["identity_sha256"] = canonical_json_sha256(value)
    return value


def _decimal_range(start: str, stop: str, step: str) -> Iterable[Decimal]:
    value = Decimal(start)
    end = Decimal(stop)
    increment = Decimal(step)
    if increment <= 0:
        raise ValueError("step must be positive")
    while value <= end:
        yield value
        value += increment


def _id_decimal(value: Decimal, places: int) -> str:
    return f"{value:.{places}f}".replace(".", "p")


def task_arithmetic_specs() -> list[CandidateSpec]:
    rows = [
        CandidateSpec(
            candidate_id=f"task_arithmetic_alpha_{_id_decimal(alpha, 2)}",
            family="task_arithmetic",
            alpha_text=f"{alpha:.2f}",
            alpha=float(alpha),
        )
        for alpha in _decimal_range("0.00", "1.00", "0.05")
    ]
    if len(rows) != 21 or rows[0].alpha_text != "0.00" or rows[-1].alpha_text != "1.00":
        raise AssertionError("Task Arithmetic grid construction drifted")
    return rows


def ties_specs() -> list[CandidateSpec]:
    rows = [
        CandidateSpec(
            candidate_id=(
                f"ties_density_{_id_decimal(density, 1)}_alpha_{_id_decimal(alpha, 1)}"
            ),
            family="ties_merging",
            alpha_text=f"{alpha:.1f}",
            alpha=float(alpha),
            density_text=f"{density:.1f}",
            density=float(density),
        )
        for density in (Decimal("0.1"), Decimal("0.2"), Decimal("0.3"))
        for alpha in _decimal_range("0.8", "3.0", "0.1")
    ]
    if len(rows) != 69:
        raise AssertionError("TIES grid construction drifted")
    return rows


def all_specs() -> list[CandidateSpec]:
    rows = task_arithmetic_specs() + ties_specs()
    if len({row.candidate_id for row in rows}) != 90:
        raise AssertionError("Candidate IDs are not unique")
    return rows


def _validate_vectors(vectors: Sequence[np.ndarray]) -> tuple[list[np.ndarray], int]:
    if not vectors:
        raise ValueError("at least one task vector is required")
    values = [np.asarray(vector) for vector in vectors]
    shape = values[0].shape
    if len(shape) != 1 or any(value.shape != shape for value in values):
        raise ValueError("task vectors must be equally shaped and flat")
    if any(not np.issubdtype(value.dtype, np.floating) for value in values):
        raise ValueError("task vectors must be floating point")
    return values, values[0].size


def exact_global_keep_threshold_radix(
    vector: np.ndarray, keep_fraction: float, *, chunk_size: int = 4 * 1024 * 1024
) -> float | None:
    """Match the audited one-based TIES threshold with bounded memory.

    Absolute finite float32 values sort in the same order as their unsigned
    IEEE-754 bit patterns.  Four byte-histogram passes recover the exact kth
    value without allocating a second full-size array or relying on a 2^31
    indexing limit.
    """
    values, length = _validate_vectors([vector])
    source = values[0]
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must lie in (0, 1]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    retained = int(length * keep_fraction)
    if retained < 1:
        raise ValueError("keep_fraction retains no coordinates")
    if retained == length:
        return None
    rank = length - retained
    prefix = 0
    for shift in (24, 16, 8, 0):
        counts = np.zeros(256, dtype=np.int64)
        prefix_shift = shift + 8
        for start in range(0, length, chunk_size):
            chunk = np.asarray(source[start : start + chunk_size], dtype=np.float32)
            if not np.isfinite(chunk).all():
                raise ValueError("task vector is non-finite")
            bits = np.abs(chunk).view(np.uint32)
            if prefix_shift < 32:
                bits = bits[(bits >> prefix_shift) == prefix]
            counts += np.bincount(
                ((bits >> shift) & np.uint32(0xFF)).astype(np.int64), minlength=256
            )
        cumulative = np.cumsum(counts)
        candidates = np.flatnonzero(cumulative >= rank)
        if candidates.size == 0:
            raise RuntimeError("radix selection lost the requested rank")
        selected = int(candidates[0])
        rank -= int(cumulative[selected - 1]) if selected else 0
        prefix = (prefix << 8) | selected
    return float(np.asarray([prefix], dtype=np.uint32).view(np.float32)[0])


def exact_global_keep_threshold_oracle(
    vector: np.ndarray, keep_fraction: float
) -> float | None:
    """Small-vector sort oracle for tests; do not use on the 2.7B domain."""
    values, length = _validate_vectors([vector])
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must lie in (0, 1]")
    retained = int(length * keep_fraction)
    if retained < 1:
        raise ValueError("keep_fraction retains no coordinates")
    if retained == length:
        return None
    # Historical kernel uses one-based kth = N-retained.
    return float(np.sort(np.abs(values[0]).astype(np.float32))[length - retained - 1])


def task_arithmetic_direction(
    vectors: Sequence[np.ndarray], *, output: np.ndarray | None = None, chunk_size: int = 1 << 20
) -> np.ndarray:
    values, length = _validate_vectors(vectors)
    if output is None:
        output = np.empty(length, dtype=np.float64)
    if output.shape != (length,) or output.dtype != np.float64:
        raise ValueError("output must be a flat float64 array of matching length")
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        combined = np.zeros(end - start, dtype=np.float64)
        for vector in values:
            combined += np.asarray(vector[start:end], dtype=np.float64)
        output[start:end] = combined
    if not np.isfinite(output).all():
        raise ValueError("Task Arithmetic direction is non-finite")
    return output


def ties_direction(
    vectors: Sequence[np.ndarray],
    *,
    keep_fraction: float,
    output: np.ndarray | None = None,
    chunk_size: int = 1 << 20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute one reusable global TIES direction for one density.

    This function performs thresholding/sign election/mean-disjoint merge once.
    Alpha is deliberately absent; every alpha candidate reuses the returned
    direction as ``base + alpha * direction``.
    """
    values, length = _validate_vectors(vectors)
    if output is None:
        output = np.empty(length, dtype=np.float64)
    if output.shape != (length,) or output.dtype != np.float64:
        raise ValueError("output must be a flat float64 array of matching length")
    thresholds = [
        exact_global_keep_threshold_radix(
            vector, keep_fraction, chunk_size=chunk_size
        )
        for vector in values
    ]
    retained_counts = [0 for _ in values]
    sign_balance = 0
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        trimmed = []
        for index, (vector, threshold) in enumerate(zip(values, thresholds, strict=True)):
            chunk = np.asarray(vector[start:end], dtype=np.float64)
            selected = chunk if threshold is None else chunk * (np.abs(chunk) >= threshold)
            retained_counts[index] += int(np.count_nonzero(selected))
            trimmed.append(selected)
        elected = np.sign(np.sum(trimmed, axis=0))
        sign_balance += int(elected.astype(np.int64).sum(dtype=np.int64))
    zero_sign = 1 if sign_balance > 0 else -1 if sign_balance < 0 else 0
    merged_nonzero = 0
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        trimmed = []
        for vector, threshold in zip(values, thresholds, strict=True):
            chunk = np.asarray(vector[start:end], dtype=np.float64)
            trimmed.append(
                chunk if threshold is None else chunk * (np.abs(chunk) >= threshold)
            )
        elected = np.sign(np.sum(trimmed, axis=0))
        if zero_sign:
            elected = np.where(elected == 0, zero_sign, elected)
        selected = [
            value * np.where(elected > 0, value > 0, value < 0)
            for value in trimmed
        ]
        combined = np.sum(selected, axis=0)
        contributors = np.sum([value != 0 for value in selected], axis=0)
        combined /= np.maximum(contributors, 1)
        output[start:end] = combined
        merged_nonzero += int(np.count_nonzero(combined))
    if not np.isfinite(output).all():
        raise ValueError("TIES direction is non-finite")
    return output, {
        "thresholds": thresholds,
        "retained_counts": retained_counts,
        "merged_nonzero": merged_nonzero,
        "zero_sign_fallback": zero_sign,
        "sign_balance": sign_balance,
        "disjoint": "mean",
        "keep_fraction": keep_fraction,
        "selection": "exact_ieee_float32_radix_global",
        "direction_reused_across_alpha": True,
        "chunk_size": chunk_size,
    }


def compose_candidate(
    base: np.ndarray, direction: np.ndarray, alpha: float, *, output_dtype: np.dtype | None = None
) -> np.ndarray:
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    anchor = np.asarray(base)
    delta = np.asarray(direction)
    if anchor.shape != delta.shape or not np.issubdtype(anchor.dtype, np.floating):
        raise ValueError("base/direction shapes or dtypes differ")
    value = anchor.astype(np.float64) + float(alpha) * delta.astype(np.float64)
    if not np.isfinite(value).all():
        raise ValueError("candidate is non-finite")
    return value.astype(output_dtype or anchor.dtype)


def verify_complete_checkpoint(candidate_root: Path) -> dict[str, Any]:
    """Validate a resumable candidate and detect model/hash tampering."""
    candidate_root = candidate_root.expanduser().resolve()
    manifest_path = candidate_root / "candidate-manifest.json"
    if not manifest_path.is_file():
        raise ContractError(f"candidate manifest is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("fastlane_version") != FASTLANE_VERSION
        or manifest.get("kind") != "iclr2027_table1_ta_ties_candidate"
        or manifest.get("status") != "complete_full_checkpoint"
        or manifest.get("finite") is not True
    ):
        raise ContractError("candidate manifest is incomplete")
    checkpoint = manifest.get("checkpoint", {})
    kind = checkpoint.get("kind")
    if kind in {
        "single_merged_dense_checkpoint",
        "common_base_hardlink_reference",
    }:
        expected_policy_root = candidate_root / "pretrained_model"
    elif kind == "adopted_v1_read_only_dense_checkpoint":
        candidate_id = str(manifest.get("candidate", {}).get("id", ""))
        expected_policy_root = CANONICAL_V1_TA_ROOT / "candidates" / candidate_id / (
            "pretrained_model"
        )
        adoption = manifest.get("source_adoption", {})
        if (
            adoption.get("mode") != "verified_v1_read_only_reference"
            or adoption.get("source_root") != str(CANONICAL_V1_TA_ROOT)
        ):
            raise ContractError("adopted candidate source provenance differs")
    else:
        raise ContractError("candidate checkpoint kind differs")
    model = Path(str(checkpoint.get("model_path", ""))).resolve()
    if model != expected_policy_root / "model.safetensors" or not model.is_file():
        raise ContractError("candidate model path is not canonical or is absent")
    expected = require_sha(checkpoint.get("model_sha256"), "candidate model")
    observed = sha256_file(model)
    if observed != expected:
        raise ContractError(
            f"candidate model SHA256 differs: expected={expected}, observed={observed}"
        )
    expected_policy = manifest.get("full_policy_identity")
    if not isinstance(expected_policy, dict):
        raise ContractError("candidate full policy identity is absent")
    observed_policy = full_policy_identity(
        expected_policy_root,
        expected,
        expected_sidecars=RUNTIME_POLICY_SIDECARS,
        model_sha256_already_verified=observed,
    )
    if observed_policy != expected_policy:
        raise ContractError("candidate full policy sidecar tree differs")
    return manifest


def primary_default_distance(spec: CandidateSpec) -> tuple[float, float]:
    if spec.family == "task_arithmetic":
        return abs(spec.alpha - 0.4), 0.0
    if spec.family == "ties_merging" and spec.density is not None:
        return abs(spec.density - 0.2), abs(spec.alpha - 1.0)
    raise ValueError(f"unsupported family: {spec.family}")
