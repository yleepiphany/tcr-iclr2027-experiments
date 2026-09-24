"""Immutable loading and selection for prospective LIBERO procedural reset banks.

The stock LIBERO initial-state files remain untouched.  A procedural bank lives in
its own directory and is accepted only when its manifest, state arrays, current
BDDL files, and current official initial-state files all match their recorded
SHA256 identities.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


BANK_SCHEMA_VERSION = 1
SELECTION_SCHEMA_VERSION = 1


class ProceduralBankError(ValueError):
    """Raised when a procedural bank or selection fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_state(state: np.ndarray) -> np.ndarray:
    value = np.asarray(state)
    if value.ndim != 1:
        raise ProceduralBankError(
            f"Expected a flat simulator state, got shape={value.shape}"
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise ProceduralBankError(
            f"Expected a floating simulator state, got dtype={value.dtype}"
        )
    if not bool(np.isfinite(value).all()):
        raise ProceduralBankError("Simulator state contains non-finite values")
    # Preserve the simulator dtype.  The dtype and shape are part of the hash domain.
    return np.ascontiguousarray(value)


def sha256_state(state: np.ndarray) -> str:
    value = canonical_state(state)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProceduralBankError(f"Cannot read JSON receipt {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProceduralBankError(f"Expected a JSON object in {path}")
    return payload


def _safe_relative_path(root: Path, raw: object, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ProceduralBankError(f"{field} must be a non-empty relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ProceduralBankError(f"Unsafe {field}: {raw!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ProceduralBankError(f"{field} escapes bank root: {raw!r}") from exc
    return resolved


def _verify_manifest_sidecar(root: Path, manifest_path: Path) -> None:
    receipt = root / "manifest.sha256"
    if not receipt.is_file():
        raise ProceduralBankError(f"Missing manifest hash receipt: {receipt}")
    fields = receipt.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != "manifest.json":
        raise ProceduralBankError(f"Malformed manifest hash receipt: {receipt}")
    observed = sha256_file(manifest_path)
    if fields[0] != observed:
        raise ProceduralBankError(
            f"Bank manifest SHA256 mismatch: receipt={fields[0]} observed={observed}"
        )


def load_bank_manifest(
    bank_root: Path, *, verify_source_files: bool = True
) -> dict[str, Any]:
    root = bank_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ProceduralBankError(f"Missing bank manifest: {manifest_path}")
    _verify_manifest_sidecar(root, manifest_path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != BANK_SCHEMA_VERSION:
        raise ProceduralBankError("Unsupported procedural-bank schema")
    if manifest.get("bank_id") != "procedural-clean-v1":
        raise ProceduralBankError(
            "Unexpected bank_id; explicit procedural-clean-v1 is required"
        )
    if manifest.get("canonical_benchmark_bank") is not False:
        raise ProceduralBankError(
            "Procedural bank must explicitly declare itself non-canonical"
        )
    if manifest.get("prospective_policy_independent_generation") is not True:
        raise ProceduralBankError(
            "Procedural bank lacks prospective policy-independent declaration"
        )
    generator = manifest.get("generator")
    if not isinstance(generator, dict):
        raise ProceduralBankError("Procedural bank lacks a generator receipt")
    frozen_generator_fields = {
        "seed_formula": "275000000 + suite_index*100000 + task_id*1000 + state_index",
        "seed_base": 275_000_000,
        "suite_stride": 100_000,
        "task_stride": 1_000,
        "suite_order": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        "final_eval_seeds": [274_001, 274_002, 274_003],
    }
    for field, expected in frozen_generator_fields.items():
        if generator.get(field) != expected:
            raise ProceduralBankError(
                f"Frozen procedural generator field differs for {field}: {generator.get(field)!r}"
            )
    if manifest.get("profile") == "full":
        expected_mapping = {
            "repeat-01": list(range(0, 10)),
            "repeat-02": list(range(10, 20)),
            "repeat-03": list(range(20, 30)),
            "reserve": list(range(30, 40)),
        }
        if (
            generator.get("states_per_task") != 40
            or generator.get("repeat_mapping") != expected_mapping
        ):
            raise ProceduralBankError(
                "Full bank does not have the frozen 30+10 state mapping"
            )
        reserve = generator.get("reserve_policy", {})
        if (
            reserve.get("indices") != list(range(30, 40))
            or reserve.get("selection") != "smallest_unused_state_index"
        ):
            raise ProceduralBankError(
                "Full bank reserve policy differs from the frozen rule"
            )
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ProceduralBankError("Procedural bank contains no task records")
    if manifest.get("profile") == "full":
        expected_tasks = {
            f"{suite}/{task_id:02d}"
            for suite in frozen_generator_fields["suite_order"]
            for task_id in range(10)
        }
        if set(tasks) != expected_tasks:
            raise ProceduralBankError(
                "Full bank does not contain exactly four suites by ten tasks"
            )

    seen_state_hashes: set[str] = set()
    for task_key, row in tasks.items():
        if not isinstance(task_key, str) or not isinstance(row, dict):
            raise ProceduralBankError("Malformed task record")
        expected_key = f"{row.get('suite')}/{int(row.get('task_id')):02d}"
        expected_suite_index = frozen_generator_fields["suite_order"].index(
            row.get("suite")
        )
        if task_key != expected_key or row.get("suite_index") != expected_suite_index:
            raise ProceduralBankError(f"Task identity/index mismatch for {task_key}")
        states_path = _safe_relative_path(
            root, row.get("states_file"), field="states_file"
        )
        if not states_path.is_file():
            raise ProceduralBankError(
                f"Missing state array for {task_key}: {states_path}"
            )
        observed_file_sha = sha256_file(states_path)
        if observed_file_sha != row.get("states_file_sha256"):
            raise ProceduralBankError(f"State-array SHA256 mismatch for {task_key}")
        try:
            states = np.load(states_path, allow_pickle=False)
        except Exception as exc:
            raise ProceduralBankError(
                f"Cannot load state array for {task_key}: {exc}"
            ) from exc
        if states.ndim != 2 or list(states.shape[1:]) != row.get("state_shape"):
            raise ProceduralBankError(
                f"State-array shape mismatch for {task_key}: {states.shape}"
            )
        if states.dtype.str != row.get("state_dtype"):
            raise ProceduralBankError(
                f"State-array dtype mismatch for {task_key}: {states.dtype.str}"
            )
        records = row.get("states")
        if not isinstance(records, list) or len(records) != states.shape[0]:
            raise ProceduralBankError(f"State-record count mismatch for {task_key}")
        if [
            record.get("index") for record in records if isinstance(record, dict)
        ] != list(range(states.shape[0])):
            raise ProceduralBankError(f"State indices are not contiguous in {task_key}")
        official_hashes = set(row.get("official_init", {}).get("raw_state_sha256", []))
        if len(official_hashes) != row.get("official_init", {}).get("state_count"):
            raise ProceduralBankError(
                f"Official-state hash receipt is incomplete for {task_key}"
            )
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ProceduralBankError(
                    f"Malformed state record for {task_key}/{index}"
                )
            expected_seed = (
                275_000_000
                + int(row.get("suite_index")) * 100_000
                + int(row.get("task_id")) * 1_000
                + index
            )
            if record.get("seed") != expected_seed:
                raise ProceduralBankError(
                    f"Frozen seed formula mismatch for {task_key}/{index}"
                )
            expected_role = (
                "repeat-01"
                if index < 10
                else (
                    "repeat-02"
                    if index < 20
                    else "repeat-03" if index < 30 else "reserve"
                )
            )
            if record.get("role") != expected_role:
                raise ProceduralBankError(
                    f"Frozen state role mismatch for {task_key}/{index}"
                )
            observed_state_sha = sha256_state(states[index])
            if observed_state_sha != record.get("raw_sha256"):
                raise ProceduralBankError(
                    f"Raw-state SHA256 mismatch for {task_key}/{index}"
                )
            if observed_state_sha in official_hashes:
                raise ProceduralBankError(
                    f"Procedural state collides with official bank: {task_key}/{index}"
                )
            if observed_state_sha in seen_state_hashes:
                raise ProceduralBankError(
                    f"Duplicate procedural state hash: {task_key}/{index}"
                )
            seen_state_hashes.add(observed_state_sha)

        if verify_source_files:
            from libero.libero import get_libero_path

            current_paths = {
                "bddl": (
                    Path(get_libero_path("bddl_files"))
                    / str(row.get("problem_folder"))
                    / str(row.get("bddl_file"))
                ).resolve(),
                "official_init": (
                    Path(get_libero_path("init_states"))
                    / str(row.get("problem_folder"))
                    / str(row.get("init_states_file"))
                ).resolve(),
            }
            for source_name, current_path in current_paths.items():
                receipt = row.get(source_name)
                if not isinstance(receipt, dict):
                    raise ProceduralBankError(
                        f"Missing {source_name} receipt for {task_key}"
                    )
                if not current_path.is_file() or sha256_file(
                    current_path
                ) != receipt.get("sha256"):
                    raise ProceduralBankError(
                        f"Current {source_name} identity differs for {task_key}"
                    )
    return manifest


@dataclass(frozen=True)
class SelectedTaskBank:
    task_key: str
    suite: str
    task_id: int
    task_name: str
    problem_folder: str
    bddl_file: str
    indices: tuple[int, ...]
    states: np.ndarray
    raw_sha256: tuple[str, ...]


@dataclass(frozen=True)
class LoadedSelection:
    bank_root: Path
    bank_manifest_sha256: str
    selection_path: Path
    selection_sha256: str
    repeat_id: str | None
    eval_seed: int | None
    tasks: dict[tuple[str, int], SelectedTaskBank]


def load_selection(
    bank_root: Path,
    selection_path: Path,
    *,
    verify_source_files: bool = True,
) -> LoadedSelection:
    root = bank_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = load_bank_manifest(root, verify_source_files=verify_source_files)
    selection_file = selection_path.expanduser().resolve()
    selection = _read_json(selection_file)
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ProceduralBankError("Unsupported procedural-bank selection schema")
    bank_sha = sha256_file(manifest_path)
    if selection.get("bank_id") != manifest["bank_id"]:
        raise ProceduralBankError("Selection bank_id differs from bank manifest")
    if selection.get("bank_manifest_sha256") != bank_sha:
        raise ProceduralBankError(
            "Selection does not bind the current bank manifest SHA256"
        )
    bindings = selection.get("tasks")
    if not isinstance(bindings, dict) or not bindings:
        raise ProceduralBankError("Selection contains no task bindings")

    repeat_id = selection.get("repeat_id")
    eval_seed = selection.get("eval_seed")
    if (repeat_id is None) != (eval_seed is None):
        raise ProceduralBankError(
            "repeat_id and eval_seed must either both be set or both be omitted"
        )
    if repeat_id is not None:
        if repeat_id not in {"repeat-01", "repeat-02", "repeat-03"}:
            raise ProceduralBankError(f"Unsupported formal repeat_id: {repeat_id!r}")
        if type(eval_seed) is not int:
            raise ProceduralBankError("Formal selection eval_seed must be an integer")
        expected_eval_seeds = dict(
            zip(
                ("repeat-01", "repeat-02", "repeat-03"),
                manifest.get("generator", {}).get("final_eval_seeds", []),
                strict=True,
            )
        )
        if eval_seed != expected_eval_seeds.get(repeat_id):
            raise ProceduralBankError(
                f"Formal eval seed differs from frozen mapping for {repeat_id}: {eval_seed}"
            )

    selected: dict[tuple[str, int], SelectedTaskBank] = {}
    for task_key, raw_indices in bindings.items():
        if task_key not in manifest["tasks"]:
            raise ProceduralBankError(f"Selection references unknown task: {task_key}")
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ProceduralBankError(
                f"Selection for {task_key} must be a non-empty index list"
            )
        if any(type(index) is not int or index < 0 for index in raw_indices):
            raise ProceduralBankError(
                f"Selection for {task_key} contains an invalid index"
            )
        if len(set(raw_indices)) != len(raw_indices):
            raise ProceduralBankError(
                f"Selection for {task_key} contains duplicate indices"
            )
        row = manifest["tasks"][task_key]
        states_path = _safe_relative_path(root, row["states_file"], field="states_file")
        states = np.load(states_path, allow_pickle=False)
        if any(index >= states.shape[0] for index in raw_indices):
            raise ProceduralBankError(f"Selection index is out of range for {task_key}")
        records = row["states"]
        selected_states = np.ascontiguousarray(states[raw_indices])
        selected_hashes = tuple(records[index]["raw_sha256"] for index in raw_indices)
        # Recheck the selected rows after advanced indexing/copying.
        for state, expected in zip(selected_states, selected_hashes, strict=True):
            if sha256_state(state) != expected:
                raise ProceduralBankError(
                    f"Selected raw-state SHA256 mismatch for {task_key}"
                )
        suite = str(row["suite"])
        task_id = int(row["task_id"])
        identity = (suite, task_id)
        if identity in selected:
            raise ProceduralBankError(
                f"Duplicate task identity in selection: {identity}"
            )
        selected[identity] = SelectedTaskBank(
            task_key=task_key,
            suite=suite,
            task_id=task_id,
            task_name=str(row["task_name"]),
            problem_folder=str(row["problem_folder"]),
            bddl_file=str(row["bddl_file"]),
            indices=tuple(raw_indices),
            states=selected_states,
            raw_sha256=selected_hashes,
        )
    return LoadedSelection(
        bank_root=root,
        bank_manifest_sha256=bank_sha,
        selection_path=selection_file,
        selection_sha256=sha256_file(selection_file),
        repeat_id=repeat_id,
        eval_seed=eval_seed,
        tasks=selected,
    )
