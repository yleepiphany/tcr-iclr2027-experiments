#!/usr/bin/env python3
"""Freeze and verify the outcome-blind OpenVLA repair development resets.

The development bank uses official LIBERO reset offsets 36--45.  It is kept
separate from the procedural formal bank and from expert-capture offsets 30/31.
Selection 40 is the first reset per task; selection 400 is all ten resets.
No policy outcome is read while creating or validating this artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_merge.libero_procedural_bank import sha256_file, sha256_state


WORK = Path(__file__).resolve().parents[3]
FORMAL_BANK = (WORK / "vla-merge-runtime/experiments/iclr2027-table1-20260910/"
               "reset-banks/libero-procedural-clean-v1")
CAPTURE = (WORK / "vla-merge-runtime/experiments/openvla-tcr-20260921/"
           "expert-ab-capture-attempt-01/capture-contract.json")
INIT_ROOT = WORK / ".datasets/LIBERO/20260919/runtime/libero/init_files"
OFFSETS = tuple(range(36, 46))
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EVAL_SEED = 315_001


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _save(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _formal_selected_hashes(bank: Path) -> set[str]:
    values: set[str] = set()
    for repeat in (1, 2, 3):
        selection = _json(bank / f"selections/repeat-0{repeat}.json")
        for key, indices in selection["tasks"].items():
            records = _json(bank / "manifest.json")["tasks"][key]["states"]
            values.update(records[index]["raw_sha256"] for index in indices)
    return values


def _capture_hashes(contract: dict[str, Any]) -> set[str]:
    return {
        pool["reset_sha256"]
        for suite in contract["tasks"].values()
        for task in suite.values()
        for pool in task["pools"].values()
    }


def _selection_payload(manifest: dict[str, Any], *, offsets: tuple[int, ...], name: str) -> dict[str, Any]:
    tasks = {}
    for key, task in manifest["tasks"].items():
        by_offset = {row["stock_offset"]: row["raw_sha256"] for row in task["states"]}
        tasks[key] = {
            "stock_offsets": list(offsets),
            "raw_sha256": [by_offset[offset] for offset in offsets],
        }
    return {
        "schema": "oft_tcr_development_selection_v1",
        "selection_id": name,
        "bank_manifest_sha256": manifest["manifest_sha256"],
        "eval_seed": EVAL_SEED,
        "tasks": tasks,
        "success_filter": False,
        "policy_outcomes_read": False,
    }


def create(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    formal_manifest_path = FORMAL_BANK / "manifest.json"
    formal_manifest = _json(formal_manifest_path)
    capture = _json(CAPTURE)
    if capture.get("schema") != "oft_expert_ab_capture_v1":
        raise ValueError("Unexpected expert capture contract")
    formal_hashes = _formal_selected_hashes(FORMAL_BANK)
    capture_hashes = _capture_hashes(capture)
    output.mkdir(parents=True)
    (output / "states").mkdir()

    tasks: dict[str, Any] = {}
    dev_hashes: set[str] = set()
    for suite in SUITES:
        (output / "states" / suite).mkdir()
        for task_id in range(10):
            key = f"{suite}/{task_id:02d}"
            source = formal_manifest["tasks"][key]
            receipt = source["official_init"]
            source_path = INIT_ROOT / source["problem_folder"] / source["init_states_file"]
            if sha256_file(source_path) != receipt["sha256"]:
                raise ValueError(f"Official reset file differs: {key}")
            states = np.asarray(torch.load(source_path, weights_only=False))
            if states.shape[0] <= OFFSETS[-1]:
                raise ValueError(f"Official bank has fewer than 46 states: {key}")
            selected = np.ascontiguousarray(states[list(OFFSETS)])
            rows = []
            for offset, state in zip(OFFSETS, selected, strict=True):
                digest = sha256_state(state)
                if digest != receipt["raw_state_sha256"][offset]:
                    raise ValueError(f"Official state receipt differs: {key}/{offset}")
                if digest in formal_hashes:
                    raise ValueError(f"Development reset overlaps formal reset: {key}/{offset}")
                if digest in capture_hashes:
                    raise ValueError(f"Development reset overlaps A/B capture: {key}/{offset}")
                if digest in dev_hashes:
                    raise ValueError(f"Duplicate development reset content: {key}/{offset}")
                dev_hashes.add(digest)
                rows.append({"stock_offset": offset, "raw_sha256": digest})
            relative = Path("states") / suite / f"task-{task_id:02d}.npy"
            np.save(output / relative, selected, allow_pickle=False)
            tasks[key] = {
                "suite": suite,
                "task_id": task_id,
                "task_name": source["task_name"],
                "problem_folder": source["problem_folder"],
                "bddl_file": source["bddl_file"],
                "states_file": str(relative),
                "states_file_sha256": sha256_file(output / relative),
                "source_file_sha256": receipt["sha256"],
                "states": rows,
            }

    manifest = {
        "schema": "oft_tcr_development_stock_bank_v1",
        "stock_offsets": list(OFFSETS),
        "development_eval_seed": EVAL_SEED,
        "tasks": tasks,
        "episodes_40": 40,
        "episodes_400": 400,
        "formal_bank_manifest_sha256": sha256_file(formal_manifest_path),
        "formal_selection_sha256": [sha256_file(FORMAL_BANK / f"selections/repeat-0{i}.json")
                                     for i in (1, 2, 3)],
        "expert_capture_contract_sha256": sha256_file(CAPTURE),
        "formal_hash_overlap": 0,
        "capture_hash_overlap": 0,
        "policy_outcomes_read": False,
        "success_filter": False,
    }
    # Bind selections to the exact manifest bytes without self-referential data.
    _save(output / "manifest.json", manifest)
    manifest_sha = sha256_file(output / "manifest.json")
    manifest["manifest_sha256"] = manifest_sha
    # Selection payloads bind the first manifest bytes through a sidecar field.
    selection40 = _selection_payload(manifest, offsets=(36,), name="development-40")
    selection400 = _selection_payload(manifest, offsets=OFFSETS, name="development-400")
    _save(output / "selection-40.json", selection40)
    _save(output / "selection-400.json", selection400)
    (output / "manifest.sha256").write_text(f"{manifest_sha}  manifest.json\n")
    result = verify(output)
    return result


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    fields = (root / "manifest.sha256").read_text().strip().split()
    if fields != [sha256_file(manifest_path), "manifest.json"]:
        raise ValueError("Development manifest sidecar differs")
    if (manifest.get("schema") != "oft_tcr_development_stock_bank_v1"
            or manifest.get("stock_offsets") != list(OFFSETS)
            or manifest.get("policy_outcomes_read") is not False
            or manifest.get("success_filter") is not False):
        raise ValueError("Development bank contract differs")
    formal_manifest = _json(FORMAL_BANK / "manifest.json")
    if manifest.get("formal_bank_manifest_sha256") != sha256_file(FORMAL_BANK / "manifest.json"):
        raise ValueError("Formal bank identity changed")
    expected_formal_selections = [
        sha256_file(FORMAL_BANK / f"selections/repeat-0{i}.json") for i in (1, 2, 3)
    ]
    if manifest.get("formal_selection_sha256") != expected_formal_selections:
        raise ValueError("Formal selection identity changed")
    if manifest.get("expert_capture_contract_sha256") != sha256_file(CAPTURE):
        raise ValueError("Expert capture identity changed")
    formal_hashes = _formal_selected_hashes(FORMAL_BANK)
    capture_hashes = _capture_hashes(_json(CAPTURE))

    seen: set[str] = set()
    for key, task in manifest["tasks"].items():
        source = formal_manifest["tasks"][key]
        source_path = INIT_ROOT / source["problem_folder"] / source["init_states_file"]
        if (task.get("source_file_sha256") != source["official_init"]["sha256"]
                or sha256_file(source_path) != task["source_file_sha256"]):
            raise ValueError(f"Official reset source changed: {key}")
        path = (root / task["states_file"]).resolve()
        path.relative_to(root)
        if sha256_file(path) != task["states_file_sha256"]:
            raise ValueError(f"Development state file differs: {key}")
        states = np.load(path, allow_pickle=False)
        if states.shape[0] != len(OFFSETS) or len(task["states"]) != len(OFFSETS):
            raise ValueError(f"Development state count differs: {key}")
        for state, row, offset in zip(states, task["states"], OFFSETS, strict=True):
            digest = sha256_state(state)
            if row != {"stock_offset": offset, "raw_sha256": digest} or digest in seen:
                raise ValueError(f"Development state identity differs: {key}/{offset}")
            if digest in formal_hashes or digest in capture_hashes:
                raise ValueError(f"Development reset is no longer held out: {key}/{offset}")
            seen.add(digest)

    if len(manifest["tasks"]) != 40 or len(seen) != 400:
        raise ValueError("Development bank is not exactly 40 tasks x 10 resets")
    manifest_sha = sha256_file(manifest_path)
    for filename, offsets, expected_id in (
            ("selection-40.json", (36,), "development-40"),
            ("selection-400.json", OFFSETS, "development-400")):
        selection = _json(root / filename)
        if (selection.get("schema") != "oft_tcr_development_selection_v1"
                or selection.get("selection_id") != expected_id
                or selection.get("bank_manifest_sha256") != manifest_sha
                or selection.get("eval_seed") != EVAL_SEED
                or selection.get("policy_outcomes_read") is not False):
            raise ValueError(f"Development selection contract differs: {filename}")
        if set(selection["tasks"]) != set(manifest["tasks"]):
            raise ValueError(f"Development selection task coverage differs: {filename}")
        for key, row in selection["tasks"].items():
            if row["stock_offsets"] != list(offsets):
                raise ValueError(f"Development offsets differ: {filename}/{key}")
            expected = {item["stock_offset"]: item["raw_sha256"]
                        for item in manifest["tasks"][key]["states"]}
            if row["raw_sha256"] != [expected[offset] for offset in offsets]:
                raise ValueError(f"Development hashes differ: {filename}/{key}")
    return {
        "accepted": True,
        "tasks": 40,
        "states": 400,
        "selection_40_episodes": 40,
        "selection_400_episodes": 400,
        "manifest_sha256": manifest_sha,
        "policy_outcomes_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify(args.output) if args.verify else create(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
