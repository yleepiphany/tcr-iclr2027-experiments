#!/usr/bin/env python3
"""Prepare matched, read-only LIBERO-PRO selections for main-table Model Soups."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile


WORK = Path(__file__).resolve().parents[3]
VLA = WORK / "vla-merge"
RUNTIME = WORK / "vla-merge-runtime"
SOURCE = RUNTIME / "experiments/claude-pro-main10k-20260923/selection-v1"
MODEL = (RUNTIME / "experiments/iclr2027-table1-20260910/libero/model-soups"
         / "repeat-shared/merge/attempt-02-peft-safe-v2/pretrained_model")
SOURCE_MANIFEST_SHA = "a0fa026982b70139c9fc52e88705947c26d818ddca2abe5a657df88714533614"
MODEL_SHA = "a92aacc43146dc41663c0057f2999bd90252fb3fb316ef963f165be67e1be01a"
MODEL_MANIFEST_SHA = "625400e9ec730ba16cbb123b0a2a7d03371c46467d2aaefdb9c4ffa8cb5c0bb4"
DIMENSIONS = ("object", "swap", "semantic", "task")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
REPEATS = ("repeat-01", "repeat-02", "repeat-03")

sys.path.insert(0, str(VLA / "src"))
from vla_merge.libero_extension import load_extension_selection  # noqa: E402
from vla_merge.libero_procedural_bank import sha256_file  # noqa: E402


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if sha256_file(SOURCE / "manifest.json") != SOURCE_MANIFEST_SHA:
        raise ValueError("Frozen matched PRO source manifest changed")
    if sha256_file(MODEL / "model.safetensors") != MODEL_SHA:
        raise ValueError("Main-table Soup checkpoint changed")
    model_manifest = MODEL / "parameter_baseline_manifest.json"
    if sha256_file(model_manifest) != MODEL_MANIFEST_SHA:
        raise ValueError("Main-table Soup provenance changed")
    identity = json.loads(model_manifest.read_text())
    if identity.get("method") != "uniform_soup" or identity.get("model_sha256") != MODEL_SHA:
        raise ValueError("Model is not the frozen four-expert uniform Soup")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        index = {}
        source_identity = None
        total_jobs = total_episodes = 0
        for repeat in REPEATS:
            reference = SOURCE / "selections/TCR-Full" / f"{repeat}.json"
            old = load_extension_selection(reference, expected_benchmark="libero_pro", verify_files=True)
            if old.checkpoint_contract != "pi05-libero-main10k-matched-v1":
                raise ValueError("PRO checkpoint contract changed")
            if source_identity is None:
                source_identity = old.source_identity
            elif old.source_identity != source_identity:
                raise ValueError("PRO source identity varies by repeat")
            counts = Counter((job.dimension, job.source_suite) for job in old.jobs)
            if counts != Counter({(d, s): 10 for d in DIMENSIONS for s in SUITES}):
                raise ValueError(f"Incomplete PRO coverage: {repeat}")
            if len({job.job_id for job in old.jobs}) != 160 or any(len(job.state_ids) != 10 for job in old.jobs):
                raise ValueError(f"Unexpected PRO job or reset count: {repeat}")
            payload = json.loads(reference.read_text())
            payload["policy_checkpoint_sha256"] = MODEL_SHA
            new = staging / "selections" / f"{repeat}.json"
            write(new, payload)
            checked = load_extension_selection(new, expected_benchmark="libero_pro", verify_files=True)
            if (checked.repeat_id != repeat or checked.eval_seed != old.eval_seed
                    or checked.source_identity != old.source_identity or checked.jobs != old.jobs
                    or checked.policy_checkpoint_sha256 != MODEL_SHA):
                raise ValueError(f"PRO Soup selection failed parity: {repeat}")
            jobs = len(checked.jobs)
            episodes = sum(len(job.state_ids) for job in checked.jobs)
            total_jobs += jobs
            total_episodes += episodes
            index[repeat] = {
                "path": str(output / new.relative_to(staging)), "sha256": sha256_file(new),
                "source_path": str(reference), "source_sha256": sha256_file(reference),
                "jobs": jobs, "episodes": episodes, "eval_seed": old.eval_seed,
            }
        if (total_jobs, total_episodes) != (480, 4800):
            raise ValueError("PRO Soup matrix must be 480 jobs and 4800 episodes")
        manifest = {
            "schema": "pi05_pro_soup_main10k_selection_v1",
            "status": "prepared_unclaimed_owner_check_no_gpu",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": "LIBERO-PRO",
            "paper_target": "tab:libero_pro_robustness / pi0.5 Model Soups",
            "method": "uniform_soup", "checkpoint_contract": "pi05-libero-main10k-matched-v1",
            "model": str(MODEL), "model_sha256": MODEL_SHA,
            "model_manifest": str(model_manifest), "model_manifest_sha256": MODEL_MANIFEST_SHA,
            "source_manifest": str(SOURCE / "manifest.json"),
            "source_manifest_sha256": SOURCE_MANIFEST_SHA,
            "source_identity": source_identity, "selection_index": index,
            "dimensions": list(DIMENSIONS), "suites": list(SUITES),
            "repeats": list(REPEATS), "jobs": total_jobs, "episodes": total_episodes,
            "gpu_launched": False, "no_training": True, "no_score_based_selection": True,
        }
        write(staging / "manifest.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"path": str(output), "manifest_sha256": sha256_file(output / "manifest.json"),
            "jobs": total_jobs, "episodes": total_episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output.expanduser().absolute()), indent=2))


if __name__ == "__main__":
    main()
