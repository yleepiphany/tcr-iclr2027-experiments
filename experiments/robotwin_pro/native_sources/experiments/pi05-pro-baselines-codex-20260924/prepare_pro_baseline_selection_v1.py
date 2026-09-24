#!/usr/bin/env python3
"""Prepare unclaimed PRO selections for the exact clean-main baseline weights."""
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
TABLE = RUNTIME / "experiments/iclr2027-table1-20260910"
SOURCE = RUNTIME / "experiments/claude-pro-main10k-20260923/selection-v1"
SOURCE_SHA = "a0fa026982b70139c9fc52e88705947c26d818ddca2abe5a657df88714533614"
ROOT = RUNTIME / "experiments/pi05-pro-baselines-codex-20260924"
REPEATS = ("repeat-01", "repeat-02", "repeat-03")
DIMENSIONS = ("object", "swap", "semantic", "task")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SPECS = {
    "ties_merging": {
        "model": TABLE / "libero/ta-ties-fastlane-v2/candidates/ties_density_0p3_alpha_0p9/pretrained_model",
        "sha256": "9382c79c9a76dea4803d71484599ec16edb77ee3cf7ce3b861fb9f0296921847",
        "clean_manifest": TABLE / "manifest-clean-formal-ta-ties-v1.json",
        "clean_manifest_sha256": "e2f971badeb4725317ad8c24d86d2631c4c1d7e305ffa78df187f932ad7b2ac6",
    },
    "regmean_pp": {
        "model": TABLE / "libero/original-regmeanpp/repeat-01/merge/attempt-03-active-support-v2-host1023-gpu2",
        "sha256": "38745ba812c2f4ac4327ab40719e0ce8da017873c0a1765608f65dd926cd4ab3",
        "clean_manifest": TABLE / "manifest-clean-formal-regmeanpp-v1.json",
        "clean_manifest_sha256": "53f39190d68ff12897f8382a563e047f82c696e1dee728129c0bc2b833511d1b",
    },
    "featcal": {
        "model": TABLE / "preflight/featcal-formal-full-v1/checkpoint/pretrained_model",
        "sha256": "e43de109844431c02e316e57f701d7c06a9a3c8feaa3c41c9f391281f8197efa",
        "clean_manifest": TABLE / "manifest-clean-fastlane-v3.json",
        "clean_manifest_sha256": "67309bf9ae7907c6b807dac272ef1933b8f82220123ea7df09cd711f61e87b1e",
    },
}

sys.path.insert(0, str(VLA / "src"))
from vla_merge.libero_extension import load_extension_selection  # noqa: E402
from vla_merge.libero_procedural_bank import sha256_file  # noqa: E402


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_model(method: str) -> dict:
    spec = SPECS[method]
    if sha256_file(SOURCE / "manifest.json") != SOURCE_SHA:
        raise ValueError("Frozen PRO reset selection source changed")
    if sha256_file(spec["model"] / "model.safetensors") != spec["sha256"]:
        raise ValueError(f"{method}: main-table model bytes differ")
    if sha256_file(spec["clean_manifest"]) != spec["clean_manifest_sha256"]:
        raise ValueError(f"{method}: clean formal identity manifest changed")
    clean = json.loads(spec["clean_manifest"].read_text())
    jobs = [job for job in clean["jobs"] if job["method"] == method]
    if len(jobs) != 12 or {job["repeat"] for job in jobs} != set(REPEATS):
        raise ValueError(f"{method}: clean formal job coverage differs")
    if Counter((job["repeat"], job["suite"]) for job in jobs) != Counter(
        {(repeat, suite): 1 for repeat in REPEATS for suite in SUITES}
    ):
        raise ValueError(f"{method}: clean formal suites differ")
    for job in jobs:
        checkpoint = job["checkpoint"]
        if (Path(checkpoint["path"]).resolve() != spec["model"].resolve()
                or checkpoint["model_sha256"] != spec["sha256"]):
            raise ValueError(f"{method}: clean formal checkpoint identity differs")
    bound = jobs[0]["checkpoint"]
    if any(job["checkpoint"] != bound for job in jobs):
        raise ValueError(f"{method}: clean formal checkpoint varies by job")
    for key in ("manifest", "verification"):
        binding = bound[key]
        if sha256_file(Path(binding["path"])) != binding["sha256"]:
            raise ValueError(f"{method}: checkpoint {key} proof changed")
    return {"path": str(spec["model"]), "sha256": spec["sha256"],
            "clean_manifest": str(spec["clean_manifest"]),
            "clean_manifest_sha256": spec["clean_manifest_sha256"],
            "checkpoint_manifest": bound["manifest"],
            "checkpoint_verification": bound["verification"]}


def prepare(method: str) -> dict:
    if method not in SPECS:
        raise ValueError(method)
    output = ROOT / method / "selection-v1"
    if output.exists():
        raise FileExistsError(output)
    model = source_model(method)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        index = {}
        source_identity = None
        jobs_total = episodes_total = 0
        for repeat in REPEATS:
            reference = SOURCE / "selections/TCR-Full" / f"{repeat}.json"
            original = load_extension_selection(reference, expected_benchmark="libero_pro",
                                                verify_files=True)
            if original.checkpoint_contract != "pi05-libero-main10k-matched-v1":
                raise ValueError("PRO checkpoint contract changed")
            if source_identity is None:
                source_identity = original.source_identity
            elif original.source_identity != source_identity:
                raise ValueError("PRO source identity varies by repeat")
            if (Counter((job.dimension, job.source_suite) for job in original.jobs)
                    != Counter({(dimension, suite): 10 for dimension in DIMENSIONS
                                for suite in SUITES})
                    or len({job.job_id for job in original.jobs}) != 160
                    or any(len(job.state_ids) != 10 for job in original.jobs)):
                raise ValueError(f"Incomplete PRO reset panel: {repeat}")
            payload = json.loads(reference.read_text())
            payload["policy_checkpoint_sha256"] = model["sha256"]
            new = staging / "selections" / f"{repeat}.json"
            write(new, payload)
            checked = load_extension_selection(new, expected_benchmark="libero_pro",
                                               verify_files=True)
            if (checked.repeat_id != repeat or checked.eval_seed != original.eval_seed
                    or checked.source_identity != original.source_identity
                    or checked.jobs != original.jobs
                    or checked.policy_checkpoint_sha256 != model["sha256"]):
                raise ValueError(f"{method}: PRO reset parity failed: {repeat}")
            count = len(checked.jobs)
            episodes = sum(len(job.state_ids) for job in checked.jobs)
            jobs_total += count
            episodes_total += episodes
            index[repeat] = {"path": str(output / new.relative_to(staging)),
                             "sha256": sha256_file(new), "source_path": str(reference),
                             "source_sha256": sha256_file(reference), "jobs": count,
                             "episodes": episodes, "eval_seed": original.eval_seed}
        if (jobs_total, episodes_total) != (480, 4800):
            raise ValueError("PRO baseline reset matrix must be 480 jobs/4800 episodes")
        manifest = {"schema": "pi05_pro_main_baseline_selection_v1",
                    "status": "prepared_unclaimed_no_gpu",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "paper_target": "tab:libero_pro_robustness",
                    "benchmark": "LIBERO-PRO", "method": method,
                    "checkpoint_contract": "pi05-libero-main10k-matched-v1",
                    "model": model, "source_manifest": str(SOURCE / "manifest.json"),
                    "source_manifest_sha256": SOURCE_SHA,
                    "source_identity": source_identity, "selection_index": index,
                    "dimensions": list(DIMENSIONS), "suites": list(SUITES),
                    "repeats": list(REPEATS), "jobs": jobs_total,
                    "episodes": episodes_total, "gpu_launched": False,
                    "no_training": True, "no_score_based_selection": True}
        write(staging / "manifest.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"method": method, "selection": str(output),
            "manifest_sha256": sha256_file(output / "manifest.json"),
            "jobs": jobs_total, "episodes": episodes_total}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=sorted(SPECS))
    args = parser.parse_args()
    print(json.dumps(prepare(args.method), sort_keys=True))
