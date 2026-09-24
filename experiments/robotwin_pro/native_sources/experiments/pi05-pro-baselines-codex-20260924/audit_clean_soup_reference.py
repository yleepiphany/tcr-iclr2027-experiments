#!/usr/bin/env python3
"""Independently bind the published clean Soup row to its exact PRO policy weight."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
BASE = WORK / "vla-merge-runtime/experiments/iclr2027-table1-20260910"
MANIFEST = BASE / "manifest-clean-fastlane-v2.json"
MANIFEST_SHA = "abc265fd9bcaed83efceabb14792f317460b96499306ab136f5177924b12714a"
MODEL_SHA = "a92aacc43146dc41663c0057f2999bd90252fb3fb316ef963f165be67e1be01a"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
REPEATS = ("repeat-01", "repeat-02", "repeat-03")
OUTPUT = WORK / "coordination/2026-09-23/local-codex-pro-soup-clean-reference.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def audit() -> dict:
    if sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Clean formal manifest changed")
    manifest = read(MANIFEST)
    jobs = [j for j in manifest["jobs"] if j["method"] == "model_soups"]
    episodes = {e["episode_key"]: e for e in manifest["episodes"] if e["method"] == "model_soups"}
    if (len(jobs) != 12 or len(episodes) != 1200
            or {(j["repeat"], j["suite"]) for j in jobs} !=
            {(r, s) for r in REPEATS for s in SUITES}):
        raise ValueError("Clean Soup coverage differs")
    model = Path(jobs[0]["checkpoint"]["path"]) / "model.safetensors"
    if sha(model) != MODEL_SHA:
        raise ValueError("Clean Soup model bytes differ from PRO model")
    used = set()
    by_repeat_suite = {}
    receipt_digests = {}
    for job in jobs:
        repeat, suite = job["repeat"], job["suite"]
        if job["checkpoint"]["model_sha256"] != MODEL_SHA or job["episodes"] != 100:
            raise ValueError("Clean job checkpoint or episode count differs")
        directory = Path(manifest["output_root"]) / job["output_relpath"] / "attempt-01"
        receipt_path = directory / "run_receipt.json"
        receipt = read(receipt_path)
        if (receipt.get("status") != "completed" or receipt.get("method") != "model_soups"
                or receipt.get("repeat") != repeat or receipt.get("suite") != suite
                or receipt.get("identities", {}).get("checkpoint_sha256") != MODEL_SHA
                or receipt.get("identities", {}).get("fastlane_manifest", {}).get("sha256") != MANIFEST_SHA
                or receipt.get("result", {}).get("episodes") != 100):
            raise ValueError(f"Clean run receipt mismatch: {repeat}/{suite}")
        for artifact in receipt["artifacts"].values():
            if sha(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError(f"Clean artifact changed: {repeat}/{suite}")
        lines = Path(receipt["artifacts"]["episode_receipts"]["path"]).read_text().splitlines()
        if len(lines) != 100:
            raise ValueError(f"Clean raw episode count differs: {repeat}/{suite}")
        successes = 0
        tasks = defaultdict(list)
        for line in lines:
            row = json.loads(line)
            key = row["episode_key"]
            expected = episodes.get(key)
            if (key in used or expected is None or row.get("status") != "completed"
                    or row.get("checkpoint_sha256") != MODEL_SHA
                    or row.get("fastlane_manifest_sha256") != MANIFEST_SHA
                    or row.get("repeat") != repeat or row.get("suite") != suite
                    or row.get("eval_seed") != expected["eval_seed"]
                    or row.get("task_id") != expected["task_id"]
                    or row.get("state_index") != expected["state_index"]
                    or row.get("reset_sha256") != expected["reset_sha256"]
                    or type(row.get("success")) is not bool):
                raise ValueError(f"Clean raw episode mismatch: {key}")
            used.add(key)
            tasks[row["task_id"]].append(row["success"])
            successes += row["success"]
        if len(tasks) != 10 or any(len(v) != 10 for v in tasks.values()):
            raise ValueError(f"Clean task coverage differs: {repeat}/{suite}")
        if (receipt["result"].get("successes") != successes
                or abs(float(receipt["result"].get("success_percent", -1)) - successes) > 1e-6):
            raise ValueError(f"Clean result disagrees with raw outcomes: {repeat}/{suite}")
        by_repeat_suite[f"{repeat}/{suite}"] = successes
        receipt_digests[f"{repeat}/{suite}"] = sha(receipt_path)
    if used != set(episodes):
        raise ValueError("Clean formal episode IDs not exactly covered")
    per_repeat = [sum(by_repeat_suite[f"{r}/{s}"] for s in SUITES) / 4.0 for r in REPEATS]
    if sum(by_repeat_suite.values()) != 538:
        raise ValueError("Clean model-soup outcome differs from accepted historical row")
    return {"schema": "pi05_pro_soup_clean_reference_identity_v1", "accepted": True,
            "benchmark": "clean_LIBERO", "model_sha256": MODEL_SHA,
            "formal_manifest_sha256": MANIFEST_SHA,
            "formal_jobs": 12, "formal_episodes": 1200,
            "by_repeat_suite_successes": by_repeat_suite,
            "repeat_macro_success_percent": per_repeat,
            "mean_percent": mean(per_repeat), "sample_std_percent": stdev(per_repeat),
            "run_receipt_sha256": receipt_digests,
            "note": "Clean reference uses its own procedural selection; it is not a reset-paired PRO degradation estimate."}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = audit()
    with OUTPUT.open("x") as stream:
        stream.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(OUTPUT), "sha256": sha(OUTPUT),
                      "accepted": True, "episodes": data["formal_episodes"]}))


if __name__ == "__main__":
    main()
