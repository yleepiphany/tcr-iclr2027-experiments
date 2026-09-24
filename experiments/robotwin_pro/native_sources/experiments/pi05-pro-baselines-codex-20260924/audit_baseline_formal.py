#!/usr/bin/env python3
"""Independent source/raw-result audit for matched 480-job PRO baselines."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
ROOT = WORK / "vla-merge-runtime/experiments/pi05-pro-baselines-codex-20260924"
SELECTION = RUN = PLAN_SHA = SELECTION_SHA = MODEL_SHA = METHOD = None
SPECS = {
    "ties_merging": {"selection_sha": "e451b150756e302201a1cce736eb04e5abaf20d228561ca70119922262807e5e",
                     "model_sha": "9382c79c9a76dea4803d71484599ec16edb77ee3cf7ce3b861fb9f0296921847",
                     "plan_sha": "eec53167c4204960155b903bea8f564f44ee79387bef1212ab3f3191576107cf"},
    "regmean_pp": {"selection_sha": "61490ca165af7ad60ba33e6f3ab1b569a681db7ebb12b4c5f29e492cfc9c33b2",
                   "model_sha": "38745ba812c2f4ac4327ab40719e0ce8da017873c0a1765608f65dd926cd4ab3",
                   "plan_sha": "697ed9707b6fb6d73657d0abcaab8cd59b923a1bcb5cd4398360cab52a9d39db"},
    "featcal": {"selection_sha": "cbd455255be21186db59117cc24eec2672e0fa598e6d147407a07a347600d1b0",
                "model_sha": "e43de109844431c02e316e57f701d7c06a9a3c8feaa3c41c9f391281f8197efa",
                "plan_sha": "25f04c5076c3195efb0bb73b3eb11beb0a3785c5f59c55ae0b9b080460b18b24"},
}
HOST = "dsw-967394-56ffd4897d-42wft"
GPU_UUID = "GPU-e027de95-4c80-d8ec-d650-30da9b6eb135"
REPEATS = ("repeat-01", "repeat-02", "repeat-03")
DIMENSIONS = ("object", "swap", "semantic", "task")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def select_method(method: str) -> None:
    global SELECTION, RUN, PLAN_SHA, SELECTION_SHA, MODEL_SHA, METHOD
    spec = SPECS[method]
    SELECTION = ROOT / method / "selection-v1"
    RUN = ROOT / method / "formal-v1"
    PLAN_SHA = spec["plan_sha"]
    SELECTION_SHA = spec["selection_sha"]
    MODEL_SHA = spec["model_sha"]
    METHOD = f"{method}-main10k-codex-v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def raw_sha(state: np.ndarray) -> str:
    value = np.asarray(state)
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError("Invalid raw simulator reset state")
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def audit_plan() -> tuple[dict, dict[str, dict]]:
    plan_path = RUN / "plan.json"
    if sha(plan_path) != PLAN_SHA or sha(SELECTION / "manifest.json") != SELECTION_SHA:
        raise ValueError("Frozen plan or selection manifest changed")
    plan = read(plan_path)
    if (plan.get("schema") != "pi05_pro_baseline_formal_shared_v1"
            or plan.get("host") != HOST or plan.get("gpu") != 7
            or plan.get("gpu_uuid") != GPU_UUID or plan.get("model_sha256") != MODEL_SHA
            or plan.get("method") != METHOD
            or plan.get("jobs_expected") != 480 or plan.get("episodes_expected") != 4800
            or not plan.get("no_retry") or not plan.get("no_score_based_scheduling")):
        raise ValueError("Formal plan protocol mismatch")
    if len(plan["jobs"]) != 480 or len({j["id"] for j in plan["jobs"]}) != 480:
        raise ValueError("Formal job coverage mismatch")
    source = {}
    for repeat in REPEATS:
        selected = SELECTION / "selections" / f"{repeat}.json"
        index = read(SELECTION / "manifest.json")["selection_index"][repeat]
        if sha(selected) != index["sha256"]:
            raise ValueError(f"Selection changed: {repeat}")
        payload = read(selected)
        original = read(Path(index["source_path"]))
        if sha(Path(index["source_path"])) != index["source_sha256"]:
            raise ValueError(f"PRO source selection changed: {repeat}")
        if (payload["jobs"] != original["jobs"]
                or payload["source_identity"] != original["source_identity"]
                or payload["policy_checkpoint_sha256"] != MODEL_SHA
                or payload["eval_seed"] != original["eval_seed"]):
            raise ValueError(f"Baseline/source reset parity failed: {repeat}")
        for item in payload["jobs"]:
            key = f"{repeat}-{item['job_id']}"
            if key in source:
                raise ValueError(f"Duplicate source job: {key}")
            source[key] = item
    if set(source) != {j["id"] for j in plan["jobs"]}:
        raise ValueError("Source/formal job IDs differ")
    if Counter((j["repeat"], j["dimension"], j["suite"]) for j in plan["jobs"]) != Counter({
            (repeat, dimension, suite): 10
            for repeat in REPEATS for dimension in DIMENSIONS for suite in SUITES}):
        raise ValueError("Incomplete repeat/dimension/suite coverage")
    for job in plan["jobs"]:
        item = source[job["id"]]
        if (job["job_id"] != item["job_id"] or job["dimension"] != item["dimension"]
                or job["suite"] != item["source_suite"] or job["task_id"] != item["task_id"]
                or job["state_ids"] != item["state_ids"]
                or job["state_raw_sha256"] != item["state_raw_sha256"]
                or len(item["state_ids"]) != 10
                or job["bddl_sha256"] != item["bddl"]["sha256"]
                or job["init_states_sha256"] != item["init_states"]["sha256"]):
            raise ValueError(f"Source/formal job identity mismatch: {job['id']}")
    return plan, source


def audit_job(job: dict, source: dict, plan: dict) -> tuple[int, dict]:
    key = job["id"]
    launch = read(RUN / "launches" / f"{key}.json")
    exit_row = read(RUN / "exits" / f"{key}.json")
    if (launch.get("id") != key or launch.get("gpu") != 7
            or launch.get("uuid") != GPU_UUID
            or launch.get("admission", {}).get("free_mib", 0) < 40960
            or launch.get("plan_sha256") != PLAN_SHA
            or launch.get("command") != job["command"]
            or exit_row.get("id") != key or exit_row.get("returncode") != 0
            or exit_row.get("accepted") is not True or exit_row.get("error") is not None):
        raise ValueError(f"Launch/exit receipt mismatch: {key}")
    for label in ("bddl", "init_states"):
        row = source[label]
        if sha(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"Raw source file changed: {key}/{label}")
    raw = np.asarray(torch.load(source["init_states"]["path"], weights_only=False))
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.floating):
        raise ValueError(f"Invalid init-state bank: {key}")
    for state_id, expected in zip(source["state_ids"], source["state_raw_sha256"], strict=True):
        if state_id >= len(raw) or raw_sha(raw[state_id]) != expected:
            raise ValueError(f"Actual selected reset SHA mismatch: {key}")
    output = Path(job["output"])
    receipt_path = output / "extension_eval_receipt.json"
    info_path = output / "eval_info.json"
    receipt = read(receipt_path)
    info = read(info_path)
    expected = {
        "benchmark": "libero_pro", "method_id": METHOD,
        "repeat_id": job["repeat"], "eval_seed": job["eval_seed"],
        "checkpoint_contract": "pi05-libero-main10k-matched-v1",
        "policy_sha256": MODEL_SHA, "policy_checkpoint_sha256": MODEL_SHA,
        "selection_sha256": job["selection_sha256"],
        "job_id": job["job_id"], "dimension": job["dimension"],
        "source_suite": job["suite"], "runtime_suite": source["runtime_suite"],
        "task_id": job["task_id"], "task_name": source["task_name"],
        "state_ids": job["state_ids"], "state_raw_sha256": job["state_raw_sha256"],
        "bddl_sha256": job["bddl_sha256"],
        "init_states_sha256": job["init_states_sha256"],
        "eval_info_sha256": sha(info_path),
        "entrypoint_sha256": plan["evaluator_sha256"],
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"Native receipt field differs: {key}/{field}")
    if (exit_row["artifact"]["receipt_sha256"] != sha(receipt_path)
            or exit_row["artifact"]["eval_info_sha256"] != sha(info_path)):
        raise ValueError(f"Exit artifact digests differ: {key}")
    rows = info.get("per_task")
    if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("task_id") != job["task_id"]:
        raise ValueError(f"Native task coverage differs: {key}")
    values = (rows[0].get("metrics") or {}).get("successes")
    if (not isinstance(values, list) or len(values) != 10
            or any(type(value) is not bool for value in values)):
        raise ValueError(f"Native ten bool outcomes missing: {key}")
    if (info.get("overall", {}).get("n_episodes") != 10
            or abs(float(info["overall"].get("pc_success", -1)) - 10 * sum(values)) > 1e-6):
        raise ValueError(f"Native aggregate differs: {key}")
    return sum(values), {"receipt_sha256": sha(receipt_path), "eval_info_sha256": sha(info_path)}


def audit_results(plan: dict, source: dict[str, dict]) -> dict:
    terminal = read(RUN / "queue-ended.json")
    state = read(RUN / "state.json")
    if (terminal.get("complete") is not True or terminal.get("stopped") is not False
            or terminal.get("accepted_jobs") != 480 or terminal.get("accepted_episodes") != 4800
            or terminal.get("failed") or terminal.get("pending_jobs") != 0
            or state.get("status") != "complete" or state.get("active") or state.get("pending")
            or state.get("failed") or len(state.get("accepted", {})) != 480):
        raise ValueError("Formal PRO Soup queue lacks clean complete terminal")
    model = Path(read(SELECTION / "manifest.json")["model"]["path"]) / "model.safetensors"
    if sha(model) != MODEL_SHA:
        raise ValueError("Model bytes changed at final audit")
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    evidence = {}
    for job in plan["jobs"]:
        successes, digests = audit_job(job, source[job["id"]], plan)
        grouped[(job["repeat"], job["dimension"], job["suite"])].append(successes)
        evidence[job["id"]] = digests
    if set(grouped) != {(repeat, dimension, suite)
                        for repeat in REPEATS for dimension in DIMENSIONS for suite in SUITES}:
        raise ValueError("Missing formal PRO Soup table cells")
    by_repeat = {}
    for (repeat, dimension, suite), values in grouped.items():
        if len(values) != 10:
            raise ValueError(f"Task-macro coverage differs: {repeat}/{dimension}/{suite}")
        by_repeat[f"{repeat}/{dimension}/{suite}"] = {
            "tasks": 10, "episodes": 100, "successes": sum(values),
            "success_percent": mean(values) * 10.0}
    dimensions = {}
    for dimension in DIMENSIONS:
        repeats = []
        for repeat in REPEATS:
            successes = sum(by_repeat[f"{repeat}/{dimension}/{suite}"]["successes"]
                            for suite in SUITES)
            repeats.append(successes / 4.0)  # 400 episodes -> percentage.
        dimensions[dimension] = {"repeat_success_percent": repeats,
                                 "mean_percent": mean(repeats),
                                 "sample_std_percent": stdev(repeats),
                                 "episodes": 1200, "repeats": 3}
    overall_repeats = [mean(dimensions[dimension]["repeat_success_percent"][index]
                            for dimension in DIMENSIONS) for index in range(3)]
    return {"schema": "pi05_pro_baseline_independent_table_data_v1", "accepted": True,
            "method": METHOD, "model_sha256": MODEL_SHA,
            "formal_plan_sha256": PLAN_SHA, "formal_terminal_sha256": sha(RUN / "queue-ended.json"),
            "selection_manifest_sha256": SELECTION_SHA,
            "independent_auditor_sha256": sha(Path(__file__)),
            "jobs": 480, "episodes": 4800, "by_repeat_suite_dimension": by_repeat,
            "dimensions": dimensions,
            "average_four_dimensions": {"repeat_success_percent": overall_repeats,
                                        "mean_percent": mean(overall_repeats),
                                        "sample_std_percent": stdev(overall_repeats)},
            "raw_receipt_digests": evidence,
            "notes": "Four equal-weight perturbations, including altered-goal task; 3 paired repeats per dimension. No PRO clean or environment perturbation is implied."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "one-accepted", "results"))
    parser.add_argument("--method", required=True, choices=sorted(SPECS))
    args = parser.parse_args()
    select_method(args.method)
    plan, source = audit_plan()
    if args.action == "preflight":
        print(json.dumps({"plan_valid": True, "jobs": 480, "episodes": 4800}))
    elif args.action == "one-accepted":
        job = plan["jobs"][0]
        audit_job(job, source[job["id"]], plan)
        print(json.dumps({"first_job_independent_technical_check": "passed"}))
    else:
        data = audit_results(plan, source)
        output = RUN / "independent-table-data-v1.json"
        with output.open("x") as stream:
            stream.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"path": str(output), "sha256": sha(output),
                          "accepted": True, "jobs": 480, "episodes": 4800}))


if __name__ == "__main__":
    main()
