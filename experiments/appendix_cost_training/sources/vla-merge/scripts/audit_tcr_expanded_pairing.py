"""Read-only D2 receipt audit. No GPU, no scores, no changes to running jobs.

This verifies recorded protocol evidence, not determinism or statistical superiority.
Checkpoint bytes and simulator state contents are outside this audit's scope.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

ARMS = ("e0", "c_e", "c_m", "n2", "featcal")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
OFFSETS = range(30, 40)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_receipt(row, offset):
    suite, task = row["suite"], row["task_id"]
    require(suite in SUITES and type(task) is int and 0 <= task < 10, "Invalid task")
    require(row["key"] == f"{suite}/{task}/off{offset}", "Invalid pairing key")
    for field in ("offset", "init_state_offset", "actual_init_state_id"):
        require(type(row[field]) is int and row[field] == offset, f"Invalid {field}")
    require(type(row["available_init_states"]) is int and row["available_init_states"] >= 40,
            "Insufficient init states")
    seed = 391600 + 1000 * (offset - 30) + 100 * SUITES.index(suite) + task
    require(type(row["environment_seed"]) is int and row["environment_seed"] == seed,
            "Environment seed mismatch")
    require(type(row["flow_seed"]) is int and row["flow_seed"] == seed + 100000,
            "Flow seed mismatch")
    values = row["successes"]
    require(isinstance(values, list) and len(values) == 1 and type(values[0]) is bool,
            "Invalid success receipt")
    hashes = row["noise_hashes"]
    require(isinstance(hashes, list) and bool(hashes), "Empty noise evidence")
    for digest in hashes:
        require(isinstance(digest, str) and len(digest) == 64 and
                all(c in "0123456789abcdef" for c in digest), "Invalid noise hash")
    return suite, task, offset


def compare_noise(left, right):
    count = min(len(left["noise_hashes"]), len(right["noise_hashes"]))
    require(count > 0, "Empty common noise prefix")
    require(left["noise_hashes"][:count] == right["noise_hashes"][:count],
            "Actual flow-noise prefix mismatch")
    require(left["available_init_states"] == right["available_init_states"],
            "Init bank length mismatch")
    return count


def receipt_lines(path, allow_partial):
    data = path.read_bytes()
    pending = bool(data) and not data.endswith(b"\n")
    require(not pending or allow_partial, "Truncated receipt file")
    lines = data.splitlines()
    if pending:
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()], pending


def audit(run, allow_incomplete=False):
    plan = json.loads((run / "plan-e0-c_e-c_m-n2-featcal-30-39.json").read_text())
    require(plan["arms"] == list(ARMS) and plan["offsets"] == list(OFFSETS),
            "D2 arm/state selection changed")
    require(plan["episodes_total"] == 2000 and plan["episodes_per_job"] == 40,
            "D2 size changed")
    for name in ("wrapper", "original_wrapper"):
        digest = hashlib.sha256(Path(plan[name]).read_bytes()).hexdigest()
        require(digest == plan[name + "_sha256"], f"Changed {name}")
    rows_by_arm = {arm: {} for arm in ARMS}
    complete_jobs = pending_lines = 0
    for arm in ARMS:
        for offset in OFFSETS:
            folder = run / arm / f"offset-{offset}"
            launch_path = folder / "launch.json"
            if not launch_path.exists():
                require(allow_incomplete, f"Missing launch: {arm}/{offset}")
                continue
            launch = json.loads(launch_path.read_text())
            require(launch["wrapper_sha256"] == plan["wrapper_sha256"], "Launch wrapper mismatch")
            require(launch["model"] == plan["models"][arm]["path"] and
                    launch["model_sha256"] == plan["models"][arm]["model_sha256"],
                    "Launch model provenance mismatch")
            require(launch["offset"] == offset, "Launch offset mismatch")
            exited = (folder / "exit.json").exists()
            if exited:
                require(json.loads((folder / "exit.json").read_text())["return_code"] == 0,
                        f"Failed job: {arm}/{offset}")
            else:
                require(allow_incomplete, f"Job has not exited: {arm}/{offset}")
            receipts_path = folder / "paired-noise.jsonl"
            if not receipts_path.exists():
                require(allow_incomplete and not exited, "Missing receipts")
                continue
            rows, pending = receipt_lines(receipts_path, allow_incomplete and not exited)
            pending_lines += int(pending)
            require(len(rows) <= 40, "Too many episode receipts")
            local = {}
            for row in rows:
                key = validate_receipt(row, offset)
                require(key not in local, "Duplicate receipt")
                local[key] = row
            rows_by_arm[arm].update(local)
            if exited:
                require(len(local) == 40, "Incomplete exited job")
                metrics = json.loads((folder / "eval_info.json").read_text())["per_task"]
                require(len(metrics) == 40, "Incomplete eval_info")
                seen = set()
                for item in metrics:
                    key = item["task_group"], item["task_id"], offset
                    require(key in local and key not in seen, "Invalid eval_info task coverage")
                    values = item["metrics"]["successes"]
                    require(isinstance(values, list) and len(values) == 1 and type(values[0]) is bool,
                            "Invalid eval_info outcome")
                    require(values == local[key]["successes"], "Result/receipt mismatch")
                    seen.add(key)
                complete_jobs += 1
    comparisons = {}
    for a, b in itertools.combinations(ARMS, 2):
        common = rows_by_arm[a].keys() & rows_by_arm[b].keys()
        calls = sum(compare_noise(rows_by_arm[a][key], rows_by_arm[b][key]) for key in common)
        comparisons[f"{a}_vs_{b}"] = {"paired_episodes_checked": len(common),
                                        "common_request_noise_hashes_checked": calls}
    return {"status": "complete_receipt_audit" if complete_jobs == 50 else "partial_receipt_audit",
            "completed_jobs": complete_jobs, "expected_jobs": 50,
            "receipts_checked": {arm: len(rows) for arm, rows in rows_by_arm.items()},
            "inflight_partial_lines_ignored": pending_lines, "pairing": comparisons,
            "checkpoint_bytes_rehashed": False, "simulator_states_rehashed": False,
            "independent_confirmation_used": False, "goal_achieved": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.allow_incomplete), indent=2))
