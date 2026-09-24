#!/usr/bin/env python3
"""Strictly audit a terminal OpenVLA repair development screen from raw rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import evaluate_development as evaluator
from vla_merge.libero_procedural_bank import sha256_file

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SHORT = {"libero_spatial": "spatial", "libero_object": "object",
         "libero_goal": "goal", "libero_10": "long"}


def audit(run: Path) -> dict:
    run = run.resolve()
    plan_path, terminal_path = run / "plan.json", run / "queue-ended.json"
    plan, terminal = json.loads(plan_path.read_text()), json.loads(terminal_path.read_text())
    models = tuple(plan["models"])
    bank, selection_path = Path(plan["selection"]).parent, Path(plan["selection"])
    selection = evaluator.load_development_selection(bank, selection_path)
    per_suite_episodes = len(selection.tasks[(SUITES[0], 0)].indices) * 10
    expected_jobs = {(model, suite) for model in models for suite in SUITES}
    planned_jobs = {(job["model"], job["suite"]) for job in plan["jobs"]}
    if (planned_jobs != expected_jobs or len(plan["jobs"]) != len(expected_jobs)
            or terminal.get("status") != "complete" or terminal.get("stopped") is not False
            or terminal.get("accepted_jobs") != len(expected_jobs)
            or terminal.get("episodes") != len(models) * per_suite_episodes * len(SUITES)):
        raise ValueError("Development plan or terminal coverage differs")

    rows_by_model: dict[str, dict[tuple, dict]] = {}
    summaries = {}
    for model in models:
        keyed = {}
        per_suite = {}
        for suite in SUITES:
            folder = run / "jobs" / f"{model}-{SHORT[suite]}" / "eval"
            contract = json.loads((folder / "contract.json").read_text())
            rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
            successes = evaluator.validate_rows(rows, selection, suite)
            summary = json.loads((folder / "summary.json").read_text())
            if (contract.get("selection_sha256") != selection.selection_sha256
                    or contract.get("bank_manifest_sha256") != selection.manifest_sha256
                    or contract.get("selection_id") != selection.selection_id
                    or contract.get("eval_seed") != selection.eval_seed
                    or contract.get("suite") != suite
                    or contract.get("formal_evaluation") is not False
                    or contract.get("used_for_candidate_selection") is not True
                    or summary.get("complete") is not True
                    or summary.get("episodes") != per_suite_episodes
                    or summary.get("successes") != successes
                    or summary.get("episodes_sha256") != sha256_file(folder / "episodes.jsonl")):
                raise ValueError(f"Development receipt differs: {model}/{suite}")
            per_suite[SHORT[suite]] = successes
            for row in rows:
                key = (suite, row["task_id"], row["episode_index"], row["raw_state_sha256"])
                if key in keyed:
                    raise ValueError(f"Duplicate development row: {model}/{key}")
                keyed[key] = row
        rows_by_model[model] = keyed
        total = sum(per_suite.values())
        summaries[model] = {"successes": total, "episodes": len(keyed),
                            "pc_success": 100.0 * total / len(keyed),
                            "per_suite_successes": per_suite}

    reference_keys = set(next(iter(rows_by_model.values())))
    if any(set(rows) != reference_keys for rows in rows_by_model.values()):
        raise ValueError("Models were not evaluated on identical development episodes")
    paired = {}
    for left in models:
        for right in models:
            if left >= right:
                continue
            wins = losses = ties = 0
            for key in reference_keys:
                a = rows_by_model[left][key]["success"]
                b = rows_by_model[right][key]["success"]
                wins += a and not b
                losses += b and not a
                ties += a == b
            paired[f"{left}_minus_{right}"] = {
                "wins": wins, "losses": losses, "ties": ties,
                "difference_pp": summaries[left]["pc_success"] - summaries[right]["pc_success"]}
    return {"schema": "openvla_repair_development_audit_v1", "accepted": True,
        "complete": True, "development_only": True, "selection_id": selection.selection_id,
        "eval_seed": selection.eval_seed, "models": list(models),
        "jobs": len(expected_jobs), "episodes": len(models) * len(reference_keys),
        "summaries": summaries, "paired": paired,
        "inputs": {"plan_sha256": sha256_file(plan_path),
                   "terminal_sha256": sha256_file(terminal_path),
                   "identity_sha256": sha256_file(run / "identities.json"),
                   "selection_sha256": selection.selection_sha256,
                   "bank_manifest_sha256": selection.manifest_sha256},
        "auditor_sha256": sha256_file(Path(__file__)),
        "audited_at": datetime.now(timezone.utc).isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temp = Path(stream.name)
    temp.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
