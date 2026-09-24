"""Summarize only complete nightly arms; incomplete arms remain explicitly incomplete."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np

from run_tcr_night_20260919 import DEFAULT, SUITES, read, sha, write
from audit_tcr_expanded_pairing import validate_receipt, compare_noise
from run_iclr2027_table1_clean_fastlane import validate_eval_info


def cluster_interval(differences):
    """Resample tasks within suites, keeping all repeated episodes of a task together."""
    rng = np.random.default_rng(20260919)
    sample = np.zeros(10000)
    for suite in SUITES:
        a = np.asarray([differences[(suite, i)] for i in range(10)], dtype=float)
        sample += a[rng.integers(0, 10, (10000, 10))].mean(axis=1) / 4
    return list(np.quantile(sample * 100, [.025, .975]))


def development_rows(folder, offset, expected):
    """Recheck results against receipts, not just the existence of a green marker."""
    from run_tcr_half_step_development import verify_job
    actual = verify_job(folder, offset)
    if actual != expected:
        raise ValueError("Development verification changed")
    return [json.loads(line) for line in (folder / "paired-noise.jsonl").read_text().splitlines()]


def development_item(rows):
    expected = {(s, t, o) for s in SUITES for t in range(10) for o in range(30, 40)}
    keys = [validate_receipt(r, r["offset"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicated development episode")
    if not set(keys) <= expected:
        raise ValueError("Unexpected development episode")
    item = {"complete": set(keys) == expected, "verified_episodes": len(rows)}
    if item["complete"]:
        item.update(successes=sum(r["successes"][0] for r in rows),
                    success_percent=sum(r["successes"][0] for r in rows)/4,
                    suite_percent={s:sum(r["successes"][0] for r in rows if r["suite"] == s)
                                   for s in SUITES})
    return item


def paired_development(left, right):
    a = {(r["suite"], r["task_id"], r["offset"]):r for r in left}
    b = {(r["suite"], r["task_id"], r["offset"]):r for r in right}
    if not development_item(left)["complete"] or not development_item(right)["complete"]:
        return {"complete": False}
    diffs = {(s,t):[] for s in SUITES for t in range(10)}
    wins = losses = 0
    for key in a:
        compare_noise(a[key], b[key])
        d = int(a[key]["successes"][0])-int(b[key]["successes"][0])
        diffs[key[:2]].append(d)
        wins += d == 1
        losses += d == -1
    return {"complete": True, "paired_wins": wins, "paired_losses": losses,
            "delta_pp": (wins-losses)/4,
            "task_cluster_95ci_delta_pp": cluster_interval({k:statistics.mean(v) for k,v in diffs.items()}),
            "claim_boundary": "Exploratory same-reset and common flow-noise-prefix comparison; not independent confirmation."}


def summarize(run):
    plan = read(run / "plan.json")
    result = {"formal": {}, "development": {}, "failures": {}, "independent_confirmation_used": False,
              "claim_boundary": "Existing two-pass and newly collected one-pass builds are different acquisitions; their gap alone is not the causal effect of a second pass. Formal reset states are paired, flow-noise streams not guaranteed paired."}
    for family, prefix in (("expert_pass2", "expert-pass2-r"), ("fresh_tcre", "tcre-r")):
        jobs = [j for j in plan["jobs"] if j["kind"] == "formal" and j["model"].startswith(prefix)]
        expected_jobs = {(r,s) for r in range(1,4) for s in SUITES}
        if len(jobs) != 12 or {(j["repeat"],j["suite"]) for j in jobs} != expected_jobs:
            raise ValueError("Formal family job allocation changed")
        completed = [j for j in jobs if (run / "jobs" / j["id"] / "verified.json").exists()]
        if len(completed) != 12:
            result["formal"][family] = {"complete": False, "verified_suite_jobs": len(completed), "required": 12}
            continue
        repeat_success = {i:0 for i in range(1,4)}
        baseline_repeat = {i:0 for i in range(1,4)}
        diffs = {(s,i): [] for s in SUITES for i in range(10)}
        wins = losses = 0
        suite_success = {s:0 for s in SUITES}
        for j in jobs:
            folder = run / "jobs" / j["id"]
            v = read(folder / "verified.json")
            if read(folder / "exit.json")["return_code"] != 0 or sha(folder / "eval/eval_info.json") != v["eval_info_sha256"]:
                raise ValueError("Completed evidence changed")
            if sha(folder / "eval/procedural_bank_receipt.json") != v["reset_sha256"]:
                raise ValueError("Reset receipt changed")
            reference = plan["baseline"][f"repeat-{j['repeat']:02d}/{j['suite']}"]
            if v["baseline_eval"] != reference["eval_info"]:
                raise ValueError("Unexpected formal reference")
            for path in (reference["eval_info"], reference["reset"]):
                if sha(path) != plan["frozen"][path]["sha256"]:
                    raise ValueError("Frozen baseline evidence changed")
            raw_result, outcomes = validate_eval_info(folder / "eval/eval_info.json", j["suite"])
            if (raw_result != v["result"] or
                    {str(k): vals for k, vals in outcomes.items()} != v["outcomes"]):
                raise ValueError("Formal verified outcomes differ from raw episodes")
            _, base_outcomes = validate_eval_info(Path(v["baseline_eval"]), j["suite"])
            actual_reset = read(folder / "eval/procedural_bank_receipt.json")
            baseline_reset = read(reference["reset"])
            for key in ("bank_manifest_sha256", "selection_sha256", "repeat_id", "eval_seed", "tasks"):
                if actual_reset[key] != baseline_reset[key]:
                    raise ValueError(f"Formal reset pairing differs: {key}")
            for task in range(10):
                a, b = outcomes[task], base_outcomes[task]
                if len(a) != 10 or len(b) != 10:
                    raise ValueError("Not ten paired episodes")
                delta = [int(x)-int(y) for x,y in zip(a,b)]
                wins += sum(x==1 for x in delta)
                losses += sum(x==-1 for x in delta)
                diffs[(j["suite"],task)].extend(delta)
                repeat_success[j["repeat"]] += sum(a)
                baseline_repeat[j["repeat"]] += sum(b)
                suite_success[j["suite"]] += sum(a)
        values = [repeat_success[i]/4 for i in range(1,4)]
        baseline_values = [baseline_repeat[i]/4 for i in range(1,4)]
        result["formal"][family] = {"complete": True, "episodes":1200, "successes":sum(repeat_success.values()),
            "mean_percent":statistics.mean(values), "std_percent":statistics.stdev(values),
            "repeat_percent":values, "suite_percent":{s:n/3 for s,n in suite_success.items()},
            "featcal_repeat_percent":baseline_values, "delta_pp":statistics.mean(values)-statistics.mean(baseline_values),
            "paired_wins":wins, "paired_losses":losses,
            "task_cluster_95ci_delta_pp":cluster_interval({k:statistics.mean(v) for k,v in diffs.items()}),
            "interval_unit":"suite-stratified tasks, all repeats/episodes retained per task"}
    all_dev_rows = {}
    for model in ("tcre-r1", "uniform-r1", "c_e_half", "c_m_half"):
        rows = []
        for j in plan["jobs"]:
            if j["kind"] != "development" or j["model"] != model:
                continue
            folder = run / "jobs" / j["id"]
            if not (folder / "verified.json").exists():
                continue
            v = read(folder / "verified.json")
            if read(folder / "exit.json")["return_code"] != 0:
                raise ValueError("Development worker failed")
            rows.extend(development_rows(folder / "eval", j["offset"], v))
        for reuse in plan["reused_half"]:
            if reuse["model"] == model:
                rows.extend(development_rows(Path(reuse["path"]), reuse["offset"], reuse["verified"]))
        all_dev_rows[model] = rows
        result["development"][model] = development_item(rows)
    result["development_comparisons"] = {
        "uniform_minus_original": paired_development(all_dev_rows["uniform-r1"], all_dev_rows["tcre-r1"])}
    for f in (run / "jobs").glob("*/failure.json"):
        result["failures"][f.parent.name] = read(f)["error"]
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=DEFAULT)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    data = summarize(args.run)
    if args.output:
        write(args.output, data)
    print(json.dumps(data, indent=2))
