#!/usr/bin/env python3
"""Finish the matched OpenVLA-OFT expert formal panel on procedural repeats 2/3.

Repeat 1 is an immutable, separately completed 400-episode run. This queue
contains only its eight missing suite/repeat cells (800 episodes); it never
runs or copies a repeat-1 episode. A separate combined audit is still required
before entering a three-repeat result into the paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "openvla-tcr-20260921"
WORK = HERE.parents[2]
sys.path[:0] = [str(HERE), str(COMMON), str(WORK / "vla-merge/src")]
import run_development40 as screen  # noqa: E402
import run_local as base  # noqa: E402

from vla_merge.libero_procedural_bank import load_selection  # noqa: E402

formal = base.formal
BANK = formal.BANK
PRIOR = (WORK / "vla-merge-runtime/experiments/openvla-tcr-20260921/"
         "local-expert-dynamic-01")
REPEATS = (2, 3)
SUITES = screen.SUITES
SHORT = screen.SHORT
ADMISSION_MIB = 56 * 1024
RESERVE_MIB = 8 * 1024
MEASURED_REQUIREMENT_MIB = 24 * 1024
MAX_WORKERS = 2


def prior_gate() -> dict:
    terminal = json.loads((PRIOR / "eval/queue-ended.json").read_text())
    if (terminal.get("status") != "complete" or terminal.get("accepted_jobs") != 4
            or terminal.get("episodes") != 400 or terminal.get("stopped") is not False):
        raise ValueError("The matched repeat-1 expert panel is incomplete")
    identity = json.loads((PRIOR / "identities.json").read_text())
    if identity.get("complete") is not True or len(identity.get("experts", {})) != 4:
        raise ValueError("The released expert identity audit is incomplete")
    base.assert_unchanged(identity["files"])
    return identity


def selections() -> dict[int, object]:
    loaded = {repeat: load_selection(BANK, BANK / f"selections/repeat-0{repeat}.json",
                                     verify_source_files=False)
              for repeat in (1, 2, 3)}
    seen = set()
    for repeat, selection in loaded.items():
        if (selection.repeat_id != f"repeat-0{repeat}"
                or selection.eval_seed != 274000 + repeat or len(selection.tasks) != 40):
            raise ValueError("Procedural selection identity differs")
        for (suite, task_id), task in selection.tasks.items():
            if suite not in SUITES or task_id not in range(10) or len(task.raw_sha256) != 10:
                raise ValueError("Procedural task coverage differs")
            for digest in task.raw_sha256:
                key = (suite, task_id, digest)
                if key in seen:
                    raise ValueError("Formal repeats reuse a raw reset identity")
                seen.add(key)
    if len(seen) != 1200:
        raise ValueError("Formal reset bank does not have 1200 distinct episode identities")
    return loaded


def jobs_for(run: Path, identity: dict) -> list[dict]:
    jobs = []
    by_suite = {row["suite"]: row for row in identity["experts"].values()}
    if set(by_suite) != set(SUITES):
        raise ValueError("Official expert suite coverage differs")
    for repeat in REPEATS:
        selection = BANK / f"selections/repeat-0{repeat}.json"
        for suite in SUITES:
            output = run / "jobs" / f"expert-{SHORT[suite]}-r0{repeat}"
            checkpoint = Path(by_suite[suite]["local_path"]).resolve()
            command = [str(base.PYTHON), str(COMMON / "evaluate.py"),
                       "--checkpoint", str(checkpoint), "--suite", suite,
                       "--bank", str(BANK), "--selection", str(selection),
                       "--output", str(output / "eval")]
            jobs.append({"id": output.name, "repeat": repeat, "suite": suite,
                         "checkpoint": str(checkpoint), "selection": str(selection),
                         "output": str(output), "identity": str(run / "identities.json"),
                         "command": command})
    if len(jobs) != 8 or len({job["id"] for job in jobs}) != 8:
        raise ValueError("Expert complement is not eight jobs")
    return jobs


def verify(job: dict) -> dict:
    folder = Path(job["output"]) / "eval"
    selected = load_selection(BANK, Path(job["selection"]), verify_source_files=True)
    contract = json.loads((folder / "contract.json").read_text())
    if (contract.get("checkpoint") != job["checkpoint"]
            or contract.get("suite") != job["suite"]
            or contract.get("repeat_id") != selected.repeat_id
            or contract.get("eval_seed") != selected.eval_seed
            or contract.get("selection_sha256") != selected.selection_sha256
            or contract.get("bank_manifest_sha256") != selected.bank_manifest_sha256
            or contract.get("entrypoint_sha256") != formal.sha(COMMON / "evaluate.py")
            or contract.get("mode") != "eval" or contract.get("hard_reset") is not True
            or contract.get("native_chunk") != 8 or contract.get("execute_actions") != 8):
        raise ValueError("OpenVLA expert formal contract differs")
    base.assert_unchanged(json.loads(Path(job["identity"]).read_text())["files"])
    rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
    from evaluate import validate_rows
    successes = validate_rows(rows, selected, job["suite"])
    summary = json.loads((folder / "summary.json").read_text())
    if (summary.get("complete") is not True or summary.get("episodes") != 100
            or summary.get("successes") != successes
            or summary.get("episodes_sha256") != formal.sha(folder / "episodes.jsonl")):
        raise ValueError("Expert formal summary differs from raw episodes")
    return {"episodes": 100, "successes": successes,
            "episodes_sha256": summary["episodes_sha256"],
            "selection_sha256": selected.selection_sha256}


def prepare(run: Path) -> None:
    if run.exists():
        raise FileExistsError(run)
    if ADMISSION_MIB < MEASURED_REQUIREMENT_MIB + RESERVE_MIB:
        raise ValueError("Expert admission lacks 8 GiB reserve")
    identity = prior_gate()
    loaded = selections()
    run.mkdir(parents=True, exist_ok=False)
    for repeat in REPEATS:
        for suite in SUITES:
            output = run / "preflight" / f"r0{repeat}-{SHORT[suite]}"
            checkpoint = next(Path(row["local_path"]) for row in identity["experts"].values()
                              if row["suite"] == suite)
            command = [str(base.PYTHON), str(COMMON / "evaluate.py"),
                       "--checkpoint", str(checkpoint), "--suite", suite,
                       "--bank", str(BANK), "--selection",
                       str(BANK / f"selections/repeat-0{repeat}.json"),
                       "--output", str(output), "--preflight-only"]
            subprocess.run(command, cwd=WORK / "vla-merge", env=screen.environment(0),
                           check=True, capture_output=True, text=True)
            contract = json.loads((output / "contract.json").read_text())
            if (contract.get("mode") != "preflight" or contract.get("suite") != suite
                    or contract.get("repeat_id") != loaded[repeat].repeat_id
                    or contract.get("selection_sha256") != loaded[repeat].selection_sha256):
                raise ValueError("Native expert preflight differs")
    files = dict(identity["files"])
    for path in (Path(__file__).resolve(), COMMON / "evaluate.py", COMMON / "native_oft.py",
                 BANK / "manifest.json", PRIOR / "eval/queue-ended.json",
                 PRIOR / "identities.json", *(BANK / f"selections/repeat-0{i}.json"
                                               for i in (1, 2, 3))):
        files[str(path.resolve())] = base.file_identity(path.resolve())
    audit = run / "identities.json"
    base.BASE_SAVE(audit, {"complete": True, "files": files,
        "experts": identity["experts"], "prior_repeat1": str(PRIOR / "eval"),
        "prior_repeat1_episodes": 400, "new_repeats": [2, 3],
        "new_episodes": 800, "selection_uses_success_values": False})
    frozen_env = screen.environment(0)
    environment_contract = {key: frozen_env[key] for key in (
        "PYTHONPATH", "LIBERO_CONFIG_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS")}
    environment_contract["CUDA_VISIBLE_DEVICES"] = "selected GPU 0-7"
    plan = {"schema": "openvla_oft_expert_formal_repeats23_v1",
            "host": socket.gethostname(), "run": str(run), "gpus": list(screen.GPUS),
            "gpu_uuids": {gpu: formal.gpu_row(gpu)["uuid"] for gpu in screen.GPUS},
            "max_workers": MAX_WORKERS, "one_worker_per_card": True,
            "min_free_mib": ADMISSION_MIB, "admission_reserve_mib": RESERVE_MIB,
            "measured_eval_requirement_mib": MEASURED_REQUIREMENT_MIB,
            "runtime_floor_mib": RESERVE_MIB, "jobs": jobs_for(run, identity),
            "episodes": 800, "reused_repeat1_episodes": 400,
            "identity_audit": str(audit), "identity_audit_sha256": formal.sha(audit),
            "evaluator_sha256": formal.sha(COMMON / "evaluate.py"),
            "environment_contract": environment_contract, "no_retry": True,
            "no_score_based_scheduling": True, "formal_evaluation": True,
            "authorization": "User requested 1200 matched OpenVLA-OFT formal episodes"}
    base.BASE_SAVE(run / "plan.json", plan)
    base.BASE_SAVE(run / "CLAIM.json", {"owner": "Claude", "host": socket.gethostname(),
        "scope": "only official experts procedural repeat 2/3, eight jobs, 800 new episodes",
        "prior_repeat1_reused": True, "no_retry": True})


def run_queue(run: Path) -> None:
    plan = json.loads((run / "plan.json").read_text())
    if (plan.get("schema") != "openvla_oft_expert_formal_repeats23_v1"
            or plan.get("host") != socket.gethostname() or (run / "started.json").exists()):
        raise ValueError("Wrong host/plan or already started")
    if formal.sha(Path(__file__).resolve()) != json.loads((run / "identities.json").read_text())["files"][str(Path(__file__).resolve())]["sha256"]:
        raise ValueError("Runner changed since preflight")
    for gpu, expected in plan["gpu_uuids"].items():
        if formal.gpu_row(int(gpu))["uuid"] != expected:
            raise ValueError("GPU UUID changed")
    base.assert_unchanged(json.loads((run / "identities.json").read_text())["files"])
    formal.GPUS = screen.DynamicCards(plan["gpus"])
    formal.FLOOR_MIB = RESERVE_MIB
    formal.EVALUATOR = COMMON / "evaluate.py"
    formal.IDENTITIES = run / "identities.json"
    formal.scientific_environment = screen.environment
    formal.verify_job = verify
    formal.save = base.save_with_actual_counts
    formal.run_queue(plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare(args.run.resolve()) if args.prepare else run_queue(args.run.resolve())


if __name__ == "__main__":
    main()
