#!/usr/bin/env python3
"""Run the corrected paper RegMean++ action arm on the frozen C146 bank."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
VLA = WORK / "vla-merge"
SOURCE = WORK / "pi05_lora_finetune_v2_20260826"
PYTHON = SOURCE / ".venv/bin/python"
WORKER = HERE / "run_candidate_actions.py"
BASE_RUN = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/candidate-actions-attempt-01"
DEFAULT_RUN = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/regmeanpp-correction-attempt-01"
MODEL = WORK / "vla-merge-runtime/experiments/table3-ablation-20260916/checkpoints/uniform"
IDENTITY = WORK / "vla-merge-runtime/experiments/table3-ablation-20260916/status.json"
MODEL_SHA = "49ea499c3ad37329b95baf597b252bc424cef40da312edd943a5102fabfe645a"
OLD_SHA = "38745ba812c2f4ac4327ab40719e0ce8da017873c0a1765608f65dd926cd4ab3"
FLOCK_DIR = WORK / "vla-merge/experiments/claude-firstpass-cause-20260920"
sys.path.insert(0, str(FLOCK_DIR))
import card_flock  # noqa: E402


def sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def gpu_rows():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, free, used, util = [value.strip() for value in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "free_mib": int(float(free)),
                     "used_mib": int(float(used)), "utilization_gpu_percent": int(float(util))})
    return sorted(rows, key=lambda row: (-row["free_mib"], row["index"]))


def environment(gpu: int):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "ITERATION_PHYSICAL_GPU": str(gpu),
                "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
                "PALIGEMMA_TOKENIZER_PATH": str(SOURCE / "assets/paligemma-3b-pt-224-tokenizer"),
                "PYTHONPATH": ":".join([str(WORK / "vla-merge-runtime/python-overlay"), str(SOURCE / "src"),
                    str(SOURCE / "lerobot/src"), str(VLA), str(VLA / "src")])})
    return env


def prepare(run: Path) -> None:
    if run.exists():
        raise FileExistsError(run)
    old_plan = json.loads((BASE_RUN / "plan.json").read_text())
    if sha(MODEL / "model.safetensors") != MODEL_SHA:
        raise ValueError("Current paper RegMean++ model hash differs")
    for filename, expected in old_plan["normalizer_identity"].items():
        if sha(MODEL / filename) != expected:
            raise ValueError(f"RegMean++ processor differs: {filename}")
    config = json.loads((MODEL / "config.json").read_text())
    if config.get("num_inference_steps") != 10 or config.get("chunk_size") != 50:
        raise ValueError("RegMean++ generation config differs")
    model = {"path": str(MODEL), "model_sha256": MODEL_SHA,
             "identity_source": str(IDENTITY), "identity_source_sha256": sha(IDENTITY),
             "group": "main", "policy_files": old_plan["normalizer_identity"]}
    plan = dict(old_plan)
    plan.update({"schema": "main_fidelity_regmeanpp_correction_v1",
                 "created_at": datetime.now(timezone.utc).isoformat(),
                 "source": str(WORKER), "source_sha256": sha(WORKER),
                 "models": {"regmean_pp": model}, "max_active": 1,
                 "correction_reason": "Replace historical SHA38745 arm with current paper 69.42 checkpoint SHA49ea.",
                 "supersedes_model_sha256": OLD_SHA})
    run.mkdir(parents=True)
    write(run / "plan.json", plan)
    write(BASE_RUN / "REGMEANPP-OLD-CHECKPOINT-SUPERSEDED.json", {
        "schema": "main_fidelity_superseded_arm_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "regmean_pp", "old_model_sha256": OLD_SHA, "correct_model_sha256": MODEL_SHA,
        "old_outputs_preserved": True, "exclude_old_arm_from_current_paper": True,
        "reason": "The prior action arm used a historical checkpoint, not the current paper 69.42 RegMean++ row."})
    print(json.dumps({"prepared": str(run), "model_sha256": MODEL_SHA}))


def take_best(admission_mib: int, label: str):
    uuids = card_flock.gpu_uuids()
    for row in gpu_rows():
        if row["free_mib"] < admission_mib or uuids.get(row["index"]) != row["uuid"]:
            continue
        lease = card_flock.take_card(row["index"], row["uuid"], label, "offline-native-forward")
        if lease is not None:
            return row, lease
    return None, None


def run_one(run: Path, smoke: bool) -> None:
    plan = json.loads((run / "plan.json").read_text())
    admission = 32 * 1024
    if not smoke:
        passed = json.loads((run / "SMOKE-PASSED.json").read_text())
        if passed.get("plan_sha256") != sha(run / "plan.json"):
            raise ValueError("Smoke is not bound to correction plan")
        admission = int(passed["full_admission_mib"])
    row, lease = take_best(admission, "main-fidelity-regmeanpp-correction")
    if lease is None:
        raise RuntimeError("No currently admissible GPU; caller may retry after a low-frequency wait")
    label = "smoke" if smoke else "full"
    log = run / f"{label}.log"
    command = [str(PYTHON), "-u", str(WORKER), "--worker", "--model", "regmean_pp", "--run", str(run)]
    if smoke:
        command.append("--smoke")
    started = time.time()
    try:
        with log.open("x") as handle:
            process = subprocess.run(command, cwd=VLA, env=environment(row["index"]),
                                     stdout=handle, stderr=subprocess.STDOUT)
        write(run / f"{label}-exit.json", {"return_code": process.returncode, "gpu": row["index"],
              "snapshot": row, "elapsed_seconds": time.time() - started, "command": command})
        if process.returncode:
            raise RuntimeError(f"RegMean++ correction {label} failed")
        manifest = json.loads((run / ("smoke" if smoke else "actions") / "regmean_pp" / "manifest.json").read_text())
        if manifest.get("model_sha256") != MODEL_SHA or manifest.get("requests") != (1 if smoke else 400):
            raise ValueError("RegMean++ correction manifest differs")
        if smoke:
            full_admission = max(32 * 1024, math.ceil((manifest["peak_reserved_gib"] + 12) * 1024))
            write(run / "SMOKE-PASSED.json", {"passed": True, "gpu": row["index"],
                  "manifest_sha256": sha(run / "smoke/regmean_pp/manifest.json"),
                  "measured_peak_reserved_gib": manifest["peak_reserved_gib"],
                  "full_admission_mib": full_admission, "runtime_floor_mib": 12 * 1024,
                  "plan_sha256": sha(run / "plan.json")})
        else:
            write(run / "queue-ended.json", {"status": "complete", "complete": True,
                  "states": {"regmean_pp": {"status": "complete", "return_code": 0,
                  "gpu": row["index"], "started_unix": started, "ended_unix": time.time()}},
                  "elapsed_seconds": time.time() - started,
                  "finished_at": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"complete": True, "phase": label, "gpu": row["index"],
                          "requests": manifest["requests"], "peak_reserved_gib": manifest["peak_reserved_gib"]}))
    finally:
        lease.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if sum((args.prepare, args.smoke, args.full)) != 1:
        parser.error("Choose exactly one mode")
    run = args.run.resolve()
    if args.prepare:
        prepare(run)
    else:
        run_one(run, args.smoke)


if __name__ == "__main__":
    main()
