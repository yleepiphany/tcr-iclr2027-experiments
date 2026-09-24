#!/usr/bin/env python3
"""Queue Fast-WAM capture acceptance, real prefix gate, then one TCR build.

No result-dependent adaptation or retries. The sequencer waits for the frozen
80-job collector, independently audits it, and dynamically admits one safe
GPU with an exclusive UUID lease and build slot for the parity/build stages.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNTIME = ROOT / "vla-merge-runtime"
BASE = RUNTIME / "experiments/claude-fastwam-tcr-20260923"
QUEUE = BASE / "capture-queue-v1"
RUN = BASE / "pipeline-v1"
PYTHON = RUNTIME / "envs/mergevla/bin/python"
ADMISSION_MIB = 48 * 1024
RUNTIME_FLOOR_MIB = 8 * 1024
GPUS = tuple(range(8))

sys.path[:0] = [str(HERE), str(ROOT / "vla-merge/experiments/claude-firstpass-cause-20260920")]
import card_flock  # noqa: E402
from audit_ab_bank import audit_all  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def gpu_rows() -> dict[int, dict]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used",
        "--format=csv,noheader,nounits"], text=True)
    result = {}
    for line in output.strip().splitlines():
        index, uuid, free, used = [part.strip() for part in line.split(",")]
        result[int(index)] = {"index": int(index), "uuid": uuid,
                              "free_mib": int(free), "used_mib": int(used)}
    return result


def code_hashes() -> dict[str, str]:
    names = ("audit_ab_bank.py", "audit_capture.py", "block_calibration.py",
             "build_two_pass.py", "capture_closed_loop.py", "check_real_prefix_parity.py",
             "freeze_bank.py", "freeze_block_plan.py", "native_trace.py",
             "replay_merged_prefix.py", "replay_native_trace.py",
             "run_capture_episode.py", "run_capture_queue.py", "run_tcr_pipeline.py")
    return {name: sha(HERE / name) for name in names}


def prepare() -> dict:
    if RUN.exists():
        raise FileExistsError(RUN)
    bank = json.loads((BASE / "calibration-bank-v1.json").read_text())
    blocks = json.loads((BASE / "block-plan-v1.json").read_text())
    queue = json.loads((QUEUE / "plan.json").read_text())
    if (bank.get("job_count") != 80 or blocks.get("blocks_per_pass") != 64 or
            blocks.get("active_linear_modules") != 613 or
            queue.get("max_workers") != 3 or queue.get("no_retry") is not True):
        raise ValueError("Frozen collector or TCR scope is incomplete")
    RUN.mkdir(parents=True)
    plan = {"schema": "fastwam_tcr_pipeline_v1", "host": socket.gethostname(),
            "bank_sha256": sha(BASE / "calibration-bank-v1.json"),
            "block_plan_sha256": sha(BASE / "block-plan-v1.json"),
            "capture_queue_plan_sha256": sha(QUEUE / "plan.json"),
            "source_code_sha256": code_hashes(),
            "capture_jobs": 80, "passes": ["A", "B"],
            "permitted_gpus": list(GPUS), "admission_min_free_mib": ADMISSION_MIB,
            "runtime_floor_mib": RUNTIME_FLOOR_MIB,
            "max_build_workers": 1, "no_retry": True,
            "success_evaluations": 0,
            "stages": ["wait_capture", "independent_ab_audit",
                       "real_expert_prefix_parity", "two_pass_build", "native_reload_gate"]}
    save(RUN / "plan.json", plan)
    save(RUN / "state.json", {"phase": "prepared", "active": None,
                              "stop_latch": False})
    return {"plan": str(RUN / "plan.json"), "stages": plan["stages"]}


def validate_frozen(plan: dict) -> None:
    if (plan.get("schema") != "fastwam_tcr_pipeline_v1" or
            plan.get("host") != socket.gethostname() or
            plan.get("bank_sha256") != sha(BASE / "calibration-bank-v1.json") or
            plan.get("block_plan_sha256") != sha(BASE / "block-plan-v1.json") or
            plan.get("capture_queue_plan_sha256") != sha(QUEUE / "plan.json") or
            plan.get("source_code_sha256") != code_hashes()):
        raise ValueError("Frozen Fast-WAM pipeline inputs or code changed")


def run() -> dict:
    plan = json.loads((RUN / "plan.json").read_text())
    validate_frozen(plan)
    with (RUN / "coordinator.lock").open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((RUN / "state.json").read_text())
        if state.get("phase") != "prepared":
            raise ValueError("Pipeline already started")
        stop_requested = False
        child = None

        def stop_handler(_signum, _frame):
            nonlocal stop_requested
            stop_requested = True

        def write_state(phase: str, **kw):
            save(RUN / "state.json", {
                "phase": phase, "supervisor_pid": os.getpid(),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "active": None if child is None else {"pid": child.pid, **kw},
                "stop_latch": stop_requested,
            })

        def wait_child(process: subprocess.Popen, gpu: int, phase: str) -> None:
            nonlocal child
            child = process
            while process.poll() is None:
                if stop_requested or gpu_rows()[gpu]["free_mib"] < RUNTIME_FLOOR_MIB:
                    os.killpg(process.pid, signal.SIGTERM)
                    stop = "operator signal" if stop_requested else "GPU runtime floor"
                    raise RuntimeError(f"{phase} stopped: {stop}")
                write_state(phase, gpu=gpu)
                time.sleep(30)
            child = None
            if process.returncode:
                raise RuntimeError(f"{phase} worker exited {process.returncode}")

        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)
        try:
            while True:
                if stop_requested:
                    raise RuntimeError("Pipeline received operator stop")
                capture_state = json.loads((QUEUE / "state.json").read_text())
                terminal = QUEUE / "ended.json"
                if terminal.exists():
                    ended = json.loads(terminal.read_text())
                    if ended.get("complete") is not True:
                        raise RuntimeError("Frozen capture queue ended incomplete; no build")
                    break
                if capture_state.get("stop_latch"):
                    raise RuntimeError("Frozen capture queue stopped; no build")
                write_state("waiting_capture", accepted=len(capture_state.get("accepted", [])))
                time.sleep(60)
            validate_frozen(plan)
            write_state("auditing_capture")
            acceptance = audit_all()
            acceptance_path = BASE / "calibration-acceptance-v1.json"
            if acceptance_path.exists():
                if json.loads(acceptance_path.read_text()) != acceptance:
                    raise ValueError("Existing A/B acceptance differs from independent audit")
            else:
                save(acceptance_path, acceptance)
            while True:
                if stop_requested:
                    raise RuntimeError("Pipeline received operator stop")
                validate_frozen(plan)
                rows = gpu_rows()
                selected = None
                for gpu in sorted(GPUS, key=lambda index: rows[index]["free_mib"], reverse=True):
                    row = rows[gpu]
                    if row["free_mib"] < ADMISSION_MIB:
                        continue
                    card = card_flock.take_card(gpu, row["uuid"],
                                               "fastwam-tcr-two-pass", "fastwam-solve")
                    if card is None:
                        continue
                    legacy = card_flock._try_lock(
                        RUNTIME / "resource-leases" / socket.gethostname() / f"gpu-{gpu}.lock",
                        "fastwam-tcr-two-pass", {"gpu": gpu})
                    if legacy is None:
                        card.release()
                        continue
                    slot = card_flock.take_build_slot("fastwam-tcr-two-pass")
                    if slot is None:
                        legacy.release()
                        card.release()
                        continue
                    again = gpu_rows()[gpu]
                    if again["uuid"] != row["uuid"] or again["free_mib"] < ADMISSION_MIB:
                        slot.release()
                        legacy.release()
                        card.release()
                        continue
                    selected = (gpu, row["uuid"], card, legacy, slot)
                    break
                if selected is not None:
                    break
                write_state("waiting_safe_gpu")
                time.sleep(60)
            gpu, uuid, card, legacy, slot = selected
            try:
                env = os.environ.copy()
                env.update({"CUDA_VISIBLE_DEVICES": str(gpu),
                            "FASTWAM_INHERITED_GPU_INDEX": str(gpu),
                            "FASTWAM_INHERITED_GPU_UUID": uuid,
                            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                            "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
                parity_path = BASE / "real-prefix-parity-v1.json"
                if parity_path.exists():
                    raise FileExistsError(parity_path)
                parity_command = [str(PYTHON), "-u", str(HERE / "check_real_prefix_parity.py"),
                                  "--gpu", str(gpu), "--min-free-mib", str(ADMISSION_MIB),
                                  "--output", str(parity_path), "--inherited-lease"]
                with (RUN / "parity.log").open("x") as log:
                    process = subprocess.Popen(parity_command, cwd=ROOT, env=env,
                                               stdin=subprocess.DEVNULL, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True,
                                               pass_fds=(card.handle, legacy.handle, slot.handle))
                    save(RUN / "parity-launch.json", {"command": parity_command,
                                                      "pid": process.pid, "gpu": gpu,
                                                      "uuid": uuid})
                    wait_child(process, gpu, "real_prefix_parity")
                parity = json.loads(parity_path.read_text())
                if parity.get("accepted") is not True or \
                        parity.get("native_selected_prediction_exact") is not True:
                    raise ValueError("Real Fast-WAM merged-prefix gate failed")
                validate_frozen(plan)
                build_output = BASE / "tcr-build-v1"
                if build_output.exists():
                    raise FileExistsError(build_output)
                build_command = [str(PYTHON), "-u", str(HERE / "build_two_pass.py"),
                                 "--acceptance", str(acceptance_path),
                                 "--block-plan", str(BASE / "block-plan-v1.json"),
                                 "--prefix-parity", str(parity_path),
                                 "--output", str(build_output)]
                with (RUN / "build.log").open("x") as log:
                    process = subprocess.Popen(build_command, cwd=ROOT, env=env,
                                               stdin=subprocess.DEVNULL, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True,
                                               pass_fds=(card.handle, legacy.handle, slot.handle))
                    save(RUN / "build-launch.json", {"command": build_command,
                                                     "pid": process.pid, "gpu": gpu,
                                                     "uuid": uuid})
                    wait_child(process, gpu, "two_pass_build")
                final = json.loads((build_output / "final-manifest.json").read_text())
                if final.get("complete") is not True or final.get("is_tcr") is not True or \
                        final.get("native_reload_action_exact") is not True or \
                        final.get("evaluation_episodes") != 0:
                    raise ValueError("Two-pass build manifest was not accepted")
            finally:
                slot.release()
                legacy.release()
                card.release()
            result = {"complete": True, "checkpoint": final["checkpoint"],
                      "native_reload_action_exact": True,
                      "success_evaluations": 0}
            save(RUN / "ended.json", result)
            write_state("complete")
            return result
        except BaseException as error:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            failure = {"complete": False, "exception": type(error).__name__,
                       "message": str(error), "no_retry": True}
            save(RUN / "ended.json", failure)
            write_state("failed")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    options = parser.parse_args()
    print(json.dumps(prepare() if options.stage == "prepare" else run(), sort_keys=True))
