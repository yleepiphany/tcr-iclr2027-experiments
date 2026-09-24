#!/usr/bin/env python3
"""Isolated RoboTwin TCR B recovery using the frozen per-replay row quota."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
RUNTIME = WORK / "vla-merge-runtime"
SOURCE_BASE = RUNTIME / "experiments/claude-robotwin-tcr-20260923"
BASE = RUNTIME / "experiments/robotwin-tcr-local-codex-20260924"
TRANSFER = BASE / "transfer-accept.json"
ORIGINAL_BUILD = SOURCE_BASE / "tcr-build-attempt-01/plan.json"
ORIGINAL_FORMAL = SOURCE_BASE / "tcr-formal-attempt-02/plan.json"
RUN = BASE / "tcr-build-attempt-04"
OLD_RUN = BASE / "tcr-build-attempt-02"
FAILED_RUN = BASE / "tcr-build-attempt-03"
MODELS = BASE / "tcr-checkpoints-v2"
AUDIT = WORK / "coordination/2026-09-23/robotwin-capture-combined-v3-final-audit.json"
PROTOCOL = SOURCE_BASE / "reset-preflight-attempt-03/candidate-protocol.json"
BANK = SOURCE_BASE / "dense-experts-v2/expert-dense-bank.json"
SOUP = SOURCE_BASE / "dense-soup-v1/COMPLETE.json"
SOURCE_CODE = WORK / "vla-merge/experiments/claude-robotwin-tcr-20260923"
SOLVER = HERE / "run_materializer_local_v4.py"
ORIGINAL_SOLVER = SOURCE_CODE / "run_materializer.py"
GENERIC = WORK / "vla-merge/experiments/claude-20260917/materialize_second_round_v3.py"
PYTHON = WORK / "pi05_lora_finetune_v2_20260826/.venv/bin/python"
MIN_FREE_MIB = 71_000
FLOOR_MIB = 8 * 1024
MAX_ACTIVE = 1
POLL_SECONDS = 60
GPUS = tuple(range(4, 8))
ROWS = {"A": 3_330_900, "B": 5_328_900}
SUPPORT_SHA = {
    "policy_preprocessor.json": "714723707403f6a5d1c18ccfa287c16c7b36e13fa1b6447a24431398a241e424",
    "policy_postprocessor.json": "9f2a2c28bd3d3ba779477b84586fec0efc98522bf4286f77ae24e1b167e42a39",
    "policy_preprocessor_step_3_normalizer_processor.safetensors": "6ce304a9f8704ed56ab9b56d5c478abd878c8a89bf4d665692b624862cd0d234",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": "b8ba9b84423b717d05abcbc2999e0b2ff89f8b1c82d2232a37d1f37c14253948",
}

sys.path.insert(0, str(SOURCE_CODE))
sys.path.insert(0, str(WORK / "vla-merge/experiments/claude-firstpass-cause-20260920"))
import card_flock  # noqa: E402
import robotwin_solver_contract as contract  # noqa: E402


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(32 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def record_board(job_id: str, row: dict) -> None:
    path = RUN / "board-samples" / f"{job_id}.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), **row},
                                    sort_keys=True) + "\n")
    except OSError:
        # A telemetry write must not leave an already spawned GPU child untracked.
        pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def checkpoint(repeat: int, which: str) -> Path:
    # The native RoboTwin evaluator resolves checkpoint/pretrained_model.
    # Materialize directly in that layout so the exact verified model is
    # loaded without an untracked copy or symlink at formal-evaluation time.
    return MODELS / f"r{repeat:02d}-pass{which}" / "pretrained_model"


def gpu_rows() -> dict[int, dict]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used",
        "--format=csv,noheader,nounits"], text=True)
    result = {}
    for line in output.strip().splitlines():
        gpu, uuid, free, used = [item.strip() for item in line.split(",")]
        result[int(gpu)] = {"gpu": int(gpu), "uuid": uuid,
                            "free_mib": int(free), "used_mib": int(used)}
    return result


def source_gate() -> tuple[dict, dict, dict, dict]:
    audit = read(AUDIT)
    protocol = read(PROTOCOL)
    bank = read(BANK)
    soup = read(SOUP)
    require(audit.get("accepted") is True and audit.get("scope") == "final"
            and audit.get("accepted_jobs") == 18 and audit.get("accepted_episodes") == 180
            and audit.get("accepted_requests") == 900 and audit.get("accepted_flow_rows") == 2700
            and audit.get("success_values_read") is False
            and audit.get("protocol_sha256") == sha(PROTOCOL),
            "Full original+recovery RoboTwin A/B capture is not independently accepted")
    require(len(protocol.get("jobs", [])) == 18 and len(audit["jobs"]) == 18,
            "Capture audit/protocol job partition differs")
    dense = contract.validate_dense_bank(BANK, verify_weights=True)
    require(set(dense) == set(contract.NAMES), "Dense expert identities differ")
    require(soup.get("status") == "passed_parameter_exact"
            and soup.get("model_sha256") == sha(Path(soup["path"]) / "model.safetensors"),
            "Exact three-expert Soup prior differs")
    reports = {item["id"]: item for item in audit["jobs"]}
    require(set(reports) == {j["id"] for j in protocol["jobs"]},
            "Capture audit does not cover exact source jobs")
    return audit, protocol, bank, soup


def transfer_gate() -> dict:
    receipt = read(TRANSFER)
    build_ids = {f"r{repeat:02d}-pass{phase}" for repeat in (1, 2, 3)
                 for phase in ("A", "B")}
    formal_ids = {f"r{repeat:02d}-{group}" for repeat in (1, 2, 3)
                  for group in ("coordination", "receptacle", "precision")}
    require(receipt.get("accepted") is True
            and receipt.get("release_id") == "ROBOTWIN-TRANSFER-201-RELEASE"
            and receipt.get("source_build_plan_sha256") == sha(ORIGINAL_BUILD)
            and receipt.get("source_formal_plan_sha256") == sha(ORIGINAL_FORMAL)
            and set(receipt.get("released_build_jobs", [])) == build_ids
            and set(receipt.get("released_formal_jobs", [])) == formal_ids,
            "RoboTwin transfer is not released and accepted for all local jobs")
    return receipt


def failure_gate() -> None:
    terminal = read(FAILED_RUN / "queue-ended.json")
    a = {f"r{repeat:02d}-passA" for repeat in (1, 2, 3)}
    b = {f"r{repeat:02d}-passB" for repeat in (1, 2, 3)}
    failed = set(terminal.get("failed", {}))
    pending = set(terminal.get("pending", []))
    require(terminal.get("complete") is False
            and set(terminal.get("accepted", {})) == a
            and not terminal.get("skipped") and failed | pending == b
            and not failed & pending and {"r01-passB", "r02-passB"} <= failed
            and (not pending or terminal.get("stopped") is True)
            and read(FAILED_RUN / "plan.json").get("recovered_A") == terminal["accepted"],
            "RoboTwin third build attempt did not fail only at pass-B row selection")
    for job in failed:
        receipt = read(FAILED_RUN / "exits" / f"{job}.json")
        log = (FAILED_RUN / "logs" / f"{job}.log").read_text(errors="replace")
        telemetry = read(FAILED_RUN / "telemetry" / f"{job}.json")
        require(receipt.get("returncode") == 1 and receipt.get("accepted") is False
                and receipt.get("error") == "child exit 1; resource_floor=False"
                and "Per-module row contract differs: model.paligemma_with_expert.paligemma.model.vision_tower" in log
                and telemetry.get("solver_completed") is False
                and telemetry.get("cuda_available") is True,
                f"{job}: third-attempt failure differs from B row quota mismatch")
    for job in pending:
        require(not (FAILED_RUN / "launches" / f"{job}.json").exists(),
                f"{job}: stopped third-attempt job was already launched")


def recovered_a() -> dict[str, dict]:
    prior = Path(read(SOUP)["path"])
    old_jobs = {job["id"]: job for job in read(OLD_RUN / "plan.json")["jobs"]}
    accepted = {}
    for repeat in (1, 2, 3):
        job_id = f"r{repeat:02d}-passA"
        launch = read(OLD_RUN / "launches" / f"{job_id}.json")
        require(launch.get("job") == job_id
                and launch.get("command") == old_jobs[job_id]["command"]
                and launch.get("gpu") in GPUS
                and launch.get("uuid") == launch.get("admission", {}).get("uuid")
                and launch["admission"]["free_mib"] >= MIN_FREE_MIB,
                f"{job_id}: original launch or GPU admission differs")
        output = checkpoint(repeat, "A")
        model = output / "model.safetensors"
        manifest = output / "block_regmeanpp_manifest.json"
        data = read(manifest)
        require(data.get("model_sha256") == sha(model)
                and len(data.get("modules") or {}) == 418
                and data.get("realized_row_total") == ROWS["A"]
                and data.get("modified_tensor_count") == 422
                and data.get("inputs", {}).get("prior_model") == str(prior),
                f"{job_id}: completed pass-A model differs")
        for filename, expected in SUPPORT_SHA.items():
            require(sha(output / filename) == expected,
                    f"{job_id}: retained Soup deployment support differs: {filename}")
        board = OLD_RUN / "board-samples" / f"{job_id}.jsonl"
        require(board.exists() and board.stat().st_size > 0,
                f"{job_id}: live board samples missing")
        accepted[job_id] = {"id": job_id, "repeat": repeat, "pass": "A",
                            "model_sha256": data["model_sha256"],
                            "manifest_sha256": sha(manifest), "modules": 418,
                            "rows": ROWS["A"], "peak_telemetry": "missing_in_attempt_02",
                            "source_launch_sha256": sha(OLD_RUN / "launches" / f"{job_id}.json"),
                            "source_exit_sha256": sha(OLD_RUN / "exits" / f"{job_id}.json"),
                            "source_board_samples_sha256": sha(board)}
    return accepted


def trace_path(report: dict) -> Path:
    return SOURCE_BASE / f"capture-v{2 if report['attempt'] == '02' else 3}/jobs" / report["id"]


def command_for(repeat: int, which: str, bank: dict, soup: dict,
                captures: dict[str, dict]) -> list[str]:
    prior = Path(soup["path"]) if which == "A" else checkpoint(repeat, "A")
    command = [str(PYTHON), "-u", str(SOLVER),
               f"--dense-expert-bank={BANK}",
               f"--base-model={bank['base']['path']}",
               f"--prior-model={prior}",
               f"--output={checkpoint(repeat, which)}",
               "--ridge-ratio=.05", "--ridge-scale=feature_energy",
               "--max-correction-ratio=3", f"--max-rows-per-sample={10 if which == 'A' else 16}",
               f"--expert-loss-normalization={'prior' if which == 'A' else 'none'}",
               "--replay-prefix=merged", "--allow-mixed-calibration-policies",
               "--allow-prior-calibration-mismatch", "--device=cuda"]
    if which == "B":
        command.append(f"--ablation-config={RUN / 'configs' / f'r{repeat:02d}-passB.json'}")
    original = {row["name"]: row["source_adapter"]["path"] for row in bank["experts"]}
    for name in contract.NAMES:
        trace = trace_path(captures[f"r{repeat:02d}-{which}-{name}"])
        command.extend((f"--expert={name}={original[name]}",
                        f"--calibration={name}={trace / 'replay.safetensors'}",
                        f"--manifest={name}={trace / 'replay.json'}"))
    return command


def prepare() -> dict:
    if RUN.exists():
        raise FileExistsError("RoboTwin TCR build attempt or output already exists")
    failure_gate()
    recovered = recovered_a()
    require(all(not checkpoint(repeat, "B").exists() for repeat in (1, 2, 3)),
            "A pass-B output already exists")
    transfer_gate()
    audit, protocol, bank, soup = source_gate()
    reports = {item["id"]: item for item in audit["jobs"]}
    bindings = {}
    for job_id, report in sorted(reports.items()):
        trace = trace_path(report)
        manifest = trace / "replay.json"
        tensor = trace / "replay.safetensors"
        require(sha(manifest) == report["manifest_sha256"]
                and sha(tensor) == report["tensor_sha256"],
                f"Capture content changed: {job_id}")
        bindings[job_id] = {"manifest": str(manifest), "manifest_sha256": report["manifest_sha256"],
                            "tensor": str(tensor), "tensor_sha256": report["tensor_sha256"],
                            "attempt": report["attempt"]}
    jobs = []
    for repeat in (1, 2, 3):
        for which in ("A", "B"):
            jobs.append({"id": f"r{repeat:02d}-pass{which}", "repeat": repeat, "pass": which,
                         "needs": [] if which == "A" else [f"r{repeat:02d}-passA"],
                         "output": str(checkpoint(repeat, which)),
                         "command": command_for(repeat, which, bank, soup, reports),
                         "expected_rows": ROWS[which], "module_count": 418,
                         "capture_jobs": [f"r{repeat:02d}-{which}-{name}" for name in contract.NAMES]})
    plan = {"schema": "robotwin_three_expert_two_pass_build_v1",
            "created_at": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
            "transfer_accept_sha256": sha(TRANSFER),
            "source_build_plan_sha256": sha(ORIGINAL_BUILD),
            "source_formal_plan_sha256": sha(ORIGINAL_FORMAL),
            "capture_audit_sha256": sha(AUDIT), "protocol_sha256": sha(PROTOCOL),
            "dense_bank_sha256": sha(BANK), "soup_complete_sha256": sha(SOUP),
            "failed_attempt_terminal_sha256": sha(FAILED_RUN / "queue-ended.json"),
            "recovered_A": recovered,
            "support_policy": "retain_exact_prior_soup_normalizer_and_deployment_support",
            "row_policy": "pass_B_16_rows_per_replay_state_matching_frozen_5328900_rows",
            "source_sha256": {str(path): sha(path) for path in (SOLVER, ORIGINAL_SOLVER, GENERIC,
                                SOURCE_CODE / "robotwin_solver_contract.py", Path(__file__))},
            "capture_bindings": bindings, "jobs": jobs,
            "max_active": MAX_ACTIVE, "min_free_mib": MIN_FREE_MIB,
            "runtime_floor_mib": FLOOR_MIB, "permitted_gpus": list(GPUS),
            "no_retry": True, "training": False, "evaluation_episodes": 0,
            "paper_scope": "three independent repeat A/B builds; full 418-module RoboTwin M=3 TCR"}
    RUN.mkdir(parents=True, exist_ok=False)
    save(RUN / "plan.json", plan)
    return {"jobs": len(jobs), "source_capture_jobs": len(bindings),
            "plan": str(RUN / "plan.json")}


def validate_plan(plan: dict) -> None:
    require(plan.get("schema") == "robotwin_three_expert_two_pass_build_v1"
            and plan.get("host") == socket.gethostname()
            and len(plan.get("jobs", [])) == 6
            and plan.get("capture_audit_sha256") == sha(AUDIT)
            and plan.get("protocol_sha256") == sha(PROTOCOL)
            and plan.get("dense_bank_sha256") == sha(BANK)
            and plan.get("soup_complete_sha256") == sha(SOUP),
            "Frozen RoboTwin TCR build plan differs")
    failure_gate()
    require(plan.get("recovered_A") == recovered_a(),
            "Completed pass-A artifacts changed after recovery plan freeze")
    require(plan.get("failed_attempt_terminal_sha256") == sha(FAILED_RUN / "queue-ended.json")
            and plan.get("support_policy") == "retain_exact_prior_soup_normalizer_and_deployment_support"
            and plan.get("row_policy") == "pass_B_16_rows_per_replay_state_matching_frozen_5328900_rows",
            "RoboTwin support recovery provenance differs")
    transfer_gate()
    require(plan.get("transfer_accept_sha256") == sha(TRANSFER)
            and plan.get("source_build_plan_sha256") == sha(ORIGINAL_BUILD)
            and plan.get("source_formal_plan_sha256") == sha(ORIGINAL_FORMAL),
            "RoboTwin transfer provenance changed")
    for path, expected in plan["source_sha256"].items():
        require(sha(Path(path)) == expected, f"Build source changed: {path}")
    for job_id, binding in plan["capture_bindings"].items():
        require(sha(Path(binding["manifest"])) == binding["manifest_sha256"]
                and sha(Path(binding["tensor"])) == binding["tensor_sha256"],
                f"Frozen capture changed: {job_id}")


def make_b_config(job: dict, accepted: dict) -> Path:
    repeat = job["repeat"]
    parent = accepted[f"r{repeat:02d}-passA"]
    start = checkpoint(repeat, "A")
    manifest = start / "block_regmeanpp_manifest.json"
    config = {"schema": "claude_second_round_v1", "arm": "robotwin_b",
              "repeat": repeat, "row_cap_per_request_module": 16,
              "expert_masses": "uniform_three", "merged_slots": [],
              "start_point": str(start), "start_point_manifest": str(manifest),
              "start_point_sha256": parent["model_sha256"]}
    contract.validate_second_round_config(config)
    path = RUN / "configs" / f"r{repeat:02d}-passB.json"
    if path.exists():
        raise FileExistsError(path)
    save(path, config)
    return path


def verify(job: dict, accepted: dict) -> dict:
    output = Path(job["output"])
    model = output / "model.safetensors"
    manifest = output / "block_regmeanpp_manifest.json"
    data = read(manifest)
    require(data.get("model_sha256") == sha(model)
            and len(data.get("modules") or {}) == 418
            and data.get("realized_row_total") == job["expected_rows"]
            and data.get("modified_tensor_count") == 422,
            f"{job['id']}: model digest or full-scope row budget differs")
    require(data.get("inputs", {}).get("prior_model") == (
        str(Path(read(SOUP)["path"])) if job["pass"] == "A"
        else str(checkpoint(job["repeat"], "A"))),
        f"{job['id']}: immutable prior differs")
    if job["pass"] == "B":
        config = RUN / "configs" / f"r{job['repeat']:02d}-passB.json"
        contract.validate_second_round_config(read(config))
        require(data.get("ablation_config_sha256") == sha(config),
                f"{job['id']}: pass-B config differs")
        prior = read(checkpoint(job["repeat"], "A") / "block_regmeanpp_manifest.json")
        require(all(data["modules"][name]["ridge"] == row["ridge"]
                    for name, row in prior["modules"].items()),
                f"{job['id']}: pass B did not reuse its own A numeric ridge")
    telemetry_path = RUN / "telemetry" / f"{job['id']}.json"
    telemetry = read(telemetry_path)
    board_path = RUN / "board-samples" / f"{job['id']}.jsonl"
    require(telemetry.get("solver_completed") is True
            and telemetry.get("cuda_available") is True
            and telemetry.get("peak_reserved_mib", 0) > 0
            and board_path.exists() and board_path.stat().st_size > 0,
            f"{job['id']}: CUDA peak telemetry missing")
    return {"id": job["id"], "model_sha256": data["model_sha256"],
            "manifest_sha256": sha(manifest), "modules": 418,
            "rows": job["expected_rows"], "repeat": job["repeat"], "pass": job["pass"],
            "telemetry_sha256": sha(telemetry_path),
            "board_samples_sha256": sha(board_path),
            "peak_allocated_mib": telemetry["peak_allocated_mib"],
            "peak_reserved_mib": telemetry["peak_reserved_mib"]}


def run() -> None:
    plan = read(RUN / "plan.json")
    if (RUN / "started.json").exists():
        raise ValueError("No restart within RoboTwin build attempt")
    validate_plan(plan)
    save(RUN / "started.json", {"pid": os.getpid(), "host": socket.gethostname(),
         "kernel_starttime": int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]),
         "started_at": datetime.now(timezone.utc).isoformat()})
    pending = {job["id"]: job for job in plan["jobs"] if job["pass"] == "B"}
    accepted: dict[str, dict] = dict(plan["recovered_A"])
    failed: dict[str, str] = {}
    skipped: dict[str, str] = {}
    active: dict[str, dict] = {}
    stopping = False

    def on_signal(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    while pending or active:
        if stopping:
            for item in active.values():
                if item["process"].poll() is None:
                    os.killpg(item["process"].pid, signal.SIGTERM)
        for job_id, item in list(active.items()):
            child = item["process"]
            code = child.poll()
            board = gpu_rows()[item["gpu"]]
            record_board(job_id, board)
            if code is None and not stopping and board["free_mib"] < FLOOR_MIB:
                os.killpg(child.pid, signal.SIGTERM)
                item["resource_floor"] = True
            if code is None:
                continue
            child.wait()
            item["log"].close()
            item["slot"].release(); item["legacy"].release(); item["card"].release()
            active.pop(job_id)
            try:
                result = verify(item["job"], accepted) if code == 0 and not item.get("resource_floor") else None
                error = None if result else f"child exit {code}; resource_floor={item.get('resource_floor', False)}"
            except Exception as exc:
                result, error = None, f"artifact gate: {type(exc).__name__}: {exc}"
            save(RUN / "exits" / f"{job_id}.json", {"job": job_id,
                 "returncode": code, "accepted": result is not None,
                 "artifact": result, "error": error,
                 "finished_at": datetime.now(timezone.utc).isoformat()})
            if result:
                accepted[job_id] = result
            else:
                failed[job_id] = error
        for job_id, job in list(pending.items()):
            if any(dep in failed or dep in skipped for dep in job["needs"]):
                skipped[job_id] = "Matching pass A failed"
                pending.pop(job_id)
        if not stopping:
            ready = [job for job in pending.values() if all(dep in accepted for dep in job["needs"])]
            if ready and len(active) < MAX_ACTIVE:
                rows = gpu_rows()
                for gpu in sorted(GPUS, key=lambda index: rows[index]["free_mib"], reverse=True):
                    if not ready or len(active) >= MAX_ACTIVE:
                        break
                    row = rows[gpu]
                    if (gpu in {item["gpu"] for item in active.values()}
                            or row["free_mib"] < MIN_FREE_MIB):
                        continue
                    job = ready[0]
                    card = card_flock.take_card(gpu, row["uuid"], job["id"], "robotwin-tcr-build")
                    if card is None:
                        continue
                    legacy = card_flock._try_lock(
                        RUNTIME / "resource-leases" / socket.gethostname() / f"gpu-{gpu}.lock",
                        job["id"], {"gpu": gpu, "job": job["id"], "stage": "robotwin-tcr-build"})
                    if legacy is None:
                        card.release()
                        continue
                    slot = card_flock.take_build_slot(job["id"])
                    again = gpu_rows()[gpu]
                    if slot is None or again["uuid"] != row["uuid"] or again["free_mib"] < MIN_FREE_MIB:
                        if slot:
                            slot.release()
                        legacy.release(); card.release()
                        continue
                    try:
                        if job["pass"] == "B":
                            make_b_config(job, accepted)
                        require(not Path(job["output"]).exists(), "Output already exists")
                        log_path = RUN / "logs" / f"{job['id']}.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log = log_path.open("x")
                        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                               "ROBOTWIN_TCR_PASS": job["pass"], "PYTHONUNBUFFERED": "1",
                               "ROBOTWIN_TCR_PEAK_PATH": str(RUN / "telemetry" / f"{job['id']}.json"),
                               "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2",
                               "MKL_NUM_THREADS": "2"}
                        child = subprocess.Popen(job["command"], cwd=WORK, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True, env=env,
                                                 pass_fds=(card.handle, legacy.handle, slot.handle))
                        save(RUN / "launches" / f"{job['id']}.json", {"job": job["id"],
                             "gpu": gpu, "uuid": row["uuid"], "pid": child.pid,
                             "command": job["command"], "admission": again,
                             "pass": job["pass"], "started_at": datetime.now(timezone.utc).isoformat()})
                        record_board(job["id"], again)
                        active[job["id"]] = {"job": job, "gpu": gpu, "card": card,
                                              "legacy": legacy, "slot": slot,
                                              "process": child, "log": log}
                        pending.pop(job["id"])
                        ready.pop(0)
                    except Exception as exc:
                        slot.release(); legacy.release(); card.release()
                        failed[job["id"]] = f"prelaunch: {type(exc).__name__}: {exc}"
                        pending.pop(job["id"])
        save(RUN / "state.json", {"accepted": accepted, "failed": failed,
             "skipped": skipped, "active": {key: {"gpu": item["gpu"], "pid": item["process"].pid}
                                                for key, item in active.items()},
             "pending": list(pending), "stopped": stopping,
             "updated_at": datetime.now(timezone.utc).isoformat()})
        if stopping and not active:
            break
        if pending or active:
            time.sleep(POLL_SECONDS)
    save(RUN / "queue-ended.json", {"complete": len(accepted) == 6 and not failed and not skipped,
         "accepted": accepted, "failed": failed, "skipped": skipped,
         "pending": list(pending), "stopped": stopping,
         "ended_at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    action = parser.parse_args().action
    print(json.dumps(prepare() if action == "prepare" else run(), sort_keys=True))
