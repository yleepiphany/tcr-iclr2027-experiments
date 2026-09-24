#!/usr/bin/env python3
"""Freeze and dynamically run the C146 held-out candidate action measurements."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
VLA = WORK / "vla-merge"
SOURCE = WORK / "pi05_lora_finetune_v2_20260826"
PYTHON = SOURCE / ".venv/bin/python"
BANK = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/heldout-raw-attempt-02/bank-index.json"
TEACHER_AUDIT = WORK / "coordination/2026-09-21/c146-teacher-actions-audit.json"
DEFAULT_RUN = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/candidate-actions-attempt-01"
BASE_PATH = VLA / "experiments/tcr-mainline-local-20260919/mainline_capture.py"
FLOCK_DIR = VLA / "experiments/claude-firstpass-cause-20260920"
CANDIDATE_GPUS = tuple(range(8))
MAX_ACTIVE = 4
SMOKE_ADMISSION_MIB = 32 * 1024
RUNTIME_FLOOR_MIB = 12 * 1024
POLICY_FILES = {
    "policy_preprocessor.json": "d46dc51bc8f13d37eec4f3d8b0488e05d31dfe334d3ad24f96e841640743cc3c",
    "policy_preprocessor_step_3_normalizer_processor.safetensors": "f98234b21ce8ffacf1b86657f3cf8e24ac6c31ebcd6a3b83cecd484439674306",
    "policy_postprocessor.json": "e6e0c0371c2ea20b0a2f99feab26cea73a7d8994eccc4446c038958f285d68ff",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": "a002c0df7f79c5b169c5a899ad151d4ea1bed246c7d82bd93ed1556558d517a9",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("main_fidelity_candidate_base", BASE_PATH)
sys.path.insert(0, str(FLOCK_DIR))
import card_flock  # noqa:E402


def sha(path: Path) -> str:
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


def tensor_hash(value) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_specs():
    table = WORK / "vla-merge-runtime/experiments/iclr2027-table1-20260910"
    tcr = WORK / "vla-merge-runtime/experiments/claude-tcr-new-methods-20260917/arms/checkpoints"
    demo = WORK / "vla-merge-runtime/experiments/claude-three-level-main-20260921/attempt-01/checkpoints"
    identities = WORK / "coordination/2026-09-20/final-ab-model-identities.json"
    demo_id = demo.parent / "DEMO-MODEL-IDENTITIES.json"
    common = {
        "model_soups": (table / "libero/model-soups/repeat-shared/merge/attempt-02-peft-safe-v2/pretrained_model", "a92aacc43146dc41663c0057f2999bd90252fb3fb316ef963f165be67e1be01a", table / "manifest-clean-fastlane-v2.json", "main"),
        "task_arithmetic": (table / "libero/ta-ties-fastlane-v1/candidates/task_arithmetic_alpha_0p35/pretrained_model", "311359cf6eb6f139e5acb1a0f87dd5f58f1d9ee7db354aff3426f7326ef7c448", table / "manifest-clean-formal-ta-ties-v1.json", "main"),
        "ties_merging": (table / "libero/ta-ties-fastlane-v2/candidates/ties_density_0p3_alpha_0p9/pretrained_model", "9382c79c9a76dea4803d71484599ec16edb77ee3cf7ce3b861fb9f0296921847", table / "manifest-clean-formal-ta-ties-v1.json", "main"),
        "wudi_merging": (table / "libero/wudi-plus/development-v2/candidates/wudi-rms-head-mean/pretrained_model", "3c4e9c35b771201a8a4211a29eff5a972e81f720d589cea176ea6187d5f97ea8", table / "manifest-clean-formal-wudiplus-rms-head-mean-v1.json", "main"),
        "knots_ties": (table / "libero/knots-ties/development-screen01/fixed-top20-scale0p6-v1/pretrained_model", "f32a3cd38122322b616b624588b0a0294f48cefc63883f6435e0ffde047abf69", table / "receipts/table1-knots-ties-select10-cumulative-accepted-20260914.json", "main"),
        "regmean": (table / "libero/original-regmean/repeat-shared/merge/attempt-07-active-support-v5-cpfs-publish-host1023-gpu1/pretrained_model", "00bed4399ce035014852b79aff00f9142a45463f474d0e2863962f3ccad613cf", table / "manifest-clean-fastlane-v4-regmean.json", "main"),
        "regmean_pp": (table / "libero/original-regmeanpp/repeat-01/merge/attempt-03-active-support-v2-host1023-gpu2", "38745ba812c2f4ac4327ab40719e0ce8da017873c0a1765608f65dd926cd4ab3", table / "manifest-clean-formal-regmeanpp-v1.json", "main"),
        "featcal": (table / "preflight/featcal-formal-full-v1/checkpoint/pretrained_model", "e43de109844431c02e316e57f701d7c06a9a3c8feaa3c41c9f391281f8197efa", table / "manifest-clean-fastlane-v3.json", "main"),
        "tcr_r01": (tcr / "r01", "4562193825dc9b23e834eb32bfc57242501c7abc3f6c561eeaeee1b5b6f42da8", identities, "full"),
        "tcr_r02": (tcr / "r02", "cc3185acd526f9fd70a0640a9932991e21b0a3f32e0ff994a4fb3fc5c250602c", identities, "full"),
        "tcr_r03": (tcr / "c_e", "3e3a55ce0acd04b3ee1e180eda2a6109b0ef6d0ea4e717f6052a27829adfce95", identities, "full"),
        "demo_r01": (demo / "demo-r01", "508bdb11f4c6c044c388950cd05c6dc35f62b0c5c9374b2ef13a485689cd49db", demo_id, "demo"),
        "demo_r02": (demo / "demo-r02", "76993b843d52e7128037d67421fe1b4c49f29c10caef3ed706089f766d9914b9", demo_id, "demo"),
        "demo_r03": (demo / "demo-r03", "b12790c4a0be4c25ccb68db272c5d6f4121e235a39496b5850bb140499f9af16", demo_id, "demo"),
    }
    return {name: {"path": str(path), "model_sha256": digest, "identity_source": str(proof), "group": group}
            for name, (path, digest, proof, group) in common.items()}


def prepare(run: Path):
    if run.exists():
        raise FileExistsError(run)
    bank = json.loads(BANK.read_text())
    teacher = json.loads(TEACHER_AUDIT.read_text())
    if bank.get("accepted") is not True or bank.get("request_count") != 400:
        raise ValueError("Raw bank is not accepted")
    if teacher.get("accepted") is not True or teacher.get("requests") != 400:
        raise ValueError("Teacher bank is not strictly accepted")
    models = model_specs()
    for name, item in models.items():
        path = Path(item["path"])
        proof = Path(item["identity_source"])
        if not proof.is_file():
            raise FileNotFoundError(f"{name}: missing identity source")
        item["identity_source_sha256"] = sha(proof)
        if sha(path / "model.safetensors") != item["model_sha256"]:
            raise ValueError(f"{name}: model hash differs")
        item["policy_files"] = {}
        for filename, expected in POLICY_FILES.items():
            actual = sha(path / filename)
            if actual != expected:
                raise ValueError(f"{name}: common processor/normalizer differs: {filename}")
            item["policy_files"][filename] = actual
        config = json.loads((path / "config.json").read_text())
        if config.get("num_inference_steps") != 10 or config.get("chunk_size") != 50:
            raise ValueError(f"{name}: native generation config differs")
    # Teacher experts use these identical processor hashes as well; the strict
    # teacher audit binds their model identities to every request.
    for row in bank["requests"]:
        path = Path(row["teacher_expert_path"])
        for filename, expected in POLICY_FILES.items():
            if sha(path / filename) != expected:
                raise ValueError(f"Teacher normalizer differs: {row['suite_short']}/{filename}")
    source = Path(__file__).resolve()
    plan = {"schema": "main_fidelity_candidate_actions_v1",
            "created_at": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
            "bank": str(BANK), "bank_sha256": sha(BANK),
            "teacher_audit": str(TEACHER_AUDIT), "teacher_audit_sha256": sha(TEACHER_AUDIT),
            "source": str(source), "source_sha256": sha(source), "models": models,
            "requests_per_model": 400, "candidate_gpus": list(CANDIDATE_GPUS),
            "max_active": MAX_ACTIVE, "one_worker_per_card": True, "no_retry": True,
            "smoke_admission_mib": SMOKE_ADMISSION_MIB, "runtime_floor_mib": RUNTIME_FLOOR_MIB,
            "normalizer_identity": POLICY_FILES, "coordinate_scale": [1.0] * 6,
            "coordinate_scale_reason": "all candidates and teacher experts have byte-identical native action normalizer and unnormalizer files",
            "continuous_dims": [0, 1, 2, 3, 4, 5], "execute_steps": 10,
            "cosine_eps": 1e-12, "native_steps": 10}
    run.mkdir(parents=True)
    write(run / "plan.json", plan)
    print(json.dumps({"prepared": str(run), "models": len(models), "requests_per_model": 400}))


def free_mib(gpu: int) -> int:
    return int(float(subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip()))


def gpu_rows():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, free, used, util = [value.strip() for value in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "free_mib": int(float(free)),
                     "used_mib": int(float(used)), "utilization_gpu_percent": int(float(util))})
    return sorted(rows, key=lambda row: (-row["free_mib"], row["index"]))


def environment(gpu: int):
    result = base.environment(gpu)
    result.update({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "OPENBLAS_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
                   "PALIGEMMA_TOKENIZER_PATH": str(SOURCE / "assets/paligemma-3b-pt-224-tokenizer"),
                   "PYTHONPATH": ":".join([str(WORK / "vla-merge-runtime/python-overlay"), str(SOURCE / "src"),
                       str(SOURCE / "lerobot/src"), str(VLA), str(VLA / "src")])})
    return result


def worker(run: Path, model_name: str, smoke: bool):
    physical = int(os.environ["ITERATION_PHYSICAL_GPU"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise ValueError("GPU visibility mismatch")
    plan = json.loads((run / "plan.json").read_text())
    if plan.get("source_sha256") != sha(Path(__file__).resolve()):
        raise ValueError("Candidate source changed after plan freeze")
    item = plan["models"][model_name]
    model_path = Path(item["path"])
    if sha(model_path / "model.safetensors") != item["model_sha256"]:
        raise ValueError("Candidate model changed")
    bank = json.loads(BANK.read_text())
    rows = sorted(bank["requests"], key=lambda row: row["request_id"])
    if smoke:
        rows = rows[:1]
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from scripts.run_pi05_featcal_execution_pilot import load_policy
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.35, 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    policy = load_policy(str(model_path))
    output = run / ("smoke" if smoke else "actions") / model_name
    output.mkdir(parents=True, exist_ok=False)
    opened, verified_files, actions, receipts = {}, set(), {}, []
    start = time.monotonic()
    try:
        for index, row in enumerate(rows):
            if free_mib(physical) < RUNTIME_FLOOR_MIB:
                raise RuntimeError("GPU dropped below runtime floor")
            tensor_path = Path(row["tensor_file"])
            if str(tensor_path) not in verified_files:
                if sha(tensor_path) != row["tensor_file_sha256"]:
                    raise ValueError("Raw tensor file changed")
                verified_files.add(str(tensor_path))
            handle = opened.get(str(tensor_path))
            if handle is None:
                handle = safe_open(str(tensor_path), framework="pt")
                opened[str(tensor_path)] = handle
            prefix = f"sample_{row['sample_index']:03d}."
            values = {key[len(prefix):]: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}
            if tensor_hash(values["x_t"]) != row["native_noise_sha256"]:
                raise ValueError("Native noise hash differs")
            batch = {key: value.to("cuda") for key, value in values.items() if key not in ("native_velocity", "time")}
            action = policy.model.sample_actions(
                images=[batch[f"image_{i}"] for i in range(3)],
                img_masks=[batch[f"image_mask_{i}"] for i in range(3)],
                tokens=batch["tokens"], masks=batch["masks"],
                states=batch.get("states"), state_masks=batch.get("state_masks"),
                noise=batch["x_t"].clone(), num_steps=10).detach().cpu().contiguous()
            if tuple(action.shape) != (1, 50, 32) or not torch.isfinite(action).all():
                raise ValueError("Candidate native action is invalid")
            if index == 0:
                repeated = policy.model.sample_actions(
                    images=[batch[f"image_{i}"] for i in range(3)],
                    img_masks=[batch[f"image_mask_{i}"] for i in range(3)],
                    tokens=batch["tokens"], masks=batch["masks"],
                    states=batch.get("states"), state_masks=batch.get("state_masks"),
                    noise=batch["x_t"].clone(), num_steps=10).detach().cpu().contiguous()
                if not torch.equal(action, repeated):
                    raise ValueError("Candidate native generation is not deterministic")
            actions[row["request_id"]] = action
            receipts.append({"request_id": row["request_id"], "raw_input_sha256": row["raw_input_sha256"],
                             "native_noise_sha256": row["native_noise_sha256"], "action_sha256": tensor_hash(action)})
            del batch, action
        tensor_path = output / "actions.safetensors"
        save_file(actions, tensor_path, metadata={"format": "pt"})
        manifest = {"schema": "main_fidelity_candidate_action_output_v1", "model_name": model_name,
                    "model_path": str(model_path), "model_sha256": item["model_sha256"], "smoke": smoke,
                    "requests": len(rows), "bank": str(BANK), "bank_sha256": sha(BANK), "native_steps": 10,
                    "determinism_first_request": True, "actions_file": str(tensor_path),
                    "actions_sha256": sha(tensor_path), "rows": receipts,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                    "elapsed_seconds": time.monotonic() - start}
        write(output / "manifest.json", manifest)
        print(json.dumps({key: manifest[key] for key in ("model_name", "smoke", "requests", "peak_reserved_gib", "elapsed_seconds")}), flush=True)
    finally:
        try:
            policy.to("cpu")
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def smoke_dispatch(run: Path):
    plan = json.loads((run / "plan.json").read_text())
    if (run / "SMOKE-PASSED.json").exists():
        raise FileExistsError("Smoke already has a terminal receipt")
    uuids = card_flock.gpu_uuids()
    lease = None
    for row in gpu_rows():
        if row["free_mib"] < SMOKE_ADMISSION_MIB or uuids.get(row["index"]) != row["uuid"]:
            continue
        lease = card_flock.take_card(row["index"], row["uuid"], "main-fidelity-candidate-smoke", "offline-native-forward")
        if lease is not None:
            gpu, snapshot = row["index"], row
            break
    if lease is None:
        raise RuntimeError("No admissible GPU for candidate smoke")
    try:
        model_name = "model_soups"
        command = [str(PYTHON), "-u", str(Path(__file__).resolve()), "--worker", "--model", model_name, "--run", str(run), "--smoke"]
        log = run / "smoke.log"
        with log.open("x") as handle:
            process = subprocess.run(command, cwd=VLA, env=environment(gpu), stdout=handle, stderr=subprocess.STDOUT)
        write(run / "smoke-exit.json", {"return_code": process.returncode, "gpu": gpu, "snapshot": snapshot})
        if process.returncode:
            raise RuntimeError("Candidate smoke failed")
        manifest = json.loads((run / "smoke" / model_name / "manifest.json").read_text())
        if manifest.get("requests") != 1 or manifest.get("determinism_first_request") is not True:
            raise ValueError("Candidate smoke audit failed")
        admission = max(SMOKE_ADMISSION_MIB, math.ceil((manifest["peak_reserved_gib"] + 12) * 1024))
        write(run / "SMOKE-PASSED.json", {"passed": True, "gpu": gpu,
              "manifest_sha256": sha(run / "smoke" / model_name / "manifest.json"),
              "measured_peak_reserved_gib": manifest["peak_reserved_gib"],
              "full_admission_mib": admission, "runtime_floor_mib": RUNTIME_FLOOR_MIB,
              "plan_sha256": sha(run / "plan.json")})
        print(json.dumps({"passed": True, "gpu": gpu, "full_admission_mib": admission}))
    finally:
        lease.release()


def full_dispatch(run: Path):
    plan = json.loads((run / "plan.json").read_text())
    smoke_path = run / "SMOKE-PASSED.json"
    if not smoke_path.is_file():
        raise ValueError("Candidate smoke has not passed")
    smoke = json.loads(smoke_path.read_text())
    if smoke.get("passed") is not True or smoke.get("plan_sha256") != sha(run / "plan.json"):
        raise ValueError("Candidate smoke is not bound to this plan")
    for marker in ("started.json", "queue-ended.json", "state.json"):
        if (run / marker).exists():
            raise FileExistsError(marker)
    admission = int(smoke["full_admission_mib"])
    pending = deque(plan["models"])
    states = {name: {"status": "pending"} for name in pending}
    active, uuids, failed = {}, card_flock.gpu_uuids(), False
    started, deadline = time.time(), time.time() + 8 * 3600
    write(run / "started.json", {"pid": os.getpid(), "kernel_start": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
          "started_at": datetime.now(timezone.utc).isoformat(), "admission_mib": admission})
    def checkpoint():
        write(run / "state.json", {"updated_at": datetime.now(timezone.utc).isoformat(), "states": states,
              "pending": list(pending), "active": [item["model"] for item in active.values()], "failed_latch": failed})
    try:
        while pending or active:
            if time.time() >= deadline:
                raise TimeoutError("Candidate action queue deadline reached")
            for pid, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                item["log_handle"].close()
                item["lease"].release()
                del active[pid]
                states[item["model"]].update(status="complete" if code == 0 else "failed",
                    return_code=code, ended_unix=time.time())
                write(run / "exits" / f"{item['model']}.json", {"model": item["model"], "gpu": item["gpu"],
                      "pid": pid, "return_code": code, "started_unix": item["started_unix"],
                      "ended_unix": time.time(), "log": str(item["log"])})
                if code:
                    failed = True
            if not failed:
                occupied = {item["gpu"] for item in active.values()}
                for row in gpu_rows():
                    if not pending or len(active) >= MAX_ACTIVE:
                        break
                    gpu = row["index"]
                    if gpu in occupied or row["free_mib"] < admission or uuids.get(gpu) != row["uuid"]:
                        continue
                    lease = card_flock.take_card(gpu, row["uuid"], "main-fidelity-candidate-actions", "offline-native-forward")
                    if lease is None:
                        continue
                    model = pending.popleft()
                    log = run / "logs" / f"{model}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    handle = log.open("x")
                    command = [str(PYTHON), "-u", str(Path(__file__).resolve()), "--worker", "--model", model, "--run", str(run)]
                    try:
                        process = subprocess.Popen(command, cwd=VLA, env=environment(gpu), stdout=handle,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                    except Exception:
                        handle.close(); lease.release(); pending.appendleft(model); raise
                    now = time.time()
                    active[process.pid] = {"process": process, "model": model, "gpu": gpu, "lease": lease,
                                           "log_handle": handle, "log": log, "started_unix": now}
                    occupied.add(gpu)
                    states[model] = {"status": "running", "gpu": gpu, "pid": process.pid,
                                     "started_unix": now, "admission_snapshot": row, "log": str(log)}
                    write(run / "launches" / f"{model}.json", {"model": model, "gpu": gpu, "pid": process.pid,
                          "started_unix": now, "snapshot": row, "command": command})
            checkpoint()
            if failed and not active:
                break
            time.sleep(5)
        status = "complete" if not failed and all(value.get("status") == "complete" for value in states.values()) else "failed"
        result = {"status": status, "complete": status == "complete", "states": states,
                  "elapsed_seconds": time.time() - started, "finished_at": datetime.now(timezone.utc).isoformat()}
        write(run / "queue-ended.json", result)
        print(json.dumps(result), flush=True)
        if status != "complete":
            raise RuntimeError("Candidate action queue did not complete")
    finally:
        for item in active.values():
            if item["process"].poll() is None:
                item["process"].terminate()
            item["log_handle"].close()
            item["lease"].release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke-dispatch", action="store_true")
    parser.add_argument("--full-dispatch", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args()
    if sum((args.prepare, args.smoke_dispatch, args.full_dispatch, args.worker)) != 1:
        parser.error("Choose exactly one mode")
    run = args.run.resolve()
    if args.prepare:
        prepare(run)
    elif args.smoke_dispatch:
        smoke_dispatch(run)
    elif args.full_dispatch:
        full_dispatch(run)
    elif args.model:
        worker(run, args.model, args.smoke)
    else:
        parser.error("Worker mode requires --model")


if __name__ == "__main__":
    main()
