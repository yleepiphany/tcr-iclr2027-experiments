"""Four-GPU bounded queue: unified formal evaluation, fresh builds, fixed dev tests.

Never overwrites old experiments. External termination cancels this queue, without
automatic restart. Only this supervisor's private child process groups are stopped.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback

WORK = Path(__file__).resolve().parents[2]
ROOT = WORK / "vla-merge"
SCRIPTS = ROOT / "scripts"
SOURCE = WORK / "pi05_lora_finetune_v2_20260826"
PYTHON = SOURCE / ".venv/bin/python"
EXP = WORK / "vla-merge-runtime/experiments"
TABLE = EXP / "iclr2027-table1-20260910"
CLAUDE = EXP / "claude-tcr-new-methods-20260917"
DATA = WORK / ".datasets/LIBERO/20260919"
DEFAULT = EXP / "tcr-unified-night-20260919"
NAMES = ("spatial", "object", "goal", "long")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
GPUS = (4, 5, 6, 7)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def identity(path, expected=None):
    p = Path(path)
    value = sha(p)
    if expected is not None and value != expected:
        raise ValueError(f"Hash mismatch: {p}")
    s = p.stat()
    return {"sha256": value, "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def snapshot():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
                                   "--format=csv,noheader,nounits"], text=True)
    return {int(x.split(',')[0]): tuple(map(int, x.split(',')[1:])) for x in raw.strip().splitlines()}


def environment(gpu, mode):
    from run_featcal_execution_pilot_lane_v2 import environment as native_env
    e = native_env(gpu)
    e.update(LIBERO_CONFIG_PATH=str(DATA / "config-standard"), HF_HUB_OFFLINE="1",
             ITERATION_PHYSICAL_GPU=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
             OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1", CLAUDE_EVAL_DUTY="0",
             PYTHONDONTWRITEBYTECODE="1")
    for key in ("LIBERO_PRO_REPO", "LIBERO_PRO_ASSET_DIR", "MUJOCO_EGL_DEVICE_ID",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        e.pop(key, None)
    if mode == "development":
        e["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    return e


def native_signature(folder):
    p = Path(folder)
    d = read(p / "config.json")
    keys = ("chunk_size", "max_action_dim", "num_inference_steps", "use_amp", "dtype",
            "rtc_config", "rtc_training_max_delay")
    norms = {x.name: sha(x) for x in p.glob("*normalizer*.safetensors")}
    if len(norms) != 2:
        raise ValueError(f"Incomplete normalizers: {p}")
    return {"config": {k: d[k] for k in keys}, "normalizers": norms}


def prepare(run):
    from pi05_tcr_e_dense_contract import validate_dense_bank
    import run_tcr_half_step_development as half
    import resume_tcr_half_step_development as recovery
    manifest_path = TABLE / "manifest-clean-fastlane-v3.json"
    m = read(manifest_path)
    frozen, models, jobs = {}, {}, []
    def freeze(p, expected=None):
        key = str(Path(p))
        if key not in frozen:
            frozen[key] = identity(p, expected)
        elif expected and frozen[key]["sha256"] != expected:
            raise ValueError("Conflicting identity requirements")
    bankpath = TABLE / "expert-dense-bank-peft-v2.json"
    bank = validate_dense_bank(bankpath, verify_weights=False)
    freeze(bankpath, m["expert_bank"]["dense_index_sha256"])
    for v in bank.values():
        freeze(Path(v["path"]) / "model.safetensors", v["model_sha256"])
    for p in [manifest_path, Path(__file__), SCRIPTS / "tcr_night_worker_20260919.py",
              SCRIPTS / "materialize_pi05_tcr_e_peft_safe.py", SCRIPTS / "materialize_tcr_night_uniform_20260919.py",
              SCRIPTS / "collect_pi05_tcr_e_dense_trace_v2.py", SCRIPTS / "pi05_tcr_e_dense_contract.py",
              SCRIPTS / "collect_pi05_block_regmeanpp_calibration.py", SCRIPTS / "materialize_pi05_block_regmeanpp.py",
              SCRIPTS / "run_iclr2027_table1_clean_fastlane.py", SCRIPTS / "eval_pi05_policy_with_procedural_bank.py",
              SCRIPTS / "eval_pi05_libero_with_init_offset.py", SCRIPTS / "audit_tcr_expanded_pairing.py",
              ROOT / "src/vla_merge/libero_procedural_bank.py",
              ROOT / "experiments/claude-20260917/eval_pi05_expanded_development.py",
              SOURCE / "src/eval_with_local_tokenizer.py", SOURCE / "src/runtime_overrides.py",
              DATA / "config-standard/config.yaml", DATA / "runtime-files.json"]:
        freeze(p)
    # Restore integrity check on every locally snapshotted environment asset.
    for row in read(DATA / "runtime-files.json"):
        freeze(row["path"], row["sha256"])
    if not read(DATA / "verify-standard.json")["passed"]:
        raise ValueError("Standard environment validation has not passed")
    recipe = read(TABLE / "libero/tcr-e/repeat-01/merge/attempt-01-peft-safe-v1/block_regmeanpp_manifest.json")
    for k in ("base_model", "prior_model"):
        freeze(Path(recipe["inputs"][k]) / "model.safetensors")
    for n, p in recipe["experts"].items():
        if "/checkpoints/010000/" not in p:
            raise ValueError("Not the uniform 10k expert bank")
        freeze(Path(p) / "adapter_model.safetensors", recipe["inputs"]["adapter_sha256"][n])
    baseline_signature = native_signature(next(j["checkpoint"]["path"] for j in m["jobs"] if j["method"] == "featcal"))
    baseline = {}
    for rep in range(1, 4):
        rn = f"repeat-{rep:02d}"
        for suite in SUITES:
            b = TABLE / "libero/clean-formal-fastlane-v3/featcal" / rn / suite / "attempt-01"
            rec = read(b / "run_receipt.json")
            if rec["status"] != "completed":
                raise ValueError("FeatCal formal receipt incomplete")
            freeze(b / "run_receipt.json")
            for artifact in ("eval_info", "procedural_bank_receipt", "episode_receipts"):
                freeze(rec["artifacts"][artifact]["path"], rec["artifacts"][artifact]["sha256"])
            for name in ("evaluator_sha256",):
                if rec["identities"][name] != sha(SCRIPTS / "eval_pi05_policy_with_procedural_bank.py"):
                    raise ValueError("Formal evaluator changed since FeatCal")
            j = next(j for j in m["jobs"] if j["method"] == "featcal" and j["repeat"] == rn and j["suite"] == suite)
            freeze(j["selection"]["path"], j["selection"]["sha256"])
            baseline[f"{rn}/{suite}"] = {"command": rec["command"], "reset": str(b / "eval/procedural_bank_receipt.json"),
                                          "eval_info": str(b / "eval/eval_info.json"), "selection": j["selection"]}
    # Three existing expert-only second-pass models; parent repeats are preserved.
    for rep, arm in ((1, "r01"), (2, "r02"), (3, "c_e")):
        key = f"expert-pass2-r{rep}"
        folder = CLAUDE / "arms/checkpoints" / arm
        mm = read(folder / "block_regmeanpp_manifest.json")
        if mm["dense_expert_bank_sha256"] != sha(bankpath) or mm["realized_row_total"] != 1331600:
            raise ValueError("C-E bank/budget differs")
        if native_signature(folder) != baseline_signature:
            raise ValueError("C-E native settings differ from FeatCal")
        freeze(folder / "model.safetensors", mm["model_sha256"])
        freeze(folder / "block_regmeanpp_manifest.json")
        models[key] = {"path": str(folder), "sha256": mm["model_sha256"], "repeat": rep,
                       "label": "existing expert-only second pass, not original TCR-E",
                       "cumulative_regression_rows": 5772800}
        for suite in SUITES:
            jobs.append({"id": f"formal-{key}-{suite}", "kind": "formal", "model": key,
                         "repeat": rep, "suite": suite, "deps": [], "priority": 2})
    for rep in range(1, 4):
        deps = []
        for name in NAMES:
            key = f"collect-r{rep}-{name}"
            deps.append(key)
            jobs.append({"id": key, "kind": "collect", "repeat": rep, "expert": name,
                         "deps": [], "priority": 0 if rep == 1 else 3})
        key = f"tcre-r{rep}"
        models[key] = {"path": str(run / "models" / key), "repeat": rep,
                       "label": "newly collected/reconstructed TCR-E; not byte-identical recovery"}
        jobs.append({"id": f"solve-{key}", "kind": "solve", "model": key, "repeat": rep,
                     "deps": deps, "priority": 1})
        for suite in SUITES:
            jobs.append({"id": f"formal-{key}-{suite}", "kind": "formal", "model": key,
                         "repeat": rep, "suite": suite, "deps": [f"solve-{key}"], "priority": 2})
    # Fixed one-factor iteration, chosen before any new scores are seen.
    models["uniform-r1"] = {"path": str(run / "models/uniform-r1"), "repeat": 1, "development_only": True}
    jobs.append({"id": "solve-uniform-r1", "kind": "uniform", "model": "uniform-r1", "repeat": 1,
                 "deps": ["solve-tcre-r1"], "priority": 4})
    for offset in range(30, 40):
        for key in ("tcre-r1", "uniform-r1"):
            jobs.append({"id": f"dev-{key}-{offset}", "kind": "development", "model": key,
                         "offset": offset, "deps": [f"solve-{key}"], "priority": 5})
    # Complete existing half-step data only after frozen provenance validation.
    half_plan = read(half.DEFAULT / "plan.json")
    half.check_dependencies(half_plan)
    half_models = read(half.DEFAULT / "build-complete.json")["models"]
    reused = []
    for arm, item in half_models.items():
        folder = Path(item["path"])
        if read(folder / "half_step_manifest.json") != item:
            raise ValueError("Half-step manifest changed")
        freeze(folder / "model.safetensors", item["model_sha256"])
        models[arm] = {"path": str(folder), "sha256": item["model_sha256"], "development_only": True}
        for offset in range(30, 40):
            found = None
            for r in (EXP / "tcr-half-eval-recovery2-20260918", recovery.DEFAULT, half.DEFAULT):
                p = r / "development" / arm / f"offset-{offset}"
                receipt = recovery.reusable(p, offset, half_plan, item)
                if receipt:
                    found = {"model": arm, "offset": offset, "path": str(p), "verified": receipt}
                    break
            if found:
                reused.append(found)
            else:
                jobs.append({"id": f"dev-{arm}-{offset}", "kind": "development", "model": arm,
                             "offset": offset, "deps": [], "priority": 6})
    result = {"schema": "tcr_unified_night_v1", "created_unix": time.time(), "host": socket.gethostname(),
              "pid": os.getpid(), "gpus": list(GPUS), "protocol": m["protocol"], "baseline": baseline,
              "native_signature": baseline_signature, "bank": bank, "bank_path": str(bankpath),
              "recipe": recipe, "models": models, "jobs": jobs, "frozen": frozen,
              "reused_half": reused, "formal_episodes": 2400,
              "new_development_episodes": 40 * sum(j["kind"] == "development" for j in jobs),
              "new_checkpoints": 4, "no_auto_retry": True, "cancel_all_on_external_kill": True,
              "no_throttle_sleep": True, "independent_confirmation_access": False,
              "formal_scores_not_used_for_iteration_selection": True,
              "old_raw_traces_missing": True, "old_models_not_overwritten": True}
    write(run / "plan.json", result)
    return result


class Queue:
    def __init__(self, run, plan):
        self.run, self.plan = run, plan
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.states = {j["id"]: "pending" for j in plan["jobs"]}
        self.children = {}
        self.model_hashes = {k:v["sha256"] for k,v in plan["models"].items() if "sha256" in v}
        self.deadline = time.time() + 12 * 3600

    def cancel(self, signum=signal.SIGTERM, frame=None):
        self.stop.set()
        with self.lock:
            for p in self.children.values():
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def next_job(self):
        with self.lock:
            for j in self.plan["jobs"]:
                if self.states[j["id"]] == "pending" and any(self.states[d] in ("failed", "blocked", "cancelled") for d in j["deps"]):
                    self.states[j["id"]] = "blocked"
            ready = [j for j in self.plan["jobs"] if self.states[j["id"]] == "pending"
                     and all(self.states[d] == "done" for d in j["deps"])]
            if ready:
                j = min(ready, key=lambda x: (x["priority"], self.plan["jobs"].index(x)))
                self.states[j["id"]] = "running"
                return j
            return None

    def check_sources(self):
        for path, info in self.plan["frozen"].items():
            s = Path(path).stat()
            if (s.st_size, s.st_mtime_ns) != (info["size"], info["mtime_ns"]):
                raise ValueError(f"Frozen input changed: {path}")

    def configure(self, j, gpu, folder):
        kind = j["kind"]
        env = environment(gpu, kind)
        args = []
        if kind == "collect":
            rep, name = j["repeat"], j["expert"]
            old = read(TABLE / f"libero/tcr-e/repeat-{rep:02d}/execution-traces-v2/{name}/receipt.json")
            out = self.run / "traces" / f"repeat-{rep:02d}" / name
            out.mkdir(parents=True, exist_ok=False)
            args = [f"--output_dir={out / 'rollout'}" if x.startswith("--output_dir=") else x for x in old["command"][2:]]
            env.update({k:v for k,v in old["environment"].items() if k.startswith(("PI05_BLOCK_", "TCR_E_", "PI05_LIBERO_INIT"))})
            env.update(PI05_BLOCK_REGMEANPP_TENSOR_OUTPUT=str(out / "replay.safetensors"),
                       PI05_BLOCK_REGMEANPP_MANIFEST_OUTPUT=str(out / "replay.json"))
            return args, env, out
        model = self.plan["models"][j["model"]]["path"]
        if kind in ("solve", "uniform"):
            r = self.plan["recipe"]
            args = [f"--dense-expert-bank={self.plan['bank_path']}", f"--base-model={r['inputs']['base_model']}",
                    f"--prior-model={r['inputs']['prior_model']}", f"--output={model}", "--ridge-ratio=.05",
                    "--ridge-scale=feature_energy", "--max-correction-ratio=3", "--max-rows-per-sample=10",
                    "--expert-loss-normalization=" + ("none" if kind == "uniform" else "prior"),
                    "--replay-prefix=merged", "--device=cuda", "--allow-mixed-calibration-policies",
                    "--allow-prior-calibration-mismatch"]
            for n in NAMES:
                d = self.run / "traces" / f"repeat-{j['repeat']:02d}" / n
                receipt = read(self.run / "jobs" / f"collect-r{j['repeat']}-{n}" / "verified.json")
                if sha(d / "replay.safetensors") != receipt["trace_sha256"]:
                    raise ValueError("Collected trace changed")
                args += [f"--expert={n}={r['experts'][n]}", f"--calibration={n}={d / 'replay.safetensors'}",
                         f"--manifest={n}={d / 'replay.json'}"]
            if kind == "uniform":
                args.append(f"--night-reference={self.run / 'models/tcre-r1/block_regmeanpp_manifest.json'}")
            return args, env, Path(model)
        if sha(Path(model) / "model.safetensors") != self.model_hashes[j["model"]]:
            raise ValueError("Evaluation model changed")
        out = folder / "eval"
        if kind == "formal":
            b = self.plan["baseline"][f"repeat-{j['repeat']:02d}/{j['suite']}"]
            args = [f"--output_dir={out}" if x.startswith("--output_dir=") else
                    f"--policy.path={model}" if x.startswith("--policy.path=") else x for x in b["command"][2:]]
            env.update(PI05_LIBERO_SUITE=j["suite"], PI05_ALLOW_GPU_SHARING="1")
        else:
            offset = j["offset"]
            args = [f"--output_dir={out}", "--env.type=libero", "--env.task=" + ",".join(SUITES),
                    "--env.task_ids=[0,1,2,3,4,5,6,7,8,9]", "--env.max_parallel_tasks=1",
                    "--eval.batch_size=1", "--eval.n_episodes=1", f"--seed={391600+1000*(offset-30)}",
                    f"--policy.path={model}", "--policy.device=cuda", "--policy.compile_model=false",
                    "--policy.gradient_checkpointing=false", "--policy.n_action_steps=10"]
            env.update(PI05_LIBERO_INIT_STATE_OFFSET=str(offset), PI05_LIBERO_INIT_STATE_COUNT="1",
                       ITERATION_NOISE_RECEIPT=str(out / "paired-noise.jsonl"))
        return args, env, out

    def verify(self, j, folder, out):
        from pi05_tcr_e_dense_contract import validate_trace, validate_realized_rows
        kind = j["kind"]
        if kind == "collect":
            m = read(out / "replay.json")
            validate_trace(m, self.plan["bank"][j["expert"]]["path"], j["expert"])
            return {"trace_sha256": sha(out / "replay.safetensors"), "manifest_sha256": sha(out / "replay.json"),
                    "new_acquisition": True, "sample_count": m["sample_count"]}
        if kind in ("solve", "uniform"):
            m = read(out / "block_regmeanpp_manifest.json")
            if validate_realized_rows(m["modules"]) != 4441200 or m["modified_tensor_count"] != 422:
                raise ValueError("Row/module scope differs")
            digest = sha(out / "model.safetensors")
            if digest != m["model_sha256"] or native_signature(out) != self.plan["native_signature"]:
                raise ValueError("Model/hash/native configuration verification failed")
            if m["dense_expert_bank_sha256"] != sha(self.plan["bank_path"]):
                raise ValueError("Solver bank identity differs")
            if kind == "uniform":
                ref = read(self.run / "models/tcre-r1/block_regmeanpp_manifest.json")
                if any(v["ridge"] != ref["modules"][k]["ridge"] for k,v in m["modules"].items()):
                    raise ValueError("Uniform frozen ridge mismatch")
                if any(any(abs(w-.25)>1e-12 for w in v["expert_objective_weights"].values()) for v in m["modules"].values()):
                    raise ValueError("Uniform mass mismatch")
            with self.lock:
                self.model_hashes[j["model"]] = digest
            return {"model_sha256": digest, "rows": 4441200, "modules": 418, "native_signature_verified": True}
        if kind == "formal":
            import run_iclr2027_table1_clean_fastlane as f
            result, outcomes = f.validate_eval_info(out / "eval_info.json", j["suite"])
            b = self.plan["baseline"][f"repeat-{j['repeat']:02d}/{j['suite']}"]
            actual, reference = read(out / "procedural_bank_receipt.json"), read(b["reset"])
            for k in ("bank_manifest_sha256", "selection_sha256", "repeat_id", "eval_seed", "tasks"):
                if actual[k] != reference[k]:
                    raise ValueError(f"Formal reset differs from FeatCal: {k}")
            return {"result": result, "outcomes": outcomes, "baseline_eval": b["eval_info"],
                    "eval_info_sha256": sha(out / "eval_info.json"), "reset_sha256": sha(out / "procedural_bank_receipt.json")}
        import run_tcr_half_step_development as h
        # Validator requires a successful exit receipt alongside the result.
        write(out / "exit.json", {"return_code": 0})
        return h.verify_job(out, j["offset"])

    def lane(self, gpu):
        lockpath = self.run / f"gpu-{gpu}.lock"
        with lockpath.open("a") as device_lock:
            fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while not self.stop.is_set():
                if time.time() > self.deadline:
                    self.stop.set()
                    break
                j = self.next_job()
                if j is None:
                    with self.lock:
                        if not any(v in ("running", "pending") for v in self.states.values()):
                            return
                    self.stop.wait(30)
                    continue
                folder = self.run / "jobs" / j["id"]
                folder.mkdir(parents=True, exist_ok=False)
                try:
                    self.check_sources()
                    required = 64*1024 if j["kind"] in ("solve", "uniform") else 32*1024
                    while snapshot()[gpu][1] < required and not self.stop.is_set():
                        print("WAIT_GPU", gpu, j["id"], flush=True)
                        self.stop.wait(300)
                    if self.stop.is_set():
                        break
                    args, env, out = self.configure(j, gpu, folder)
                    command = [str(PYTHON), "-u", str(SCRIPTS / "tcr_night_worker_20260919.py"),
                               "--parent", str(os.getpid()), "--mode", j["kind"],
                               "--resources", str(folder / "resources.json"), *args]
                    write(folder / "launch.json", {"job": j, "gpu": gpu, "host": socket.gethostname(),
                          "command": command, "gpu_snapshot": snapshot(), "started_unix": time.time(),
                          "plan_sha256": sha(self.run / "plan.json"), "output": str(out),
                          "model_sha256": self.model_hashes.get(j.get("model")),
                          "runtime_config": env["LIBERO_CONFIG_PATH"], "duty_sleep": 0})
                    begin = time.monotonic()
                    with (folder / "worker.log").open("x") as log:
                        with self.lock:
                            if self.stop.is_set():
                                break
                            p = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                            self.children[j["id"]] = p
                        write(folder / "worker.json", {"pid": p.pid, "pgid": p.pid, "host": socket.gethostname()})
                        print("START", gpu, j["id"], "pid", p.pid, flush=True)
                        code = p.wait()
                    write(folder / "exit.json", {"return_code": code, "seconds": time.monotonic()-begin})
                    with self.lock:
                        self.children.pop(j["id"], None)
                    if code < 0 or code in (137, 143, 130):
                        self.cancel()
                        with self.lock:
                            self.states[j["id"]] = "cancelled"
                        raise RuntimeError("Externally terminated: cancelling entire nightly queue")
                    if code:
                        raise RuntimeError(f"Worker exited {code}; no automatic retry")
                    if not read(folder / "resources.json")["completed"]:
                        raise ValueError("Worker completion accounting absent")
                    verified = self.verify(j, folder, out)
                    write(folder / "verified.json", verified)
                    with self.lock:
                        self.states[j["id"]] = "done"
                    print("DONE", gpu, j["id"], flush=True)
                except Exception as exc:
                    write(folder / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc(), "no_retry": True})
                    with self.lock:
                        if self.states[j["id"]] != "cancelled":
                            self.states[j["id"]] = "failed"
                    print("FAILED", gpu, j["id"], repr(exc), flush=True)

    def execute(self):
        signal.signal(signal.SIGTERM, self.cancel)
        signal.signal(signal.SIGINT, self.cancel)
        with ThreadPoolExecutor(max_workers=4) as pool:
            fs = [pool.submit(self.lane, g) for g in GPUS]
            for f in fs:
                f.result()
        write(self.run / "queue-ended.json", {"states": self.states, "stopped": self.stop.is_set(),
              "ended_unix": time.time(), "all_successful": all(x=="done" for x in self.states.values())})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-prepared", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.execute_prepared:
        raise ValueError("Choose one stage")
    if args.execute_prepared:
        plan = read(args.run / "plan.json")
        if (args.run / "started.json").exists():
            raise ValueError("This queue was already started; automatic restart forbidden")
    else:
        args.run.mkdir(parents=True, exist_ok=False)
        plan = prepare(args.run)
    if args.prepare_only:
        print("PREPARED", len(plan["jobs"]), "jobs", flush=True)
        return
    write(args.run / "started.json", {"pid": os.getpid(), "host": socket.gethostname(), "time": time.time()})
    Queue(args.run, plan).execute()


if __name__ == "__main__":
    main()
