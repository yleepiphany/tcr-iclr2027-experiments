"""GPU2/3: three fixed controls; one solve/card, then two evaluations/card.

Independent outputs, preserved old data, no automatic retry or restart. Cancellation
uses the audited local queue's owned-child handling, never process-name killing.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

import run_tcr_night_20260919 as night
from tcr_pass2_controls_contract_20260919 import SCHEMA, ARMS, ROWS, attach_request_slots, validate_config

RUN = night.EXP / "tcr-pass2-controls-20260919"
WORKER = night.SCRIPTS / "tcr_pass2_controls_worker_20260919.py"
CONTRACT = night.SCRIPTS / "tcr_pass2_controls_contract_20260919.py"
SOLVER = night.ROOT / "experiments/claude-20260917/materialize_second_round_v3.py"
GPUS = (2, 3)


def prepare(run):
    original = night.read(night.DEFAULT / "plan.json")
    prior_audit = night.read(night.WORK / "coordination/2026-09-19/local-main-completion-audit.json")
    if not prior_audit["passed"]:
        raise ValueError("Fresh original campaign has not passed audit")
    plan = copy.deepcopy(original)
    plan.update(schema=SCHEMA, jobs=[], models={}, reused_half=[], gpu_ids=list(GPUS),
                solve_workers_per_gpu=1, eval_workers_per_gpu=2,
                new_checkpoints=3, new_development_episodes=1200,
                new_formal_episodes=0, reuse_unchanged_start_episodes=400,
                claim_boundary="Explored D2; controlled second-stage input/initialization comparisons, not a same-total-cost first-vs-second-pass causal test.")
    frozen = plan["frozen"]
    def freeze(path, expected=None):
        frozen[str(path)] = night.identity(path, expected)
    # Old frozen inputs are verified by unchanged size/mtime against their audited
    # identities. The critical new start and all two cache families are rehashed.
    for path, item in frozen.items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size"], item["mtime_ns"]):
            raise ValueError(f"Original frozen dependency changed: {path}")
    for path in (Path(__file__).resolve(), WORKER, CONTRACT, SOLVER,
                 night.SCRIPTS / "pi05_table3_contract.py"):
        freeze(path)
    start = night.DEFAULT / "models/tcre-r1"
    source_path = start / "block_regmeanpp_manifest.json"
    source = night.read(source_path)
    start_sha = prior_audit["models"]["tcre-r1"]["model_sha256"]
    freeze(start / "model.safetensors", start_sha)
    freeze(source_path)
    soup = Path(source["inputs"]["prior_model"])
    freeze(soup / "model.safetensors")
    soup_sha = frozen[str(soup / "model.safetensors")]["sha256"]
    for folder in (start, soup):
        if night.native_signature(folder) != plan["native_signature"]:
            raise ValueError("Start/Soup native config differs")
        for path in (folder / "config.json", *folder.glob("*normalizer*.safetensors")):
            freeze(path)
    calibration = {}
    for family in ("fresh_repeat01", "table3_across"):
        calibration[family] = {}
        for name in night.NAMES:
            folder = (night.DEFAULT / "traces/repeat-01" / name if family == "fresh_repeat01"
                      else night.EXP / "table3-ablation-20260916/inputs" / name / "across")
            tensor, manifest = folder / "replay.safetensors", folder / "replay.json"
            freeze(tensor)
            freeze(manifest)
            source_manifest = night.read(manifest)
            decorated = attach_request_slots(source_manifest)
            if Path(decorated["calibration_policy"]).resolve() != Path(plan["bank"][name]["path"]).resolve():
                raise ValueError("Trace source is not the registered expert")
            decorated["pass2_control_metadata"] = {
                "source_manifest":str(manifest), "source_manifest_sha256":night.sha(manifest),
                "tensor_sha256":frozen[str(tensor)]["sha256"], "no_tensor_or_state_order_changes":True,
                "slot_rule":"ascending request_index within each prompt, exactlyfive requests"}
            target = run / "inputs" / family / f"{name}.json"
            night.write(target, decorated)
            freeze(target)
            calibration[family][name] = dict(tensor=str(tensor), manifest=str(target),
                                             tensor_sha256=frozen[str(tensor)]["sha256"],
                                             manifest_sha256=frozen[str(target)]["sha256"])
    plan["reused_baseline"] = []
    from run_tcr_half_step_development import verify_job
    for offset in range(30, 40):
        folder = night.DEFAULT / "jobs" / f"dev-tcre-r1-{offset}" / "eval"
        record = verify_job(folder, offset)
        plan["reused_baseline"].append(dict(offset=offset, folder=str(folder), verified=record))
        for name in ("eval_info.json", "paired-noise.jsonl", "exit.json"):
            freeze(folder / name)
    for arm in ARMS:
        initial, digest = (soup, soup_sha) if arm == "soup" else (start, start_sha)
        family = "fresh_repeat01" if arm == "reuse" else "table3_across"
        reference = run / "references" / f"{arm}.json"
        # Not a forged checkpoint manifest: this minimal explicit reference records
        # the actual initialization separately from the shared numeric ridge anchor.
        night.write(reference, dict(schema="explicit_start_and_separate_numeric_ridge_reference_v1",
                    model_sha256=digest, initial_model=str(initial),
                    numeric_ridge_source=str(source_path), numeric_ridge_source_sha256=night.sha(source_path),
                    modules={k:{"ridge":v["ridge"]} for k,v in source["modules"].items()}))
        freeze(reference)
        config = dict(schema=SCHEMA, arm=arm, repeat=1, row_cap_per_request_module=16,
                      expert_masses="uniform_quarter", merged_slots=[], expected_realized_rows=ROWS,
                      start_point=str(initial), start_point_sha256=digest, start_point_manifest=str(reference),
                      numeric_ridge_source=str(source_path), numeric_ridge_source_sha256=night.sha(source_path),
                      base_model=source["inputs"]["base_model"], bank_path=plan["bank_path"],
                      initialization="soup" if arm == "soup" else "fresh_tcre_r1",
                      calibration_family=family, calibration=calibration[family])
        validate_config(config)
        config_path = run / "configs" / f"{arm}.json"
        night.write(config_path, config)
        freeze(config_path)
        plan["models"][arm] = dict(path=str(run / "models" / arm), config=str(config_path),
                                   start_model_sha256=digest, development_only=True)
        plan["jobs"].append(dict(id=f"solve-{arm}", kind="solve", model=arm, deps=[], priority=0))
    # Interleave arms at each offset; this is not score-adaptive scheduling.
    for offset in range(30, 40):
        for arm in ARMS:
            plan["jobs"].append(dict(id=f"dev-{arm}-{offset}", kind="development", model=arm,
                                     offset=offset, deps=[f"solve-{arm}"], priority=1))
    plan["prelaunch_gpu_snapshot"] = night.snapshot()
    night.write(run / "plan.json", plan)
    return plan


class Queue(night.Queue):
    def configure(self, job, gpu, folder):
        if job["kind"] != "solve":
            return super().configure(job, gpu, folder)
        item = self.plan["models"][job["model"]]
        config = night.read(item["config"])
        recipe = self.plan["recipe"]
        args = [f"--ablation-config={item['config']}", f"--dense-expert-bank={config['bank_path']}",
                f"--base-model={config['base_model']}", f"--prior-model={config['start_point']}",
                f"--output={item['path']}", "--ridge-ratio=.05", "--ridge-scale=feature_energy",
                "--max-correction-ratio=3", "--max-rows-per-sample=16",
                "--expert-loss-normalization=none", "--replay-prefix=merged", "--device=cuda",
                "--allow-mixed-calibration-policies", "--allow-prior-calibration-mismatch"]
        for name, source in config["calibration"].items():
            args += [f"--expert={name}={recipe['experts'][name]}",
                     f"--calibration={name}={source['tensor']}", f"--manifest={name}={source['manifest']}"]
        return args, night.environment(gpu, "solve"), Path(item["path"])

    def verify(self, job, folder, out):
        if job["kind"] != "solve":
            return super().verify(job, folder, out)
        from pi05_table3_contract import validate_rows
        result = night.read(out / "block_regmeanpp_manifest.json")
        config = night.read(self.plan["models"][job["model"]]["config"])
        source = night.read(config["numeric_ridge_source"])
        digest = night.sha(out / "model.safetensors")
        if (validate_rows(result["modules"], night.NAMES, None) != ROWS or
                result["realized_row_total"] != ROWS or result["modified_tensor_count"] != 422 or
                result["ablation"] != config or result["model_sha256"] != digest or
                result["inputs"]["prior_model"] != config["start_point"] or
                result["dense_expert_bank_sha256"] != night.sha(config["bank_path"]) or
                night.native_signature(out) != self.plan["native_signature"]):
            raise ValueError("Export/recipe/native identity differs")
        if set(result["modules"]) != set(source["modules"]):
            raise ValueError("Module scope differs")
        for key, value in result["modules"].items():
            if (value["ridge"] != source["modules"][key]["ridge"] or
                    value["expert_objective_weights"] != {n:.25 for n in night.NAMES}):
                raise ValueError("Numeric ridge or uniform mass differs")
        with self.lock:
            self.model_hashes[job["model"]] = digest
        return dict(model_sha256=digest, rows=ROWS, modules=418,
                    equals_start=digest == config["start_point_sha256"], uniform_masses=True,
                    exact_numeric_ridge_anchor=config["numeric_ridge_source"])

    def next_phase_job(self, phase):
        with self.lock:
            if self.stop.is_set():
                return None
            for job in self.plan["jobs"]:
                jid = job["id"]
                if job["kind"] != phase or self.states[jid] != "pending":
                    continue
                if any(self.states[d] in ("failed", "cancelled", "blocked") for d in job["deps"]):
                    self.states[jid] = "blocked"
                    continue
                if all(self.states[d] == "done" for d in job["deps"]):
                    self.states[jid] = "running"
                    return job
        return None

    def lane(self, gpu, phase, slot):
        while not self.stop.is_set():
            if time.time() > self.deadline:
                self.cancel()
                return
            job = self.next_phase_job(phase)
            if job is None:
                return
            folder = self.run / "jobs" / job["id"]
            folder.mkdir(parents=True, exist_ok=False)
            try:
                self.check_sources()
                required = (64 if phase == "solve" else 32)*1024
                while night.snapshot()[gpu][1] < required and not self.stop.is_set():
                    if time.time() > self.deadline:
                        self.cancel()
                        break
                    print("WAIT_GPU", gpu, job["id"], flush=True)
                    self.stop.wait(120)
                if self.stop.is_set():
                    break
                args, env, out = self.configure(job, gpu, folder)
                command = [str(night.PYTHON), "-u", str(WORKER), "--parent", str(os.getpid()),
                           "--mode", phase, "--resources", str(folder / "resources.json"), *args]
                night.write(folder / "launch.json", dict(job=job, gpu=gpu, slot=slot,
                            host=socket.gethostname(), command=command, gpu_snapshot=night.snapshot(),
                            started_unix=time.time(), plan_sha256=night.sha(self.run / "plan.json"),
                            output=str(out), model_sha256=self.model_hashes.get(job["model"]),
                            runtime_config=env["LIBERO_CONFIG_PATH"], duty_sleep=0))
                begin = time.monotonic()
                with (folder / "worker.log").open("x") as log:
                    with self.lock:
                        if self.stop.is_set():
                            break
                        process = subprocess.Popen(command, cwd=night.ROOT, env=env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
                        self.children[job["id"]] = process
                    night.write(folder / "worker.json", dict(pid=process.pid, pgid=process.pid,
                                                              host=socket.gethostname()))
                    print("START", gpu, slot, job["id"], "pid", process.pid, flush=True)
                    code = process.wait()
                night.write(folder / "exit.json", dict(return_code=code, seconds=time.monotonic()-begin))
                with self.lock:
                    self.children.pop(job["id"], None)
                if code < 0 or code in (137,143,130):
                    self.cancel()
                    with self.lock:
                        self.states[job["id"]] = "cancelled"
                    raise RuntimeError("Signal termination: entire queue cancelled, no restart")
                if code or not night.read(folder / "resources.json")["completed"]:
                    raise RuntimeError(f"Worker failure{code}, no retry")
                verified = self.verify(job, folder, out)
                night.write(folder / "verified.json", verified)
                with self.lock:
                    self.states[job["id"]] = "done"
                print("DONE", gpu, slot, job["id"], flush=True)
            except Exception as error:
                night.write(folder / "failure.json", dict(error=repr(error), traceback=traceback.format_exc(), no_retry=True))
                with self.lock:
                    if self.states[job["id"]] != "cancelled":
                        self.states[job["id"]] = "failed"
                print("FAILED", job["id"], repr(error), flush=True)

    def execute(self):
        signal.signal(signal.SIGTERM, self.cancel)
        signal.signal(signal.SIGINT, self.cancel)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(self.lane, gpu, "solve", 0) for gpu in GPUS]:
                future.result()
        if not self.stop.is_set():
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(self.lane, gpu, "development", slot) for gpu in GPUS for slot in range(2)]
                for future in futures:
                    future.result()
        with self.lock:
            if self.stop.is_set():
                for key, state in self.states.items():
                    if state in ("pending", "running"):
                        self.states[key] = "cancelled"
        night.write(self.run / "queue-ended.json", dict(states=self.states, stopped=self.stop.is_set(),
                    ended_unix=time.time(), all_successful=all(v == "done" for v in self.states.values())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        args.run.mkdir(parents=True, exist_ok=False)
        plan = prepare(args.run)
        print("PREPARED", len(plan["jobs"]), "jobs", flush=True)
        return
    plan = night.read(args.run / "plan.json")
    if plan["schema"] != SCHEMA or plan["gpu_ids"] != list(GPUS):
        raise ValueError("Unexpected prepared run")
    # Exclusive creation prohibits any automatic restart of this prepared campaign.
    night.write(args.run / "started.json", dict(pid=os.getpid(), host=socket.gethostname(), time=time.time()))
    queue = Queue(args.run, plan)
    queue.check_sources()
    queue.execute()


if __name__ == "__main__":
    main()
