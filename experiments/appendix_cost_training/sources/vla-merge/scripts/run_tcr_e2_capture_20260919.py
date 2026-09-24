"""New E2 occupancy collection, queued behind current local GPU2/3 evaluations.

Not a restart of failed reuse. The known normal reserve-check failure is preserved;
any signal termination, new failure or changed predecessor identity prevents launch.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import socket
import time

import run_tcr_night_20260919 as night
import run_tcr_pass2_controls_20260919 as controls
import queue_tcr_10k_occupancy_collection as capture
from run_regmeanpp_missing_20260919 import process_identity

RUN = night.EXP / "tcr-e2-occupancy-20260919"
WORKER = night.SCRIPTS / "tcr_e2_capture_worker_20260919.py"
START = night.EXP / "claude-tcr-new-methods-20260917/arms/checkpoints/r01"
START_SHA = "4562193825dc9b23e834eb32bfc57242501c7abc3f6c561eeaeee1b5b6f42da8"
SCHEMA = "tcr_e2_occupancy_capture_v1"


def predecessor_gate(ended, alive, exit_codes):
    if any(code < 0 or code in (130,137,143) for code in exit_codes):
        return "cancel"
    if ended is None:
        return "wait" if alive else "cancel"
    expected = {f"solve-{a}":("failed" if a == "reuse" else "done")
                for a in ("reuse","alternate","soup")}
    expected.update({f"dev-{a}-{o}":("blocked" if a == "reuse" else "done")
                     for a in ("reuse","alternate","soup") for o in range(30,40)})
    if ended.get("stopped") or ended.get("states") != expected:
        return "cancel"
    return "wait" if alive else "ready"


def prepare(run):
    parent = night.read(controls.RUN / "started.json")
    identity = process_identity(parent["pid"])
    if parent["host"] != socket.gethostname() or identity is None:
        raise ValueError("Prepare requires the currently live local predecessor")
    command = Path(f"/proc/{parent['pid']}/cmdline").read_bytes()
    if str(Path(controls.__file__).resolve()).encode() not in command.split(b"\0"):
        raise ValueError("Unexpected predecessor command")
    plan = copy.deepcopy(night.read(controls.RUN / "plan.json"))
    plan.update(schema=SCHEMA, jobs=[], models={}, reused_baseline=[],
                new_development_episodes=0, new_formal_episodes=0,
                calibration_episodes=40, collection_only=True,
                parent_campaign=dict(run=str(controls.RUN),pid=parent["pid"],start_ticks=identity,
                                     host=parent["host"],known_non_signal_failure="solve-reuse"),
                capture_model=dict(path=str(START),model_sha256=START_SHA),
                interpretation="E2=r01, the existing two-pass start of audited L0. Collection outcomes are not evaluation. No claim that TCR-O is ineffective or effective.")
    frozen = plan["frozen"]
    for path, item in frozen.items():
        stat = Path(path).stat()
        if (stat.st_size,stat.st_mtime_ns) != (item["size"],item["mtime_ns"]):
            raise ValueError(f"Frozen predecessor dependency changed: {path}")
    for path in (Path(__file__).resolve(),WORKER,Path(capture.__file__).resolve(),
                 capture.COLLECTOR,night.SCRIPTS/"run_regmeanpp_missing_20260919.py",
                 START/"block_regmeanpp_manifest.json",START/"config.json",
                 night.WORK/"coordination/2026-09-19/claude-r01-three-arm-audit.json"):
        frozen[str(path)] = night.identity(path)
    frozen[str(START/"model.safetensors")] = night.identity(START/"model.safetensors",START_SHA)
    if night.native_signature(START) != plan["native_signature"]:
        raise ValueError("E2 native configuration differs")
    for path in START.glob("*normalizer*.safetensors"):
        frozen[str(path)] = night.identity(path)
    ref = night.read(START/"block_regmeanpp_manifest.json")
    if ref["dense_expert_bank_sha256"] != night.sha(plan["bank_path"]):
        raise ValueError("E2 built from a different expert bank")
    fail = controls.RUN/"jobs/solve-reuse"
    if (night.read(fail/"exit.json")["return_code"] != 1 or
            "GPU free reserve below12GiB" not in (fail/"worker.log").read_text()):
        raise ValueError("Predecessor's allowed failure differs; no implicit recovery")
    for path in (fail/"exit.json",fail/"failure.json",fail/"resources.json"):
        frozen[str(path)] = night.identity(path)
    for name,suite in capture.table3.SUITES.items():
        spec = capture.capture_command(plan["capture_model"],run/"traces",name,suite)
        plan["jobs"].append(dict(id=f"collect-{name}",kind="collect",model="e2",
                                 name=name,deps=[],priority=0,spec=spec))
    night.write(run/"plan.json",plan)
    return plan


class Queue(controls.Queue):
    def configure(self, job, gpu, folder):
        spec = job["spec"]
        env = night.environment(gpu,"collect")
        env.update(spec["environment"],TCR_OCCUPANCY_BANK_SHA256=night.sha(self.plan["bank_path"]))
        return spec["command"][3:], env, Path(spec["output"])

    def verify(self, job, folder, out):
        result = capture.verify_capture(job["spec"],self.plan["capture_model"],night.sha(self.plan["bank_path"]))
        result["collection_only"] = True
        return result

    def execute(self):
        signal.signal(signal.SIGTERM,self.cancel)
        signal.signal(signal.SIGINT,self.cancel)
        parent = self.plan["parent_campaign"]
        while not self.stop.is_set():
            observed = process_identity(parent["pid"])
            if observed is not None and observed != parent["start_ticks"]:
                raise RuntimeError("Predecessor PID reused; no automatic launch")
            path = controls.RUN/"queue-ended.json"
            ended = night.read(path) if path.exists() else None
            codes = [night.read(p)["return_code"] for p in (controls.RUN/"jobs").glob("*/exit.json")]
            verdict = predecessor_gate(ended,observed is not None,codes)
            if verdict == "cancel" or time.time() > self.deadline:
                self.cancel()
                break
            if verdict == "ready":
                # Revalidate all required complete evaluation jobs, not just states.
                from run_tcr_half_step_development import verify_job
                for arm in ("alternate","soup"):
                    for offset in range(30,40):
                        folder=controls.RUN/"jobs"/f"dev-{arm}-{offset}"
                        if verify_job(folder/"eval",offset) != night.read(folder/"verified.json"):
                            raise ValueError("Predecessor raw evaluation mismatch")
                night.write(self.run/"predecessor-released.json",dict(time=time.time(),
                            terminal_sha256=night.sha(path),known_failure_preserved=True))
                break
            print("WAIT_PREDECESSOR",parent["pid"],flush=True)
            self.stop.wait(300)
        if not self.stop.is_set():
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(self.lane,gpu,"collect",slot) for gpu in (2,3) for slot in range(2)]
                for future in futures:
                    future.result()
        if self.stop.is_set():
            for key,state in self.states.items():
                if state in ("pending","running"):
                    self.states[key]="cancelled"
        night.write(self.run/"queue-ended.json",dict(states=self.states,stopped=self.stop.is_set(),
                    all_successful=all(v=="done" for v in self.states.values()),ended_unix=time.time(),
                    collection_only=True,no_success_rate_evaluation=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    if args.prepare_only:
        RUN.mkdir(parents=True,exist_ok=False)
        plan=prepare(RUN)
        print("PREPARED",len(plan["jobs"]),"collection jobs",flush=True)
        return
    plan=night.read(RUN/"plan.json")
    if plan["schema"] != SCHEMA or plan["gpu_ids"] != [2,3]:
        raise ValueError("Wrong prepared collection plan")
    night.write(RUN/"started.json",dict(pid=os.getpid(),host=socket.gethostname(),time=time.time()))
    controls.WORKER=WORKER  # Independent process-local adapter, no source mutation.
    queue=Queue(RUN,plan)
    queue.check_sources()
    queue.execute()


if __name__ == "__main__":
    main()
