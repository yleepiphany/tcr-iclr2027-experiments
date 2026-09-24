"""Three missing formal baseline jobs, only after the current nightly queue succeeds.

Independent recovery directory, no edits to the active queue or old results. A killed
or failed parent campaign never triggers this supplement. No automatic restart.
"""
import argparse
import copy
import os
from pathlib import Path
import signal
import socket
import threading
import time

import run_tcr_night_20260919 as night

RUN = night.EXP / "regmeanpp-missing-night-20260919"
MANIFEST = night.TABLE / "manifest-clean-formal-regmeanpp-v1.json"
SUITES = ("libero_object", "libero_goal", "libero_10")


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return stat[19]
    except FileNotFoundError:
        return None


def gate(ended, parent_alive):
    if ended is not None:
        if (ended.get("stopped") or not ended.get("all_successful") or
                not ended.get("states") or any(s != "done" for s in ended["states"].values())):
            return "cancel"
        return "wait" if parent_alive else "ready"
    return "wait" if parent_alive else "cancel"


def canonical(arguments):
    return [a for a in arguments if not a.startswith(("--output_dir=", "--policy.path="))]


def prepare(run):
    parent = night.read(night.DEFAULT / "started.json")
    if parent["host"] != socket.gethostname():
        raise ValueError("Parent identity belongs to another host")
    start_id = process_identity(parent["pid"])
    if start_id is None:
        raise ValueError("Prepare requires live nightly supervisor")
    original = night.read(MANIFEST)
    plan = copy.deepcopy(night.read(night.DEFAULT / "plan.json"))
    plan.update(schema="regmeanpp_missing_night_v1", models={}, jobs=[], reused_half=[],
                parent_campaign={"run":str(night.DEFAULT), "pid":parent["pid"],
                                 "proc_start_ticks":start_id, "host":parent["host"],
                                 "deadline":parent["time"] + 12*3600},
                claim_boundary="Recovery of three incomplete baseline evaluations, not a new method or parameter selection.")
    plan["frozen"][str(Path(__file__).resolve())] = night.identity(Path(__file__).resolve())
    plan["frozen"][str(MANIFEST)] = night.identity(MANIFEST)
    chosen = [j for j in original["jobs"] if j["repeat"] == "repeat-03" and j["suite"] in SUITES]
    if len(chosen) != 3 or {j["suite"] for j in chosen} != set(SUITES):
        raise ValueError("Unexpected supplement allocation")
    for j in chosen:
        folder = Path(original["output_root"]) / j["output_relpath"] / "attempt-01"
        if (folder / "run_receipt.json").exists():
            raise ValueError("Old completion appeared; audit rather than duplicate")
        pre = night.read(folder / "prelaunch_receipt.json")
        if pre["host"] != socket.gethostname():
            raise ValueError("Old process ownership cannot be checked locally")
        old_pid = pre["resource"]["lease"]["pid"]
        if process_identity(old_pid) is not None:
            raise ValueError("Old PID exists; verify identity manually before recovery")
        checkpoint = Path(j["checkpoint"]["path"])
        key = "regmeanpp-r3"
        digest = j["checkpoint"]["model_sha256"]
        plan["frozen"][str(checkpoint / "model.safetensors")] = night.identity(checkpoint / "model.safetensors", digest)
        if night.native_signature(checkpoint) != plan["native_signature"]:
            raise ValueError("Native policy signature differs from nightly protocol")
        for path in [checkpoint / "config.json", *checkpoint.glob("*normalizer*.safetensors"),
                     folder / "prelaunch_receipt.json"]:
            plan["frozen"][str(path)] = night.identity(path)
        baseline = plan["baseline"][f"repeat-03/{j['suite']}"]
        if canonical(pre["command"][2:]) != canonical(baseline["command"][2:]):
            raise ValueError("Old baseline command differs beyond model/output path")
        if pre["identities"]["checkpoint_sha256"] != digest:
            raise ValueError("Old checkpoint identity mismatch")
        plan["models"][key] = {"path":str(checkpoint), "sha256":digest}
        plan["jobs"].append(dict(id=f"formal-regmeanpp-r3-{j['suite']}", kind="formal",
            model=key, repeat=3, suite=j["suite"], deps=[], priority=0,
            original_incomplete_attempt=str(folder)))
    night.write(run / "plan.json", plan)
    return plan


def execute(run):
    plan = night.read(run / "plan.json")
    if (run / "started.json").exists():
        raise ValueError("Already started; no automatic restart")
    night.write(run / "started.json", {"host":socket.gethostname(), "pid":os.getpid(), "time":time.time()})
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_:stop.set())
    signal.signal(signal.SIGINT, lambda *_:stop.set())
    parent = plan["parent_campaign"]
    ended_path = Path(parent["run"]) / "queue-ended.json"
    print("WAIT_PARENT_CAMPAIGN", parent["pid"], flush=True)
    while not stop.is_set() and time.time() < parent["deadline"]:
        ended = night.read(ended_path) if ended_path.exists() else None
        alive = process_identity(parent["pid"]) == parent["proc_start_ticks"]
        state = gate(ended, alive)
        if state == "cancel":
            night.write(run / "not-launched.json", {"reason":"Parent did not finish successfully; no recovery/restart", "gpu_jobs_started":0})
            return
        if state == "ready":
            break
        stop.wait(300)
    else:
        night.write(run / "not-launched.json", {"reason":"Wait cancelled or overnight deadline reached", "gpu_jobs_started":0})
        return
    # Recheck that no competing old result arrived while waiting.
    if any((Path(j["original_incomplete_attempt"]) / "run_receipt.json").exists() for j in plan["jobs"]):
        raise ValueError("Old completion appeared while waiting; refusing duplicate")
    queue = night.Queue(run, plan)
    queue.check_sources()
    print("START_SUPPLEMENT", len(plan["jobs"]), flush=True)
    queue.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        args.run.mkdir(parents=True, exist_ok=False)
        prepare(args.run)
        print("PREPARED_THREE_MISSING_JOBS", flush=True)
    else:
        execute(args.run)
