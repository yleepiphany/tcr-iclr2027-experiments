#!/usr/bin/env python3
"""Read-only preflight, then GPU3 OFT smoke/expert eval after cap24 completes.

Reuses the tested UUID leases, child ownership, death guard and stop latch of the
formal runner. No local GPU2, retries, expert retraining, or TCR-result claims.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
OFT = WORK / "vla-merge_table4/OpenVLA-OFT"
BASE = WORK / "vla-merge/experiments/claude-final-ab-20260920/run_final_ab.py"
PYTHON = WORK / "vla-merge-runtime/envs/mergevla/bin/python"
HOST = "dsw-967394-56ffd4897d-42wft"
UUID = "GPU-21ccffbd-eb6c-73a9-d63a-a5d200607a84"
DEPENDENCY = WORK / "vla-merge-runtime/experiments/local-budget-eval-20260921/attempt-01"

spec = importlib.util.spec_from_file_location("oft_formal", BASE)
formal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formal)
BASE_SAVE = formal.save


def save_with_actual_counts(path, value):
    if Path(path).name == "queue-ended.json":
        value = dict(value)
        value["episodes"] = sum(row.get("artifacts", {}).get("episodes", 0)
                                for row in value.get("finished", []))
        value["accepted_jobs"] = sum(row.get("outcome") == "success" and "artifacts" in row
                                     for row in value.get("finished", []))
        value["note"] = "Smoke jobs count as zero evaluation episodes; failed jobs are not accepted."
    BASE_SAVE(path, value)


def environment(gpu):
    if gpu not in (0, 3):  # formal runner probes its environment with 0; launches only 3
        raise ValueError("OFT local queue only authorizes GPU3")
    env = dict(os.environ)
    paths = [OFT / "dependencies/openvla_transformers_runtime",
             OFT / "dependencies/transformers-openvla-oft/src", OFT / "python_overlay",
             OFT / "source", WORK / "vla-merge-runtime/references/dlimp-openvla",
             WORK / "vla-merge-runtime/references/LIBERO-MergeVLA"]
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               PYTHONPATH=":".join(map(str, paths)), PYTHONUNBUFFERED="1",
               LIBERO_CONFIG_PATH=str(formal.LIBERO_CONFIG), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HOME=str(OFT / "cache/huggingface"),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2", TOKENIZERS_PARALLELISM="false")
    for key in (*formal.FORBIDDEN_ENV, "LIBERO_PRO_REPO", "LIBERO_PRO_ASSET_DIR", "MUJOCO_EGL_DEVICE_ID"):
        env.pop(key, None)
    return env


def file_identity(path):
    before = path.stat()
    digest = formal.sha(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during identity audit: {path}")
    return {"sha256": digest, "size": after.st_size, "mtime_ns": after.st_mtime_ns}


def assert_unchanged(files):
    for name, record in files.items():
        path = Path(name)
        s = path.stat()
        if (s.st_size, s.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError(f"Frozen input changed: {name}")
        if s.st_size < 64 * 1024 * 1024 and formal.sha(path) != record["sha256"]:
            raise ValueError(f"Frozen input content changed: {name}")


def command(model, suite, output, smoke):
    result = [str(PYTHON), str(HERE / "evaluate.py"), "--checkpoint", str(model),
              "--suite", suite, "--bank", str(formal.BANK), "--selection",
              str(formal.BANK / "selections/repeat-01.json"), "--output", str(output / "eval")]
    if smoke:
        result.append("--smoke")
    return result


def prepare(run):
    from safetensors import safe_open
    if socket.gethostname() != HOST or formal.gpu_row(3)["uuid"] != UUID:
        raise ValueError("Wrong local host/GPU")
    run.mkdir(parents=True, exist_ok=False)
    manifest = OFT / "manifests/official_release_checkpoints.json"
    experts = json.loads(manifest.read_text())["experts"]
    files = {str(manifest): file_identity(manifest)}
    reference_shapes = None
    for name, expert in experts.items():
        model = Path(expert["local_path"])
        index_path = model / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())["weight_map"]
        if len(index) != 982:
            raise ValueError("Unexpected expert backbone key count")
        shapes = {}
        for shard in sorted(set(index.values())):
            path = (model / shard).resolve()
            if path.parent != model.resolve():
                raise ValueError("Invalid shard path")
            files[str(path)] = file_identity(path)
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if index.get(key) != shard or key in shapes:
                        raise ValueError("Shard/key index mismatch")
                    shapes[key] = (f.get_slice(key).get_shape(), f.get_slice(key).get_dtype())
        if set(shapes) != set(index):
            raise ValueError("Missing backbone keys")
        if reference_shapes is None:
            reference_shapes = shapes
        elif shapes != reference_shapes:
            raise ValueError("Experts have different tensor shapes/dtypes")
        for head in ("action_head", "proprio_projector"):
            matches = list(model.glob(f"{head}--*_checkpoint.pt"))
            if len(matches) != 1:
                raise ValueError(f"Expected one {head} in {model}")
            files[str(matches[0])] = file_identity(matches[0])
        for path in model.iterdir():
            if path.is_file() and (path.suffix in (".json", ".model") or path.name == "config.json"):
                files[str(path)] = file_identity(path)
        print(json.dumps({"expert_audited": name, "backbone_keys": len(shapes)}), flush=True)
    # Bind code actually imported (including solver-independent native image helpers).
    paths = list((OFT / "source/experiments/robot").rglob("*.py"))
    paths += list((OFT / "source/prismatic/extern/hf").glob("*.py"))
    paths += [OFT / "source/prismatic/models/action_heads.py", OFT / "source/prismatic/models/projectors.py",
              OFT / "source/prismatic/vla/constants.py", formal.LIBERO_CONFIG / "config.yaml",
              HERE / "evaluate.py", HERE / "native_oft.py", Path(__file__), BASE]
    for path in paths:
        files[str(path)] = file_identity(path)
    identity = run / "identities.json"
    formal.save(identity, {"files": files, "experts": experts, "complete": True,
                           "training_steps": "official 150k/150k/50k/150k, not unified"})
    jobs = []
    for smoke in (True, False):
        for name, expert in experts.items():
            out = run / "jobs" / f'{"smoke" if smoke else "expert"}-{name}-r01'
            jobs.append({"id": out.name, "output": str(out), "suite": expert["suite"],
                         "expert": name, "smoke": smoke, "identity": str(identity),
                         "command": command(Path(expert["local_path"]), expert["suite"], out, smoke)})
    plan = {"schema": "oft_native_smoke_and_expert_v1", "host": HOST, "run": str(run),
            "gpus": [3], "gpu_uuid": UUID, "max_workers": 1, "min_free_mib": 32768,
            "runtime_floor_mib": 12288, "jobs": jobs, "episodes": 400,
            "identity_audit_sha256": formal.sha(identity), "identity_audit": str(identity),
            "evaluator_sha256": formal.sha(HERE / "evaluate.py"),
            "dependency": str(DEPENDENCY), "no_retry": True,
            "environment_contract": {k: environment(3)[k] for k in (
                "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "LIBERO_CONFIG_PATH", "OMP_NUM_THREADS",
                "HF_HUB_OFFLINE", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}}
    formal.save(run / "plan.json", plan)
    formal.save(run / "CLAIM.json", {"owner": "Codex", "host": HOST, "gpu": 3, "uuid": UUID,
                "scope": "4 native smoke, then expert 400; no TCR checkpoint exists yet",
                "dependency": str(DEPENDENCY)})
    print(json.dumps({"prepared": True, "smokes": 4, "expert_episodes": 400}), flush=True)


def verify(job):
    import sys
    sys.path.insert(0, str(HERE))
    from evaluate import load_selection, validate_rows
    folder = Path(job["output"]) / "eval"
    selection = load_selection(formal.BANK, formal.BANK / "selections/repeat-01.json",
                               verify_source_files=False)
    contract = json.loads((folder / "contract.json").read_text())
    if contract["selection_sha256"] != selection.selection_sha256 or contract["bank_manifest_sha256"] != selection.bank_manifest_sha256:
        raise ValueError("Wrong bank/selection")
    if contract["suite"] != job["suite"] or contract["eval_seed"] != 274001:
        raise ValueError("Wrong suite/seed")
    if (contract["native_chunk"] != 8 or contract["execute_actions"] != 8
            or contract["hard_reset"] is not True
            or contract["mode"] != ("smoke" if job["smoke"] else "eval")):
        raise ValueError("Native execution contract changed")
    assert_unchanged(json.loads(Path(job["identity"]).read_text())["files"])
    if job["smoke"]:
        result = json.loads((folder / "smoke.json").read_text())
        if result["identity_max_abs"] != 0.0 or result["action_shape"] != [8, 7] or not result["linear_call_order"]:
            raise ValueError("Native request smoke failed")
        return {"smoke_passed": True, "episodes": 0, "receipt_sha256": formal.sha(folder / "smoke.json")}
    rows = [json.loads(s) for s in (folder / "episodes.jsonl").read_text().splitlines()]
    successes = validate_rows(rows, selection, job["suite"])
    summary = json.loads((folder / "summary.json").read_text())
    if (summary["successes"] != successes or summary["episodes"] != 100 or
            summary["episodes_sha256"] != formal.sha(folder / "episodes.jsonl")):
        raise ValueError("Summary differs from actual episodes")
    return {"episodes": 100, "successes": successes, "pc_success": successes,
            "episodes_sha256": summary["episodes_sha256"]}


def wait_dependency(run):
    stop = formal.BatchStop(report_path=run / "dependency-stop.json")
    stop.install_signal_handlers()
    if (run / "waiting.json").exists():
        raise FileExistsError("This attempt was already started; no automatic restart")
    formal.save(run / "waiting.json", {"pid": os.getpid(), "kernel_starttime": formal.start_time(os.getpid()),
                "host": socket.gethostname(), "dependency": str(DEPENDENCY), "started_unix": time.time()})
    deadline = time.time() + 12 * 3600
    while not stop.stopped:
        terminal = DEPENDENCY / "queue-ended.json"
        if terminal.exists():
            value = json.loads(terminal.read_text())
            if value.get("status") != "complete" or value.get("completed_jobs") != 4:
                raise RuntimeError("Preceding local batch did not finish healthily; stop, do not restart")
            return
        if (DEPENDENCY / "batch-stop-report.json").exists():
            raise RuntimeError("Preceding local batch stopped; no new GPU jobs")
        if time.time() >= deadline:
            raise TimeoutError("Dependency did not complete within 12h")
        time.sleep(30)
    raise RuntimeError(stop.reason)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    args = p.parse_args()
    run = args.run.resolve()
    if args.prepare:
        prepare(run)
        return
    plan = json.loads((run / "plan.json").read_text())
    if socket.gethostname() != HOST or formal.gpu_row(3)["uuid"] != UUID or plan["gpus"] != [3]:
        raise ValueError("Host/GPU changed")
    wait_dependency(run)
    assert_unchanged(json.loads((run / "identities.json").read_text())["files"])
    formal.IDENTITIES = run / "identities.json"
    formal.EVALUATOR = HERE / "evaluate.py"
    formal.GPUS, formal.MAX_WORKERS = (3,), 1
    formal.FLOOR_MIB = 12288
    formal.scientific_environment = environment
    formal.verify_job = verify
    formal.save = save_with_actual_counts
    formal.run_queue(plan)
    # Publish the scope explicitly: these are expert results, never TCR results.
    terminal = json.loads((run / "queue-ended.json").read_text())
    actual = sum(row.get("artifacts", {}).get("episodes", 0) for row in terminal["finished"])
    formal.save(run / "oft-completion.json", {"status": terminal["status"],
                "actual_evaluation_episodes": actual, "smoke_jobs": 4, "tcr_evaluated": False})


if __name__ == "__main__":
    main()
