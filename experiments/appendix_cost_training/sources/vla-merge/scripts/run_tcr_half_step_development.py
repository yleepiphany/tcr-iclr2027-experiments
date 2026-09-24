"""Two fixed half-step checkpoints, then serial expanded development.

No alpha search, no new solver, no confirmation access, no process termination.
Preserves live D2 and the formal/repeat02 lanes. Emits no partial success scores.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from run_featcal_execution_pilot_lane_v2 import environment
from audit_tcr_expanded_pairing import validate_receipt, compare_noise, receipt_lines

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
PYTHON = WORK / "pi05_lora_finetune_v2_20260826/.venv/bin/python"
SOURCE = ROOT / "experiments/claude-20260917"
D2 = WORK / "vla-merge-runtime/experiments/claude-tcr-new-methods-20260917/expanded-development"
DEFAULT = WORK / "vla-merge-runtime/experiments/tcr-half-step-development-20260917"
OCCUPANCY = WORK / "vla-merge-runtime/experiments/tcr-10k-occupancy-repeat02-recovery-20260917"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ARMS = ("c_e_half", "c_m_half")
PINNED = {
    "e0": "48007623a8dea0659bd2e7e6820729a15c81f872022ded366bb97a22bb834904",
    "c_e": "3e3a55ce0acd04b3ee1e180eda2a6109b0ef6d0ea4e717f6052a27829adfce95",
    "c_m": "df29be55a439c45087291ab59df47790e2ea47b4e2647fdd60e001eb892b5a7a",
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def half_tensor(base, candidate, alpha=.5):
    import torch
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Invalid interpolation coefficient")
    if base.shape != candidate.shape or base.dtype != candidate.dtype:
        raise ValueError("Tensor shape/dtype mismatch")
    if not torch.is_floating_point(base):
        if not torch.equal(base, candidate):
            raise ValueError("Non-floating buffer changed")
        return base.clone()
    if not torch.isfinite(base).all() or not torch.isfinite(candidate).all():
        raise ValueError("Non-finite source tensor")
    if alpha == 0:
        return base.clone()
    if alpha == 1:
        return candidate.clone()
    precision = torch.float64 if base.dtype == torch.float64 else torch.float32
    result = (base.to(precision) + alpha * (candidate.to(precision) - base.to(precision))).to(base.dtype)
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite interpolated tensor")
    return result.contiguous()


def module_scope(manifest):
    modules = manifest["modules"]
    return {name + ".weight" for name in modules} | {
        name + ".bias" for name, item in modules.items() if item["kind"] == "multi_dense_augmented"}


def source_files(folder):
    return {str(p.relative_to(folder)): p for p in folder.rglob("*") if p.is_file()
            and p.name not in {"model.safetensors", "block_regmeanpp_manifest.json"}}


def freeze(run, poll):
    if poll < 60:
        raise ValueError("Use low-frequency monitoring")
    reference = D2 / "plan-e0-c_e-c_m-n2-featcal-30-39.json"
    original = read(reference)
    models = {name: original["models"][name] for name in PINNED}
    for name, item in models.items():
        if item["model_sha256"] != PINNED[name]:
            raise ValueError("Unexpected source checkpoint identity")
    wrapper = SOURCE / "eval_pi05_expanded_development.py"
    if sha(wrapper) != original["wrapper_sha256"]:
        raise ValueError("D2 wrapper changed")
    dependencies = [Path(__file__), wrapper, ROOT / "scripts/audit_tcr_expanded_pairing.py",
                    ROOT / "scripts/run_featcal_execution_pilot_lane_v2.py",
                    ROOT / "scripts/eval_pi05_libero_with_init_offset.py"]
    plan = {"experiment": "Fixed half-step of the existing second-round corrections",
            "alpha": .5, "arms": list(ARMS), "models": models,
            "offsets": list(range(30, 40)), "episodes_per_arm": 400, "new_episodes": 800,
            "sequential_workers": 1, "wrapper": str(wrapper), "duty": .25,
            "source_plan": str(reference), "source_plan_sha256": sha(reference),
            "dependencies": {str(p): sha(p) for p in dependencies},
            "formula": "E0 + 0.5 * (second_round - E0), FP32 then original stored dtype",
            "primary_contrasts": ["c_m_half vs c_m", "c_e_half vs c_e", "c_m_half vs c_e_half"],
            "utility_contrasts": ["c_m_half vs featcal", "c_e_half vs featcal"],
            "reference_evaluation": str(D2), "development_only": True,
            "offset_boundary": "All30..39 are development; report30 and31..39 separately, no subset selection",
            "no_automatic_extra_rounds": True, "no_alpha_search": True, "no_kill": True,
            "poll_seconds": poll, "minimum_free_mib": 30 * 1024,
            "preserve_one_44gib_device_for_unfinished_repeat02": True,
            "independent_confirmation_used": False, "goal_achieved": False,
            "created_unix": time.time(), "hostname": socket.gethostname(), "pid": os.getpid()}
    run.mkdir(parents=True, exist_ok=False)
    write(run / "plan.json", plan)
    return plan


def build(plan, arm, run):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    start = time.monotonic()
    identity = {name: plan["models"][name] for name in ("e0", arm.removesuffix("_half"))}
    folders = {name: Path(item["path"]) for name, item in identity.items()}
    manifests = {name: read(p / "block_regmeanpp_manifest.json") for name, p in folders.items()}
    names = list(identity)
    base_name, target_name = names
    for name, folder in folders.items():
        if sha(folder / "model.safetensors") != identity[name]["model_sha256"]:
            raise ValueError(f"Source weight hash mismatch: {name}")
    base_manifest, target_manifest = (manifests[name] for name in names)
    if (base_manifest["dense_expert_bank_sha256"] != target_manifest["dense_expert_bank_sha256"] or
            sha(base_manifest["dense_expert_bank"]) != base_manifest["dense_expert_bank_sha256"]):
        raise ValueError("Expert bank changed")
    if base_manifest["realized_row_total"] != 4441200 or target_manifest["realized_row_total"] != 1331600:
        raise ValueError("Unexpected source recipe")
    scope = module_scope(target_manifest)
    if scope != module_scope(base_manifest):
        raise ValueError("Correction scope changed")
    assets = {name: source_files(p) for name, p in folders.items()}
    if assets[base_name].keys() != assets[target_name].keys():
        raise ValueError("Configuration asset coverage differs")
    for relative in assets[base_name]:
        a, b = (assets[name][relative] for name in names)
        if relative == "config.json":
            x, y = read(a), read(b)
            x.pop("pretrained_path", None)
            y.pop("pretrained_path", None)
            if x != y:
                raise ValueError("Native model configurations differ")
        elif sha(a) != sha(b):
            raise ValueError(f"Processing assets differ: {relative}")
    output = run / "checkpoints" / arm
    output.mkdir(parents=True, exist_ok=False)
    state, changed = {}, []
    with ExitStack() as stack:
        handles = {name: stack.enter_context(safe_open(str(p / "model.safetensors"), framework="pt", device="cpu"))
                   for name, p in folders.items()}
        keys = set(handles[base_name].keys())
        if set(handles[target_name].keys()) != keys or not scope <= keys:
            raise ValueError("State dict key mismatch")
        for key in sorted(keys):
            a, b = (handles[name].get_tensor(key) for name in names)
            if key not in scope and not torch.equal(a, b):
                raise ValueError(f"Unexpected change outside calibrated scope: {key}")
            value = half_tensor(a, b, plan["alpha"])
            if not torch.equal(value, a):
                changed.append(key)
            state[key] = value
        save_file(state, str(output / "model.safetensors"))
        with safe_open(str(output / "model.safetensors"), framework="pt", device="cpu") as saved:
            if set(saved.keys()) != keys or any(not torch.equal(saved.get_tensor(k), v) for k, v in state.items()):
                raise ValueError("Reload differs from exported values")
    del state
    for relative, source in assets[base_name].items():
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    receipt = {"path": str(output), "model_sha256": sha(output / "model.safetensors"),
               "source_models": identity, "alpha": plan["alpha"], "changed_tensors": changed,
               "scope_tensors": len(scope), "reload_bitwise_equal": True,
               "outside_scope_unchanged": True, "expert_bank_sha256": base_manifest["dense_expert_bank_sha256"],
               "cumulative_regression_rows": 5772800, "new_regression_rows": 0,
               "build_cpu_wall_seconds": time.monotonic() - start,
               "additional_checkpoint_bytes": (output / "model.safetensors").stat().st_size}
    write(output / "half_step_manifest.json", receipt)
    print("BUILD_COMPLETE", arm, flush=True)
    return receipt


def select_gpu(rows, protect_large):
    reserved = None
    if protect_large:
        large = [r for r in rows if r[2] >= 44 * 1024]
        if large:
            reserved = max(large, key=lambda r: r[2])[0]
    eligible = [r for r in rows if r[2] >= 30 * 1024 and r[0] != reserved]
    return min(eligible, key=lambda r: (r[1], r[0]))[0] if eligible else None


def protect_repeat02():
    if (OCCUPANCY / "failure.json").exists():
        return False
    status = OCCUPANCY / "status.json"
    return not status.exists() or read(status).get("stage") != "complete"


def check_dependencies(plan):
    if any(sha(path) != expected for path, expected in plan["dependencies"].items()):
        raise ValueError("Frozen execution dependency changed; no new jobs launched")


def eval_command(plan, out, model, offset):
    return [str(PYTHON), "-u", plan["wrapper"], f"--output_dir={out}", "--env.type=libero",
            "--env.task=" + ",".join(SUITES), "--env.task_ids=[0,1,2,3,4,5,6,7,8,9]",
            "--env.max_parallel_tasks=1", "--eval.batch_size=1", "--eval.n_episodes=1",
            f"--seed={391600 + 1000 * (offset - 30)}", f"--policy.path={model}",
            "--policy.device=cuda", "--policy.compile_model=false", "--policy.gradient_checkpointing=false",
            "--policy.n_action_steps=10"]


def verify_job(folder, offset):
    if read(folder / "exit.json")["return_code"] != 0:
        raise ValueError("Evaluation failed; preserve artifacts, no retry")
    rows, _ = receipt_lines(folder / "paired-noise.jsonl", False)
    keys = [validate_receipt(row, offset) for row in rows]
    if len(keys) != 40 or len(set(keys)) != 40:
        raise ValueError("Missing or duplicate paired receipts")
    outcomes = {}
    for item in read(folder / "eval_info.json")["per_task"]:
        key = item["task_group"], item["task_id"], offset
        values = item["metrics"]["successes"]
        if key in outcomes or len(values) != 1 or type(values[0]) is not bool:
            raise ValueError("Invalid result row")
        outcomes[key] = values[0]
    if outcomes != {key: row["successes"][0] for key, row in zip(keys, rows)}:
        raise ValueError("Result/receipt disagreement")
    return {"episodes": 40, "eval_info_sha256": sha(folder / "eval_info.json"),
            "noise_receipt_sha256": sha(folder / "paired-noise.jsonl")}


def run_all(args):
    os.nice(10)
    plan = freeze(args.run, args.poll_seconds)
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        models = {arm: build(plan, arm, args.run) for arm in ARMS}
        write(args.run / "build-complete.json", {"models": models, "ended_unix": time.time()})
        for offset in plan["offsets"]:
            for arm in ARMS:
                while True:
                    check_dependencies(plan)
                    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
                                                   "--format=csv,noheader,nounits"], text=True)
                    rows = [tuple(map(int, line.split(","))) for line in raw.strip().splitlines()]
                    gpu = select_gpu(rows, protect_repeat02())
                    if gpu is not None:
                        break
                    print("WAIT_GPU", arm, offset, flush=True)
                    time.sleep(args.poll_seconds)
                out = args.run / "development" / arm / f"offset-{offset}"
                out.mkdir(parents=True, exist_ok=False)
                model = models[arm]
                command = eval_command(plan, out, model["path"], offset)
                env = environment(gpu)
                env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                           TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="1", PI05_LIBERO_INIT_STATE_OFFSET=str(offset),
                           PI05_LIBERO_INIT_STATE_COUNT="1", PI05_TASK_TEXT_MODE="correct",
                           CLAUDE_EVAL_DUTY=str(plan["duty"]), ITERATION_PHYSICAL_GPU=str(gpu),
                           ITERATION_NOISE_RECEIPT=str(out / "paired-noise.jsonl"))
                write(out / "launch.json", {"gpu": gpu, "gpu_snapshot": rows, "command": command,
                      "model": model["path"], "model_sha256": model["model_sha256"], "offset": offset,
                      "wrapper_sha256": plan["dependencies"][plan["wrapper"]],
                      "hostname": socket.gethostname(), "started_unix": time.time(), "duty": plan["duty"]})
                begin = time.monotonic()
                print("EVAL_START", arm, offset, "gpu", gpu, flush=True)
                with (out / "worker.log").open("x") as log:
                    code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                write(out / "exit.json", {"return_code": code, "wall_seconds": time.monotonic() - begin})
                write(out / "verified.json", verify_job(out, offset))
                print("EVAL_COMPLETE", arm, offset, flush=True)
        write(args.run / "complete.json", {"new_episodes": 800, "ended_unix": time.time(),
              "status": "evaluations_complete_pending_paired_analysis", "goal_achieved": False,
              "reference_run": str(D2), "independent_confirmation_used": False})
    except Exception as exc:
        write(args.run / "failure.json", {"error": repr(exc), "ended_unix": time.time(), "no_retry": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT)
    parser.add_argument("--poll-seconds", type=int, default=900)
    run_all(parser.parse_args())
