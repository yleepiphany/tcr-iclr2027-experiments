#!/usr/bin/env python3
"""One Fast-WAM expert closed-loop calibration episode from a demo reset.

This calls the stock LIBERO evaluator's episode function. Demo initial states
are screened against the 50 stock evaluation reset states for the same task.
No demonstration action is read or used to generate a calibration target.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FAST = ROOT / "vla-merge_table4/Fast-WAM"
RUNTIME = ROOT / "vla-merge-runtime"
SOURCE = FAST / "source"
SUITES = {"spatial": "libero_spatial", "object": "libero_object",
          "goal": "libero_goal", "long": "libero_10"}
STEPS = {"spatial": 12000, "object": 10000, "goal": 10000, "long": 9000}


def gpu_row(gpu: int) -> dict:
    line = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=index,uuid,memory.free,memory.used",
        "--format=csv,noheader,nounits"], text=True).strip()
    index, uuid, free, used = [part.strip() for part in line.split(",")]
    return {"index": int(index), "uuid": uuid,
            "free_mib": int(free), "used_mib": int(used)}


def select_demo_reset(task_suite, task_id: int, demo_index: int):
    import h5py
    import numpy as np

    suite = task_suite.get_task(task_id).problem_folder
    demo_path = Path(os.environ["FASTWAM_LIBERO_DEMONSTRATIONS"]) / \
        task_suite.get_task_demonstration(task_id)
    with h5py.File(demo_path) as handle:
        key = f"demo_{demo_index}"
        if key not in handle["data"]:
            raise ValueError(f"Missing demo reset {key}: {demo_path}")
        state = np.ascontiguousarray(handle["data"][key].attrs["init_state"])
    stock = np.asarray(task_suite.get_task_init_states(task_id))
    if state.shape != stock.shape[1:] or any(np.array_equal(state, row) for row in stock):
        raise ValueError("Demo reset shape/identity overlaps the stock evaluation bank")
    return state, {"source": str(demo_path), "demo_index": demo_index,
                   "suite_directory": suite,
                   "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                   "stock_state_count": len(stock),
                   "different_from_all_stock_states": True}


def run(args: argparse.Namespace) -> dict:
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != "fastwam_four_expert_ab_calibration_bank_v1" or \
            plan.get("job_count") != 80 or len(plan.get("jobs", [])) != 80:
        raise ValueError("Frozen Fast-WAM A/B calibration bank is invalid")
    job_id = f"{args.round}-{args.expert}-task{args.task:02d}"
    matching = [row for row in plan["jobs"] if row["id"] == job_id]
    if len(matching) != 1:
        raise ValueError(f"Job is not unique in the frozen bank: {job_id}")
    job = matching[0]
    if (job["demo_index"] != args.demo_index or job["seed"] != args.seed
            or job["suite"] != SUITES[args.expert]):
        raise ValueError("Requested calibration identity differs from frozen bank")
    sys.path[:0] = [str(HERE), str(FAST / ".python-packages"), str(SOURCE / "src"),
                    str(SOURCE), str(SOURCE / "experiments/libero"),
                    str(RUNTIME / "references/LIBERO-MergeVLA"),
                    str(ROOT / "vla-merge/experiments/claude-firstpass-cause-20260920")]
    import card_flock

    before = gpu_row(args.gpu)
    with ExitStack() as locks:
        if args.inherited_lease:
            if (os.environ.get("FASTWAM_INHERITED_GPU_INDEX") != str(args.gpu) or
                    os.environ.get("FASTWAM_INHERITED_GPU_UUID") != before["uuid"]):
                raise RuntimeError("Inherited GPU lease identity is missing or differs")
        else:
            lease = card_flock.take_card(args.gpu, before["uuid"],
                                        f"fastwam-capture-{args.expert}-{args.task}-{args.demo_index}",
                                        "fastwam-calibration")
            if lease is None:
                raise RuntimeError("GPU has a project lease")
            locks.enter_context(lease)
            legacy = card_flock._try_lock(
                RUNTIME / "resource-leases" / os.uname().nodename / f"gpu-{args.gpu}.lock",
                "fastwam-calibration", {"gpu": args.gpu, "expert": args.expert})
            if legacy is None:
                raise RuntimeError("GPU has a legacy project lease")
            locks.enter_context(legacy)
        admission = gpu_row(args.gpu)
        if admission["uuid"] != before["uuid"] or admission["free_mib"] < args.min_free_mib:
            raise RuntimeError(f"GPU identity/free-memory gate failed: {admission}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["LIBERO_CONFIG_PATH"] = str(FAST.parent / "DreamZero/run/libero_config")
        os.environ["FASTWAM_LIBERO_DEMONSTRATIONS"] = str(
            ROOT / ".datasets/LIBERO/20260919/raw-demonstrations")
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / ".datasets/FastWAM/models"))
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)

        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from libero.libero import benchmark
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        import eval_libero_single as native
        from capture_closed_loop import ClosedLoopEpisodeCapture

        torch.backends.cuda.matmul.allow_tf32 = False
        with initialize_config_dir(version_base="1.3", config_dir=str(SOURCE / "configs")):
            cfg = compose(config_name="sim_libero.yaml", overrides=[
                "task=libero_uncond_2cam224_1e-4",
                f"EVALUATION.task_suite_name={SUITES[args.expert]}",
                f"EVALUATION.task_id={args.task}",
                f"seed={args.seed}"])
        assert not bool(cfg.EVALUATION.visualize_future_video)
        assert int(cfg.EVALUATION.num_inference_steps) == 10
        task_suite = benchmark.get_benchmark_dict()[SUITES[args.expert]]()
        reset, reset_identity = select_demo_reset(task_suite, args.task, args.demo_index)
        if reset_identity["state_sha256"] != job["reset_sha256"]:
            raise ValueError("Demo reset changed since frozen bank")
        checkpoint = FAST / f"weights/{args.expert}/checkpoints/weights/step_{STEPS[args.expert]:06d}.pt"
        stats_path = FAST / f"weights/{args.expert}/dataset_stats.json"
        stats_sha = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        if (str(checkpoint) != job["checkpoint"]["path"] or
                checkpoint.stat().st_size != job["checkpoint"]["size"] or
                checkpoint.stat().st_mtime_ns != job["checkpoint"]["mtime_ns"] or
                stats_sha != job["stats_sha256"]):
            raise ValueError("Checkpoint or normalizer differs from frozen bank")
        model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
        weights = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
        model.mot.load_state_dict(weights["mot"], strict=True)
        if model.proprio_encoder is None or "proprio_encoder" not in weights:
            raise ValueError("Checkpoint lacks proprio encoder")
        model.proprio_encoder.load_state_dict(weights["proprio_encoder"], strict=True)
        del weights
        model = model.to("cuda:0").eval()
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
        native._install_robosuite_mujoco_compatibility()
        task = task_suite.get_task(args.task)
        env, description = native.get_libero_env(task, native.LIBERO_ENV_RESOLUTION, cfg.seed)
        identity = {"suite": SUITES[args.expert], "task_id": args.task,
                    "expert": args.expert, "checkpoint": str(checkpoint),
                    "checkpoint_size": checkpoint.stat().st_size,
                    "stats_path": str(stats_path), "stats_sha256": stats_sha,
                    "seed": args.seed, "reset": reset_identity,
                    "gpu": admission, "calibration_round": args.round,
                    "bank_plan": str(args.plan), "bank_job_id": job_id}
        try:
            with ClosedLoopEpisodeCapture(native, model, args.output, identity) as capture:
                result = native.run_single_episode(
                    env=env, initial_state=reset, task_description=description,
                    model=model, processor=processor, cfg=cfg, episode_idx=0,
                    action_horizon=int(cfg.data.train.num_frames) - 1,
                    input_w=int(cfg.data.train.video_size[1]),
                    input_h=int(cfg.data.train.video_size[0]),
                    model_device="cuda:0")
            sealed = capture.complete(initial_state=reset, success=bool(result[0]))
        finally:
            env.close()
        return {"complete": True, "episode": str(args.output / "episode.json"),
                "requests": sealed["request_count"],
                "selected": sealed["selected_request_indices"],
                "peak_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--expert", choices=sorted(SUITES), required=True)
    parser.add_argument("--task", type=int, choices=range(10), required=True)
    parser.add_argument("--demo-index", type=int, required=True)
    parser.add_argument("--round", choices=("A", "B"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--min-free-mib", type=int, default=40 * 1024)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=RUNTIME /
                        "experiments/claude-fastwam-tcr-20260923/calibration-bank-v1.json")
    parser.add_argument("--inherited-lease", action="store_true")
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    print(json.dumps(run(options), sort_keys=True))
