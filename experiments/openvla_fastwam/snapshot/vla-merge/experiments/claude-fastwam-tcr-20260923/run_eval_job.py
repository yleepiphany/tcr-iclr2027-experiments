#!/usr/bin/env python3
"""Run one frozen Fast-WAM TCR task on native LIBERO stock resets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FAST = ROOT / "vla-merge_table4/Fast-WAM"
SOURCE = FAST / "source"
RUNTIME = ROOT / "vla-merge-runtime"
BASE = RUNTIME / "experiments/claude-fastwam-tcr-20260923"
sys.path[:0] = [str(FAST / ".python-packages"), str(SOURCE / "src"), str(SOURCE),
                str(SOURCE / "experiments/libero"),
                str(RUNTIME / "references/LIBERO-MergeVLA")]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    bank = json.loads(args.bank.read_text())
    if bank.get("schema") != "fastwam_tcr_eval_bank_v1" or len(bank.get("jobs", [])) != 80:
        raise ValueError("Frozen evaluation bank is incomplete")
    jobs = [job for job in bank["jobs"] if job["id"] == args.job_id]
    if len(jobs) != 1:
        raise ValueError("Evaluation job absent or duplicated")
    job = jobs[0]
    pipeline = json.loads((BASE / "pipeline-v1/ended.json").read_text())
    final = json.loads((BASE / "tcr-build-v1/final-manifest.json").read_text())
    if (pipeline.get("complete") is not True or final.get("complete") is not True
            or final.get("is_tcr") is not True
            or final.get("native_reload_action_exact") is not True
            or final.get("checkpoint") != pipeline.get("checkpoint")):
        raise ValueError("TCR build has not passed its final native gate")
    checkpoint = Path(final["checkpoint"]["path"])
    if checkpoint != args.checkpoint or not checkpoint.is_file():
        raise ValueError("Checkpoint differs from accepted TCR build")
    if sha(checkpoint) != final["checkpoint"]["sha256"]:
        raise ValueError("Final checkpoint hash differs")
    stats_path = Path(job["stats_path"])
    if sha(stats_path) != job["stats_sha256"]:
        raise ValueError("Suite normalizer changed")
    if (os.environ.get("FASTWAM_INHERITED_GPU_INDEX") != str(args.gpu)
            or not os.environ.get("FASTWAM_INHERITED_GPU_UUID")):
        raise RuntimeError("Evaluation worker requires inherited GPU lease")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["LIBERO_CONFIG_PATH"] = str(FAST.parent / "DreamZero/run/libero_config")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / ".datasets/FastWAM/models"))
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)

    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from libero.libero import benchmark
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    import eval_libero_single as native

    torch.backends.cuda.matmul.allow_tf32 = False
    with initialize_config_dir(version_base="1.3", config_dir=str(SOURCE / "configs")):
        cfg = compose(config_name="sim_libero.yaml", overrides=[
            "task=libero_uncond_2cam224_1e-4",
            f"EVALUATION.task_suite_name={job['suite']}",
            f"EVALUATION.task_id={job['task_id']}",
            f"seed={job['seed']}"])
    if bool(cfg.EVALUATION.visualize_future_video) or int(cfg.EVALUATION.num_inference_steps) != 10:
        raise ValueError("Native action-only evaluation configuration drift")
    suite = benchmark.get_benchmark_dict()[job["suite"]]()
    states = np.asarray(suite.get_task_init_states(job["task_id"]))
    for index, expected in zip(job["stock_indices"], job["reset_sha256"]):
        observed = hashlib.sha256(np.ascontiguousarray(states[index]).tobytes()).hexdigest()
        if observed != expected:
            raise ValueError("Frozen LIBERO reset changed")
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
    weights = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
    model.mot.load_state_dict(weights["mot"], strict=True)
    if model.proprio_encoder is None or "proprio_encoder" not in weights:
        raise ValueError("TCR checkpoint lacks the proprio encoder")
    model.proprio_encoder.load_state_dict(weights["proprio_encoder"], strict=True)
    del weights
    model = model.to("cuda:0").eval()
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    native._install_robosuite_mujoco_compatibility()
    native.set_global_seed(job["seed"], get_worker_init_fn=False)
    task = suite.get_task(job["task_id"])
    env, description = native.get_libero_env(task, native.LIBERO_ENV_RESOLUTION, cfg.seed)
    args.output.mkdir(parents=True)
    (args.output / "started.json").write_text(json.dumps({
        "job_id": args.job_id, "bank_sha256": sha(args.bank),
        "checkpoint": str(checkpoint), "checkpoint_sha256": final["checkpoint"]["sha256"],
        "stats_sha256": job["stats_sha256"], "gpu": args.gpu,
        "gpu_uuid": os.environ["FASTWAM_INHERITED_GPU_UUID"]}, sort_keys=True) + "\n")
    results = []
    try:
        with (args.output / "episodes.jsonl").open("x") as stream:
            for index, expected in zip(job["stock_indices"], job["reset_sha256"]):
                success, _, _, _ = native.run_single_episode(
                    env=env, initial_state=states[index], task_description=description,
                    model=model, processor=processor, cfg=cfg, episode_idx=index,
                    action_horizon=int(cfg.data.train.num_frames) - 1,
                    input_w=int(cfg.data.train.video_size[1]),
                    input_h=int(cfg.data.train.video_size[0]), model_device="cuda:0")
                row = {"job_id": args.job_id, "stage": job["stage"],
                       "suite": job["suite"], "task_id": job["task_id"],
                       "stock_state_index": index, "state_sha256": expected,
                       "seed": job["seed"], "success": bool(success)}
                results.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
    finally:
        env.close()
    if len(results) != len(job["stock_indices"]):
        raise ValueError("Incomplete evaluation job")
    complete = {"complete": True, "job_id": args.job_id,
                "episodes": len(results), "episodes_sha256": sha(args.output / "episodes.jsonl"),
                "checkpoint_sha256": final["checkpoint"]["sha256"],
                "peak_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2)}
    (args.output / "complete.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=BASE / "evaluation-bank-v1.json")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))
