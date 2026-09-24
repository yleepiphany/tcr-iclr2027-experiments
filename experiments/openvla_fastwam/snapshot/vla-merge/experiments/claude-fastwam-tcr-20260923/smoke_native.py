#!/usr/bin/env python3
"""One real-checkpoint Fast-WAM action-only trace/replay parity gate.

Synthetic input checks the native implementation path only; it is not a
calibration row, a closed-loop trajectory, or an evaluation result.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FAST = ROOT / "vla-merge_table4/Fast-WAM"
RUNTIME = ROOT / "vla-merge-runtime"
CHECKPOINTS = {
    "spatial": FAST / "weights/spatial/checkpoints/weights/step_012000.pt",
    "object": FAST / "weights/object/checkpoints/weights/step_010000.pt",
    "goal": FAST / "weights/goal/checkpoints/weights/step_010000.pt",
    "long": FAST / "weights/long/checkpoints/weights/step_009000.pt",
}


def gpu_row(gpu: int) -> dict:
    line = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=index,uuid,memory.free,memory.used",
        "--format=csv,noheader,nounits"], text=True).strip()
    index, uuid, free, used = [part.strip() for part in line.split(",")]
    return {"index": int(index), "uuid": uuid, "free_mib": int(free),
            "used_mib": int(used)}


def run(args: argparse.Namespace) -> dict:
    sys.path[:0] = [str(HERE), str(FAST / ".python-packages"),
                    str(FAST / "source/src"), str(FAST / "source"),
                    str(ROOT / "vla-merge/experiments/claude-firstpass-cause-20260920")]
    import card_flock

    before = gpu_row(args.gpu)
    lease = card_flock.take_card(args.gpu, before["uuid"],
                                "fastwam-native-parity", "fastwam-smoke")
    if lease is None:
        raise RuntimeError("GPU card already has a project lease")
    with lease:
        legacy = card_flock._try_lock(
            RUNTIME / "resource-leases" / os.uname().nodename / f"gpu-{args.gpu}.lock",
            "fastwam-native-parity", {"job": "fastwam-native-parity", "gpu": args.gpu})
        if legacy is None:
            raise RuntimeError("GPU card already has a legacy project lease")
        with legacy:
            admission = gpu_row(args.gpu)
            if admission["uuid"] != before["uuid"] or admission["free_mib"] < args.min_free_mib:
                raise RuntimeError(f"GPU identity/free-memory gate failed: {admission}")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / ".datasets/FastWAM/models"))
            os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            import torch
            from hydra import compose, initialize_config_dir
            from hydra.utils import instantiate
            from native_trace import NativeActionTrace
            from replay_native_trace import replay_native_trace

            torch.backends.cuda.matmul.allow_tf32 = False
            with initialize_config_dir(version_base="1.3", config_dir=str(FAST / "source/configs")):
                cfg = compose(config_name="sim_libero.yaml", overrides=[
                    "task=libero_uncond_2cam224_1e-4"])
            assert int(cfg.eval_num_inference_steps) == 10
            assert not bool(cfg.EVALUATION.visualize_future_video)
            checkpoint = CHECKPOINTS[args.expert]
            stat = checkpoint.stat()
            model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
            payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
            model.mot.load_state_dict(payload["mot"], strict=True)
            if model.proprio_encoder is None or "proprio_encoder" not in payload:
                raise ValueError("Fast-WAM checkpoint lacks the required proprio encoder")
            model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            del payload
            model = model.to("cuda:0").eval()
            request = {
                "prompt": "Perform the robot manipulation task.",
                "input_image": torch.zeros(1, 3, 224, 448, dtype=torch.bfloat16),
                "action_horizon": 32,
                "proprio": torch.zeros(1, 8, dtype=torch.bfloat16),
                "num_inference_steps": 10,
                "seed": 20260923,
                "rand_device": "cpu",
            }
            with torch.inference_mode():
                with NativeActionTrace(model) as trace:
                    native_action = model.infer_action(**request)["action"]
                native_trace = trace.payload(native_action=native_action)
                replay = replay_native_trace(model, native_trace, device="cuda:0")
                repeat_action = model.infer_action(**request)["action"]
            if not torch.equal(native_action, repeat_action):
                raise ValueError("Native same-input same-noise action changed")
            if checkpoint.stat().st_size != stat.st_size or \
                    checkpoint.stat().st_mtime_ns != stat.st_mtime_ns:
                raise ValueError("Source checkpoint changed during native smoke")
            return {
                "complete": True, "formal_evaluation": False,
                "calibration_row": False, "synthetic_input": True,
                "expert": args.expert, "checkpoint": str(checkpoint),
                "checkpoint_size": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns,
                "gpu": admission, "num_inference_steps": 10,
                "native_action_exact_repeat": True,
                "replay": replay,
                "peak_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--expert", choices=sorted(CHECKPOINTS), default="spatial")
    parser.add_argument("--min-free-mib", type=int, default=40 * 1024)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    result = run(options)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"complete": result["complete"], "gpu": result["gpu"],
                      "peak_allocated_mib": result["peak_allocated_mib"],
                      "replay": result["replay"]}, sort_keys=True))
