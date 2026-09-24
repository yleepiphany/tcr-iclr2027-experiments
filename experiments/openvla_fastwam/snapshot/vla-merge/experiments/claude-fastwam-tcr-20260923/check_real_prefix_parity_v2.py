#!/usr/bin/env python3
"""Real expert request: original Fast-WAM action equals merged-prefix replay."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FAST = ROOT / "vla-merge_table4/Fast-WAM"
RUNTIME = ROOT / "vla-merge-runtime"
BASE = RUNTIME / "experiments/claude-fastwam-tcr-20260923"
CAPTURE = BASE / "calibration-v1/A-spatial-task00"
BANK = BASE / "calibration-bank-v1.json"


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path[:0] = [str(HERE), str(FAST / ".python-packages"),
                    str(FAST / "source/src"), str(FAST / "source"),
                    str(ROOT / "vla-merge/experiments/claude-firstpass-cause-20260920")]
    import card_flock
    from audit_capture import audit
    from smoke_native import gpu_row

    source_audit = audit(BANK, CAPTURE, "A-spatial-task00")
    if source_audit["selected_request_indices"][0] != 0:
        raise ValueError("Frozen pilot must include its first real request")
    prior = json.loads((BASE / "capture-queue-v1/audits/A-spatial-task00.json").read_text())
    if source_audit != prior:
        raise ValueError("Real-request audit changed")
    before = gpu_row(args.gpu)
    with ExitStack() as stack:
        if args.inherited_lease:
            if (os.environ.get("FASTWAM_INHERITED_GPU_INDEX") != str(args.gpu) or
                    os.environ.get("FASTWAM_INHERITED_GPU_UUID") != before["uuid"]):
                raise RuntimeError("Inherited GPU lease identity is missing or differs")
        else:
            card = card_flock.take_card(args.gpu, before["uuid"],
                                       "fastwam-real-prefix-parity", "fastwam-native-gate")
            if card is None:
                raise RuntimeError("GPU has a project lease")
            stack.enter_context(card)
            legacy = card_flock._try_lock(
                RUNTIME / "resource-leases" / os.uname().nodename / f"gpu-{args.gpu}.lock",
                "fastwam-real-prefix-parity", {"gpu": args.gpu})
            if legacy is None:
                raise RuntimeError("GPU has a legacy project lease")
            stack.enter_context(legacy)
        admission = gpu_row(args.gpu)
        if admission["uuid"] != before["uuid"] or admission["free_mib"] < args.min_free_mib:
            raise RuntimeError("GPU identity/free-memory gate failed")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / ".datasets/FastWAM/models"))
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from hydra import compose, initialize_config_dir
        from build_two_pass_v2 import load_native_model
        from replay_merged_prefix import _device, replay_selected_expert_states

        torch.backends.cuda.matmul.allow_tf32 = False
        with initialize_config_dir(version_base="1.3", config_dir=str(FAST / "source/configs")):
            cfg = compose(config_name="sim_libero.yaml", overrides=[
                "task=libero_uncond_2cam224_1e-4"])
        expert_checkpoint = FAST / "weights/spatial/checkpoints/weights/step_012000.pt"
        model = load_native_model(torch, cfg, expert_checkpoint)
        request_path = CAPTURE / "request-000000.pt"
        row = torch.load(request_path, map_location="cpu", weights_only=True)
        with torch.no_grad():
            actual_action = model.infer_action(**_device(row["native_request"], "cuda:0"))["action"]
            predictions, prefix = replay_selected_expert_states(
                model, row["native_request"], row["trace"], device="cuda:0")
        if not torch.equal(actual_action, row["trace"]["native_action"]):
            raise ValueError("Full original expert action did not reproduce exactly")
        errors = []
        for prediction, captured in zip(predictions, row["trace"]["calls"]):
            expected = captured["prediction"].to(prediction.device)
            difference = float((prediction.float() - expected.float()).abs().max().item())
            errors.append(difference)
            if not torch.equal(prediction, expected):
                raise ValueError("Merged-prefix native selected prediction did not reproduce exactly")
        result = {
            "accepted": True, "source_expert": "spatial",
            "source_checkpoint": str(expert_checkpoint),
            "source_capture": str(request_path),
            "source_capture_sha256": source_audit["request_files"][0]["sha256"],
            "native_full_action_exact": True,
            "native_selected_prediction_exact": True,
            "selected_call_indices": [0, 5, 9],
            "selected_max_absolute_errors": errors,
            "visual_cache_layers": len(prefix["video_kv_cache"]),
            "gpu": admission,
            "peak_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
            "formal_evaluation": False, "calibration_rows_added": 0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--min-free-mib", type=int, default=40 * 1024)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inherited-lease", action="store_true")
    options = parser.parse_args()
    result = run(options)
    print(json.dumps({"accepted": result["accepted"],
                      "native_full_action_exact": result["native_full_action_exact"],
                      "max_errors": result["selected_max_absolute_errors"]}))
