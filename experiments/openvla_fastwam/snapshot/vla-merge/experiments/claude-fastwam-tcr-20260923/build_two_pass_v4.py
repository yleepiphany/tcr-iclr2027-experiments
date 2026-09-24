#!/usr/bin/env python3
"""Build one static Fast-WAM TCR checkpoint from the frozen A/B bank.

This worker owns no GPU scheduling or success evaluation. It requires the
independent 80-job capture acceptance and a real expert-prefix parity gate.
Every active action-only Linear is solved in forward block order; all other
tensors remain at the fixed Soup prior unless already changed by pass A.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FAST = ROOT / "vla-merge_table4/Fast-WAM"
RUNTIME = ROOT / "vla-merge-runtime"
BASE = RUNTIME / "experiments/claude-fastwam-tcr-20260923"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_records(round_name: str, accepted: dict, torch) -> dict[str, list[dict]]:
    if round_name not in ("A", "B"):
        raise ValueError("Only frozen A/B pools may be used")
    records = {name: [] for name in ("spatial", "object", "goal", "long")}
    found = {name: set() for name in records}
    for job in accepted["jobs_detail"]:
        if job["round"] != round_name:
            continue
        expert = job["expert"]
        task = job["task_id"]
        if expert not in records or task in found[expert]:
            raise ValueError("Duplicate or unfamiliar expert/task calibration cell")
        found[expert].add(task)
        hashes = {row["index"]: row["sha256"] for row in job["request_file_sha256"]}
        if len(hashes) != job["request_count"]:
            raise ValueError("Accepted native request hash list has gaps")
        for index in job["selected_request_indices"]:
            path = BASE / "calibration-v1" / job["id"] / f"request-{index:06d}.pt"
            if sha(path) != hashes[index]:
                raise ValueError(f"Selected native request changed: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["request_index"] != index:
                raise ValueError("Selected request index differs")
            records[expert].append({
                "id": f"{round_name}/{expert}/task-{task:02d}/request-{index:06d}",
                "native_request": payload["native_request"],
                "trace": payload["trace"],
            })
    if any(found[name] != set(range(10)) or len(records[name]) < 10
           for name in records):
        raise ValueError("Frozen A/B expert request coverage is incomplete")
    return records


def load_native_model(torch, cfg, checkpoint: Path):
    from hydra.utils import instantiate

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
    model.mot.load_state_dict(payload["mot"], strict=True)
    if model.proprio_encoder is None or "proprio_encoder" not in payload:
        raise ValueError("Full Fast-WAM checkpoint must include proprio encoder")
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    model = model.to("cuda:0").eval()
    for component in ("mot", "proprio_encoder"):
        current = getattr(model, component).state_dict()
        if set(current) != set(payload[component]) or any(
                not torch.equal(value.detach().cpu(), payload[component][key].to(value.dtype))
                for key, value in current.items()):
            raise ValueError(f"Native loaded tensor values differ: {component}")
    return model


def export_native(torch, model, output: Path, provenance: dict) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mot": {key: value.detach().cpu().clone() for key, value in model.mot.state_dict().items()},
        "proprio_encoder": {key: value.detach().cpu().clone()
                            for key, value in model.proprio_encoder.state_dict().items()},
        "torch_dtype": "torch.bfloat16", "tcr_provenance": provenance,
    }
    tmp = output.with_name(output.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, output)
    del payload
    gc.collect()
    check = torch.load(str(output), map_location="cpu", weights_only=True, mmap=True)
    for component in ("mot", "proprio_encoder"):
        current = getattr(model, component).state_dict()
        if set(check[component]) != set(current) or any(
                not torch.equal(check[component][key], value.detach().cpu())
                for key, value in current.items()):
            raise ValueError("Exported native checkpoint differs from solved model")
    del check
    return {"path": str(output), "sha256": sha(output),
            "size_bytes": output.stat().st_size, "exact_readback": True}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError("One TCR build attempt cannot overwrite or resume")
    acceptance = json.loads(args.acceptance.read_text())
    plan = json.loads(args.block_plan.read_text())
    bank = json.loads((BASE / "calibration-bank-v1.json").read_text())
    parity = json.loads(args.prefix_parity.read_text())
    if (acceptance.get("schema") != "fastwam_ab_calibration_acceptance_v1" or
            acceptance.get("accepted") is not True or acceptance.get("jobs") != 80 or
            acceptance.get("bank_sha256") != sha(BASE / "calibration-bank-v1.json") or
            plan.get("schema") != "fastwam_action_only_tcr_block_plan_v1" or
            plan.get("active_linear_modules") != 613 or plan.get("blocks_per_pass") != 64 or
            parity.get("accepted") is not True or parity.get("source_expert") != "spatial" or
            parity.get("native_selected_prediction_exact") is not True):
        raise ValueError("Frozen Fast-WAM source/capture/prefix gate is incomplete")
    soup = json.loads((RUNTIME / "experiments/openvla-fastwam-diagnostics-20260923/fastwam-soup/manifest.json").read_text())
    if (soup.get("is_tcr") is not False or soup.get("complete") is not True or
            soup["checkpoint"]["sha256"] != plan["source_soup_sha256"]):
        raise ValueError("Fixed uniform-Soup prior changed")
    args.output.mkdir(parents=True)
    save(args.output / "started.json", {
        "calibration_acceptance_sha256": sha(args.acceptance),
        "block_plan_sha256": sha(args.block_plan),
        "prefix_parity_sha256": sha(args.prefix_parity),
        "soup_manifest_sha256": sha(RUNTIME / "experiments/openvla-fastwam-diagnostics-20260923/fastwam-soup/manifest.json"),
        "success_evaluation": False, "no_retry": True, "passes": ["A", "B"],
        "start_unix": time.time()})
    sys.path[:0] = [str(HERE), str(FAST / ".python-packages"),
                    str(FAST / "source/src"), str(FAST / "source")]
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / ".datasets/FastWAM/models"))
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from hydra import compose, initialize_config_dir
    from block_calibration_v3 import calibrate_group

    torch.backends.cuda.matmul.allow_tf32 = False
    with initialize_config_dir(version_base="1.3", config_dir=str(FAST / "source/configs")):
        cfg = compose(config_name="sim_libero.yaml", overrides=[
            "task=libero_uncond_2cam224_1e-4"])
    model = load_native_model(torch, cfg, Path(soup["checkpoint"]["path"]))
    experts = {}
    for name in ("spatial", "object", "goal", "long"):
        row = soup["sources"][name]
        source = Path(row["path"])
        stat = source.stat()
        if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"] or \
                row["sha256"] != plan["expert_checkpoint_sha256"][name]:
            raise ValueError(f"Source expert checkpoint changed: {name}")
        experts[name] = torch.load(str(source), map_location="cpu", weights_only=True, mmap=True)
    torch.cuda.reset_peak_memory_stats()
    prior_checkpoint = Path(soup["checkpoint"]["path"])
    outputs = []
    try:
        for round_name, mass_rule, seed in (("A", "relative", 2026092301),
                                            ("B", "uniform", 2026092302)):
            requests = load_records(round_name, acceptance, torch)
            pass_root = args.output / f"pass-{round_name}"
            pass_root.mkdir()
            reports = []
            for index, group in enumerate(plan["block_order"]):
                report = calibrate_group(
                    model, group, plan["blocks"][group], experts, requests,
                    mass_rule=mass_rule, seed=seed,
                    cap=plan["row_cap_per_selected_request_per_linear"],
                    ridge_multiplier=plan[f"pass_{round_name}"]["ridge_multiplier"],
                    max_correction_ratio=plan[f"pass_{round_name}"]["max_correction_ratio"],
                    device="cuda:0")
                save(pass_root / f"block-{index:02d}.json", report)
                reports.append(report)
            if len(reports) != 64 or sum(len(row["modules"]) for row in reports) != 613:
                raise ValueError("Full Fast-WAM action-only linear scope was not solved")
            checkpoint = export_native(
                torch, model, pass_root / "checkpoint.pt",
                {"pass": round_name, "source": str(prior_checkpoint),
                 "recipe": plan[f"pass_{round_name}"],
                 "blocks": 64, "active_linears": 613})
            result = {"complete": True, "pass": round_name,
                      "checkpoint": checkpoint, "blocks": 64, "active_linears": 613,
                      "realized_rows": sum(row["rows"] for row in reports),
                      "mass_rule": mass_rule, "success_evaluated": False}
            save(pass_root / "manifest.json", result)
            outputs.append(result)
            prior_checkpoint = Path(checkpoint["path"])
            del requests
            gc.collect()
        # Validate a real captured request before and after native model reload.
        verification_records = load_records("B", acceptance, torch)
        sample = verification_records["spatial"][0]["native_request"]
        from replay_merged_prefix import _device
        with torch.no_grad():
            native_before = model.infer_action(**_device(sample, "cuda:0"))["action"]
        del model, experts, verification_records
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = load_native_model(torch, cfg, Path(outputs[-1]["checkpoint"]["path"]))
        with torch.no_grad():
            native_after = reloaded.infer_action(**_device(sample, "cuda:0"))["action"]
        if not torch.equal(native_before, native_after):
            raise ValueError("Fast-WAM final native action differs after checkpoint reload")
        result = {"complete": True, "is_tcr": True, "passes": ["A", "B"],
                  "pass_A": outputs[0], "pass_B": outputs[1],
                  "checkpoint": outputs[1]["checkpoint"],
                  "native_reload_action_exact": True,
                  "active_linear_modules_per_pass": 613,
                  "blocks_per_pass": 64,
                  "inactive_video_generation_head": plan["inactive_video_generation_linear"],
                  "success_evaluated": False, "evaluation_episodes": 0,
                  "peak_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2)}
        save(args.output / "final-manifest.json", result)
        return result
    except BaseException as error:
        save(args.output / "failed.json", {
            "exception": type(error).__name__, "message": str(error),
            "completed_passes": len(outputs), "no_retry": True,
            "checkpoint_accepted": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--block-plan", type=Path, required=True)
    parser.add_argument("--prefix-parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    result = run(options)
    print(json.dumps({"complete": result["complete"], "is_tcr": result["is_tcr"],
                      "checkpoint": result["checkpoint"]["path"]}, sort_keys=True))
