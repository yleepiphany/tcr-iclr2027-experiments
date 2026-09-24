"""One native request and one Linear solve before the bounded two-pass build."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FAST = ROOT / "vla-merge_table4/Fast-WAM"
sys.path[:0] = [str(HERE), str(ROOT / "vla-merge/experiments/claude-fastwam-tcr-20260923"),
                str(ROOT / "vla-merge/experiments/fastwam-local-diagnosis-20260924"),
                str(FAST / ".python-packages"), str(FAST / "source/src"), str(FAST / "source")]
import build_two_pass_fixed as backend
import block_calibration_v4 as block
import pi05_solver_bridge as bridge


def main(args):
    if os.environ.get("FASTWAM_FIXED_INTERFACE_AUTH") != "local-4-7-offline-v1":
        raise RuntimeError("Missing frozen local launch")
    import torch
    from hydra import compose, initialize_config_dir
    from replay_merged_prefix import _device

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(40 * 1024**3 /
                                               torch.cuda.get_device_properties(0).total_memory)
    with initialize_config_dir(version_base="1.3", config_dir=str(FAST / "source/configs")):
        cfg = compose(config_name="sim_libero.yaml", overrides=["task=libero_uncond_2cam224_1e-4"])
    if int(cfg.EVALUATION.replan_steps) != 10:
        raise ValueError("Native execution prefix differs")
    plan = json.loads(args.block_plan.read_text())
    acceptance = json.loads(args.acceptance.read_text())
    soup = json.loads((backend.RUNTIME / "experiments/openvla-fastwam-diagnostics-20260923/fastwam-soup/manifest.json").read_text())
    if plan["active_linear_modules"] != 613 or plan["blocks_per_pass"] != 64 or not acceptance["accepted"]:
        raise ValueError("Frozen source contract differs")
    model = backend.load_native_model(torch, cfg, Path(soup["checkpoint"]["path"]))
    experts = {name: torch.load(soup["sources"][name]["path"], map_location="cpu",
                                weights_only=True, mmap=True)
               for name in ("spatial", "object", "goal", "long")}
    spatial = experts["spatial"]
    backend.copy_fixed_interfaces(torch, model, spatial)
    views = backend.fixed_expert_views(experts)
    group = "03-action-pre"
    descriptors = [d for d in plan["blocks"][group]
                   if d["module"] == "mot.mixtures.action.text_embedding.0"]
    if len(descriptors) != 1:
        raise ValueError("Smoke Linear scope differs")
    for name in views:
        with block.expert_group_state(model, group, views[name]):
            backend.verify_fixed_interfaces(torch, model, spatial)
        backend.verify_fixed_interfaces(torch, model, spatial)
    requests = {name: rows[:1] for name, rows in backend.load_records("A", acceptance, torch).items()}
    sample = requests["spatial"][0]["native_request"]
    with torch.inference_mode():
        before = model.infer_action(**_device(sample, "cuda:0"))["action"]
    if tuple(before.shape[-2:]) != (32, 7):
        raise ValueError("Native action must be [32,7]")
    original = block.solve_module
    block.solve_module = bridge.solve_module
    try:
        report = block.calibrate_group(model, group, descriptors, views, requests,
                                       mass_rule="relative", seed=2026092301,
                                       cap=plan["row_cap_per_selected_request_per_linear"],
                                       ridge_multiplier=.05, max_correction_ratio=3., device="cuda:0")
    finally:
        block.solve_module = original
    backend.verify_fixed_interfaces(torch, model, spatial)
    if not report["complete"] or len(report["modules"]) != 1:
        raise ValueError("Smoke solve incomplete")
    with torch.inference_mode():
        after = model.infer_action(**_device(sample, "cuda:0"))["action"]
    if tuple(after.shape[-2:]) != (32, 7) or not torch.isfinite(after).all():
        raise ValueError("Post-solve native action invalid")
    backend.save(args.output, {"accepted": True, "one_native_request_per_expert": True,
                               "one_linear_solved": descriptors[0]["module"],
                               "interface_equal_during_all_four_swaps": True,
                               "interface_equal_after_solve": True,
                               "native_action_shape": [32, 7], "executed_prefix": 10,
                               "ridge": report["modules"][descriptors[0]["module"]]["ridge"],
                               "peak_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2)})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--acceptance", type=Path, required=True)
    p.add_argument("--block-plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
