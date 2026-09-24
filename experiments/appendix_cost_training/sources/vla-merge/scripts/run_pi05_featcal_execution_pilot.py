#!/usr/bin/env python3
"""FeatCal + actual specialist execution inputs, isolated from Table 1 artifacts.

Retains baseline Soup/base anchor, alpha/rho/lambda, 418 weight scope, 47-stage
schedule, exact solver, physical-camera quotas and bias fallback. Only the
source of paired call inputs changes. Uses helpers, not the old r7 executable
or its authorization entrypoint. Does not edit or reuse old feature shards.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "pi05_lora_finetune_v2_20260826"
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(SOURCE / "src"), str(SOURCE / "lerobot/src")]

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from scripts import run_pi05_featcal_formal_full as baseline
from vla_merge.featcal_cache_plan import (
    MultiLayerStreamingRowCollector, build_call_slots, build_shard_index,
    sha256_file, tensor_sha256, write_completed_shard_atomic,
)
from vla_merge.featcal_execution import validate_raw_manifest, stack_records, explicit_velocity_forward
from vla_merge.featcal_prefix_chain import SingleStepSelectedRowCollector

TABLE1 = baseline.TABLE1
PILOT = ROOT.parent / "vla-merge-runtime/experiments/featcal-execution-pilot-20260916"
BUILD = PILOT / "build"
EXPERTS = baseline.EXPERT_ORDER
FLOWS = (0, 5, 9)


def event(**values):
    baseline.append_jsonl(BUILD / "progress.jsonl", {"time": time.time(), **values})
    print(json.dumps(values, sort_keys=True), flush=True)


def implementations():
    paths = [Path(__file__), ROOT / "src/vla_merge/featcal_execution.py",
             Path(baseline.__file__), Path(baseline.trace_runner.__file__),
             Path(baseline.teacher_runner.__file__),
             ROOT / "src/vla_merge/featcal_cache_plan.py",
             ROOT / "src/vla_merge/featcal_forward_order.py",
             ROOT / "src/vla_merge/featcal_hybrid_solve.py",
             ROOT / "src/vla_merge/featcal_prefix_chain.py",
             SOURCE / "lerobot/src/lerobot/policies/pi05/modeling_pi05.py"]
    return {str(p.resolve()): sha256_file(p) for p in paths}


def prepare():
    if BUILD.exists():
        raise FileExistsError(BUILD)
    if shutil.disk_usage(PILOT).free < 100 * 1024**3:
        raise RuntimeError("Need at least 100 GiB free for separate FeatCal build")
    models, _ = baseline.trace_runner._model_contract()
    raw = {}
    for expert in EXPERTS:
        path = PILOT / "raw-inputs" / expert / "calls.json"
        manifest = baseline.load_json(path)
        validate_raw_manifest(manifest)
        if manifest["source_policy"] != models["experts"][expert]["path"]:
            raise ValueError("Raw capture does not use the identical 10k dense expert")
        tensor_path = Path(manifest["tensor_file"])
        if sha256_file(tensor_path) != manifest["tensor_sha256"]:
            raise ValueError("Raw capture tensor hash differs")
        raw[expert] = {"manifest": str(path), "manifest_sha256": sha256_file(path),
                       "tensor_file": str(tensor_path), "tensor_sha256": manifest["tensor_sha256"]}
    # Full weight verification is paid once before use, rather than trusting names.
    for item in [models["uniform_soup"], *models["experts"].values()]:
        if sha256_file(Path(item["path"]) / "model.safetensors") != item["model_sha256"]:
            raise ValueError("Input checkpoint hash differs")
    base_file = Path(models["base"]["path"]) / "model.safetensors"
    models["base_file_sha256"] = sha256_file(base_file)
    plan = {"schema_version": 1, "method": "FeatCal + specialist execution inputs",
            "experiment_kind": "exploratory_source_transfer_not_main_table",
            "models": models, "raw": raw, "implementations": implementations(),
            "alpha": .3, "rho": 2., "lambda": .05, "covariance_eps": 1e-8,
            "solved_weights": 418, "bias_policy": "retain Soup, identical to baseline full runner",
            "realized_rows": 4441200, "stages": 47, "calls_per_expert": 150,
            "source_intervention": "demo observations + interpolated flow -> expert observations + native flow",
            "claim_boundary": "Does not isolate outer observation source from inner generation path",
            "raw_teacher_parity": {"max_relative_l2": .02, "max_abs": .1},
            "evaluation": {"kind": "development", "seed": 260101,
                           "init_state_offset": 30, "episodes_per_task": 3,
                           "episodes_total": 120, "formal_evaluation": False},
            "host": socket.gethostname(), "gpus": [6, 7], "allocator_fraction": .35}
    BUILD.mkdir()
    baseline.write_exclusive_json(BUILD / "plan.json", plan)
    plan_sha = sha256_file(BUILD / "plan.json")
    steps = baseline.build_pi05_adapted_linear_forward_plan(baseline.expected_pi05_adapted_weight_keys())
    quotas = baseline.teacher_runner.load_frozen_module_quotas(steps)
    slots = build_call_slots()
    index = build_shard_index(quotas, steps, slots, contract_sha256=plan_sha,
        expert_model_sha256=baseline.trace_runner.EXPERT_MODEL_SHA256,
        initial_student_sha256=baseline.STUDENT_SHA256)
    baseline.write_exclusive_json(BUILD / "shard-index.json", index)
    event(event="prepared", rows=plan["realized_rows"], plan_sha256=plan_sha)


def contract():
    plan = baseline.load_json(BUILD / "plan.json")
    if plan["implementations"] != implementations():
        raise ValueError("Implementation changed after plan freeze")
    if socket.gethostname() != plan["host"] or os.environ.get("CUDA_VISIBLE_DEVICES") not in {"6", "7"}:
        raise ValueError("Pilot is restricted to the user-authorized host/GPU pair")
    torch.cuda.set_per_process_memory_fraction(plan["allocator_fraction"], 0)
    torch.set_num_threads(4)
    for raw in plan["raw"].values():
        if sha256_file(Path(raw["manifest"])) != raw["manifest_sha256"]:
            raise ValueError("Raw input manifest changed")
        if sha256_file(Path(raw["tensor_file"])) != raw["tensor_sha256"]:
            raise ValueError("Raw input tensor changed")
    plan_sha = sha256_file(BUILD / "plan.json")
    # Configure the generic shard/solve helpers with this NEW input identity.
    # The old revision-7 executable/manifest and cache remain untouched.
    baseline.CALL_STATE_MANIFEST_SHA256 = plan_sha
    specs = baseline._spec_map(baseline.load_json(BUILD / "shard-index.json"))
    steps = baseline.build_pi05_adapted_linear_forward_plan(baseline.expected_pi05_adapted_weight_keys())
    quotas = baseline.teacher_runner.load_frozen_module_quotas(steps)
    return plan, plan_sha, specs, steps, quotas


def load_policy(path):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.configs.policies import PreTrainedConfig
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = "cuda"
    cfg.compile_model = False
    cfg.gradient_checkpointing = False
    cfg.pretrained_path = Path(path)
    return PI05Policy.from_pretrained(path, config=cfg).to("cuda").eval()


class RawInputs:
    def __init__(self, plan):
        self.stack = ExitStack()
        self.handles = {e: self.stack.enter_context(safe_open(plan["raw"][e]["tensor_file"], framework="pt", device="cpu")) for e in EXPERTS}
        self.samples = {e: validate_raw_manifest(baseline.load_json(Path(plan["raw"][e]["manifest"]))) for e in EXPERTS}
        self.keys = {e: list(h.keys()) for e, h in self.handles.items()}

    def batch(self, expert, task, flow):
        rows = sorted([r for r in self.samples[expert] if r["task_slot"] == task and r["flow_index"] == flow], key=lambda r: r["task_ordinal"])
        if len(rows) != 5:
            raise ValueError("Need exactly five native requests per flow")
        records = []
        for row in rows:
            prefix = f"sample_{row['index']:03d}."
            records.append({key[len(prefix):]: self.handles[expert].get_tensor(key) for key in self.keys[expert] if key.startswith(prefix)})
        return stack_records(records, "cuda"), rows

    def close(self):
        self.stack.close()


def row_provider(batch):
    return baseline.teacher_runner.build_streaming_row_mask_provider(
        image_masks=[batch[f"image_mask_{i}"] for i in range(3)], token_mask=batch["masks"],
        action_valid_mask=torch.ones(batch["x_t"].shape[:2], device="cuda", dtype=torch.bool))


def slots(expert, task, rows):
    return [f"expert={expert}/task={task:02d}/episode=00/request={r['task_ordinal']//3:02d}/flow={r['flow_index']:02d}" for r in rows]


@torch.inference_mode()
def teachers(names):
    plan, plan_sha, specs, steps, quotas = contract()
    cache = BUILD / "cache"
    raw = RawInputs(plan)
    try:
        for expert in names:
            model = load_policy(plan["models"]["experts"][expert]["path"])
            for task in range(10):
                task_specs = [specs[("teacher", expert, task, step.step)] for step in steps]
                done = [baseline._validated_or_missing(cache, s, plan_sha) for s in task_specs]
                if all(done):
                    continue
                captures = {q.module_path: [] for q in quotas}
                keys = {q.module_path: [] for q in quotas}
                for flow in FLOWS:
                    batch, rows = raw.batch(expert, task, flow)
                    output = {}
                    def forward():
                        output["velocity"] = explicit_velocity_forward(model, batch)
                        return output["velocity"]
                    collector = MultiLayerStreamingRowCollector(model, steps, quotas,
                        call_slot_ids=slots(expert, task, rows),
                        call_ordinals_within_task=[r["task_ordinal"] for r in rows],
                        row_mask_provider=row_provider(batch), selection_seed=baseline.teacher_runner.SELECTION_SEED,
                        max_selected_bytes=2*1024**3)
                    capture = collector.capture(forward)
                    ref = batch["native_velocity"].float()
                    delta = output["velocity"].float() - ref
                    relative = float(delta.norm() / ref.norm().clamp_min(1e-8))
                    maximum = float(delta.abs().max())
                    event(event="teacher_native_parity", expert=expert, task=task, flow=flow,
                          relative_l2=relative, max_abs=maximum)
                    if relative > plan["raw_teacher_parity"]["max_relative_l2"] or maximum > plan["raw_teacher_parity"]["max_abs"]:
                        raise ValueError("Full-joint/native velocity parity gate failed; do not solve")
                    for path in captures:
                        captures[path].append(capture.inputs_by_module[path])
                        keys[path].append(capture.row_keys_by_module[path])
                    del batch, output, collector, capture, ref, delta
                for spec, old in zip(task_specs, done, strict=True):
                    if old is not None:
                        continue
                    tensors = baseline.teacher_runner.assemble_step_tensors(spec, captures, keys)
                    write_completed_shard_atomic(cache, spec, tensors, plan_sha256=plan_sha,
                        call_state_manifest_sha256=plan_sha, source_checkpoint_sha256=spec["source_checkpoint_sha256"])
                event(event="teacher_task_complete", expert=expert, task=task)
                del captures, keys, tensors
                gc.collect()
                torch.cuda.empty_cache()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        baseline.write_exclusive_json(BUILD / f"teachers-{'-'.join(names)}.json", {"status": "complete", "experts": names, "plan_sha256": plan_sha})
    finally:
        raw.close()


def capture_student(model, step, quotas, spec, teacher_keys, raw, expert, task):
    captures = {p: [] for p in step.target_module_paths}
    keys = {p: [] for p in step.target_module_paths}
    for flow in FLOWS:
        batch, rows = raw.batch(expert, task, flow)
        collector = SingleStepSelectedRowCollector(model, step, quotas,
            teacher_row_keys=teacher_keys, call_slot_ids=slots(expert, task, rows),
            call_ordinals_within_task=[r["task_ordinal"] for r in rows],
            row_mask_provider=row_provider(batch), max_selected_bytes=512*1024**2)
        cap = collector.capture(lambda: explicit_velocity_forward(model, batch))
        for path in captures:
            captures[path].append(cap.inputs_by_module[path])
            keys[path].append(cap.row_keys_by_module[path])
        del batch, collector, cap
    tensors = baseline.teacher_runner.assemble_step_tensors(spec, captures, keys)
    for target in spec["targets"]:
        if not torch.equal(tensors[target["row_keys_tensor"]], teacher_keys[target["module_path"]]):
            raise ValueError("Student/expert sampled rows are not paired")
    return tensors


def export(model, paths, prefix, plan_sha):
    output = BUILD / "checkpoint/pretrained_model"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for path in baseline.STUDENT.iterdir():
        if path.name == "model.safetensors" or "manifest" in path.name:
            continue
        if path.is_dir():
            shutil.copytree(path, output / path.name)
        else:
            shutil.copy2(path, output / path.name)
    modules = dict(model.named_modules())
    with safe_open(baseline.STUDENT / "model.safetensors", framework="pt", device="cpu") as soup:
        tensors = {key: (modules[key[:-7]].weight.detach().cpu().contiguous()
                        if key.endswith(".weight") and key[:-7] in paths else soup.get_tensor(key).contiguous())
                   for key in soup.keys()}
    save_file(tensors, output / "model.safetensors", metadata={"format": "pt"})
    receipt = {"method": "FeatCal + expert execution", "experiment_kind": "exploratory",
               "path": str(output), "model_sha256": sha256_file(output / "model.safetensors"),
               "plan_sha256": plan_sha, "final_prefix_sha256": prefix, "target_weights": len(paths),
               "tensor_count": len(tensors), "formal_result": False}
    baseline.write_exclusive_json(output / "featcal_execution_manifest.json", receipt)
    return receipt


@torch.inference_mode()
def solve():
    plan, plan_sha, specs, steps, quotas = contract()
    cache = BUILD / "cache"
    for names in ("spatial-goal", "object-long"):
        done = baseline.load_json(BUILD / f"teachers-{names}.json")
        if done["status"] != "complete" or done["plan_sha256"] != plan_sha:
            raise ValueError("Teacher workers are incomplete")
    raw = RawInputs(plan)
    started = time.monotonic()
    try:
        model = load_policy(str(baseline.STUDENT))
        next_step, prefix, previous = baseline._load_completed_prefix(model, cache)
        quota_map = {q.module_path: q for q in quotas}
        with ExitStack() as stack:
            soup = stack.enter_context(safe_open(baseline.STUDENT / "model.safetensors", framework="pt", device="cpu"))
            base = stack.enter_context(safe_open(Path(plan["models"]["base"]["path"]) / "model.safetensors", framework="pt", device="cpu"))
            experts = {e: stack.enter_context(safe_open(Path(plan["models"]["experts"][e]["path"]) / "model.safetensors", framework="pt", device="cpu")) for e in EXPERTS}
            for step in steps[next_step:]:
                for expert in EXPERTS:
                    for task in range(10):
                        spec = specs[("student", expert, task, step.step)]
                        old = baseline._validated_or_missing(cache, spec, plan_sha)
                        if old is not None:
                            if old["student_prefix_input_sha256"] != prefix:
                                raise ValueError("Student prefix receipt mismatch")
                            continue
                        teacher_spec = specs[("teacher", expert, task, step.step)]
                        teacher_inputs, teacher_keys, _ = baseline._load_shard_inputs(cache, teacher_spec, plan_sha)
                        del teacher_inputs
                        tensors = capture_student(model, step, [quota_map[p] for p in step.target_module_paths],
                                                  spec, teacher_keys, raw, expert, task)
                        write_completed_shard_atomic(cache, spec, tensors, plan_sha256=plan_sha,
                            call_state_manifest_sha256=plan_sha, source_checkpoint_sha256=prefix,
                            student_prefix_input_sha256=prefix,
                            previous_step_solve_receipt=previous["path"] if previous else None,
                            previous_step_solve_receipt_sha256=previous["sha256"] if previous else None)
                        del tensors, teacher_keys
                prefix, previous, _ = baseline._solve_formal_step(cache_root=cache, student=model,
                    step=step, specs=specs, plan_sha256=plan_sha, current_prefix=prefix,
                    previous_solve=previous, soup_handle=soup, base_handle=base, expert_handles=experts)
                event(event="pilot_step_complete", step=step.step, prefix=prefix,
                      elapsed_seconds=time.monotonic()-started)
                gc.collect()
                torch.cuda.empty_cache()
        checkpoint = export(model, {p for step in steps for p in step.target_module_paths}, prefix, plan_sha)
        batch, _ = raw.batch("spatial", 0, 5)
        expected = explicit_velocity_forward(model, batch).cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = load_policy(checkpoint["path"])
        actual = explicit_velocity_forward(reloaded, batch).cpu()
        if not torch.equal(expected, actual):
            raise ValueError("Reload-forward is not bitwise equal")
        baseline.write_exclusive_json(BUILD / "complete.json", {
            "status": "complete", "checkpoint": checkpoint, "reload_bitwise_equal": True,
            "elapsed_seconds": time.monotonic()-started, "formal_evaluation": False})
        event(event="pilot_checkpoint_complete", checkpoint=checkpoint)
    finally:
        raw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "teachers", "solve"], required=True)
    parser.add_argument("--experts", default="spatial,goal")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "teachers":
        names = args.experts.split(",")
        if names not in [["spatial", "goal"], ["object", "long"]]:
            raise ValueError("Use the two fixed non-overlapping teacher partitions")
        teachers(names)
    else:
        solve()
