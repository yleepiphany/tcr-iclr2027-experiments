#!/usr/bin/env python3
"""Run one manifest-authorized Spatial/task0 FeatCal teacher mini cache benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "pi05_lora_finetune_v2_20260826"
RUNTIME = ROOT.parent / "vla-merge-runtime"
TABLE1 = RUNTIME / "experiments/iclr2027-table1-20260910"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(SOURCE / "lerobot/src"))

from scripts import build_iclr2027_featcal_call_state_manifest as state_builder  # noqa: E402
from scripts import run_pi05_featcal_forward_order_engineering as trace_runner  # noqa: E402
from vla_merge.featcal_cache_plan import (  # noqa: E402
    MultiLayerStreamingRowCollector,
    derive_module_quotas,
    scan_resume_state,
    sha256_file,
    write_completed_shard_atomic,
)
from vla_merge.featcal_forward_order import (  # noqa: E402
    build_pi05_adapted_linear_forward_plan,
    expected_pi05_adapted_weight_keys,
)


ATTEMPT_NAME = "featcal-spatial-task0-15state-teacher-mini-v2"
EXPECTED_TABLE1_REVISION = 5
EXPERT_BANK = TABLE1 / "expert-bank-manifest.json"
EXPERT_BANK_SHA256 = "295efdfca18b0c83497bfe499e0ad771a5a8b33aaf86ba424dab7607094d53c1"
FORMAL_PLAN_ROOT = TABLE1 / "preflight/featcal-formal-cache-plan-v3"
FORMAL_PLAN = FORMAL_PLAN_ROOT / "plan.json"
FORMAL_PLAN_SHA256 = "876367b0e38ba54bd75f3781b6e3ac85e072151730dcbc899cfd6493d19e81bd"
FORMAL_INDEX = FORMAL_PLAN_ROOT / "shard-index.json"
FORMAL_INDEX_SHA256 = "f955bec8267252aebfd838394ccdc7bef73273dafd2fa0b538aaa1b4d3698fd3"
CALL_STATE_MANIFEST = (
    TABLE1
    / "preflight/featcal-call-states-training-demo-v1/artifacts/manifest.json"
)
CALL_STATE_MANIFEST_SHA256 = "b18b82300036ded14caa7c6431015acc84368606460437e495c13f424a9ad285"
SELECTION_SEED = 2026091204
GPU_LOCK = RUNTIME / "resource-leases/gpu-6.lock"
SELECTED_EXPERT = "spatial"
SELECTED_TASK = 0
SELECTED_EPISODE = 1261
EXPECTED_BATCH_SIZE = 5
EXPECTED_BATCHES = 3
EXPECTED_SHARDS = 47
STATE_BUILDER_SHA256 = "cb61109fdff885f441da4240a9f576d21dfb118a2a96d6d0e2a53832ddbb2107"
TRACE_RUNNER_SHA256 = "32270f3e15b783738a0996ea1e5b118d43fa07ff882fb84d1fc4780708471ed7"
FORWARD_ORDER_SHA256 = "712282ed7d9fad08c20f55e4330b2b29545ac5180f41a80301c36279762c9ede"
CACHE_PLAN_SHA256 = "4b5aa596dd39b857b5da60fdd3a4080233153ce32f04cab660cc844e42f67d13"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _verify_frozen_inputs(
    table1_manifest: Path,
    expected_table1_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    table1_manifest = table1_manifest.expanduser().resolve()
    if len(expected_table1_manifest_sha256) != 64:
        raise ValueError("Expected Table-1 manifest SHA256 is invalid")
    identities = {
        table1_manifest: expected_table1_manifest_sha256,
        EXPERT_BANK: EXPERT_BANK_SHA256,
        FORMAL_PLAN: FORMAL_PLAN_SHA256,
        FORMAL_INDEX: FORMAL_INDEX_SHA256,
        CALL_STATE_MANIFEST: CALL_STATE_MANIFEST_SHA256,
        ROOT / "scripts/build_iclr2027_featcal_call_state_manifest.py": STATE_BUILDER_SHA256,
        ROOT / "scripts/run_pi05_featcal_forward_order_engineering.py": TRACE_RUNNER_SHA256,
        ROOT / "src/vla_merge/featcal_forward_order.py": FORWARD_ORDER_SHA256,
        ROOT / "src/vla_merge/featcal_cache_plan.py": CACHE_PLAN_SHA256,
    }
    for path, expected in identities.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen mini input differs: {path}")
    revision = load_json(table1_manifest)
    authorization = revision.get("featcal_engineering_revision5", {}).get("authorization", {})
    if (
        revision.get("immutable_revision") != EXPECTED_TABLE1_REVISION
        or authorization.get("authorized") is not True
        or authorization.get("authorization_count") != 1
        or authorization.get("attempt") != ATTEMPT_NAME
        or authorization.get("expert") != SELECTED_EXPERT
        or authorization.get("local_task") != SELECTED_TASK
        or authorization.get("episode_index") != SELECTED_EPISODE
        or authorization.get("flow_indices") != [0, 5, 9]
        or authorization.get("call_state_count") != 15
        or authorization.get("batch_size") != EXPECTED_BATCH_SIZE
        or authorization.get("batches") != EXPECTED_BATCHES
        or authorization.get("student_collection") is not False
        or authorization.get("solve") is not False
        or authorization.get("checkpoint") is not False
        or authorization.get("rollout") is not False
    ):
        raise ValueError("Table-1 revision-5 mini authorization differs")
    states = load_json(CALL_STATE_MANIFEST)
    mini_states = [
        row
        for row in states["states"]
        if row["expert"] == SELECTED_EXPERT and row["task"] == SELECTED_TASK
    ]
    if len(mini_states) != 15:
        raise ValueError("Authorized mini call-state count differs")
    if (
        [row["slot_id"] for row in mini_states] != authorization.get("call_slot_ids")
        or {row["episode_identity"]["episode_index"] for row in mini_states}
        != {SELECTED_EPISODE}
        or {row["suite_task_id"] for row in mini_states} != {30}
        or {row["source_expert_sha256"] for row in mini_states}
        != {states["expert_model_sha256"][SELECTED_EXPERT]}
    ):
        raise ValueError("Authorized mini concrete state identities differ")
    index = load_json(FORMAL_INDEX)
    mini_shards = [
        row
        for row in index["shards"]
        if row["role"] == "teacher"
        and row["expert"] == SELECTED_EXPERT
        and row["task"] == SELECTED_TASK
    ]
    if len(mini_shards) != EXPECTED_SHARDS:
        raise ValueError("Authorized mini shard count differs")
    if (
        [row["step"] for row in mini_shards] != list(range(EXPECTED_SHARDS))
        or any(row["call_slot_ids"] != authorization["call_slot_ids"] for row in mini_shards)
        or any(row["source_checkpoint_sha256"] != mini_states[0]["source_expert_sha256"] for row in mini_shards)
    ):
        raise ValueError("Authorized mini shard identities differ")
    return authorization, {"manifest": states, "mini_states": mini_states}, {
        "formal": index,
        "mini_shards": mini_shards,
    }


def build_plan(
    attempt_root: Path,
    *,
    table1_manifest: Path,
    expected_table1_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    table1_manifest = table1_manifest.expanduser().resolve()
    authorization, state_data, index_data = _verify_frozen_inputs(
        table1_manifest,
        expected_table1_manifest_sha256,
    )
    script = Path(__file__).resolve()
    cache_module = ROOT / "src/vla_merge/featcal_cache_plan.py"
    mini_index = {
        "schema_version": 1,
        "kind": "pi05_featcal_teacher_mini_shard_index",
        "formal_index_sha256": FORMAL_INDEX_SHA256,
        "shard_count": len(index_data["mini_shards"]),
        "shards": index_data["mini_shards"],
    }
    plan = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_single_authorized_teacher_mini_not_executed",
        "attempt": ATTEMPT_NAME,
        "attempt_root": str(attempt_root.resolve()),
        "claim_boundary": (
            "Teacher-only Spatial/task0 engineering cache benchmark; no student, solve, "
            "checkpoint, candidate, score, or rollout. Failure is preserved and not rerun."
        ),
        "authorization": authorization,
        "frozen_inputs": {
            "table1_manifest": str(table1_manifest),
            "table1_manifest_sha256": expected_table1_manifest_sha256,
            "table1_immutable_revision": EXPECTED_TABLE1_REVISION,
            "expert_bank_sha256": EXPERT_BANK_SHA256,
            "formal_plan_sha256": FORMAL_PLAN_SHA256,
            "formal_index_sha256": FORMAL_INDEX_SHA256,
            "call_state_manifest_sha256": CALL_STATE_MANIFEST_SHA256,
            "spatial_expert_sha256": state_data["mini_states"][0]["source_expert_sha256"],
        },
        "scope": {
            "expert": SELECTED_EXPERT,
            "task": SELECTED_TASK,
            "episode": SELECTED_EPISODE,
            "requests": list(range(5)),
            "flow_indices": [0, 5, 9],
            "call_states": 15,
            "batch_size": EXPECTED_BATCH_SIZE,
            "batches": EXPECTED_BATCHES,
            "modules": 418,
            "steps": 47,
            "teacher_only": True,
        },
        "selection_seed": SELECTION_SEED,
        "memory": {
            "selected_only_cap_bytes": 2 * 1024**3,
            "expected_task_factor_bytes": sum(
                row["factor_bytes_float32"] for row in index_data["mini_shards"]
            ),
            "expected_task_row_key_bytes": sum(
                row["row_key_bytes_int64"] for row in index_data["mini_shards"]
            ),
        },
        "implementation": {
            "executor": str(script),
            "executor_sha256": sha256_file(script),
            "cache_streaming_module": str(cache_module.resolve()),
            "cache_streaming_module_sha256": sha256_file(cache_module),
            "forward_order_module_sha256": sha256_file(
                ROOT / "src/vla_merge/featcal_forward_order.py"
            ),
            "call_state_builder_sha256": sha256_file(
                ROOT / "scripts/build_iclr2027_featcal_call_state_manifest.py"
            ),
            "engineering_helper_sha256": sha256_file(
                ROOT / "scripts/run_pi05_featcal_forward_order_engineering.py"
            ),
        },
        "outputs_allowed": authorization["allowed_outputs"],
        "student_collection": False,
        "solve": False,
        "checkpoint": False,
        "rollout": False,
        "table1_revision5_modified": False,
    }
    return plan, mini_index


def prepare_attempt(
    attempt_root: Path,
    *,
    table1_manifest: Path,
    expected_table1_manifest_sha256: str,
) -> dict[str, Any]:
    attempt_root = attempt_root.expanduser().absolute()
    if attempt_root.name != ATTEMPT_NAME:
        raise ValueError(f"Attempt root must end in {ATTEMPT_NAME}")
    if attempt_root.exists():
        raise FileExistsError(attempt_root)
    plan, index = build_plan(
        attempt_root,
        table1_manifest=table1_manifest,
        expected_table1_manifest_sha256=expected_table1_manifest_sha256,
    )
    attempt_root.mkdir(parents=True)
    index_path = attempt_root / "shard-index.json"
    plan_path = attempt_root / "plan.json"
    write_exclusive_json(index_path, index)
    plan["shard_index"] = {"path": str(index_path), "sha256": sha256_file(index_path)}
    write_exclusive_json(plan_path, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def _gpu_admission() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise ValueError("CUDA_VISIBLE_DEVICES must be exactly 6")
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "6",
            "--query-gpu=memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    memory_mib, utilization, power_watts = (
        float(value.strip()) for value in query.split(",")
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "6",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if processes:
        raise RuntimeError(f"GPU 6 has compute processes before admission: {processes}")
    if memory_mib > 64 or utilization > 5:
        raise RuntimeError(
            f"GPU 6 is not idle: memory_mib={memory_mib}, utilization={utilization}"
        )
    return {
        "memory_used_mib": memory_mib,
        "utilization_percent": utilization,
        "power_watts": power_watts,
        "compute_processes": [],
    }


def _all_target_hook_count(root, forward_plan) -> int:
    modules = dict(root.named_modules())
    targets = {
        path for step in forward_plan for path in step.target_module_paths
    }
    return sum(len(modules[path]._forward_pre_hooks) for path in targets)


def load_frozen_module_quotas(forward_plan):
    """Load quota shapes from the logical expert contract, never the dense index."""

    if not EXPERT_BANK.is_file() or sha256_file(EXPERT_BANK) != EXPERT_BANK_SHA256:
        raise ValueError("Frozen logical expert-bank contract differs")
    quotas = derive_module_quotas(load_json(EXPERT_BANK), forward_plan)
    if len(quotas) != 418:
        raise ValueError("Frozen logical expert-bank quota cardinality differs")
    return quotas


def build_streaming_row_mask_provider(
    *,
    image_masks: list[torch.Tensor],
    token_mask: torch.Tensor,
    action_valid_mask: torch.Tensor,
):
    """Build the frozen physical-vision/valid-language/action mask contract."""

    from vla_merge.featcal_forward_order import EXPERT_ROOT, LANGUAGE_ROOT

    frozen_image_masks = [value.detach().cpu().bool() for value in image_masks]
    frozen_token_mask = token_mask.detach().cpu().bool()
    frozen_action_mask = action_valid_mask.detach().cpu().bool()

    def provider(path: str, occurrence: int, value: torch.Tensor) -> torch.Tensor:
        shape = tuple(value.shape[:-1])
        batch = shape[0]
        if ".vision_tower.vision_model.encoder.layers." in path:
            # The registered 4500-row quota is over all three physical calls,
            # including the zero-image placeholder camera.  Masking it here
            # would silently change the preregistered quota contract.
            if len(shape) != 2 or occurrence not in (0, 1, 2):
                raise ValueError(f"Vision streaming row contract differs: {path} {shape}")
            return torch.ones(shape, dtype=torch.bool)
        if path.startswith(LANGUAGE_ROOT):
            if len(shape) != 2 or batch != frozen_token_mask.shape[0]:
                raise ValueError(f"Language streaming row contract differs: {path} {shape}")
            visual_rows = shape[1] - frozen_token_mask.shape[1]
            if visual_rows <= 0 or visual_rows % len(frozen_image_masks) != 0:
                raise ValueError("Language prefix cannot be split into image/token rows")
            rows_per_image = visual_rows // len(frozen_image_masks)
            pieces = [
                mask[:, None].expand(batch, rows_per_image)
                for mask in frozen_image_masks
            ]
            return torch.cat([*pieces, frozen_token_mask], dim=1)
        if path.startswith(EXPERT_ROOT) or path in (
            "model.action_in_proj",
            "model.action_out_proj",
        ):
            if tuple(frozen_action_mask.shape) != shape:
                raise ValueError(f"Action streaming row contract differs: {path} {shape}")
            return frozen_action_mask
        if path in ("model.time_mlp_in", "model.time_mlp_out"):
            if shape != (batch,):
                raise ValueError(f"Time streaming row contract differs: {path} {shape}")
            return torch.ones(shape, dtype=torch.bool)
        raise ValueError(f"No streaming row-mask rule for {path}")

    return provider


def _load_and_verify_minibatch(states: list[dict[str, Any]]):
    """Decode the five requests and reproduce every frozen CPU tensor hash."""

    from lerobot.policies.common.vla_utils import pad_vector
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    from lerobot.utils.collate import lerobot_collate_fn

    request_rows = {request: [] for request in range(5)}
    for state in states:
        request_rows[int(state["request"])].append(state)
    if any(
        [row["flow_index"] for row in rows] != [0, 5, 9]
        for rows in request_rows.values()
    ):
        raise ValueError("Mini state request/flow order differs")

    dataset, preprocessor = state_builder._dataset_and_preprocessor([SELECTED_EPISODE])
    frame_table = dataset.hf_dataset.select_columns(
        ["episode_index", "frame_index"]
    ).to_pandas()
    camera_keys = list(dataset.meta.camera_keys)
    if camera_keys != ["observation.images.image", "observation.images.image2"]:
        raise ValueError(f"LIBERO camera keys differ: {camera_keys}")

    items = []
    raw_action_hashes = []
    dataset_indices = []
    for request in range(5):
        expected = request_rows[request][0]
        identity = expected["episode_identity"]
        matches = frame_table.index[
            (frame_table.episode_index == identity["episode_index"])
            & (frame_table.frame_index == identity["frame_index"])
        ].tolist()
        if len(matches) != 1:
            raise ValueError(f"Mini frame lookup differs for request {request}: {matches}")
        item = dataset[int(matches[0])]
        if (
            int(item["index"]) != identity["dataset_index"]
            or item["task"] != expected["task_text"]
            or int(item["task_index"]) != expected["suite_task_id"]
            or tuple(item["action"].shape) != (50, 7)
            or int(item["action_is_pad"].sum()) != 0
        ):
            raise ValueError(f"Mini dataset item contract differs for request {request}")
        raw_hash = trace_runner.tensor_sha256(item["action"].cpu().contiguous())
        if raw_hash != expected["action_chunk_sha256"]:
            raise ValueError(f"Mini raw action hash differs for request {request}")
        items.append(item)
        raw_action_hashes.append(raw_hash)
        dataset_indices.append(int(item["index"]))

    batch = lerobot_collate_fn(items)
    batch = _preprocess_dataset_batch(batch, camera_keys, {}, preprocessor)
    padded_action = pad_vector(batch["action"], 32).cpu().contiguous()
    if tuple(padded_action.shape) != (5, 50, 32):
        raise ValueError(f"Mini padded action shape differs: {tuple(padded_action.shape)}")

    noises = []
    request_hashes = []
    for request in range(5):
        rows = request_rows[request]
        expected = rows[0]
        image_bundle = {key: batch[key][request : request + 1].cpu().contiguous() for key in camera_keys}
        image_bundle.update(
            {
                "image_mask_0": torch.tensor([True]),
                "image_mask_1": torch.tensor([True]),
                "image_mask_2": torch.tensor([False]),
            }
        )
        language_bundle = {
            "tokens": batch["observation.language.tokens"][request : request + 1]
            .cpu()
            .contiguous(),
            "mask": batch["observation.language.attention_mask"][request : request + 1]
            .cpu()
            .contiguous(),
        }
        image_hash = state_builder.hash_tensor_bundle(image_bundle)
        language_hash = state_builder.hash_tensor_bundle(language_bundle)
        padded_hash = trace_runner.tensor_sha256(padded_action[request : request + 1])
        generator = torch.Generator(device="cpu").manual_seed(int(expected["noise_seed"]))
        noise = torch.randn((1, 50, 32), generator=generator, dtype=torch.float32)
        noise_hash = trace_runner.tensor_sha256(noise)
        observed = {
            "processed_images_and_masks_sha256": image_hash,
            "language_tokens_and_mask_sha256": language_hash,
            "padded_normalized_action_sha256": padded_hash,
            "noise_sha256": noise_hash,
        }
        for row in rows:
            if any(row[key] != value for key, value in observed.items()):
                raise ValueError(f"Frozen request tensor hash differs: {row['slot_id']}")
            timestep_value = float(row["timestep"])
            timestep = torch.tensor([timestep_value], dtype=torch.float32)
            x_t = timestep_value * noise + (1.0 - timestep_value) * padded_action[
                request : request + 1
            ]
            if (
                trace_runner.tensor_sha256(timestep) != row["timestep_sha256"]
                or trace_runner.tensor_sha256(x_t) != row["x_t_sha256"]
            ):
                raise ValueError(f"Frozen flow tensor hash differs: {row['slot_id']}")
        noises.append(noise)
        request_hashes.append({"request": request, **observed})

    return {
        "dataset": dataset,
        "batch_cpu": batch,
        "padded_action_cpu": padded_action,
        "noise_cpu": torch.cat(noises, dim=0),
        "request_rows": request_rows,
        "request_hashes": request_hashes,
        "raw_action_hashes": raw_action_hashes,
        "dataset_indices": dataset_indices,
        "camera_keys": camera_keys,
    }


def _prepare_core_for_flow(
    policy, batch_cpu, padded_action_cpu, noise_cpu, timestep: float
):
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    batch = {
        key: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for key, value in batch_cpu.items()
    }
    images, image_masks = policy._preprocess_images(batch)
    states, state_masks = policy._prepare_memory_states(batch)
    actions = policy.prepare_action(batch)
    noise = noise_cpu.to(device="cuda", dtype=torch.float32)
    time_tensor = torch.full(
        (actions.shape[0],), float(timestep), device="cuda", dtype=torch.float32
    )
    if tuple(actions.shape) != (5, 50, 32) or tuple(noise.shape) != tuple(actions.shape):
        raise ValueError("Mini core action/noise shape differs")
    if trace_runner.tensor_sha256(actions.detach().cpu()) != trace_runner.tensor_sha256(
        padded_action_cpu
    ):
        raise ValueError("Mini core prepared action differs from frozen CPU action")
    if not all(torch.isfinite(value).all() for value in (actions, noise, time_tensor)):
        raise ValueError("Mini core action/noise/time is non-finite")
    action_valid = ~batch["action_is_pad"].bool()
    return {
        "core": {
            "images": images,
            "img_masks": image_masks,
            "tokens": batch[OBS_LANGUAGE_TOKENS],
            "masks": batch[OBS_LANGUAGE_ATTENTION_MASK],
            "actions": actions,
            "noise": noise,
            "time": time_tensor,
            "prefix_mask": None,
            "states": states,
            "state_masks": state_masks,
        },
        "action_valid_mask": action_valid,
    }


def assemble_step_tensors(
    spec: dict[str, Any], captures_by_module: dict[str, list[torch.Tensor]],
    keys_by_module: dict[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for target in spec["targets"]:
        path = target["module_path"]
        if path not in captures_by_module or path not in keys_by_module:
            raise ValueError(f"Mini capture target is missing: {path}")
        inputs = torch.cat(captures_by_module[path], dim=0).contiguous()
        row_keys = torch.cat(keys_by_module[path], dim=0).contiguous()
        if tuple(inputs.shape) != (
            int(target["expected_rows"]),
            int(target["input_width"]),
        ):
            raise ValueError(f"Mini capture input shape differs: {path} {tuple(inputs.shape)}")
        tensors[target["input_tensor"]] = inputs
        tensors[target["row_keys_tensor"]] = row_keys
    return tensors


def execute_attempt(attempt_root: Path) -> dict[str, Any]:
    attempt_root = attempt_root.expanduser().resolve()
    plan_path = attempt_root / "plan.json"
    index_path = attempt_root / "shard-index.json"
    receipt_path = attempt_root / "execution-receipt.json"
    progress_path = attempt_root / "progress.jsonl"
    if not plan_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("Prepared mini plan/index is missing")
    if receipt_path.exists() or progress_path.exists() or (attempt_root / "cache").exists():
        raise FileExistsError("Mini execution outputs already exist; rerun is forbidden")
    plan = load_json(plan_path)
    index = load_json(index_path)
    if (
        plan.get("attempt") != ATTEMPT_NAME
        or plan.get("implementation", {}).get("executor_sha256")
        != sha256_file(Path(__file__).resolve())
        or plan.get("implementation", {}).get("cache_streaming_module_sha256")
        != sha256_file(ROOT / "src/vla_merge/featcal_cache_plan.py")
        or plan.get("implementation", {}).get("forward_order_module_sha256")
        != sha256_file(ROOT / "src/vla_merge/featcal_forward_order.py")
        or plan.get("implementation", {}).get("call_state_builder_sha256")
        != sha256_file(ROOT / "scripts/build_iclr2027_featcal_call_state_manifest.py")
        or plan.get("implementation", {}).get("engineering_helper_sha256")
        != sha256_file(ROOT / "scripts/run_pi05_featcal_forward_order_engineering.py")
        or plan.get("shard_index", {}).get("sha256") != sha256_file(index_path)
        or index.get("shard_count") != EXPECTED_SHARDS
    ):
        raise ValueError("Prepared mini implementation/index identity differs")

    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = GPU_LOCK.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_stream.close()
        raise RuntimeError(f"GPU 6 lock is busy: {GPU_LOCK}") from error
    try:
        admission = _gpu_admission()
        append_jsonl(
            progress_path,
            {
                "event": "execution_started",
                "at": datetime.now(timezone.utc).isoformat(),
                "gpu_admission": admission,
            },
        )
        print(json.dumps({"gpu_admission": admission}, sort_keys=True), flush=True)
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise RuntimeError("CUDA visibility/device is not exactly one A100")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

        frozen = plan["frozen_inputs"]
        authorization, state_data, _ = _verify_frozen_inputs(
            Path(frozen["table1_manifest"]),
            frozen["table1_manifest_sha256"],
        )
        forward_plan = build_pi05_adapted_linear_forward_plan(
            expected_pi05_adapted_weight_keys()
        )
        quotas = load_frozen_module_quotas(forward_plan)
        _, matching_expert_path = trace_runner._model_contract()
        mini = _load_and_verify_minibatch(state_data["mini_states"])
        expert = trace_runner._load_policy(matching_expert_path, mini["dataset"].meta)
        hooks_before = _all_target_hook_count(expert, forward_plan)
        if hooks_before != 0:
            raise ValueError(f"Expert starts with unexpected hooks: {hooks_before}")

        captures_by_module: dict[str, list[torch.Tensor]] = {
            quota.module_path: [] for quota in quotas
        }
        keys_by_module: dict[str, list[torch.Tensor]] = {
            quota.module_path: [] for quota in quotas
        }
        flow_receipts = []
        output_hashes = []
        started = time.monotonic()
        for flow_index in (0, 5, 9):
            rows = [
                mini["request_rows"][request][(0, 5, 9).index(flow_index)]
                for request in range(5)
            ]
            core_data = _prepare_core_for_flow(
                expert,
                mini["batch_cpu"],
                mini["padded_action_cpu"],
                mini["noise_cpu"],
                float(rows[0]["timestep"]),
            )
            if any(float(row["timestep"]) != float(rows[0]["timestep"]) for row in rows):
                raise ValueError("Flow minibatch timestep differs across requests")
            row_mask_provider = build_streaming_row_mask_provider(
                image_masks=core_data["core"]["img_masks"],
                token_mask=core_data["core"]["masks"],
                action_valid_mask=core_data["action_valid_mask"],
            )
            output_holder: dict[str, torch.Tensor] = {}

            def forward():
                output = trace_runner._core_forward(expert, core_data["core"])
                output_holder["value"] = output.detach().cpu()
                return output

            collector = MultiLayerStreamingRowCollector(
                expert,
                forward_plan,
                quotas,
                call_slot_ids=[row["slot_id"] for row in rows],
                call_ordinals_within_task=[int(row["task_ordinal"]) for row in rows],
                row_mask_provider=row_mask_provider,
                selection_seed=SELECTION_SEED,
                max_selected_bytes=int(plan["memory"]["selected_only_cap_bytes"]),
            )
            capture = collector.capture(forward)
            hooks_after_flow = _all_target_hook_count(expert, forward_plan)
            if hooks_after_flow != hooks_before:
                raise ValueError(f"Streaming hooks leaked after flow {flow_index}")
            for path in captures_by_module:
                captures_by_module[path].append(capture.inputs_by_module[path])
                keys_by_module[path].append(capture.row_keys_by_module[path])
            output_hash = trace_runner.tensor_sha256(output_holder["value"])
            output_hashes.append(output_hash)
            flow_row = {
                "flow_index": flow_index,
                "timestep": float(rows[0]["timestep"]),
                "call_slot_ids": [row["slot_id"] for row in rows],
                "call_ordinals_within_task": [int(row["task_ordinal"]) for row in rows],
                "trace_length": len(capture.trace),
                "trace_sha256": hashlib.sha256(
                    (json.dumps(list(capture.trace)) + "\n").encode("utf-8")
                ).hexdigest(),
                "selected_bytes": capture.selected_bytes,
                "output_sha256": output_hash,
                "hooks_restored": True,
            }
            flow_receipts.append(flow_row)
            append_jsonl(progress_path, {"event": "flow_captured", **flow_row})
            print(json.dumps(flow_row, sort_keys=True), flush=True)
            del capture, collector, core_data, row_mask_provider, output_holder

        if len(captures_by_module) != 418 or len(forward_plan) != EXPECTED_SHARDS:
            raise ValueError("Mini capture module/step cardinality differs")
        if len({row["trace_sha256"] for row in flow_receipts}) != 1:
            raise ValueError("Mini full-forward trace changed across flow batches")
        if len(set(output_hashes)) != 3:
            raise ValueError("Mini flow batches did not produce three distinct outputs")
        observed_selected_bytes = sum(row["selected_bytes"] for row in flow_receipts)
        expected_selected_bytes = (
            int(plan["memory"]["expected_task_factor_bytes"])
            + int(plan["memory"]["expected_task_row_key_bytes"])
        )
        if observed_selected_bytes != expected_selected_bytes:
            raise ValueError(
                "Mini selected byte contract differs: "
                f"{observed_selected_bytes} != {expected_selected_bytes}"
            )
        cache_root = attempt_root / "cache"
        empty_scan = scan_resume_state(
            cache_root,
            index,
            plan_sha256=sha256_file(plan_path),
            call_state_manifest_sha256=CALL_STATE_MANIFEST_SHA256,
            full_tensor_check=True,
        )
        if empty_scan["completed_shards"] != 0 or empty_scan["pending_shards"] != 47:
            raise ValueError("Fresh mini cache resume scan differs")
        shard_receipts = []
        for spec in index["shards"]:
            tensors = assemble_step_tensors(spec, captures_by_module, keys_by_module)
            validation = write_completed_shard_atomic(
                cache_root,
                spec,
                tensors,
                plan_sha256=sha256_file(plan_path),
                call_state_manifest_sha256=CALL_STATE_MANIFEST_SHA256,
                source_checkpoint_sha256=plan["frozen_inputs"]["spatial_expert_sha256"],
            )
            shard_receipts.append(validation)
            append_jsonl(
                progress_path,
                {"event": "teacher_shard_written", **validation},
            )
            print(json.dumps({"written": validation["shard_id"]}, sort_keys=True), flush=True)
            del tensors
        full_scan_started = time.monotonic()
        final_scan = scan_resume_state(
            cache_root,
            index,
            plan_sha256=sha256_file(plan_path),
            call_state_manifest_sha256=CALL_STATE_MANIFEST_SHA256,
            full_tensor_check=True,
        )
        full_scan_seconds = time.monotonic() - full_scan_started
        if (
            final_scan["completed_shards"] != 47
            or final_scan["pending_shards"] != 0
            or final_scan["next_phase"] != "complete"
        ):
            raise ValueError("Completed mini cache resume scan differs")
        hooks_final = _all_target_hook_count(expert, forward_plan)
        if hooks_final != hooks_before:
            raise ValueError("Streaming hooks leaked at completion")
        frozen_after = {
            "table1_manifest_sha256": sha256_file(Path(frozen["table1_manifest"])),
            "expert_bank_sha256": sha256_file(EXPERT_BANK),
            "formal_plan_sha256": sha256_file(FORMAL_PLAN),
            "formal_index_sha256": sha256_file(FORMAL_INDEX),
            "call_state_manifest_sha256": sha256_file(CALL_STATE_MANIFEST),
        }
        if frozen_after != {
            "table1_manifest_sha256": frozen["table1_manifest_sha256"],
            "expert_bank_sha256": EXPERT_BANK_SHA256,
            "formal_plan_sha256": FORMAL_PLAN_SHA256,
            "formal_index_sha256": FORMAL_INDEX_SHA256,
            "call_state_manifest_sha256": CALL_STATE_MANIFEST_SHA256,
        }:
            raise ValueError("Frozen Table-1 inputs changed during mini execution")
        elapsed = time.monotonic() - started
        append_jsonl(
            progress_path,
            {
                "event": "execution_completed",
                "at": datetime.now(timezone.utc).isoformat(),
                "shards": 47,
                "elapsed_seconds": elapsed,
            },
        )
        receipt = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "featcal_spatial_task0_teacher_mini_passed_nonfinal",
            "claim_boundary": plan["claim_boundary"],
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "shard_index": {"path": str(index_path), "sha256": sha256_file(index_path)},
            "gpu_admission": admission,
            "device": {
                "visible": os.environ["CUDA_VISIBLE_DEVICES"],
                "name": torch.cuda.get_device_name(0),
            },
            "scope": plan["scope"],
            "authorization": authorization,
            "dataset_indices": mini["dataset_indices"],
            "raw_action_sha256": mini["raw_action_hashes"],
            "request_tensor_hashes": mini["request_hashes"],
            "flows": flow_receipts,
            "output_hashes": output_hashes,
            "selected_bytes_total": observed_selected_bytes,
            "cache_bytes": sum(
                path.stat().st_size for path in cache_root.rglob("*") if path.is_file()
            ),
            "shard_pairs": len(shard_receipts),
            "shards": shard_receipts,
            "initial_resume_scan": empty_scan,
            "final_resume_scan": final_scan,
            "full_tensor_resume_scan_seconds": full_scan_seconds,
            "elapsed_seconds": elapsed,
            "hooks_before": hooks_before,
            "hooks_after": hooks_final,
            "frozen_inputs_after_execution": frozen_after,
            "formal_cache_created": False,
            "student_collection": False,
            "solve": False,
            "checkpoint": False,
            "rollout": False,
            "table1_revision5_modified": False,
            "progress": {"path": str(progress_path), "sha256": sha256_file(progress_path)},
        }
        write_exclusive_json(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}), flush=True)
        return receipt
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def finalize_attempt(attempt_root: Path) -> dict[str, Any]:
    attempt_root = attempt_root.expanduser().resolve()
    output = attempt_root / "final-receipt.json"
    if output.exists():
        raise FileExistsError(output)
    plan_path = attempt_root / "plan.json"
    resource_path = attempt_root / "resource-accounting.json"
    log_path = attempt_root / "combined.log"
    execution_path = attempt_root / "execution-receipt.json"
    progress_path = attempt_root / "progress.jsonl"
    resource = load_json(resource_path)
    if resource.get("schema_version") != 2:
        raise ValueError("Resource report schema differs")
    log_identity = {
        "path": str(log_path),
        "bytes": log_path.stat().st_size,
        "sha256": sha256_file(log_path),
    }
    if resource.get("combined_log") != log_identity:
        raise ValueError("Resource report combined log identity differs")
    success = resource.get("return_code") == 0
    execution = load_json(execution_path) if execution_path.is_file() else None
    if success and (
        not isinstance(execution, dict)
        or execution.get("status") != "featcal_spatial_task0_teacher_mini_passed_nonfinal"
        or execution.get("shard_pairs") != 47
        or execution.get("student_collection") is not False
        or execution.get("solve") is not False
        or execution.get("checkpoint") is not False
        or execution.get("rollout") is not False
    ):
        raise ValueError("Successful mini resource report lacks a passed execution receipt")
    final = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "featcal_spatial_task0_teacher_mini_attempt_passed_nonfinal"
            if success
            else "featcal_spatial_task0_teacher_mini_attempt_failed_preserved"
        ),
        "claim_boundary": (
            "One authorized teacher-only engineering mini; no formal/full cache, student, "
            "solve, checkpoint, candidate, score, or rollout."
        ),
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "resource": {"path": str(resource_path), "sha256": sha256_file(resource_path)},
        "combined_log": log_identity,
        "progress": (
            {"path": str(progress_path), "sha256": sha256_file(progress_path)}
            if progress_path.is_file()
            else None
        ),
        "execution": (
            {"path": str(execution_path), "sha256": sha256_file(execution_path)}
            if execution_path.is_file()
            else None
        ),
        "return_code": resource.get("return_code"),
        "formal_cache_created": False,
        "student_collection": False,
        "solve": False,
        "checkpoint": False,
        "rollout": False,
        "table1_revision5_modified": False,
    }
    write_exclusive_json(output, final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--table1-manifest", type=Path)
    parser.add_argument("--expected-table1-manifest-sha256")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare:
        if args.table1_manifest is None or args.expected_table1_manifest_sha256 is None:
            raise ValueError("--prepare requires Table-1 manifest path and expected SHA256")
        prepare_attempt(
            args.attempt_root,
            table1_manifest=args.table1_manifest,
            expected_table1_manifest_sha256=args.expected_table1_manifest_sha256,
        )
    elif args.execute:
        execute_attempt(args.attempt_root)
    else:
        finalize_attempt(args.attempt_root)


if __name__ == "__main__":
    main()
