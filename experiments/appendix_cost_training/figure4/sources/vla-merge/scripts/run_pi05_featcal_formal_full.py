#!/usr/bin/env python3
"""Run the single r7-authorized PI0.5 FeatCal formal cache/solve lane.

The lane is intentionally sequential because every student layer consumes the
fully solved prefix from the preceding layer.  Teacher task shards, student
task shards, per-step solve receipts, and per-step weight snapshots are durable
resume barriers.  A restart may only continue the same prepared attempt.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "pi05_lora_finetune_v2_20260826"
RUNTIME = ROOT.parent / "vla-merge-runtime"
TABLE1 = RUNTIME / "experiments/iclr2027-table1-20260910"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(SOURCE / "lerobot/src"))

from scripts import run_pi05_featcal_forward_order_engineering as trace_runner  # noqa: E402
from scripts import run_pi05_featcal_prefix47_engineering as prefix_runner  # noqa: E402
from scripts import run_pi05_featcal_teacher_mini_v2 as teacher_runner  # noqa: E402
from vla_merge.featcal_cache_plan import (  # noqa: E402
    MultiLayerStreamingRowCollector,
    sha256_file,
    tensor_sha256,
    validate_completed_shard,
    write_completed_shard_atomic,
)
from vla_merge.featcal_forward_order import (  # noqa: E402
    FeatCalRowFactors,
    build_pi05_adapted_linear_forward_plan,
    expected_pi05_adapted_weight_keys,
    forward_plan_sha256,
)
from vla_merge.featcal_hybrid_solve import featcal_hybrid_linear_weight  # noqa: E402
from vla_merge.featcal_prefix_chain import (  # noqa: E402
    prefix_chain_sha256,
    rollback_safe_step_load,
)


ATTEMPT_NAME = "featcal-formal-full-v1"
EXPECTED_REVISION = 7
EXPERT_ORDER = ["spatial", "object", "goal", "long"]
FORMAL_PLAN = TABLE1 / "preflight/featcal-formal-cache-plan-v3/plan.json"
FORMAL_PLAN_SHA256 = "876367b0e38ba54bd75f3781b6e3ac85e072151730dcbc899cfd6493d19e81bd"
FORMAL_INDEX = TABLE1 / "preflight/featcal-formal-cache-plan-v3/shard-index.json"
FORMAL_INDEX_SHA256 = "f955bec8267252aebfd838394ccdc7bef73273dafd2fa0b538aaa1b4d3698fd3"
CALL_STATE_MANIFEST = (
    TABLE1 / "preflight/featcal-call-states-training-demo-v1/artifacts/manifest.json"
)
CALL_STATE_MANIFEST_SHA256 = "b18b82300036ded14caa7c6431015acc84368606460437e495c13f424a9ad285"
SCOPE_OVERRIDE = TABLE1 / "receipts/table1-scope-override-drop-libero-plus-20260913.json"
SCOPE_OVERRIDE_SHA256 = "c0764efacfb189395db209bb0b724b49f4ca6e98b486bd3a2b6afdffd620d96b"
PREFIX_FINAL = (
    TABLE1
    / "preflight/featcal-spatial-task0-15state-prefix47-r6000-v1/final-receipt.json"
)
PREFIX_FINAL_SHA256 = "b2cebf044fa8f10d73cc7070db0b0984298b72fdaf590ca1729a1eacbf363059"
STUDENT = trace_runner.STUDENT
STUDENT_SHA256 = trace_runner.STUDENT_MODEL_SHA256
TEACHER_ALPHA = 0.3
RIDGE_LAMBDA = 0.05
ANCHOR_BLEND_RHO = 2.0
COVARIANCE_EPS = 1e-8
OUTPUT_CHUNK = 64
WORKSPACE_CAP = 2 * 1024**3
MINIMUM_FREE_BYTES = 92_472_864_000


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


def cache_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        digest.update(
            f"{sha256_file(path)}  ./{path.relative_to(root)}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _scope_is_drop_plus(receipt: dict[str, Any]) -> bool:
    table1 = receipt.get("table1", {})
    return (
        receipt.get("status") == "libero_plus_removed_from_iclr2027_execution_scope"
        and receipt.get("decision_source") == "explicit_user_instruction"
        and table1.get("clean_libero_required") is True
        and table1.get("libero_pro_required") is True
        and table1.get("libero_plus_required") is False
        and table1.get("libero_plus_formal_rollout_authorized") is False
        and table1.get("libero_plus_reporting_authorized") is False
    )


def _validate_revision(
    manifest_path: Path, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    identities = {
        manifest_path.expanduser().resolve(): expected_manifest_sha256,
        FORMAL_PLAN: FORMAL_PLAN_SHA256,
        FORMAL_INDEX: FORMAL_INDEX_SHA256,
        CALL_STATE_MANIFEST: CALL_STATE_MANIFEST_SHA256,
        SCOPE_OVERRIDE: SCOPE_OVERRIDE_SHA256,
        PREFIX_FINAL: PREFIX_FINAL_SHA256,
    }
    for path, expected in identities.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen formal-lane input differs: {path}")
    revision = load_json(manifest_path)
    authorization = revision.get("featcal_formal_revision7", {}).get(
        "authorization", {}
    )
    host = socket.gethostname()
    gpu = authorization.get("physical_gpu")
    expected_lock = f"resource-leases/{host}-gpu-{gpu}.lock"
    if (
        revision.get("immutable_revision") != EXPECTED_REVISION
        or authorization.get("authorization_count") != 1
        or authorization.get("authorized") is not True
        or authorization.get("attempt") != ATTEMPT_NAME
        or authorization.get("authorized_host") != host
        or not isinstance(gpu, int)
        or gpu < 0
        or gpu > 7
        or authorization.get("cuda_visible_devices") != str(gpu)
        or authorization.get("requires_host_qualified_gpu_lock") != expected_lock
        or authorization.get("call_states") != 600
        or authorization.get("experts") != EXPERT_ORDER
        or authorization.get("tasks_per_expert") != 10
        or authorization.get("steps") != 47
        or authorization.get("teacher_shard_pairs") != 1880
        or authorization.get("student_shard_pairs") != 1880
        or authorization.get("teacher_collection") is not True
        or authorization.get("student_collection") is not True
        or authorization.get("rowspace_solve") is not True
        or authorization.get("checkpoint_materialization") is not True
        or authorization.get("checkpoint_reload_forward_required") is not True
        or authorization.get("formal_evaluation") is not False
        or authorization.get("libero_plus_collection") is not False
        or authorization.get("libero_plus_rollout") is not False
        or authorization.get("libero_plus_reporting") is not False
    ):
        raise ValueError("Table-1 revision-7 FeatCal authorization differs")
    if not _scope_is_drop_plus(load_json(SCOPE_OVERRIDE)):
        raise ValueError("Bound drop-LIBERO-Plus receipt differs")
    prefix_final = load_json(PREFIX_FINAL)
    if (
        prefix_final.get("status")
        != "featcal_prefix47_engineering_attempt_passed_nonfinal"
        or prefix_final.get("return_code") != 0
        or prefix_final.get("initial_soup_restored") is not True
    ):
        raise ValueError("Bound prefix47 final receipt differs")
    formal_plan = load_json(FORMAL_PLAN)
    formal_index = load_json(FORMAL_INDEX)
    shards = formal_index.get("shards", [])
    if (
        len(shards) != 3760
        or sum(row.get("role") == "teacher" for row in shards) != 1880
        or sum(row.get("role") == "student" for row in shards) != 1880
        or formal_plan.get("contract", {}).get("total_rows") != 4_441_200
    ):
        raise ValueError("Formal cache plan/index cardinality differs")
    executor = revision["featcal_formal_revision7"]["evidence"]["formal_executor"]
    hybrid = revision["featcal_formal_revision7"]["evidence"][
        "formal_hybrid_solver"
    ]
    if (
        Path(executor["path"]).resolve() != Path(__file__).resolve()
        or executor["sha256"] != sha256_file(Path(__file__).resolve())
    ):
        raise ValueError("Formal executor identity differs")
    hybrid_path = (ROOT / "src/vla_merge/featcal_hybrid_solve.py").resolve()
    if (
        Path(hybrid["path"]).resolve() != hybrid_path
        or hybrid["sha256"] != sha256_file(hybrid_path)
    ):
        raise ValueError("Formal hybrid solver identity differs")
    return authorization, formal_plan, formal_index


def build_plan(
    attempt_root: Path,
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    authorization, formal_plan, formal_index = _validate_revision(
        manifest_path, expected_manifest_sha256
    )
    forward_plan = build_pi05_adapted_linear_forward_plan(
        expected_pi05_adapted_weight_keys()
    )
    if len(forward_plan) != 47:
        raise ValueError("Formal forward plan does not have 47 steps")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_featcal_formal_full_lane",
        "attempt": ATTEMPT_NAME,
        "attempt_root": str(attempt_root.expanduser().absolute()),
        "authorization": authorization,
        "frozen_inputs": {
            "table1_manifest": str(manifest_path.expanduser().resolve()),
            "table1_manifest_sha256": expected_manifest_sha256,
            "formal_plan_sha256": FORMAL_PLAN_SHA256,
            "formal_index_sha256": FORMAL_INDEX_SHA256,
            "call_state_manifest_sha256": CALL_STATE_MANIFEST_SHA256,
            "scope_override_sha256": SCOPE_OVERRIDE_SHA256,
            "prefix47_final_sha256": PREFIX_FINAL_SHA256,
            "initial_student_sha256": STUDENT_SHA256,
            "dense_bank_sha256": formal_plan["contract"]["dense_bank_sha256"],
        },
        "implementation": {
            "executor": str(Path(__file__).resolve()),
            "executor_sha256": sha256_file(Path(__file__).resolve()),
            "hybrid_solver": str(
                (ROOT / "src/vla_merge/featcal_hybrid_solve.py").resolve()
            ),
            "hybrid_solver_sha256": sha256_file(
                ROOT / "src/vla_merge/featcal_hybrid_solve.py"
            ),
            "forward_plan_sha256": forward_plan_sha256(forward_plan),
        },
        "schedule": {
            "teacher": "expert-major/task-major; one all-layer capture per flow batch",
            "student": "step-major; all 40 task shards then one exact solve barrier",
            "resume": (
                "validated shard pairs, sequential solve receipts, and per-step "
                "weight snapshots are authoritative"
            ),
        },
        "expected": {
            "teacher_full_forwards": 120,
            "student_full_forwards": 5_640,
            "teacher_shard_pairs": 1_880,
            "student_shard_pairs": 1_880,
            "steps": 47,
            "target_weights": 418,
            "minimum_cache_bytes": 73_978_291_200,
            "minimum_free_bytes_at_admission": MINIMUM_FREE_BYTES,
        },
        "formal_cache": True,
        "checkpoint": True,
        "evaluation": False,
        "libero_plus": False,
        "formal_index_summary": {
            "shards": formal_index["shard_count"],
            "contract_sha256": formal_index["contract_sha256"],
        },
    }


def prepare_attempt(
    attempt_root: Path,
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    attempt_root = attempt_root.expanduser().absolute()
    if attempt_root.name != ATTEMPT_NAME or attempt_root.exists():
        raise ValueError("Formal attempt root must be fresh and end in the authorized name")
    plan = build_plan(
        attempt_root,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    attempt_root.mkdir(parents=True)
    write_exclusive_json(attempt_root / "plan.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    return plan


def _gpu_admission(physical_gpu: int) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES differs from r7 authorization")
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-gpu=uuid,memory.total,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    uuid, total, used, utilization, power = [value.strip() for value in query.split(",")]
    processes = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if processes or float(used) > 512 or float(utilization) > 5:
        raise RuntimeError(
            f"Authorized GPU is not idle: used={used}, util={utilization}, pids={processes}"
        )
    return {
        "physical_gpu": physical_gpu,
        "uuid": uuid,
        "total_memory_mib": float(total),
        "memory_used_mib": float(used),
        "utilization_percent": float(utilization),
        "power_watts": float(power),
        "compute_processes": [],
    }


def _spec_map(index: dict[str, Any]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    result = {}
    for spec in index["shards"]:
        key = (spec["role"], spec["expert"], int(spec["task"]), int(spec["step"]))
        if key in result:
            raise ValueError(f"Duplicate formal shard spec: {key}")
        result[key] = spec
    if len(result) != 3760:
        raise ValueError("Formal shard identity map differs")
    return result


def _state_map(states: dict[str, Any]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    result = {(expert, task): [] for expert in EXPERT_ORDER for task in range(10)}
    for row in states["states"]:
        result[(row["expert"], int(row["task"]))].append(row)
    if any(len(rows) != 15 for rows in result.values()):
        raise ValueError("Formal call-state task partition differs")
    return result


def _load_task_batch(rows: list[dict[str, Any]]):
    episodes = {int(row["episode_identity"]["episode_index"]) for row in rows}
    if len(episodes) != 1:
        raise ValueError("Formal task has more than one episode")
    previous = teacher_runner.SELECTED_EPISODE
    teacher_runner.SELECTED_EPISODE = next(iter(episodes))
    try:
        return teacher_runner._load_and_verify_minibatch(rows)
    finally:
        teacher_runner.SELECTED_EPISODE = previous


def _validated_or_missing(
    cache_root: Path, spec: dict[str, Any], plan_sha256: str
) -> dict[str, Any] | None:
    data = cache_root / spec["data_file"]
    manifest = cache_root / spec["manifest_file"]
    if data.exists() != manifest.exists():
        raise ValueError(f"Half-written formal shard: {spec['shard_id']}")
    if not data.exists():
        return None
    return validate_completed_shard(
        cache_root,
        spec,
        plan_sha256=plan_sha256,
        call_state_manifest_sha256=CALL_STATE_MANIFEST_SHA256,
        full_tensor_check=True,
    )


def _capture_teacher_task(policy, mini, forward_plan, quotas):
    captures = {quota.module_path: [] for quota in quotas}
    keys = {quota.module_path: [] for quota in quotas}
    flows = []
    hooks_before = teacher_runner._all_target_hook_count(policy, forward_plan)
    for flow_index in (0, 5, 9):
        rows = [
            mini["request_rows"][request][(0, 5, 9).index(flow_index)]
            for request in range(5)
        ]
        core = teacher_runner._prepare_core_for_flow(
            policy,
            mini["batch_cpu"],
            mini["padded_action_cpu"],
            mini["noise_cpu"],
            float(rows[0]["timestep"]),
        )
        provider = teacher_runner.build_streaming_row_mask_provider(
            image_masks=core["core"]["img_masks"],
            token_mask=core["core"]["masks"],
            action_valid_mask=core["action_valid_mask"],
        )
        output = {}

        def forward():
            value = trace_runner._core_forward(policy, core["core"])
            output["value"] = value.detach().cpu()
            return value

        collector = MultiLayerStreamingRowCollector(
            policy,
            forward_plan,
            quotas,
            call_slot_ids=[row["slot_id"] for row in rows],
            call_ordinals_within_task=[int(row["task_ordinal"]) for row in rows],
            row_mask_provider=provider,
            selection_seed=teacher_runner.SELECTION_SEED,
            max_selected_bytes=2 * 1024**3,
        )
        capture = collector.capture(forward)
        if teacher_runner._all_target_hook_count(policy, forward_plan) != hooks_before:
            raise ValueError("Formal teacher hooks leaked")
        for path in captures:
            captures[path].append(capture.inputs_by_module[path])
            keys[path].append(capture.row_keys_by_module[path])
        flows.append(
            {
                "flow_index": flow_index,
                "selected_bytes": capture.selected_bytes,
                "trace_sha256": hashlib.sha256(
                    (json.dumps(list(capture.trace)) + "\n").encode("utf-8")
                ).hexdigest(),
                "output_sha256": tensor_sha256(output["value"]),
            }
        )
        del core, provider, collector, capture, output
    if len({row["trace_sha256"] for row in flows}) != 1:
        raise ValueError("Formal teacher trace differs across flow batches")
    return captures, keys, flows


def _collect_teacher(
    *,
    cache_root: Path,
    progress: Path,
    plan_sha256: str,
    formal_plan: dict[str, Any],
    specs: dict[tuple[str, str, int, int], dict[str, Any]],
    states: dict[tuple[str, int], list[dict[str, Any]]],
) -> None:
    forward_plan = build_pi05_adapted_linear_forward_plan(
        expected_pi05_adapted_weight_keys()
    )
    quotas = teacher_runner.load_frozen_module_quotas(forward_plan)
    for expert_name in EXPERT_ORDER:
        tasks = []
        for task in range(10):
            task_specs = [specs[("teacher", expert_name, task, step)] for step in range(47)]
            completed = [
                _validated_or_missing(cache_root, spec, plan_sha256)
                for spec in task_specs
            ]
            if any(value is None for value in completed):
                tasks.append((task, task_specs, completed))
        if not tasks:
            continue
        first_mini = _load_task_batch(states[(expert_name, tasks[0][0])])
        expert_path = Path(formal_plan["model_identities"]["experts"][expert_name]["path"])
        policy = trace_runner._load_policy(expert_path, first_mini["dataset"].meta)
        for position, (task, task_specs, completed) in enumerate(tasks):
            mini = first_mini if position == 0 else _load_task_batch(states[(expert_name, task)])
            captures, keys, flows = _capture_teacher_task(
                policy, mini, forward_plan, quotas
            )
            for spec, old in zip(task_specs, completed, strict=True):
                if old is not None:
                    continue
                tensors = teacher_runner.assemble_step_tensors(spec, captures, keys)
                receipt = write_completed_shard_atomic(
                    cache_root,
                    spec,
                    tensors,
                    plan_sha256=plan_sha256,
                    call_state_manifest_sha256=CALL_STATE_MANIFEST_SHA256,
                    source_checkpoint_sha256=spec["source_checkpoint_sha256"],
                )
                append_jsonl(progress, {"event": "teacher_shard_complete", **receipt})
                del tensors
            append_jsonl(
                progress,
                {
                    "event": "teacher_task_complete",
                    "expert": expert_name,
                    "task": task,
                    "flows": flows,
                },
            )
            del mini, captures, keys, flows
            gc.collect()
            torch.cuda.empty_cache()
        del policy, first_mini
        gc.collect()
        torch.cuda.empty_cache()


def _load_shard_inputs(
    cache_root: Path,
    spec: dict[str, Any],
    plan_sha256: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    validation = validate_completed_shard(
        cache_root,
        spec,
        plan_sha256=plan_sha256,
        call_state_manifest_sha256=CALL_STATE_MANIFEST_SHA256,
        full_tensor_check=True,
    )
    inputs, row_keys = {}, {}
    with safe_open(cache_root / spec["data_file"], framework="pt", device="cpu") as handle:
        for target in spec["targets"]:
            path = target["module_path"]
            inputs[path] = handle.get_tensor(target["input_tensor"]).contiguous()
            row_keys[path] = handle.get_tensor(target["row_keys_tensor"]).contiguous()
    return inputs, row_keys, validation


def _snapshot_paths(cache_root: Path, step: int) -> tuple[Path, Path]:
    return (
        cache_root / f"prefix-snapshots/step_{step:02d}.safetensors",
        cache_root / f"solves/step_{step:02d}.json",
    )


def _load_completed_prefix(student, cache_root: Path) -> tuple[int, str, dict[str, str] | None]:
    modules = dict(student.named_modules())
    current = STUDENT_SHA256
    previous = None
    for step in range(47):
        snapshot_path, solve_path = _snapshot_paths(cache_root, step)
        if snapshot_path.exists() != solve_path.exists():
            raise ValueError(f"Half-written formal solve barrier at step {step}")
        if not solve_path.exists():
            return step, current, previous
        solve = load_json(solve_path)
        if (
            solve.get("status") != "complete"
            or solve.get("step") != step
            or solve.get("input_student_prefix_sha256") != current
            or solve.get("previous_solve") != previous
            or solve.get("snapshot", {}).get("sha256") != sha256_file(snapshot_path)
        ):
            raise ValueError(f"Formal solve chain differs at step {step}")
        expected_hashes = solve.get("solved_tensor_sha256", {})
        replacements = {}
        with safe_open(snapshot_path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(expected_hashes):
                raise ValueError(f"Formal prefix snapshot keys differ at step {step}")
            for path in handle.keys():
                value = handle.get_tensor(path).contiguous()
                if tensor_sha256(value) != expected_hashes[path]:
                    raise ValueError(f"Formal prefix snapshot hash differs: {path}")
                replacements[path] = value
        rollback_safe_step_load({path: modules[path] for path in replacements}, replacements)
        current = solve["output_student_prefix_sha256"]
        previous = {"path": str(solve_path.relative_to(cache_root)), "sha256": sha256_file(solve_path)}
    return 47, current, previous


def _publish_solve_barrier(
    *,
    cache_root: Path,
    step,
    replacements: dict[str, torch.Tensor],
    solve: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    snapshot_path, solve_path = _snapshot_paths(cache_root, step.step)
    if snapshot_path.exists() or solve_path.exists():
        raise FileExistsError(f"Formal solve barrier exists at step {step.step}")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    solve_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".featcal-solve-", dir=cache_root) as tmp:
        temporary = Path(tmp)
        temp_snapshot = temporary / "snapshot.safetensors"
        temp_solve = temporary / "solve.json"
        save_file({path: value.cpu().contiguous() for path, value in replacements.items()}, temp_snapshot)
        solve = {
            **solve,
            "snapshot": {
                "path": str(snapshot_path.relative_to(cache_root)),
                "sha256": sha256_file(temp_snapshot),
            },
        }
        temp_solve.write_text(json.dumps(solve, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for path in (temp_snapshot, temp_solve):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.link(temp_snapshot, snapshot_path)
        os.link(temp_solve, solve_path)
    return solve, {"path": str(solve_path.relative_to(cache_root)), "sha256": sha256_file(solve_path)}


def _solve_formal_step(
    *,
    cache_root: Path,
    student,
    step,
    specs,
    plan_sha256: str,
    current_prefix: str,
    previous_solve: dict[str, str] | None,
    soup_handle,
    base_handle,
    expert_handles,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    modules = dict(student.named_modules())
    replacements, metadata_by_target = {}, {}
    # Each step shard contains every target in that atomic forward layer.  Read
    # each of the 80 teacher/student files once; reopening a joint-layer shard
    # once per target would multiply hundreds of GiB of I/O by fourteen.
    loaded_step = {}
    for expert_name in EXPERT_ORDER:
        for task in range(10):
            teacher_spec = specs[("teacher", expert_name, task, step.step)]
            student_spec = specs[("student", expert_name, task, step.step)]
            teacher_inputs, teacher_keys, _ = _load_shard_inputs(
                cache_root, teacher_spec, plan_sha256
            )
            student_inputs, student_keys, _ = _load_shard_inputs(
                cache_root, student_spec, plan_sha256
            )
            if any(
                not torch.equal(teacher_keys[path], student_keys[path])
                for path in step.target_module_paths
            ):
                raise ValueError(
                    f"Formal teacher/student row keys differ: {expert_name}/{task}"
                )
            loaded_step[(expert_name, task)] = (teacher_inputs, student_inputs)
            del teacher_keys, student_keys
    for target_index, path in enumerate(step.target_module_paths):
        expert_weights, factors = [], []
        for expert_name in EXPERT_ORDER:
            teacher_parts, student_parts = [], []
            for task in range(10):
                teacher_inputs, student_inputs = loaded_step[(expert_name, task)]
                teacher_parts.append(teacher_inputs[path])
                student_parts.append(student_inputs[path])
            teacher_input = torch.cat(teacher_parts, dim=0).contiguous()
            student_input = torch.cat(student_parts, dim=0).contiguous()
            target_input = TEACHER_ALPHA * teacher_input + (1.0 - TEACHER_ALPHA) * student_input
            factors.append(
                FeatCalRowFactors(
                    student=student_input,
                    target=target_input,
                    row_weights=torch.ones(student_input.shape[0], dtype=torch.float64),
                    forward_identity=hashlib.sha256(
                        f"{current_prefix}/{step.step}/{path}/{expert_name}".encode("utf-8")
                    ).hexdigest(),
                )
            )
            expert_weights.append(expert_handles[expert_name].get_tensor(f"{path}.weight"))
        key = f"{path}.weight"
        soup = soup_handle.get_tensor(key)
        base = base_handle.get_tensor(key)
        current = modules[path].weight.detach().cpu().contiguous()
        if not torch.equal(current, soup.to(dtype=current.dtype)):
            raise ValueError(f"Unsolved formal target is not initial Soup: {path}")
        solved, metadata = featcal_hybrid_linear_weight(
            expert_weights,
            factors,
            soup_weight=soup,
            base_weight=base,
            ridge_lambda=RIDGE_LAMBDA,
            anchor_blend_rho=ANCHOR_BLEND_RHO,
            covariance_eps=COVARIANCE_EPS,
            solve_dtype=torch.float64,
            solve_device="cuda",
            output_chunk_size=OUTPUT_CHUNK,
            max_workspace_bytes=WORKSPACE_CAP,
            temporary_root=cache_root,
        )
        residual = float(metadata["relative_primal_residual"])
        if (
            not torch.isfinite(solved).all()
            or not math.isfinite(residual)
            or residual > 1e-8
            or int(metadata["estimated_workspace_bytes"]) >= WORKSPACE_CAP
        ):
            raise ValueError(f"Formal exact solver gate differs: {path}")
        replacements[path] = solved
        metadata_by_target[path] = {
            **metadata,
            "target_index": target_index,
            "solved_tensor_sha256": tensor_sha256(solved),
            "expert_weight_sha256": [tensor_sha256(value) for value in expert_weights],
            "soup_weight_sha256": tensor_sha256(soup),
            "base_weight_sha256": tensor_sha256(base),
        }
        del expert_weights, factors, solved
        gc.collect()
        torch.cuda.empty_cache()
    del loaded_step
    loaded = rollback_safe_step_load(
        {path: modules[path] for path in step.target_module_paths}, replacements
    )
    solved_hashes = {path: tensor_sha256(loaded["loaded"][path]) for path in replacements}
    output_prefix = prefix_chain_sha256(current_prefix, step.step, solved_hashes)
    solve = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "step": step.step,
        "layer_id": step.layer_id,
        "input_student_prefix_sha256": current_prefix,
        "output_student_prefix_sha256": output_prefix,
        "previous_solve": previous_solve,
        "solved_tensor_sha256": solved_hashes,
        "solver_targets": metadata_by_target,
        "atomic_step_load": True,
        "formal_cache": True,
        "checkpoint_created": False,
        "evaluation": False,
    }
    solve, solve_identity = _publish_solve_barrier(
        cache_root=cache_root,
        step=step,
        replacements=replacements,
        solve=solve,
    )
    return output_prefix, solve_identity, solve


def _collect_student_and_solve(
    *,
    cache_root: Path,
    progress: Path,
    plan_sha256: str,
    formal_plan: dict[str, Any],
    specs,
    states,
):
    first_mini = _load_task_batch(states[("spatial", 0)])
    student = trace_runner._load_policy(STUDENT, first_mini["dataset"].meta)
    steps = build_pi05_adapted_linear_forward_plan(expected_pi05_adapted_weight_keys())
    quotas = teacher_runner.load_frozen_module_quotas(steps)
    quota_by_path = {row.module_path: row for row in quotas}
    next_step, current_prefix, previous_solve = _load_completed_prefix(student, cache_root)
    dense_bank = load_json(trace_runner.DENSE_BANK)
    soup_path = STUDENT / "model.safetensors"
    base_path = Path(dense_bank["base"]["path"]) / "model.safetensors"
    with ExitStack() as stack:
        soup_handle = stack.enter_context(safe_open(soup_path, framework="pt", device="cpu"))
        base_handle = stack.enter_context(safe_open(base_path, framework="pt", device="cpu"))
        expert_handles = {
            expert: stack.enter_context(
                safe_open(
                    Path(formal_plan["model_identities"]["experts"][expert]["path"])
                    / "model.safetensors",
                    framework="pt",
                    device="cpu",
                )
            )
            for expert in EXPERT_ORDER
        }
        for step in steps[next_step:]:
            for expert_name in EXPERT_ORDER:
                for task in range(10):
                    spec = specs[("student", expert_name, task, step.step)]
                    old = _validated_or_missing(cache_root, spec, plan_sha256)
                    if old is not None:
                        if old["student_prefix_input_sha256"] != current_prefix:
                            raise ValueError("Existing formal student shard has wrong prefix")
                        continue
                    teacher_spec = specs[("teacher", expert_name, task, step.step)]
                    teacher_inputs, teacher_keys, _ = _load_shard_inputs(
                        cache_root, teacher_spec, plan_sha256
                    )
                    mini = (
                        first_mini
                        if expert_name == "spatial" and task == 0 and step.step == next_step
                        else _load_task_batch(states[(expert_name, task)])
                    )
                    cores = {}
                    for flow_index in (0, 5, 9):
                        flow_rows = [
                            row for row in states[(expert_name, task)]
                            if int(row["flow_index"]) == flow_index
                        ]
                        cores[flow_index] = teacher_runner._prepare_core_for_flow(
                            student,
                            mini["batch_cpu"],
                            mini["padded_action_cpu"],
                            mini["noise_cpu"],
                            float(flow_rows[0]["timestep"]),
                        )
                    student_tensors, flows = prefix_runner._capture_student_step(
                        student,
                        step,
                        [quota_by_path[path] for path in step.target_module_paths],
                        spec,
                        teacher_keys,
                        mini,
                        cores,
                    )
                    previous_path = previous_solve["path"] if previous_solve else None
                    previous_sha = previous_solve["sha256"] if previous_solve else None
                    receipt = write_completed_shard_atomic(
                        cache_root,
                        spec,
                        student_tensors,
                        plan_sha256=plan_sha256,
                        call_state_manifest_sha256=CALL_STATE_MANIFEST_SHA256,
                        source_checkpoint_sha256=current_prefix,
                        student_prefix_input_sha256=current_prefix,
                        previous_step_solve_receipt=previous_path,
                        previous_step_solve_receipt_sha256=previous_sha,
                    )
                    append_jsonl(
                        progress,
                        {
                            "event": "student_shard_complete",
                            **receipt,
                            "flows": flows,
                        },
                    )
                    del teacher_inputs, teacher_keys, mini, cores, student_tensors, flows
                    gc.collect()
                    torch.cuda.empty_cache()
            current_prefix, previous_solve, solve = _solve_formal_step(
                cache_root=cache_root,
                student=student,
                step=step,
                specs=specs,
                plan_sha256=plan_sha256,
                current_prefix=current_prefix,
                previous_solve=previous_solve,
                soup_handle=soup_handle,
                base_handle=base_handle,
                expert_handles=expert_handles,
            )
            append_jsonl(
                progress,
                {
                    "event": "formal_step_complete",
                    "step": step.step,
                    "layer_id": step.layer_id,
                    "output_student_prefix_sha256": current_prefix,
                    "solve_receipt": previous_solve,
                    "strategies": {
                        path: row["hybrid_selected_strategy"]
                        for path, row in solve["solver_targets"].items()
                    },
                },
            )
    return student, current_prefix, first_mini


def _materialize_checkpoint(
    attempt_root: Path, student, target_paths: list[str], final_prefix: str
) -> dict[str, Any]:
    output = attempt_root / "checkpoint/pretrained_model"
    manifest_path = attempt_root / "checkpoint/featcal-formal-checkpoint.json"
    if output.exists() != manifest_path.exists():
        raise ValueError("Half-written formal checkpoint barrier")
    if output.exists():
        receipt = load_json(manifest_path)
        model = output / "model.safetensors"
        if (
            receipt.get("status")
            != "featcal_formal_checkpoint_materialized_pending_reload"
            or receipt.get("final_prefix_sha256") != final_prefix
            or receipt.get("target_weight_count") != len(target_paths)
            or not model.is_file()
            or receipt.get("model_sha256") != sha256_file(model)
        ):
            raise ValueError("Existing formal checkpoint barrier differs")
        return {**receipt, "path": str(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".featcal-checkpoint-", dir=output.parent))
    modules = dict(student.named_modules())
    try:
        for source in STUDENT.iterdir():
            if source.name != "model.safetensors":
                destination = temporary / source.name
                if source.is_dir():
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
        tensors = {}
        with safe_open(STUDENT / "model.safetensors", framework="pt", device="cpu") as soup:
            for key in soup.keys():
                path = key[: -len(".weight")] if key.endswith(".weight") else None
                tensors[key] = (
                    modules[path].weight.detach().cpu().contiguous()
                    if path in target_paths
                    else soup.get_tensor(key).contiguous()
                )
        save_file(tensors, temporary / "model.safetensors", metadata={"format": "pt"})
        model_sha = sha256_file(temporary / "model.safetensors")
        receipt = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "featcal_formal_checkpoint_materialized_pending_reload",
            "initialization_sha256": STUDENT_SHA256,
            "final_prefix_sha256": final_prefix,
            "target_weight_count": len(target_paths),
            "tensor_count": len(tensors),
            "model_sha256": model_sha,
            "formal_cache": True,
            "evaluation": False,
            "libero_plus": False,
        }
        (temporary / "featcal_formal_manifest.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
        write_exclusive_json(manifest_path, receipt)
        return {**receipt, "path": str(output)}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def execute_attempt(attempt_root: Path) -> dict[str, Any]:
    attempt_root = attempt_root.expanduser().resolve()
    plan_path = attempt_root / "plan.json"
    progress = attempt_root / "progress.jsonl"
    execution_path = attempt_root / "execution-receipt.json"
    cache_root = attempt_root / "cache"
    if not plan_path.is_file() or execution_path.exists():
        raise ValueError("Formal prepared plan is missing or execution is already complete")
    plan = load_json(plan_path)
    if (
        plan.get("attempt") != ATTEMPT_NAME
        or plan.get("implementation", {}).get("executor_sha256")
        != sha256_file(Path(__file__).resolve())
        or plan.get("implementation", {}).get("hybrid_solver_sha256")
        != sha256_file(ROOT / "src/vla_merge/featcal_hybrid_solve.py")
    ):
        raise ValueError("Prepared formal implementation differs")
    authorization, formal_plan, formal_index = _validate_revision(
        Path(plan["frozen_inputs"]["table1_manifest"]),
        plan["frozen_inputs"]["table1_manifest_sha256"],
    )
    free_bytes = shutil.disk_usage(attempt_root).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(f"Insufficient formal cache space: {free_bytes}")
    lock_path = RUNTIME / authorization["requires_host_qualified_gpu_lock"]
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_stream.close()
        raise RuntimeError(f"Host-qualified GPU lock is busy: {lock_path}") from error
    try:
        admission = _gpu_admission(authorization["physical_gpu"])
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Formal lane visibility is not exactly one A100")
        append_jsonl(
            progress,
            {
                "event": "formal_execution_or_resume_started",
                "at": datetime.now(timezone.utc).isoformat(),
                "gpu_admission": admission,
                "free_bytes": free_bytes,
            },
        )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        plan_sha = sha256_file(plan_path)
        specs = _spec_map(formal_index)
        states = _state_map(load_json(CALL_STATE_MANIFEST))
        started = time.monotonic()
        _collect_teacher(
            cache_root=cache_root,
            progress=progress,
            plan_sha256=plan_sha,
            formal_plan=formal_plan,
            specs=specs,
            states=states,
        )
        student, final_prefix, first_mini = _collect_student_and_solve(
            cache_root=cache_root,
            progress=progress,
            plan_sha256=plan_sha,
            formal_plan=formal_plan,
            specs=specs,
            states=states,
        )
        steps = build_pi05_adapted_linear_forward_plan(expected_pi05_adapted_weight_keys())
        target_paths = [path for step in steps for path in step.target_module_paths]
        checkpoint = _materialize_checkpoint(
            attempt_root, student, target_paths, final_prefix
        )
        reload_rows = states[("spatial", 0)]
        cores = {}
        for flow_index in (5,):
            flow_rows = [row for row in reload_rows if int(row["flow_index"]) == flow_index]
            cores[flow_index] = teacher_runner._prepare_core_for_flow(
                student,
                first_mini["batch_cpu"],
                first_mini["padded_action_cpu"],
                first_mini["noise_cpu"],
                float(flow_rows[0]["timestep"]),
            )
        expected_output = trace_runner._core_forward(student, cores[5]["core"]).detach().cpu()
        del student, cores
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = trace_runner._load_policy(Path(checkpoint["path"]), first_mini["dataset"].meta)
        reload_core = teacher_runner._prepare_core_for_flow(
            reloaded,
            first_mini["batch_cpu"],
            first_mini["padded_action_cpu"],
            first_mini["noise_cpu"],
            float([row for row in reload_rows if int(row["flow_index"]) == 5][0]["timestep"]),
        )
        observed_output = trace_runner._core_forward(reloaded, reload_core["core"]).detach().cpu()
        if not torch.equal(expected_output, observed_output):
            raise ValueError("Formal checkpoint reload forward is not bitwise equal")
        receipt = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "featcal_formal_cache_solve_checkpoint_reload_passed",
            "plan": {"path": str(plan_path), "sha256": plan_sha},
            "gpu_admission": admission,
            "final_prefix_sha256": final_prefix,
            "teacher_shard_pairs": 1880,
            "student_shard_pairs": 1880,
            "solve_steps": 47,
            "checkpoint": checkpoint,
            "reload_forward": {
                "bitwise_equal": True,
                "output_sha256": tensor_sha256(observed_output),
                "finite": bool(torch.isfinite(observed_output).all()),
            },
            "cache_bytes": sum(path.stat().st_size for path in cache_root.rglob("*") if path.is_file()),
            "cache_tree_sha256": cache_tree_sha256(cache_root),
            "elapsed_seconds": time.monotonic() - started,
            "formal_evaluation": False,
            "libero_plus": False,
            "progress": {"path": str(progress), "sha256": sha256_file(progress)},
        }
        write_exclusive_json(execution_path, receipt)
        print(json.dumps({"status": receipt["status"], "receipt": str(execution_path)}), flush=True)
        return receipt
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def finalize_attempt(attempt_root: Path) -> dict[str, Any]:
    attempt_root = attempt_root.expanduser().resolve()
    output = attempt_root / "final-receipt.json"
    execution_path = attempt_root / "execution-receipt.json"
    if output.exists():
        raise FileExistsError(output)
    if not execution_path.is_file():
        raise ValueError("Formal execution is not complete; exact resume remains pending")
    execution = load_json(execution_path)
    if (
        execution.get("status")
        != "featcal_formal_cache_solve_checkpoint_reload_passed"
        or execution.get("reload_forward", {}).get("bitwise_equal") is not True
    ):
        raise ValueError("Formal execution receipt differs")
    segment_rows = []
    for segment in sorted((attempt_root / "segments").glob("segment-*")):
        resource_path = segment / "resource-accounting.json"
        log_path = segment / "combined.log"
        if not resource_path.is_file() or not log_path.is_file():
            raise ValueError(f"Incomplete resource-accounting segment: {segment}")
        resource = load_json(resource_path)
        if resource.get("schema_version") != 2:
            raise ValueError(f"Resource schema differs: {segment}")
        segment_rows.append(
            {
                "segment": segment.name,
                "return_code": resource.get("return_code"),
                "resource": {"path": str(resource_path), "sha256": sha256_file(resource_path)},
                "combined_log": {"path": str(log_path), "sha256": sha256_file(log_path)},
            }
        )
    if not segment_rows or segment_rows[-1]["return_code"] != 0:
        raise ValueError("Final formal segment is absent or unsuccessful")
    final = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "featcal_formal_attempt_passed_checkpoint_ready_for_evaluation_authorization",
        "return_code": 0,
        "segments": segment_rows,
        "execution": {"path": str(execution_path), "sha256": sha256_file(execution_path)},
        "formal_evaluation": False,
        "libero_plus": False,
    }
    write_exclusive_json(output, final)
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)
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
            raise ValueError("--prepare requires the revision-7 path and SHA256")
        prepare_attempt(
            args.attempt_root,
            manifest_path=args.table1_manifest,
            expected_manifest_sha256=args.expected_table1_manifest_sha256,
        )
    elif args.execute:
        execute_attempt(args.attempt_root)
    else:
        finalize_attempt(args.attempt_root)


if __name__ == "__main__":
    main()
