#!/usr/bin/env python3
"""Run the one authorized Table-5 RoboTwin native joint/RGB integration smoke.

The module intentionally imports no CUDA-capable library at import time. Static manifest,
plan, hash, host, lock, and admission gates run before torch or the policy are imported.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Callable


ATTEMPT_ID = "robotwin-native-joint-rgb-smoke-v2"
EXPECTED_MANIFEST_REVISION = 8
TASK = "handover_block"
EXPECTED_HOST = "dsw-824375-57c745db88-n6tv9"
PHYSICAL_GPU = 4
MAX_IDLE_MEMORY_MIB = 64.0
MAX_STEPS = 10
SEED = 0
CAMERA_MAPPING = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
RUNTIME_IMAGE_SHAPE = (3, 240, 320)
TRAINING_IMAGE_SHAPE = (3, 480, 640)
ACTION_LOW = [-10.0] * 6 + [0.0] + [-10.0] * 6 + [0.0]
ACTION_HIGH = [10.0] * 6 + [1.0] + [10.0] * 6 + [1.0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class ProgressJournal:
    """Exclusive, fsync-backed JSONL evidence that survives a later runtime failure."""

    def __init__(self, path: Path):
        self.path = path
        self.last_completed_phase = "none"
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        os.close(descriptor)

    def append(self, phase: str, **payload: Any) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            **payload,
        }
        data = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
        with self.path.open("ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.last_completed_phase = phase


def validate_native_contract(contract: dict[str, Any], output_root: Path) -> None:
    expected = {
        "authorized": True,
        "authorization_count": 1,
        "authorized_attempt": ATTEMPT_ID,
        "task": TASK,
        "episodes": 1,
        "max_steps": MAX_STEPS,
        "full_horizon": False,
        "success_is_gate": False,
        "requires_resource_lock": True,
        "requires_resource_accounting": True,
        "requires_progress_journal": True,
        "progress_journal_filename": "progress.jsonl",
        "requires_fresh_output": True,
        "overwrite_any_existing_output": False,
        "performance_claim_authorized": False,
        "failure_consumes_authorization": True,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Authorization differs at {key}: {contract.get(key)!r} != {value!r}")
    if Path(str(contract.get("output_root", ""))).resolve() != output_root:
        raise ValueError("Authorization output_root differs from plan")


def validate_plan(plan_path: Path, *, runner_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_json(plan_path)
    if plan.get("status") != "authorized_native_smoke_plan_ready":
        raise ValueError("Plan is not launchable")
    if plan.get("attempt_id") != ATTEMPT_ID:
        raise ValueError("Plan attempt id differs")
    scope = plan.get("scope", {})
    expected_scope = {
        "task": TASK,
        "episodes": 1,
        "max_steps": MAX_STEPS,
        "full_horizon": False,
        "success_is_gate": False,
        "task_config": "demo_clean",
        "seed": SEED,
        "action_mode": "joint",
    }
    for key, value in expected_scope.items():
        if scope.get(key) != value:
            raise ValueError(f"Plan scope differs at {key}")

    output_root = Path(plan["output"]["root"]).resolve()
    if output_root.name != ATTEMPT_ID:
        raise ValueError("Output root is not named by the authorized attempt")
    manifest_binding = plan.get("authorization", {}).get("manifest", {})
    manifest_path = Path(manifest_binding.get("path", "")).resolve()
    if sha256_file(manifest_path) != manifest_binding.get("sha256"):
        raise ValueError("Manifest SHA256 differs")
    manifest = load_json(manifest_path)
    if manifest.get("immutable_revision") != EXPECTED_MANIFEST_REVISION:
        raise ValueError("Manifest is not revision 7")
    contract = manifest.get("authorization", {}).get("native_joint_rgb_simulator_smoke", {})
    validate_native_contract(contract, output_root)
    if manifest.get("launch_gates", {}).get("joint_action_contract") != "accepted_arm_fail_closed_gripper_saturation_audited":
        raise ValueError("Manifest joint-action gate is not accepted")

    execution = plan.get("execution", {})
    if execution.get("hostname") != EXPECTED_HOST or execution.get("physical_gpu") != PHYSICAL_GPU:
        raise ValueError("Plan host/GPU differs")
    if execution.get("max_idle_memory_mib") != MAX_IDLE_MEMORY_MIB or execution.get("required_idle_utilization_percent") != 0:
        raise ValueError("Plan GPU admission threshold differs")
    if plan.get("camera_mapping") != CAMERA_MAPPING:
        raise ValueError("Plan camera mapping differs")
    action_contract = plan.get("action_contract", {})
    if action_contract.get("low") != ACTION_LOW or action_contract.get("high") != ACTION_HIGH:
        raise ValueError("Plan action bounds differ")
    if action_contract.get("arm_clip") is not False or "indices 6 and 13" not in str(
        action_contract.get("gripper_saturation")
    ):
        raise ValueError("Plan gripper saturation contract differs")
    resolution = plan.get("image_resolution_contract", {})
    if tuple(resolution.get("training_chw", ())) != TRAINING_IMAGE_SHAPE:
        raise ValueError("Plan training image shape differs")
    if tuple(resolution.get("runtime_chw", ())) != RUNTIME_IMAGE_SHAPE:
        raise ValueError("Plan runtime image shape differs")
    if tuple(resolution.get("model_hw", ())) != (224, 224) or resolution.get("explicit_eval_override") is not True:
        raise ValueError("Plan model resize/override contract differs")

    for label, binding in plan.get("bindings", {}).items():
        path = Path(binding["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Bound file is missing ({label}): {path}")
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"Bound file SHA256 differs: {label}")
    if plan["bindings"]["runner"]["sha256"] != sha256_file(runner_path):
        raise ValueError("Running runner is not the plan-bound runner")
    if output_root.exists():
        raise FileExistsError(output_root)
    if Path(plan["output"]["resource_report"]).exists():
        raise FileExistsError(plan["output"]["resource_report"])
    return plan, manifest


@contextmanager
def exclusive_gpu_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"GPU resource lock is held: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def check_gpu_admission(gpu: int) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    name, memory, utilization = [part.strip() for part in query.split(",")]
    memory_mib = float(memory)
    utilization_percent = float(utilization)
    processes = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pids = [int(line.strip()) for line in processes.splitlines() if line.strip().isdigit()]
    if memory_mib > MAX_IDLE_MEMORY_MIB or utilization_percent != 0 or pids:
        raise RuntimeError(
            f"GPU admission failed: memory={memory_mib} MiB, utilization={utilization_percent}%, pids={pids}"
        )
    return {"name": name, "memory_used_mib": memory_mib, "utilization_percent": utilization_percent, "compute_pids": pids}


def apply_runtime_visual_override(policy_cfg: Any) -> dict[str, list[int]]:
    visual_keys = tuple(f"observation.images.{name}" for name in CAMERA_MAPPING.values())
    observed = {}
    for key in visual_keys:
        feature = policy_cfg.input_features.get(key)
        if feature is None or tuple(feature.shape) != TRAINING_IMAGE_SHAPE:
            raise ValueError(f"Checkpoint training visual contract differs for {key}: {feature}")
        observed[key] = list(feature.shape)
        policy_cfg.input_features[key] = replace(feature, shape=RUNTIME_IMAGE_SHAPE)
    if tuple(policy_cfg.input_features["observation.state"].shape) != (14,):
        raise ValueError("Checkpoint state feature differs")
    if tuple(policy_cfg.output_features["action"].shape) != (14,):
        raise ValueError("Checkpoint action feature differs")
    return observed


def _finite_numpy(value: Any) -> bool:
    import numpy as np

    return bool(np.isfinite(np.asarray(value)).all())


def validate_raw_observation(observation: dict[str, Any]) -> None:
    import numpy as np

    if set(observation) != {"pixels", "agent_pos"}:
        raise ValueError(f"Raw observation keys differ: {sorted(observation)}")
    if set(observation["pixels"]) != set(CAMERA_MAPPING.values()):
        raise ValueError("Raw camera keys differ")
    for key, image in observation["pixels"].items():
        array = np.asarray(image)
        if array.shape != (1, 240, 320, 3) or array.dtype != np.uint8:
            raise ValueError(f"Raw image differs for {key}: {array.shape}/{array.dtype}")
    state = np.asarray(observation["agent_pos"])
    if state.shape != (1, 14) or not np.isfinite(state).all():
        raise ValueError("Raw joint state differs or is non-finite")


def observation_digest(observation: dict[str, Any]) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for key in sorted(observation["pixels"]):
        digest.update(key.encode())
        digest.update(np.asarray(observation["pixels"][key]).tobytes())
    digest.update(np.asarray(observation["agent_pos"]).tobytes())
    return digest.hexdigest()


def _mode_name(value: Any) -> str:
    return getattr(value, "name", str(value).split(".")[-1])


def unbatch_info(value: Any) -> Any:
    """Convert Gymnasium's one-env dict-of-arrays info back to scalar Python values."""
    import numpy as np

    if isinstance(value, dict):
        return {key: unbatch_info(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.shape[0] == 1:
            return unbatch_info(value[0])
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_control_loop(
    *,
    env: Any,
    policy: Any,
    torch_module: Any,
    preprocess_observation: Callable[[dict[str, Any]], dict[str, Any]],
    env_preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    postprocessor: Callable[[Any], Any],
    env_postprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    action_key: str,
    progress: ProgressJournal,
) -> dict[str, Any]:
    import numpy as np

    if _mode_name(env.autoreset_mode) != "NEXT_STEP":
        raise ValueError(f"Vector autoreset mode differs: {env.autoreset_mode}")
    policy.reset()
    raw_observation, reset_info = env.reset(seed=SEED)
    validate_raw_observation(raw_observation)
    initial_digest = observation_digest(raw_observation)
    reset_values = unbatch_info(reset_info)
    progress.append(
        "env_reset",
        seed=SEED,
        episode_horizon=reset_values.get("episode_horizon"),
        registered_episode_horizon=reset_values.get("registered_episode_horizon"),
        observation_sha256=initial_digest,
        camera_keys=sorted(raw_observation["pixels"]),
        state_shape=list(raw_observation["agent_pos"].shape),
    )
    step_records = []
    done = False
    success = False

    for step in range(1, MAX_STEPS + 1):
        model_observation = preprocess_observation(raw_observation)
        model_observation["task"] = list(env.call("task_description"))
        model_observation = env_preprocessor(model_observation)
        model_observation = preprocessor(model_observation)
        for key in (f"observation.images.{name}" for name in CAMERA_MAPPING.values()):
            if tuple(model_observation[key].shape) != (1, *RUNTIME_IMAGE_SHAPE):
                raise ValueError(f"Processed runtime image shape differs for {key}")
            if not bool(torch_module.isfinite(model_observation[key]).all()):
                raise ValueError(f"Processed runtime image is non-finite: {key}")

        with torch_module.inference_mode():
            normalized_action = policy.select_action(model_observation)
        if tuple(normalized_action.shape) != (1, 14) or not bool(torch_module.isfinite(normalized_action).all()):
            raise ValueError("Policy normalized action differs or is non-finite")
        action = postprocessor(normalized_action)
        action = env_postprocessor({action_key: action})[action_key]
        action_numpy = action.to("cpu").numpy()
        if action_numpy.shape != (1, 14) or not np.isfinite(action_numpy).all():
            raise ValueError("Postprocessed action differs or is non-finite")
        postprocessed = action_numpy[0].astype(np.float32, copy=False)
        arm_indices = np.asarray([*range(6), *range(7, 13)], dtype=np.int64)
        low = np.asarray(ACTION_LOW, dtype=np.float32)
        high = np.asarray(ACTION_HIGH, dtype=np.float32)
        if np.any(postprocessed[arm_indices] < low[arm_indices]) or np.any(
            postprocessed[arm_indices] > high[arm_indices]
        ):
            raise ValueError(f"Postprocessed arm action is outside the pinned environment contract: {postprocessed.tolist()}")
        expected_executed = postprocessed.copy()
        expected_executed[[6, 13]] = np.clip(expected_executed[[6, 13]], 0.0, 1.0)
        saturation_indices = [index for index in (6, 13) if expected_executed[index] != postprocessed[index]]
        expected_saturation = {
            "indices": saturation_indices,
            "original": [float(postprocessed[index]) for index in saturation_indices],
            "executed": [float(expected_executed[index]) for index in saturation_indices],
            "max_delta": max(
                (abs(float(expected_executed[index] - postprocessed[index])) for index in saturation_indices),
                default=0.0,
            ),
        }
        normalized_values = normalized_action.detach().float().cpu().numpy()[0].tolist()
        progress.append(
            "step_action_ready",
            step=step,
            normalized_action=normalized_values,
            postprocessed_action=postprocessed.tolist(),
            expected_executed_action=expected_executed.tolist(),
            expected_saturation=expected_saturation,
        )

        raw_observation, reward, terminated, truncated, info = env.step(action_numpy)
        validate_raw_observation(raw_observation)
        step_info = unbatch_info(info)
        observed_saturation = step_info.get("action_saturation")
        if not isinstance(observed_saturation, dict):
            raise ValueError("RoboTwin step info lacks action_saturation audit")
        if observed_saturation.get("indices") != expected_saturation["indices"]:
            raise ValueError("Observed gripper saturation indices differ")
        for key in ("original", "executed"):
            if not np.allclose(observed_saturation.get(key, []), expected_saturation[key], atol=1e-8, rtol=0):
                raise ValueError(f"Observed gripper saturation {key} differs")
        if not np.isclose(
            float(observed_saturation.get("max_delta", -1.0)),
            expected_saturation["max_delta"],
            atol=1e-8,
            rtol=0,
        ):
            raise ValueError("Observed gripper saturation max_delta differs")
        done = bool(np.asarray(terminated)[0] or np.asarray(truncated)[0])
        success = bool(step_info.get("is_success", False))
        record = {
            "step": step,
            "normalized_action_min": min(normalized_values),
            "normalized_action_max": max(normalized_values),
            "normalized_action": normalized_values,
            "postprocessed_action_min": float(postprocessed.min()),
            "postprocessed_action_max": float(postprocessed.max()),
            "postprocessed_action": postprocessed.tolist(),
            "executed_action": expected_executed.tolist(),
            "action_saturation": observed_saturation,
            "reward": float(np.asarray(reward)[0]),
            "terminated": bool(np.asarray(terminated)[0]),
            "truncated": bool(np.asarray(truncated)[0]),
            "terminal": done,
            "success": success,
            "observation_sha256": observation_digest(raw_observation),
        }
        if done:
            final_info = step_info.get("final_info", {})
            if final_info.get("action_saturation") != observed_saturation:
                raise ValueError("Terminal final_info saturation audit differs")
        step_records.append(record)
        progress.append("step_completed", **record)
        if done:
            break

    if not done:
        raise RuntimeError("The explicit ten-step horizon did not produce a terminal transition")
    subenv = env.envs[0]
    if getattr(subenv, "_step_count", None) != len(step_records):
        raise RuntimeError("Underlying step count differs; possible same-step autoreset")
    if getattr(subenv, "_next_seed", None) != SEED + 1:
        raise RuntimeError("Seed lane advanced more than once; possible implicit autoreset")
    return {
        "steps": len(step_records),
        "success": success,
        "success_is_gate": False,
        "initial_observation_sha256": initial_digest,
        "terminal_observation_sha256": observation_digest(raw_observation),
        "terminal_observation_returned": True,
        "autoreset_mode": "NEXT_STEP",
        "implicit_autoreset_observed": False,
        "reset_info_keys": sorted(reset_info),
        "step_records": step_records,
    }


def execute_runtime(
    plan: dict[str, Any], admission: dict[str, Any], progress: ProgressJournal
) -> dict[str, Any]:
    import gymnasium as gym
    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.configs import RoboTwinEnvConfig
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401
    from lerobot.utils.constants import ACTION

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device is required")
    if torch.cuda.get_device_capability(0) != (8, 0):
        raise RuntimeError("Visible device is not A100 sm80")

    checkpoint = Path(plan["checkpoint"]["path"])
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    training_shapes = apply_runtime_visual_override(policy_cfg)
    policy_cfg.pretrained_path = checkpoint
    policy_cfg.device = "cuda"
    env_cfg = RoboTwinEnvConfig(
        task=TASK,
        episode_length=MAX_STEPS,
        task_config="demo_clean",
        action_mode="joint",
        observation_height=240,
        observation_width=320,
    )
    # The factory requires exactly one feature source. Passing the real eval env config
    # both satisfies that gate and makes the 240x320 runtime camera contract explicit.
    policy = make_policy(policy_cfg, env_cfg=env_cfg, rename_map={})
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    progress.append(
        "model_loaded",
        checkpoint=str(checkpoint),
        checkpoint_training_visual_shapes=training_shapes,
        eval_runtime_visual_shape=list(RUNTIME_IMAGE_SHAPE),
        model_resize_hw=[224, 224],
        visible_gpu_name=torch.cuda.get_device_name(0),
    )
    env = make_env(env_cfg, n_envs=1, use_async_envs=False)[TASK][0]
    try:
        loop = run_control_loop(
            env=env,
            policy=policy,
            torch_module=torch,
            preprocess_observation=preprocess_observation,
            env_preprocessor=env_preprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
            action_key=ACTION,
            progress=progress,
        )
    finally:
        env.close()
    return {
        "status": "native_joint_rgb_integration_passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attempt_id": ATTEMPT_ID,
        "host": socket.gethostname(),
        "physical_gpu": PHYSICAL_GPU,
        "visible_gpu_name": torch.cuda.get_device_name(0),
        "gpu_admission": admission,
        "checkpoint_training_visual_shapes": training_shapes,
        "eval_runtime_visual_shape": list(RUNTIME_IMAGE_SHAPE),
        "model_resize_hw": [224, 224],
        "loop": loop,
        "claim_boundary": "Infrastructure integration only; success is recorded but is not an acceptance or performance gate.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner_path = Path(__file__).resolve()
    plan, _ = validate_plan(args.plan.resolve(), runner_path=runner_path)
    if args.validate_only:
        print(json.dumps({"status": "native_smoke_plan_validation_passed_no_gpu", "attempt_id": ATTEMPT_ID}, indent=2))
        return

    if socket.gethostname() != EXPECTED_HOST:
        raise RuntimeError(f"Execution host differs: {socket.gethostname()} != {EXPECTED_HOST}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly {PHYSICAL_GPU}")
    output_root = Path(plan["output"]["root"]).resolve()
    lock_path = Path(plan["execution"]["resource_lock"]).resolve()
    with exclusive_gpu_lock(lock_path):
        admission = check_gpu_admission(PHYSICAL_GPU)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=False, exist_ok=False)
        write_json_exclusive(
            output_root / "authorization-consumed.json",
            {
                "attempt_id": ATTEMPT_ID,
                "manifest_revision": EXPECTED_MANIFEST_REVISION,
                "manifest_sha256": plan["authorization"]["manifest"]["sha256"],
                "consumed_at": datetime.now(timezone.utc).isoformat(),
                "failure_consumes_authorization": True,
            },
        )
        progress = ProgressJournal(output_root / "progress.jsonl")
        progress.append(
            "authorization_consumed",
            attempt_id=ATTEMPT_ID,
            manifest_revision=EXPECTED_MANIFEST_REVISION,
            manifest_sha256=plan["authorization"]["manifest"]["sha256"],
            gpu_admission=admission,
        )
        os.environ.update(plan["execution"]["runtime_environment"])
        robotwin_root = Path(plan["robotwin"]["root"]).resolve()
        sys.path.insert(0, str(robotwin_root))
        sys.path.insert(0, str(Path(plan["lerobot"]["root"]) / "src"))
        os.chdir(robotwin_root)
        try:
            result = execute_runtime(plan, admission, progress)
            progress.append(
                "smoke_completed",
                steps=result["loop"]["steps"],
                success=result["loop"]["success"],
                success_is_gate=False,
            )
            result["progress"] = {
                "path": str(progress.path),
                "sha256": sha256_file(progress.path),
                "last_completed_phase": progress.last_completed_phase,
            }
            write_json_exclusive(output_root / "result.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
        except BaseException as error:
            last_completed_phase = progress.last_completed_phase
            progress.append(
                "failure",
                error_type=type(error).__name__,
                error=str(error),
                last_completed_phase=last_completed_phase,
            )
            write_json_exclusive(
                output_root / "failure.json",
                {
                    "attempt_id": ATTEMPT_ID,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "authorization_consumed": True,
                    "progress": {
                        "path": str(progress.path),
                        "sha256": sha256_file(progress.path),
                        "last_completed_phase": last_completed_phase,
                        "journal_final_phase": progress.last_completed_phase,
                    },
                },
            )
            raise


if __name__ == "__main__":
    main()
