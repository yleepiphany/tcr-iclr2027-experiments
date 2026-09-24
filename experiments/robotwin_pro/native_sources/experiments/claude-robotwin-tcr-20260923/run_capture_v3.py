#!/usr/bin/env python3
"""Collect strict native RoboTwin PI0.5 execution inputs for M=3 TCR."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
VLA = HERE.parents[1]
WORK = VLA.parent
RUNTIME = WORK / "vla-merge-runtime"
TABLE5 = RUNTIME / "experiments/iclr2027-table5-20260910"
PROTOCOL_DEFAULT = RUNTIME / "experiments/claude-robotwin-tcr-20260923/capture-v1/protocol.json"
ROBOTWIN_PYTHON = RUNTIME / "envs/iclr2027-robotwin2-py312-mplib-curobo-v3/bin/python"
sys.path[:0] = [str(HERE), str(VLA), str(VLA / "scripts"),
                str(WORK / "pi05_lora_finetune_v2_20260826/src")]

import capture_contract as contract  # noqa: E402
import fast_official_instruction as fast_instruction  # noqa: E402


def resolve_native_policy(policy: Any) -> Any:
    """Return the PI05Policy instance that owns reset/predict_action_chunk.

    PEFT exposes delegated methods on its outer wrapper, but ``select_action``
    executes with the inner PI05Policy as ``self``.  Hooks installed on the
    wrapper therefore never observe requests.  Bind the collector to the one
    concrete native policy that the delegated method actually uses.
    """
    matches = [
        module for module in policy.modules()
        if type(module).__name__ == "PI05Policy"
        and callable(getattr(module, "predict_action_chunk", None))
        and callable(getattr(module, "reset", None))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one PI05Policy interface, found {len(matches)}")
    return matches[0]


def resolve_noise_generator(policy: Any) -> Any:
    """Return the one PI0.5 core whose native sampler consumes noise.

    RoboTwin expert checkpoints are PEFT checkpoints, so ``policy.model`` is
    the intermediate PI05Policy wrapper and the generator is one level below
    it.  Resolve by the concrete core interface instead of assuming a wrapper
    depth; refusing zero or multiple matches prevents a silent no-op patch.
    """
    matches = [
        module for module in policy.modules()
        if type(module).__name__ == "PI05Pytorch"
        and callable(getattr(module, "sample_noise", None))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one PI05Pytorch noise generator, found {len(matches)}"
        )
    return matches[0]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_protocol(path: Path) -> dict[str, Any]:
    value = contract.read(path)
    if value.get("schema") != "robotwin_tcr_native_ab_capture_v1":
        raise ValueError("Wrong RoboTwin TCR capture protocol")
    if value.get("episodes") != 180 or value.get("requests") != 900 or value.get("flow_rows") != 2700:
        raise ValueError("RoboTwin TCR capture coverage differs")
    formal = value["formal_protocol"]
    if contract.bind(Path(formal["path"])) != formal:
        raise ValueError("Frozen RoboTwin formal-v2 protocol changed")
    keys = []
    for job in value["jobs"]:
        if job.get("episodes") != 10 or job.get("requests") != 50 or job.get("flow_rows") != 150:
            raise ValueError(f"Capture job coverage differs: {job.get('id')}")
        if contract.bind(Path(job["parent"]["path"])) != job["parent"]:
            raise ValueError(f"Capture parent changed: {job.get('id')}")
        keys.extend((row["task_index"], row["seed"]) for row in job["tasks"])
    if len(value["jobs"]) != 18 or len(keys) != 180 or len(set(keys)) != 180:
        raise ValueError("Capture job/reset identity coverage differs")
    if "reset_preflight" in value:
        bound = value["reset_preflight"].get("instruction_contract") or {}
        source = Path(bound.get("path", "/nonexistent"))
        if not source.is_file() or contract.bind(source) != bound:
            raise ValueError("Amended capture lacks its exact instruction contract")
        instruction = contract.read(source)
        if (instruction.get("schema") != "robotwin_exact_lazy_official_instruction_v1"
                or instruction.get("upstream_max_descriptions") != 1_000_000
                or instruction.get("lazy_implementation") != contract.bind(HERE / "fast_official_instruction.py")):
            raise ValueError("Amended capture instruction implementation differs")
    return value


def find_job(protocol: dict[str, Any], job_id: str) -> dict[str, Any]:
    matches = [row for row in protocol["jobs"] if row["id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate capture job: {job_id}")
    return dict(matches[0])


def make_collector_class(base: Any, torch: Any):
    class RoboTwinQuantileCollector(base.BlockCalibrationCollector):
        def __init__(self, policy: Any, job: dict[str, Any], expected_tasks: int) -> None:
            self.robotwin_job = job
            self.expected_tasks = expected_tasks
            self.task_bindings: dict[int, dict[str, Any]] = {}
            self.noise_hashes: list[str] = []
            self.original_sample_noise = None
            self.noise_generator = None
            super().__init__(policy)

        def bind_current_task(self, task: dict[str, Any]) -> None:
            signature = self.latest_prompt_signature
            if signature is None:
                raise RuntimeError("Task text was not captured before binding RoboTwin identity")
            value = {"task": task["task"], "task_index": int(task["task_index"]), "seed": int(task["seed"])}
            previous = self.task_bindings.setdefault(signature, value)
            if previous != value:
                raise ValueError("One task prompt was bound to multiple RoboTwin identities")

        def install(self) -> None:
            super().install()
            self.noise_generator = resolve_noise_generator(self.policy)
            self.original_sample_noise = self.noise_generator.sample_noise
            generator = torch.Generator(device="cuda").manual_seed(int(self.robotwin_job["flow_seed"]))

            def sample_noise(shape: Any, device: Any):
                value = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
                self.noise_hashes.append(hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest())
                return value

            self.noise_generator.sample_noise = sample_noise

        def save(self) -> None:
            groups = sorted(self.records_by_prompt.items())
            if len(groups) != self.expected_tasks:
                raise ValueError(f"Expected {self.expected_tasks} task prompts, observed {len(groups)}")
            selected_records: dict[int, list[dict[str, Any]]] = {}
            selected_metadata: dict[int, list[dict[str, Any]]] = {}
            provenance: dict[str, Any] = {}
            for signature, records in groups:
                metadata = self.record_metadata_by_prompt.get(signature, [])
                chosen, rows, selection = contract.select_request_quantiles(records, metadata)
                if selection["available_request_count"] >= int(self.requests_per_episode):
                    raise ValueError("Native request cap reached; full task trajectory is not certified")
                binding = self.task_bindings.get(signature)
                if binding is None:
                    raise ValueError("Captured prompt lacks RoboTwin task/reset binding")
                rows = [{**row, **binding, "group": self.robotwin_job["group"],
                         "repeat": self.robotwin_job["repeat"], "pool": self.robotwin_job["pool"]}
                        for row in rows]
                selected_records[signature] = chosen
                selected_metadata[signature] = rows
                provenance[str(signature)] = {**selection, **binding}
            self.records_by_prompt = selected_records
            self.record_metadata_by_prompt = selected_metadata
            self.requests_per_episode = contract.REQUESTS_PER_TASK
            try:
                super().save()
            finally:
                if self.original_sample_noise is not None and self.noise_generator is not None:
                    self.noise_generator.sample_noise = self.original_sample_noise
            manifest = contract.read(self.manifest_output)
            samples = manifest.get("samples") or []
            expected_rows = self.expected_tasks * contract.REQUESTS_PER_TASK * len(contract.FLOW_INDICES)
            if len(samples) != expected_rows:
                raise ValueError(f"Expected {expected_rows} selected flow rows, observed {len(samples)}")
            manifest.update({
                "robotwin_capture_version": 1,
                "source_kind": "physical_simulator_expert_execution",
                "benchmark": "RoboTwin 2.0", "group": self.robotwin_job["group"],
                "repeat": self.robotwin_job["repeat"], "pool": self.robotwin_job["pool"],
                "flow_seed": self.robotwin_job["flow_seed"],
                "native_noise_sha256": self.noise_hashes,
                "request_selection": "full-trajectory quantiles 0,25,50,75,100 percent",
                "selected_requests": provenance, "success_filtering": False,
                "demonstration_actions_used": False,
                "closed_loop_expert_execution": True,
                "task_reset_keys": [{"task": row["task"], "task_index": row["task_index"], "seed": row["seed"]}
                                    for row in self.robotwin_job["tasks"][:self.expected_tasks]],
            })
            self.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return RoboTwinQuantileCollector


def execute(protocol_path: Path, job_id: str, output: Path, smoke: bool) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    job = find_job(protocol, job_id)
    if output.exists():
        raise FileExistsError(output)
    parent = contract.read(Path(job["parent"]["path"]))
    if parent.get("group") != job["group"] or parent.get("checkpoint") != job["checkpoint"]:
        raise ValueError("Capture job differs from frozen expert parent")
    tasks = [dict(row) for row in job["tasks"]]
    if smoke:
        tasks = [dict(tasks[0], seed=contract.uint31(contract.SEED_NAMESPACE, "technical-smoke", job_id, tasks[0]["task_index"]))]
        job = {**job, "tasks": tasks, "flow_seed": contract.uint31(contract.SEED_NAMESPACE, "technical-smoke-flow", job_id)}

    # The existing native runner owns the audited environment setup and camera/action contract.
    import importlib.util
    formal_path = VLA / "scripts/watch_robotwin_experts_formal_v2.py"
    spec = importlib.util.spec_from_file_location("robotwin_capture_formal", formal_path)
    formal = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(formal)
    formal.slicer.runtime_setup(parent)

    import torch
    import yaml
    from scripts import run_iclr2027_robotwin_native_pi05_smoke_v2 as native
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation, robotwin
    from lerobot.envs.configs import RoboTwinEnvConfig
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401
    from lerobot.utils.constants import ACTION
    import collect_pi05_block_regmeanpp_calibration as collector_base

    os.environ[robotwin.OFFICIAL_INSTRUCTION_MAX_ENV] = "1000000"
    fast_instruction.install()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device is required")
    checkpoint = Path(job["checkpoint"]) / "pretrained_model"
    config = PreTrainedConfig.from_pretrained(checkpoint)
    native.apply_runtime_visual_override(config)
    if int(config.n_action_steps) != 10 or int(config.chunk_size) != 50:
        raise ValueError("RoboTwin PI0.5 action interface differs from frozen 10/50 contract")
    config.pretrained_path, config.device = checkpoint, "cuda"
    first_cfg = RoboTwinEnvConfig(task=tasks[0]["task"], episode_length=None,
                                  task_config="demo_clean", action_mode="joint")
    policy = make_policy(config, env_cfg=first_cfg, rename_map={})
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    horizon_path = Path(parent["robotwin"]["root"]) / "task_config/_eval_step_limit.yml"
    horizons = yaml.safe_load(horizon_path.read_text())

    output.mkdir(parents=True, exist_ok=False)
    tensor_output = output / "replay.safetensors"
    manifest_output = output / "replay.json"
    os.environ.update({
        "PI05_BLOCK_REGMEANPP_TASK": job["group"],
        "PI05_BLOCK_REGMEANPP_TENSOR_OUTPUT": str(tensor_output),
        "PI05_BLOCK_REGMEANPP_MANIFEST_OUTPUT": str(manifest_output),
        "PI05_BLOCK_REGMEANPP_CALIBRATION_POLICY": str(checkpoint),
        "PI05_BLOCK_REGMEANPP_MAX_CALLS": "3840",
        "PI05_BLOCK_REGMEANPP_MAX_CALLS_PER_PROMPT": "1",
        "PI05_BLOCK_REGMEANPP_REQUESTS_PER_EPISODE": "256",
        "PI05_BLOCK_REGMEANPP_EPISODE_AWARE": "1",
        "PI05_BLOCK_REGMEANPP_FULL_PREFIX": "1",
        "PI05_BLOCK_REGMEANPP_FLOW_INDICES": "0,5,9",
        "PI05_BLOCK_REGMEANPP_REQUEST_MODE": "initial",
        "PI05_BLOCK_REGMEANPP_START_SEED": str(tasks[0]["seed"]),
        "PI05_LIBERO_INIT_STATE_OFFSET": "0",
    })
    Collector = make_collector_class(collector_base, torch)
    collector = Collector(resolve_native_policy(policy), job, len(tasks))
    collector.install()
    started = {
        "schema": "robotwin_tcr_capture_attempt_v1", "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": contract.bind(protocol_path), "job_id": job_id, "smoke": smoke,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "checkpoint": contract.bind(checkpoint / "adapter_model.safetensors"),
        "tasks": tasks, "success_not_used": True,
        "official_instruction_max_descriptions": 1_000_000,
        "lazy_instruction_implementation": contract.bind(HERE / "fast_official_instruction.py"),
        "reset_preflight_instruction_contract": protocol.get("reset_preflight", {}).get("instruction_contract"),
    }
    save(output / "started.json", started)
    rows = []
    try:
        torch.cuda.reset_peak_memory_stats()
        for task in tasks:
            task_name = task["task"]
            env_cfg = RoboTwinEnvConfig(task=task_name, episode_length=None,
                                        task_config="demo_clean", action_mode="joint")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=config)

            def capture_preprocessor(batch: dict[str, Any], *, selected=task):
                collector.capture_task_texts(batch["task"])
                collector.bind_current_task(selected)
                return preprocessor(batch)

            env = make_env(env_cfg, n_envs=1, use_async_envs=False)[task_name][0]
            try:
                native.SEED = int(task["seed"])
                native.MAX_STEPS = int(horizons[task_name])
                journal = native.ProgressJournal(output / f"{task_name}-{task['seed']}.jsonl")
                begin = time.monotonic()
                loop = native.run_control_loop(
                    env=env, policy=policy, torch_module=torch,
                    preprocess_observation=preprocess_observation,
                    env_preprocessor=env_preprocessor, preprocessor=capture_preprocessor,
                    postprocessor=postprocessor, env_postprocessor=env_postprocessor,
                    action_key=ACTION, progress=journal,
                )
                row = {"task": task_name, "task_index": task["task_index"], "seed": task["seed"],
                       "success": loop["success"], "steps": loop["steps"], "horizon": native.MAX_STEPS,
                       "seconds": time.monotonic() - begin,
                       "initial_observation_sha256": loop["initial_observation_sha256"]}
                save(output / f"{task_name}-{task['seed']}.json", row)
                rows.append(row)
            finally:
                env.close()
        collector.save()
        manifest = contract.read(manifest_output)
        expected_rows = len(tasks) * 15
        if manifest.get("sample_count") != expected_rows or manifest.get("success_filtering") is not False:
            raise ValueError("Saved capture manifest coverage/scope differs")
        result = {
            "status": "technical_smoke_complete" if smoke else "capture_job_complete",
            "formal_result": False, "job_id": job_id, "smoke": smoke,
            "episodes": len(rows), "requests": len(tasks) * 5, "flow_rows": expected_rows,
            "replay_manifest": contract.bind(manifest_output), "replay_tensor": contract.bind(tensor_output),
            "closed_loop_rows": [contract.bind(output / f"{row['task']}-{row['seed']}.json") for row in rows],
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "success_values_not_used_for_acceptance_or_scheduling": True,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        save(output / "complete.json", result)
        return result
    except BaseException as error:
        save(output / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                                       "finished_at": datetime.now(timezone.utc).isoformat()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "run"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--job-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        result = contract.build_protocol(args.protocol.parent)
        print(json.dumps({"protocol": str(args.protocol), "jobs": len(result["jobs"]),
                          "episodes": result["episodes"], "flow_rows": result["flow_rows"]}))
        return
    protocol = load_protocol(args.protocol.resolve())
    if args.action == "validate":
        print(json.dumps({"status": "valid_no_gpu", "jobs": len(protocol["jobs"]),
                          "episodes": protocol["episodes"], "flow_rows": protocol["flow_rows"]}))
        return
    if not args.job_id or args.output is None:
        parser.error("run requires --job-id and --output")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit():
        raise ValueError("run requires one explicit numeric CUDA_VISIBLE_DEVICES")
    result = execute(args.protocol.resolve(), args.job_id, args.output.resolve(), args.smoke)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
