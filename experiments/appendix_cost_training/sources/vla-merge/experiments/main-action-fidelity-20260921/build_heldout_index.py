#!/usr/bin/env python3
"""Strictly assemble the complete 400-request C146 held-out bank index."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
RUN = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/heldout-raw-attempt-02"
RESET_AUDIT = WORK / "coordination/2026-09-21/c146-heldout-reset-hash-audit.json"
OUT = RUN / "bank-index.json"
SUITES = {"spatial": "libero_spatial", "object": "libero_object",
          "goal": "libero_goal", "long": "libero_10"}
REQUIRED = ("image_0", "image_1", "image_2", "image_mask_0", "image_mask_1",
            "image_mask_2", "tokens", "masks", "x_t", "time", "native_velocity")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def request_hash(values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode())
        digest.update(tensor_hash(values[name]).encode())
    return digest.hexdigest()


def task_languages():
    os.environ['LIBERO_CONFIG_PATH'] = str(WORK/'.datasets/LIBERO/20260919/config-standard')
    from libero.libero import benchmark
    result = {}
    for short, suite_name in SUITES.items():
        suite = benchmark.get_benchmark_dict()[suite_name]()
        mapping = {suite.get_task(task_id).language: task_id for task_id in range(10)}
        if len(mapping) != 10:
            raise ValueError(f"{suite_name} has duplicate task language")
        result[short] = mapping
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan_path, terminal_path = RUN/'plan.json', RUN/'queue-ended.json'
    plan, terminal = json.loads(plan_path.read_text()), json.loads(terminal_path.read_text())
    if terminal.get('status') != 'complete' or terminal.get('stopped') is not False:
        raise ValueError("Raw held-out queue is not a healthy terminal run")
    if len(terminal.get('states', {})) != 8 or any(
            row.get('status') != 'complete' for row in terminal['states'].values()):
        raise ValueError("Raw held-out queue lacks eight accepted jobs")
    reset = json.loads(RESET_AUDIT.read_text())
    if reset.get('accepted') is not True or reset.get('plan_sha256') != sha(plan_path):
        raise ValueError("Reset content audit is absent or bound to another plan")
    languages = task_languages()
    experts = plan['expert_bank']['experts']
    rows = []
    files = []
    for episode in ('heldout-D1', 'heldout-D2'):
        for short, suite_name in SUITES.items():
            folder = RUN/episode/'inputs'/short/'raw-heldout'
            manifest_path, tensor_path = folder/'requests.json', folder/'requests.safetensors'
            manifest = json.loads(manifest_path.read_text())
            if manifest['tensor_sha256'] != sha(tensor_path):
                raise ValueError(f"Tensor hash differs: {tensor_path}")
            if manifest['episode_id'] != episode or manifest['expert'] != short:
                raise ValueError("Manifest episode/expert differs")
            if manifest['source_policy_sha256'] != experts[short]['dense_checkpoint']['model_sha256']:
                raise ValueError("Expert model identity differs")
            file_row = {"episode_id": episode, "suite": suite_name,
                        "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
                        "tensor": str(tensor_path), "tensor_sha256": sha(tensor_path)}
            files.append(file_row)
            with safe_open(str(tensor_path), framework='pt') as handle:
                keys = set(handle.keys())
                for sample in manifest['samples']:
                    prefix = f"sample_{sample['index']:03d}."
                    present = sorted(key[len(prefix):] for key in keys if key.startswith(prefix))
                    if not set(REQUIRED) <= set(present):
                        raise ValueError(f"Incomplete native fields: {episode}/{short}/{sample['index']}")
                    values = {name: handle.get_tensor(prefix+name) for name in present}
                    task_id = languages[short].get(sample['prompt'])
                    if task_id is None:
                        raise ValueError(f"Prompt does not map to LIBERO task: {sample['prompt']}")
                    identity = f"{short}-t{task_id:02d}-{episode}-q{sample['request_slot']}"
                    rows.append({
                        "request_id": identity, "suite": suite_name, "suite_short": short,
                        "task_id": task_id, "episode_id": episode,
                        "request_slot": sample['request_slot'], "request_index": sample['request_index'],
                        "prompt": sample['prompt'], "prompt_signature": sample['prompt_signature'],
                        "init_state_id": sample['init_state_id'],
                        "initial_observation_sha256": sample['initial_observation_sha256'],
                        "simulator_seed": sample['simulator_seed'],
                        "tensor_file": str(tensor_path), "tensor_file_sha256": file_row['tensor_sha256'],
                        "sample_index": sample['index'], "fields": present,
                        "raw_input_sha256": request_hash(values),
                        "native_noise_sha256": tensor_hash(values['x_t']),
                        "captured_native_velocity_sha256": tensor_hash(values['native_velocity']),
                        "teacher_expert_path": experts[short]['dense_checkpoint']['path'],
                        "teacher_expert_sha256": experts[short]['dense_checkpoint']['model_sha256'],
                    })
    expected = {(suite, task) for suite in SUITES.values() for task in range(10)}
    grouped = {key: [] for key in expected}
    for row in rows:
        grouped[(row['suite'],row['task_id'])].append(row)
    if len(rows) != 400 or len({row['request_id'] for row in rows}) != 400:
        raise ValueError("Expected 400 unique request identities")
    for key, values in grouped.items():
        if len(values) != 10:
            raise ValueError(f"{key} does not have ten requests")
        if {(v['episode_id'],v['request_slot']) for v in values} != {
                (episode,slot) for episode in ('heldout-D1','heldout-D2') for slot in range(5)}:
            raise ValueError(f"{key} lacks 2x5 request coverage")
        observations = {v['episode_id']: v['initial_observation_sha256'] for v in values}
        if len(observations) != 2 or len(set(observations.values())) != 2:
            raise ValueError(f"{key} episodes do not have distinct initial observations")
    output = {
        "schema": "main_action_fidelity_heldout_bank_v1", "accepted": True,
        "usage": "heldout action-fidelity only; forbidden for calibration, tuning, or model selection",
        "plan": str(plan_path), "plan_sha256": sha(plan_path),
        "queue_terminal": str(terminal_path), "queue_terminal_sha256": sha(terminal_path),
        "reset_content_audit": str(RESET_AUDIT), "reset_content_audit_sha256": sha(RESET_AUDIT),
        "request_count": len(rows), "task_count": len(grouped),
        "episodes_per_task": 2, "requests_per_episode": 5,
        "complete_native_inputs": True, "teacher_actions_status": "pending",
        "coordinate_metric": {"execute_steps": 10, "continuous_dims": [0,1,2,3,4,5],
                              "coordinate_scale": "pending normalizer identity audit",
                              "cosine_norm_epsilon": 1e-12},
        "files": files, "requests": sorted(rows,key=lambda r:r['request_id']),
    }
    OUT.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps({"accepted":True,"requests":len(rows),"tasks":len(grouped),
                      "files":len(files),"output":str(OUT)}))


if __name__ == '__main__': main()
