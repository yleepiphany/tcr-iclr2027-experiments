#!/usr/bin/env python3
"""Finalize already materialized RoboTwin dense experts with exact per-expert audits."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from safetensors import safe_open
import torch


HERE = Path(__file__).resolve().parent
VLA = HERE.parents[1]
WORK = VLA.parent
RUNTIME = WORK / "vla-merge-runtime"
SOURCE = RUNTIME / "experiments/claude-robotwin-tcr-20260923/dense-experts-v1"
OUTPUT = RUNTIME / "experiments/claude-robotwin-tcr-20260923/dense-experts-v2"
PROTOCOL = RUNTIME / "experiments/claude-robotwin-tcr-20260923/capture-v1/protocol.json"
BASE = WORK / "pi05_lora_finetune_v2_20260826/ckpt/theta0/checkpoints/000200/pretrained_model"
PREFLIGHT = VLA / "scripts/preflight_pi05_adapter_dense_equivalence.py"
GROUPS = ("coordination", "receptacle", "precision")
PARITY_FILES = (
    "adapter_config.json", "policy_preprocessor.json", "policy_postprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def bind(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def adapters() -> dict[str, Path]:
    protocol = json.loads(PROTOCOL.read_text())
    result = {}
    for job in protocol["jobs"]:
        result.setdefault(job["group"], (Path(job["checkpoint"]) / "pretrained_model").resolve())
    if tuple(result) != GROUPS:
        raise ValueError("Frozen expert membership/order differs")
    return result


def config_parity(paths: dict[str, Path]) -> dict:
    reference_name = GROUPS[0]
    reference = json.loads((paths[reference_name] / "config.json").read_text())
    differences = {}
    for name in GROUPS[1:]:
        value = json.loads((paths[name] / "config.json").read_text())
        changed = sorted(key for key in set(reference) | set(value) if reference.get(key) != value.get(key))
        differences[name] = changed
        if changed != ["pretrained_path"]:
            raise ValueError(f"{name} config differs beyond pretrained_path: {changed}")
    file_hashes = {}
    for filename in PARITY_FILES:
        values = {name: sha256(path / filename) for name, path in paths.items()}
        if filename.endswith(".safetensors"):
            tensor_values = {}
            for name, path in paths.items():
                with safe_open(path / filename, framework="pt", device="cpu") as handle:
                    tensor_values[name] = {key: handle.get_tensor(key) for key in handle.keys()}
            reference_tensors = tensor_values[reference_name]
            for name in GROUPS[1:]:
                current = tensor_values[name]
                if set(current) != set(reference_tensors):
                    raise ValueError(f"Normalizer keys differ: {filename}/{name}")
                for key, expected in reference_tensors.items():
                    actual = current[key]
                    if (actual.dtype != expected.dtype or actual.numel() != expected.numel()
                            or not torch.equal(actual.reshape(-1), expected.reshape(-1))):
                        raise ValueError(f"Normalizer values differ: {filename}/{name}/{key}")
            file_hashes[filename] = {"semantic_values_equal": True, "file_sha256": values}
        else:
            if len(set(values.values())) != 1:
                raise ValueError(f"Deployment metadata differs: {filename}")
            file_hashes[filename] = {"byte_identical": True,
                                     "sha256": next(iter(values.values()))}
    return {"config_differences": differences,
            "only_allowed_config_difference": "pretrained_path",
            "byte_identical_deployment_files": file_hashes}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not (SOURCE / "FAILED.json").is_file():
        raise ValueError("Expected preserved v1 joint-preflight failure")
    paths = adapters()
    parity = config_parity(paths)
    OUTPUT.mkdir(parents=True)
    save(OUTPUT / "parity.json", parity)
    experts = []
    for name, adapter in paths.items():
        exit_receipt = json.loads((SOURCE / f"{name}.exit.json").read_text())
        if exit_receipt.get("returncode") != 0:
            raise ValueError(f"{name} materialization did not finish successfully")
        dense = SOURCE / name / "pretrained_model"
        receipt_path = OUTPUT / f"{name}-exact.json"
        command = [sys.executable, str(PREFLIGHT), "--base-model", str(BASE),
                   "--adapter", f"{name}={adapter}", "--dense-checkpoint", f"{name}={dense}",
                   "--expected-lora-pairs", "414", "--expected-direct-tensors", "8",
                   "--expected-logical-adapted-tensors", "422", "--output", str(receipt_path)]
        with (OUTPUT / f"{name}-exact.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            raise RuntimeError(f"Exact tensor audit failed for {name}")
        receipt = json.loads(receipt_path.read_text())
        checks = receipt.get("dense_content_checks") or []
        if len(checks) != 1 or checks[0].get("status") != "passed_exact" or checks[0].get("exact_tensor_count") != 813:
            raise ValueError(f"{name} exact tensor receipt differs")
        experts.append({
            "name": name,
            "source_adapter": {"path": str(adapter),
                               "model_sha256": sha256(adapter / "adapter_model.safetensors")},
            "dense_checkpoint": {"path": str(dense.resolve()),
                                 "model_sha256": sha256(dense / "model.safetensors"),
                                 "manifest": bind(dense / "dense_equivalent_manifest.json"),
                                 "verification": bind(receipt_path)},
        })
    bank = {
        "schema_version": 2, "kind": "robotwin_three_expert_peft_safe_dense_bank",
        "status": "passed_parameter_exact", "created_at": datetime.now(timezone.utc).isoformat(),
        "deployment_policy": "dense_only_no_unmerged_adapter_fallback",
        "source_materialization_attempt": str(SOURCE.resolve()),
        "source_joint_preflight_failure_preserved": True,
        "packaging_parity": bind(OUTPUT / "parity.json"),
        "protocol": bind(PROTOCOL),
        "base": {"path": str(BASE.resolve()), "model_sha256": sha256(BASE / "model.safetensors"),
                 "tensor_count": 813},
        "expert_order": list(GROUPS), "experts": experts,
    }
    save(OUTPUT / "expert-dense-bank.json", bank)
    save(OUTPUT / "COMPLETE.json", {"status": "complete", "experts": 3,
                                    "bank": bind(OUTPUT / "expert-dense-bank.json"),
                                    "finished_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "complete", "experts": 3,
                      "bank": str(OUTPUT / "expert-dense-bank.json")}, indent=2))


if __name__ == "__main__":
    main()
