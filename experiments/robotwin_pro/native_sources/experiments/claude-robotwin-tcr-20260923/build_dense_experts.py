#!/usr/bin/env python3
"""Materialize and exactly audit the three frozen RoboTwin 15k experts on CPU."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
VLA = HERE.parents[1]
WORK = VLA.parent
RUNTIME = WORK / "vla-merge-runtime"
PROTOCOL = RUNTIME / "experiments/claude-robotwin-tcr-20260923/capture-v1/protocol.json"
ROOT = RUNTIME / "experiments/claude-robotwin-tcr-20260923/dense-experts-v1"
BASE = WORK / "pi05_lora_finetune_v2_20260826/ckpt/theta0/checkpoints/000200/pretrained_model"
MATERIALIZER = VLA / "scripts/materialize_pi05_lora_mean_dense.py"
PREFLIGHT = VLA / "scripts/preflight_pi05_adapter_dense_equivalence.py"
GROUPS = ("coordination", "receptacle", "precision")


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


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def frozen_adapters() -> dict[str, Path]:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("schema") != "robotwin_tcr_native_ab_capture_v1":
        raise ValueError("Wrong capture protocol")
    result: dict[str, Path] = {}
    for job in protocol["jobs"]:
        path = (Path(job["checkpoint"]) / "pretrained_model").resolve()
        previous = result.setdefault(job["group"], path)
        if previous != path:
            raise ValueError(f"Multiple expert checkpoints for {job['group']}")
    if tuple(result) != GROUPS:
        raise ValueError(f"Expert order differs: {tuple(result)}")
    for name, path in result.items():
        config = json.loads((path / "adapter_config.json").read_text())
        if Path(config.get("base_model_name_or_path", "")).resolve() != BASE.resolve():
            raise ValueError(f"{name} base model differs")
        if int(config.get("r", -1)) != 64 or float(config.get("lora_alpha", -1)) != 64:
            raise ValueError(f"{name} LoRA contract differs")
    return result


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(ROOT)
    adapters = frozen_adapters()
    ROOT.mkdir(parents=True)
    plan = {
        "schema": "robotwin_three_expert_dense_build_v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "cpu_only": True,
        "protocol": binding(PROTOCOL), "base_model": str(BASE.resolve()),
        "base_model_sha256": sha256(BASE / "model.safetensors"),
        "materializer": binding(MATERIALIZER), "preflight": binding(PREFLIGHT),
        "experts": [
            {"name": name, "adapter": str(path),
             "adapter_model_sha256": sha256(path / "adapter_model.safetensors"),
             "output": str((ROOT / name / "pretrained_model").resolve())}
            for name, path in adapters.items()
        ],
        "one_at_a_time": True, "training": False,
    }
    save(ROOT / "plan.json", plan)
    finished = []
    for row in plan["experts"]:
        command = [sys.executable, str(MATERIALIZER), "--adapter", row["adapter"],
                   "--base-model", str(BASE), "--output", row["output"]]
        log = ROOT / f"{row['name']}.log"
        started = datetime.now(timezone.utc).isoformat()
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
        receipt = {"name": row["name"], "command": command, "started_at": started,
                   "finished_at": datetime.now(timezone.utc).isoformat(),
                   "returncode": result.returncode, "log": binding(log)}
        save(ROOT / f"{row['name']}.exit.json", receipt)
        if result.returncode != 0:
            save(ROOT / "FAILED.json", receipt)
            raise RuntimeError(f"Dense materialization failed for {row['name']}")
        finished.append(receipt)
    verification = ROOT / "exact-verification.json"
    command = [sys.executable, str(PREFLIGHT), "--base-model", str(BASE)]
    for name, adapter in adapters.items():
        command.extend(["--adapter", f"{name}={adapter}"])
    for name in adapters:
        command.extend(["--dense-checkpoint", f"{name}={ROOT/name/'pretrained_model'}"])
    command.extend(["--expected-lora-pairs", "414", "--expected-direct-tensors", "8",
                    "--expected-logical-adapted-tensors", "422", "--output", str(verification)])
    log = ROOT / "verification.log"
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        failure = {"returncode": result.returncode, "command": command, "log": binding(log)}
        save(ROOT / "FAILED.json", failure)
        raise RuntimeError("Exact dense verification failed")
    verified = json.loads(verification.read_text())
    if verified.get("status") != "passed" or len(verified.get("dense_content_checks") or []) != 3:
        raise ValueError("Exact dense verification receipt differs")
    by_name = {row["name"]: row for row in verified["dense_content_checks"]}
    experts = []
    for name, adapter in adapters.items():
        dense = ROOT / name / "pretrained_model"
        check = by_name[name]
        if check.get("status") != "passed_exact" or check.get("exact_tensor_count") != 813:
            raise ValueError(f"{name} dense expert was not exact")
        experts.append({
            "name": name,
            "source_adapter": {"path": str(adapter),
                               "model_sha256": sha256(adapter / "adapter_model.safetensors")},
            "dense_checkpoint": {"path": str(dense.resolve()),
                                 "model_sha256": sha256(dense / "model.safetensors"),
                                 "manifest": binding(dense / "dense_equivalent_manifest.json")},
        })
    bank = {
        "schema_version": 1, "kind": "robotwin_three_expert_peft_safe_dense_bank",
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "passed_parameter_exact",
        "deployment_policy": "dense_only_no_unmerged_adapter_fallback",
        "protocol": binding(PROTOCOL),
        "base": {"path": str(BASE.resolve()), "model_sha256": sha256(BASE / "model.safetensors"),
                 "tensor_count": 813},
        "expert_order": list(GROUPS), "experts": experts,
        "verification": binding(verification), "training": False,
    }
    save(ROOT / "expert-dense-bank.json", bank)
    save(ROOT / "COMPLETE.json", {"status": "complete", "experts": 3,
                                  "bank": binding(ROOT / "expert-dense-bank.json"),
                                  "finished_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "complete", "experts": 3,
                      "bank": str(ROOT / "expert-dense-bank.json")}, indent=2))


if __name__ == "__main__":
    main()
