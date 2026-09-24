#!/usr/bin/env python3
"""Materialize the frozen three-expert RoboTwin uniform Soup as an exact dense prior."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
VLA = HERE.parents[1]
WORK = VLA.parent
RUNTIME = WORK / "vla-merge-runtime"
CAPTURE = RUNTIME / "experiments/claude-robotwin-tcr-20260923/capture-v1/protocol.json"
SOURCE = (RUNTIME / "experiments/iclr2027-table5-20260910/evaluation-queues/"
          "robotwin-three-expert-table5-formal-methods-v1/materialized/model_soups/pretrained_model")
BASE = WORK / "pi05_lora_finetune_v2_20260826/ckpt/theta0/checkpoints/000200/pretrained_model"
ROOT = RUNTIME / "experiments/claude-robotwin-tcr-20260923/dense-soup-v1"
MATERIALIZER = VLA / "scripts/materialize_pi05_lora_mean_dense.py"
PREFLIGHT = VLA / "scripts/preflight_pi05_adapter_dense_equivalence.py"


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


def validate_source() -> dict:
    protocol = json.loads(CAPTURE.read_text())
    manifest = json.loads((SOURCE / "fusion_manifest.json").read_text())
    expected = {}
    for job in protocol["jobs"]:
        expected.setdefault(job["group"], str((Path(job["checkpoint"]) / "pretrained_model").resolve()))
    if manifest.get("method") != "pi05_exact_multi_lora_task_delta":
        raise ValueError("Not the frozen exact RoboTwin Model Soup")
    if manifest.get("expert_count") != 3 or manifest.get("experts") != expected:
        raise ValueError("Soup expert identity differs from the capture protocol")
    if manifest.get("coefficients") != {name: 1 / 3 for name in expected}:
        raise ValueError("Soup is not uniform over the three experts")
    if Path(manifest.get("base_model", "")).resolve() != BASE.resolve():
        raise ValueError("Soup base differs")
    if sha256(SOURCE / "adapter_model.safetensors") != manifest.get("adapter_sha256"):
        raise ValueError("Soup adapter hash differs")
    return manifest


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(ROOT)
    manifest = validate_source()
    ROOT.mkdir(parents=True)
    output = ROOT / "pretrained_model"
    plan = {
        "schema": "robotwin_three_expert_dense_soup_build_v1", "cpu_only": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE.resolve()), "source_adapter_sha256": manifest["adapter_sha256"],
        "base": str(BASE.resolve()), "base_sha256": sha256(BASE / "model.safetensors"),
        "coefficients": manifest["coefficients"], "output": str(output.resolve()),
    }
    save(ROOT / "plan.json", plan)
    command = [sys.executable, str(MATERIALIZER), "--adapter", str(SOURCE),
               "--base-model", str(BASE), "--output", str(output)]
    with (ROOT / "materialize.log").open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
    save(ROOT / "materialize-exit.json", {"returncode": result.returncode, "command": command,
                                          "finished_at": datetime.now(timezone.utc).isoformat()})
    if result.returncode:
        raise RuntimeError("Dense Soup materialization failed")
    verification = ROOT / "exact-verification.json"
    command = [sys.executable, str(PREFLIGHT), "--base-model", str(BASE),
               "--adapter", f"soup={SOURCE}", "--dense-checkpoint", f"soup={output}",
               "--expected-lora-pairs", "414", "--expected-direct-tensors", "8",
               "--expected-logical-adapted-tensors", "422", "--output", str(verification)]
    with (ROOT / "verification.log").open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError("Dense Soup exact verification failed")
    receipt = json.loads(verification.read_text())
    checks = receipt.get("dense_content_checks") or []
    if len(checks) != 1 or checks[0].get("status") != "passed_exact" or checks[0].get("exact_tensor_count") != 813:
        raise ValueError("Dense Soup verification receipt differs")
    final = {
        "schema": "robotwin_three_expert_dense_soup_v1", "status": "passed_parameter_exact",
        "path": str(output.resolve()), "model_sha256": sha256(output / "model.safetensors"),
        "source_adapter": str(SOURCE.resolve()), "source_adapter_sha256": manifest["adapter_sha256"],
        "coefficients": manifest["coefficients"], "verification": str(verification.resolve()),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    save(ROOT / "COMPLETE.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
