#!/usr/bin/env python3
"""Build the RoboTwin Table-5 TIES row from the frozen 15k expert bank.

Uses the same global PEFT-safe task-vector TIES implementation as Table 1.
The Table-1 selected density=0.3 and alpha=0.9 transfer without a RoboTwin
development sweep.  No policy rollout or outcome influences materialization.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys


REPO = Path("/mnt/workspace/Wilson/parameter-fusion/vla-merge")
TABLE = Path("/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/iclr2027-table5-20260910")
PROTOCOL = TABLE / "evaluation-queues/robotwin-three-expert-formal-v2/protocol.json"
PROTOCOL_SHA = "ed47fc5340cfb53a4a1c497a08ef3aca984c0ad50182748ac3df3ca7e060d0de"
BANK = TABLE / "evaluation-queues/robotwin-three-expert-table5-master-v1/expert-bank-015000.json"
BANK_SHA = "a69322b7d4e83f92716b2c01fa6640dcd068b96f60048a90926fa5e92808be10"
OUTPUT = TABLE / "evaluation-queues/robotwin-three-expert-table5-formal-methods-v1/materialized/ties/pretrained_model"
BASE = Path("/mnt/workspace/Wilson/parameter-fusion/pi05_lora_finetune_v2_20260826/ckpt/theta0/checkpoints/000200/pretrained_model")
NAMES = ("precision", "receptacle", "coordination")

spec = importlib.util.spec_from_file_location(
    "pi05_parameter_baselines",
    REPO / "scripts/materialize_pi05_multiexpert_parameter_baseline.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("baseline implementation missing")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def validate_bank(
    path: Path,
    expected_sha256: str,
    *,
    base_root: Path,
    base_model_sha256: str,
    experts: list,
    adapted_keys: list[str],
) -> dict:
    if path.resolve() != PROTOCOL.resolve() or expected_sha256 != PROTOCOL_SHA:
        raise ValueError("formal RoboTwin protocol binding differs")
    if module.sha256(PROTOCOL) != PROTOCOL_SHA or module.sha256(BANK) != BANK_SHA:
        raise ValueError("formal protocol or frozen checkpoint inventory changed")
    protocol = json.loads(PROTOCOL.read_text())
    bank = json.loads(BANK.read_text())
    if (
        protocol.get("status") != "robotwin_three_expert_formal_protocol_frozen_amendment1"
        or protocol.get("episodes_total") != 540
        or protocol.get("method") != "Experts"
        or tuple(bank.get("groups", ())) != NAMES
        or bank.get("source_task_count") != 30
        or Path(bank.get("base_model", "")).resolve() != base_root
        or bank["base_model_file"]["sha256"] != base_model_sha256
        or tuple(expert.name for expert in experts) != NAMES
    ):
        raise ValueError("RoboTwin formal expert identity differs")
    by_name = {row["group"]: row for row in bank["experts"]}
    for expert in experts:
        recorded = by_name[expert.name]
        if (
            recorded["step"] != 15000
            or (Path(recorded["checkpoint"]) / "pretrained_model").resolve() != expert.root
            or recorded["adapter"]["sha256"] != expert.adapter_sha256
        ):
            raise ValueError(f"RoboTwin {expert.name} expert differs")
    if len(adapted_keys) != 422 or len(experts[0].lora_a) != 414 or len(experts[0].saved_tensors) != 8:
        raise ValueError("RoboTwin adapted tensor scope differs")
    roots = TABLE / "evaluation-queues/robotwin-three-expert-formal-v2/runs"
    for group in ("coordination", "receptacle", "precision"):
        for repeat in (1, 2, 3):
            receipt = roots / group / f"repeat-{repeat:02d}/complete.json"
            if not receipt.is_file():
                raise RuntimeError(f"Experts formal job incomplete: {group}/{repeat}")
            value = json.loads(receipt.read_text())
            if value.get("formal_result") is not True or value.get("status") != "robotwin_experts_formal_job_complete":
                raise ValueError(f"Experts receipt invalid: {receipt}")
    return {
        "kind": "robotwin_three_expert_15k_formal_bank",
        "path": str(PROTOCOL),
        "sha256": PROTOCOL_SHA,
        "source_bank_inventory": str(BANK),
        "source_bank_inventory_sha256": BANK_SHA,
        "formal_experts_jobs": 9,
        "formal_experts_episodes": 540,
        "expert_order": list(NAMES),
        "checkpoint_step": 15000,
        "base_model_sha256": base_model_sha256,
        "adapted_tensor_count": len(adapted_keys),
        "lora_pair_count": 414,
        "direct_tensor_count": 8,
    }


module.validate_expert_bank_manifest = validate_bank
module.__file__ = __file__


def main() -> None:
    if OUTPUT.exists():
        manifest = OUTPUT / "parameter_baseline_manifest.json"
        if manifest.is_file():
            value = json.loads(manifest.read_text())
            if value.get("method") == "ties" and value.get("model_sha256") == module.sha256(OUTPUT / "model.safetensors"):
                print(json.dumps({"status": "already_complete", "output": str(OUTPUT)}))
                return
        raise FileExistsError(f"incomplete or foreign TIES output: {OUTPUT}")
    bank = {row["group"]: row for row in json.loads(BANK.read_text())["experts"]}
    argv = [
        str(Path(__file__)),
        "--base-model", str(BASE),
        "--expert-bank-manifest", str(PROTOCOL),
        "--expected-expert-bank-manifest-sha256", PROTOCOL_SHA,
        "--method", "ties",
        "--ties-keep-fraction", "0.3",
        "--alpha", "0.9",
        "--disjoint", "mean",
        "--device", "cpu",
        "--output", str(OUTPUT),
    ]
    for name in NAMES:
        argv.extend(("--expert", f"{name}={bank[name]['checkpoint']}/pretrained_model"))
    if os.environ.get("ROBOTWIN_TIES_PLAN_ONLY") == "1":
        argv.append("--plan-only")
    sys.argv = argv
    module.main()


if __name__ == "__main__":
    main()
