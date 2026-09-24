#!/usr/bin/env python3
"""One frozen A-request OpenVLA path diagnosis; CPU only, no rollout/build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("Set CUDA_VISIBLE_DEVICES='' before this CPU diagnostic")

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "openvla-tcr-20260921"
sys.path[:0] = [str(HERE), str(COMMON)]

import torch

from diagnose_block_paths import compare_paths
from expert_bank import ExpertBank, resolve_block
from native_oft import NativePolicy

WORK = HERE.parents[2]
OLD_ROOT = WORK / "vla-merge-runtime/experiments/openvla-tcr-20260921"
REPAIR_ROOT = WORK / "vla-merge-runtime/experiments/claude-openvla-tcr-repair-20260922"
CAPTURE = OLD_ROOT / "expert-ab-capture-attempt-01"
LEDGER = OLD_ROOT / "local-expert-dynamic-01/identities.json"
FINAL = REPAIR_ROOT / "candidate-build-attempt-02/jobs/R2/build/final-manifest.json"
ACCEPTANCE = WORK / "coordination/2026-09-21/openvla-ab-capture-acceptance.json"
SUITES = {"spatial": "libero_spatial", "object": "libero_object",
          "goal": "libero_goal", "long": "libero_10"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(expert, block_name, output):
    if torch.cuda.is_available():
        raise RuntimeError("CUDA unexpectedly available in CPU diagnostic")
    torch.set_num_threads(2)
    if json.loads(ACCEPTANCE.read_text()).get("accepted") is not True:
        raise ValueError("Frozen A/B native request bank was not accepted")
    final = json.loads(FINAL.read_text())
    if (final.get("complete") is not True or final.get("candidate") != "R2"
            or final.get("native_reload_verified") is not True):
        raise ValueError("R2 checkpoint is not independently accepted")
    ledger = json.loads(LEDGER.read_text())
    if ledger.get("complete") is not True or expert not in ledger["experts"]:
        raise ValueError("Expert source identity missing")
    request_file = CAPTURE / "jobs" / f"A-{expert}" / "capture/task-00.pt"
    payload = torch.load(str(request_file), map_location="cpu", weights_only=False)
    if (payload["pool"], payload["suite"], payload["task_id"]) != ("A", SUITES[expert], 0):
        raise ValueError("Frozen native request identity differs")
    record = payload["records"][0]
    if record["identity_max_abs"] != 0.0:
        raise ValueError("Native request replay mismatch")
    request = record["inputs"]
    checkpoint = Path(final["checkpoint"])
    expert_checkpoint = Path(ledger["experts"][expert]["local_path"])
    policy = NativePolicy(checkpoint, SUITES[expert])
    # Historical expert auxiliary .pt files retain CUDA storage tags.  The
    # production loader calls torch.load(weights_only=True) without a device
    # mapping; scope a CPU mapping to this diagnostic's expert load only.
    original_load = torch.load

    def cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return original_load(*args, **kwargs)

    with patch.object(torch, "load", cpu_load):
        expert_policy = NativePolicy(expert_checkpoint, SUITES[expert])
    block = resolve_block(policy, block_name)
    names = [name for name, _module in policy.linear_modules()
             if name.startswith(block_name + ".")]
    if not names:
        raise ValueError("No native auxiliary Linears reached")
    with ExpertBank(LEDGER) as bank:
        expert_state, conversions = bank.block_state(expert, block_name, block)
        report = compare_paths(policy, expert_policy, block, expert_state,
                               request, names)
    result = {"cpu_only": True, "success_evaluation": False,
              "candidate": "R2", "expert": expert, "block": block_name,
              "checkpoint": str(checkpoint), "final_manifest_sha256": sha(FINAL),
              "expert_checkpoint": str(expert_checkpoint),
              "expert_ledger_sha256": sha(LEDGER),
              "request_file": str(request_file), "request_sha256": sha(request_file),
              "request_index": record["request_index"],
              "native_dtype_conversions": conversions,
              "source_code": str(Path(__file__).resolve()),
              "linear_paths": report,
              "interpretation_limit": "One frozen A request; path difference is not causal success evidence"}
    output = Path(output)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", choices=tuple(SUITES), required=True)
    parser.add_argument("--block", choices=("action_head", "proprio_projector"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.expert, args.block, args.output)
    print(json.dumps({"output": str(args.output), "linears": len(result["linear_paths"]),
                      "cpu_only": True}))
