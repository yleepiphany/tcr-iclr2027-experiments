#!/usr/bin/env python3
"""Build one frozen OpenVLA TCR repair candidate and verify native reload.

This worker owns no scheduling.  It consumes the independently accepted A/B
request capture, preserves both pass exports, and never runs success episodes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from expert_bank import ExpertBank
from materialize_soup import ORDER, sha, write
from native_oft import NativePolicy
from pass_engine import run_pass


SUITES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}


def _payload_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def load_requests(capture: Path, pool: str) -> tuple[dict[str, list[dict]], dict]:
    if pool not in ("A", "B"):
        raise ValueError("Only the frozen A/B pools are allowed")
    result: dict[str, list[dict]] = {}
    files: list[Path] = []
    identities: set[str] = set()
    for expert in ORDER:
        suite = SUITES[expert]
        folder = capture / "jobs" / f"{pool}-{expert}" / "capture"
        paths = sorted(folder.glob("task-*.pt"))
        if len(paths) != 10:
            raise ValueError(f"Expected ten task payloads for {pool}/{expert}")
        records = []
        for expected_task, path in enumerate(paths):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (payload.get("pool"), payload.get("suite"), payload.get("task_id")) != (
                    pool, suite, expected_task):
                raise ValueError("Captured request identity differs from frozen order")
            if len(payload.get("records", [])) != 5:
                raise ValueError("Each captured task must provide exactly five requests")
            for ordinal, record in enumerate(payload["records"]):
                if record.get("identity_max_abs") != 0.0 or not isinstance(record.get("inputs"), dict):
                    raise ValueError("Captured native replay was not exact")
                identity = (f"{pool}/{suite}/task-{expected_task:02d}/"
                            f"request-{record['request_index']:04d}/ordinal-{ordinal}")
                if identity in identities:
                    raise ValueError("Duplicate request identity")
                identities.add(identity)
                records.append({"id": identity, "inputs": record["inputs"]})
            files.append(path)
        if len(records) != 50:
            raise ValueError("Frozen pass requires 50 requests per expert")
        result[expert] = records
    return result, {
        "pool": pool,
        "payload_files": len(files),
        "requests": sum(map(len, result.values())),
        "payload_digest": _payload_digest(files),
    }


def recipe(candidate: str, pass_id: str, provenance: dict) -> dict:
    if candidate not in ('R1', 'R2'):
        raise ValueError('Only R1/R2 repair candidates are frozen')
    return {
        "candidate": candidate,
        "row_mode": "uniform" if candidate == "R1" else "llm_action_4_context_4",
        "pass_id": pass_id,
        "pool": pass_id,
        "mass_rule": "relative" if pass_id == "A" else "uniform",
        "row_seed": 274001 if pass_id == "A" else 274002,
        "cap": 8,
        "ridge_multiplier": 0.05,
        "max_correction_ratio": 3.0,
        "input_provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--candidate", choices=("R1", "R2"), required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-acceptance", type=Path, required=True)
    parser.add_argument("--block-gate", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("No resume, overwrite, or retry within one build attempt")
    acceptance = json.loads(args.capture_acceptance.read_text())
    gate = json.loads(args.block_gate.read_text())
    plan = json.loads(args.plan.read_text())
    if (acceptance.get("accepted") is not True or acceptance.get("retained_requests") != 400
            or acceptance.get("tcr_checkpoint_built") is not False):
        raise ValueError("A/B capture lacks independent acceptance")
    if (gate.get("complete") is not True or gate.get("is_tcr") is not False
            or gate.get("episodes") != 0 or gate.get("modules") != plan.get("linear_count")
            or gate.get("blocks") != plan.get("block_count")
            or gate.get("native_actions_restored_exactly") is not True):
        raise ValueError("Corrected native block gate was not accepted")
    if (plan.get("planned_rows_per_pass_if_all_linears_solved") != 698000
            or plan.get("row_cap_per_request_across_calls") != 8
            or plan.get("requests_per_expert") != 50):
        raise ValueError("Frozen OpenVLA row plan changed")

    args.output.mkdir(parents=True)
    write(args.output / "started.json", {
        "started_unix": time.time(),
        "prior": str(args.prior.resolve()),
        "ledger": str(args.ledger.resolve()),
        "capture": str(args.capture.resolve()),
        "capture_acceptance_sha256": sha(args.capture_acceptance),
        "block_gate_sha256": sha(args.block_gate),
        "block_plan_sha256": sha(args.plan),
        "candidate": args.candidate,
        "recipes": [recipe(args.candidate, "A", {"pending": True}),
                    recipe(args.candidate, "B", {"pending": True})],
        "success_evaluation": False,
        "no_retry": True,
    })

    policy = NativePolicy(args.prior, "libero_spatial")
    torch.cuda.reset_peak_memory_stats()
    with ExpertBank(args.ledger) as bank:
        requests_a, provenance_a = load_requests(args.capture, "A")
        provenance_a.update({"capture_acceptance_sha256": sha(args.capture_acceptance)})
        pass_a = run_pass(policy, bank, requests_a, plan,
                          recipe(args.candidate, "A", provenance_a),
                          template=args.prior, run=args.output / "pass-A", device="cuda")
        del requests_a
        gc.collect()

        requests_b, provenance_b = load_requests(args.capture, "B")
        provenance_b.update({"capture_acceptance_sha256": sha(args.capture_acceptance)})
        validation_request = next(iter(requests_b["spatial"]))["inputs"]
        pass_a_manifest = args.output / "pass-A/manifest.json"
        ridge_source = {
            "candidate": args.candidate,
            "pass_id": "A",
            "manifest": str(pass_a_manifest.resolve()),
            "manifest_sha256": sha(pass_a_manifest),
        }
        pass_b = run_pass(policy, bank, requests_b, plan,
                          recipe(args.candidate, "B", provenance_b),
                          template=Path(pass_a["checkpoint"]), run=args.output / "pass-B",
                          fixed_ridges=pass_a["ridge_map"],
                          fixed_ridge_source=ridge_source, device="cuda")
        native_before_reload = policy.replay(validation_request)
        del requests_b

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    final_checkpoint = Path(pass_b["checkpoint"])
    del policy
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = NativePolicy(final_checkpoint, "libero_spatial")
    native_after_reload = reloaded.replay(validation_request)
    if not np.array_equal(native_before_reload, native_after_reload):
        raise ValueError("Exported final checkpoint changed native actions after reload")
    del reloaded, validation_request
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "complete": True,
        "candidate": args.candidate,
        "row_mode": "uniform" if args.candidate == "R1" else "llm_action_4_context_4",
        "is_tcr": True,
        "passes": ["A", "B"],
        "pass_A_checkpoint": pass_a["checkpoint"],
        "checkpoint": pass_b["checkpoint"],
        "rows_per_pass": 698000,
        "rows_total": 1396000,
        "blocks_per_pass": plan["block_count"],
        "modules_per_pass": plan["linear_count"],
        "native_reload_verified": True,
        "native_actions_reload_exact": True,
        "pass_B_reuses_exact_pass_A_ridges": True,
        "pass_A_ridge_map_sha256": hashlib.sha256(json.dumps(
            pass_a["ridge_map"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "success_evaluation": False,
        "episodes": 0,
        "process_peak_allocated_bytes": peak_allocated,
        "process_peak_reserved_bytes": peak_reserved,
        "pass_A_manifest_sha256": sha(args.output / "pass-A/manifest.json"),
        "pass_B_manifest_sha256": sha(args.output / "pass-B/manifest.json"),
    }
    write(args.output / "final-manifest.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
