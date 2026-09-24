#!/usr/bin/env python3
"""Run one frozen RoboTwin development slice under an audited shared-GPU gate.

This runner never clears a GPU or signals another process. It admits only
an explicit safe set, caps whole-device memory before model loading, records all NVML compute
PIDs, and rejects a target with any RoboTwin training process mapped to it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SLICER_PATH = ROOT / "scripts/run_iclr2027_robotwin_development_slices.py"
ALLOWED_GPUS = frozenset((2, 3, 4, 5, 6, 7))
DEFAULT_MAX_USED_MIB = 40_000.0

SPEC = importlib.util.spec_from_file_location("robotwin_development_slices", SLICER_PATH)
slicer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(slicer)


def _read_cmdline(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def _read_visible_devices(pid: int) -> str | None:
    try:
        entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    prefix = b"CUDA_VISIBLE_DEVICES="
    for entry in entries:
        if entry.startswith(prefix):
            return entry[len(prefix):].decode("utf-8", errors="replace")
    return None


def robotwin_training_processes_on_gpu(index: int) -> list[dict]:
    matches = []
    training_tokens = (
        "train_pi05_robotwin_train40_full_expert.py",
        "train_pi05_robotwin_train40_full_expert_world4.py",
        "train_pi05_robotwin_task_balanced.py",
    )
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        command = _read_cmdline(pid)
        if not command or not any(token in command for token in training_tokens):
            continue
        visible = _read_visible_devices(pid)
        if visible is None:
            if "robotwin-world4-remaining-experts-v1/precision" in command:
                physical = {0, 1, 2, 3}
            else:
                physical = set(range(8))
        else:
            physical = {int(item) for item in visible.split(",") if item.strip().isdigit()}
        if index in physical:
            matches.append(
                {"pid": pid, "cuda_visible_devices": visible, "command": command}
            )
    return sorted(matches, key=lambda item: item["pid"])


def probe_shared_admission(index: int, max_used_mib: float = DEFAULT_MAX_USED_MIB) -> dict:
    if index not in ALLOWED_GPUS:
        raise ValueError(f"Shared RoboTwin evaluation GPU is outside the audited safe set: {index}")
    if max_used_mib <= 0 or max_used_mib > DEFAULT_MAX_USED_MIB:
        raise ValueError(f"Shared evaluation memory ceiling must be in (0,{DEFAULT_MAX_USED_MIB}]")
    query = subprocess.run(
        [
            "nvidia-smi", "-i", str(index),
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [part.strip() for part in query.stdout.strip().split(",")]
    if len(fields) != 5 or int(fields[0]) != index:
        raise RuntimeError(f"Unexpected nvidia-smi inventory row: {query.stdout!r}")
    processes = subprocess.run(
        [
            "nvidia-smi", "-i", str(index),
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    compute = []
    for line in processes.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            raise RuntimeError(f"Unexpected nvidia-smi compute row: {line!r}")
        compute.append({"pid": int(parts[0]), "used_memory_mib": float(parts[1])})
    training = robotwin_training_processes_on_gpu(index)
    admission = {
        "schema_version": 1,
        "shared_eval": True,
        "physical_gpu": index,
        "gpu_uuid": fields[1],
        "memory_used_mib": float(fields[2]),
        "memory_total_mib": float(fields[3]),
        "utilization_percent": float(fields[4]),
        "max_used_mib": float(max_used_mib),
        "compute_processes": compute,
        "robotwin_training_processes": training,
        "external_processes_untouched": True,
    }
    if admission["memory_used_mib"] > max_used_mib:
        raise RuntimeError(
            f"Shared GPU memory gate failed: used={admission['memory_used_mib']} max={max_used_mib}"
        )
    if training:
        raise RuntimeError(f"RoboTwin training is active on GPU{index}: {training}")
    return admission


def run(manifest: Path, output: Path, gpu: int, max_used_mib: float) -> None:
    slice_receipt, parent = slicer.validate_slice(manifest)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES must match explicit physical --gpu")
    from scripts.run_iclr2027_robotwin_native_pi05_smoke_v2 import exclusive_gpu_lock

    with exclusive_gpu_lock(slicer.base.RUNTIME / f"resource-leases/gpu-{gpu}.lock"):
        admission = probe_shared_admission(gpu, max_used_mib)
        output.mkdir(parents=True, exist_ok=False)
        slicer.base.write(
            output / "started.json",
            {
                "slice_manifest": slicer.base.bind(manifest),
                "parent": slice_receipt["parent"],
                "admission": admission,
                "shared_eval": True,
                "formal_result": False,
            },
        )
        try:
            slicer.runtime_setup(parent)
            import time
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ValueError("Exactly one CUDA device required")
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            result = slicer.base.simulator_audit(
                slicer.selected_manifest(parent, slice_receipt["task_seed_keys"]), output, False
            )
            if result["episodes"] != len(slice_receipt["task_seed_keys"]):
                raise ValueError("Incomplete shared evaluation slice")
            row_files = [
                output / f'{task["task"]}-{seed}.json'
                for task in parent["tasks"]
                for seed in task["seeds"]
                if [task["task_index"], seed] in slice_receipt["task_seed_keys"]
            ]
            slicer.base.write(
                output / "complete.json",
                {
                    "status": "development_slice_complete",
                    "formal_result": False,
                    "plateau_eligible": False,
                    "shared_eval": True,
                    "admission": admission,
                    "slice_manifest": slicer.base.bind(manifest),
                    "parent": slice_receipt["parent"],
                    "runner": slicer.base.bind(Path(__file__)),
                    "slice_index": slice_receipt["slice_index"],
                    "slice_count": slice_receipt["slice_count"],
                    "rows": [slicer.base.bind(path) for path in row_files],
                    "result": result,
                    "seconds": time.monotonic() - begin,
                    "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
                },
            )
        except BaseException as error:
            slicer.base.write(
                output / "failure.json",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "shared_eval": True,
                    "admission": admission,
                },
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-used-mib", type=float, default=DEFAULT_MAX_USED_MIB)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    admission = probe_shared_admission(args.gpu, args.max_used_mib)
    if args.probe_only:
        print(json.dumps(admission, indent=2, sort_keys=True))
        return
    run(args.manifest.resolve(), args.output.resolve(), args.gpu, args.max_used_mib)


if __name__ == "__main__":
    main()
