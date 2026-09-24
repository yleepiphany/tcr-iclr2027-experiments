#!/usr/bin/env python3
"""Run the frozen RoboTwin solver while retaining its prior's deployment support."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
VLA = WORK / "vla-merge"
SOURCE = WORK / "pi05_lora_finetune_v2_20260826"
sys.path[:0] = [str(VLA / "experiments/claude-robotwin-tcr-20260923"),
                str(VLA / "experiments/claude-20260917"),
                str(VLA / "scripts"), str(VLA / "src"), str(SOURCE / "src")]

import robotwin_solver_contract as contract  # noqa: E402
import materialize_second_round_v3 as materializer  # noqa: E402
import pi05_table3_contract as table3_contract  # noqa: E402

SOUP = WORK / "vla-merge-runtime/experiments/claude-robotwin-tcr-20260923/dense-soup-v1/pretrained_model"
MODELS = WORK / "vla-merge-runtime/experiments/robotwin-tcr-local-codex-20260924/tcr-checkpoints-v2"
SUPPORT_SHA = {
    "policy_preprocessor.json": "714723707403f6a5d1c18ccfa287c16c7b36e13fa1b6447a24431398a241e424",
    "policy_postprocessor.json": "9f2a2c28bd3d3ba779477b84586fec0efc98522bf4286f77ae24e1b167e42a39",
    "policy_preprocessor_step_3_normalizer_processor.safetensors": "6ce304a9f8704ed56ab9b56d5c478abd878c8a89bf4d665692b624862cd0d234",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": "b8ba9b84423b717d05abcbc2999e0b2ff89f8b1c82d2232a37d1f37c14253948",
}
OTHER_NORMALIZER_SHA = "792f8ad0cba252068cc6f2acaaa2a6d7fa0494a29a08426ffff3ad61c7a56b81"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arg_path(name: str) -> Path:
    prefix = f"--{name}="
    values = [item[len(prefix):] for item in sys.argv[1:] if item.startswith(prefix)]
    if len(values) != 1:
        raise ValueError(f"Expected one {name} argument")
    return Path(values[0]).resolve()


def retained_support(experts: dict[str, Path], output: Path, prior: Path) -> None:
    if tuple(experts) != contract.NAMES:
        raise ValueError("RoboTwin expert order differs")
    if prior != SOUP.resolve() and not (MODELS.resolve() in prior.parents
                                       and prior.name == "pretrained_model"
                                       and prior.parent.name.endswith("-passA")):
        raise ValueError("RoboTwin prior is neither frozen Soup nor same-repeat pass A")
    for filename, expected in SUPPORT_SHA.items():
        if sha(prior / filename) != expected:
            raise ValueError(f"RoboTwin prior support changed: {filename}")
        for name, expert in experts.items():
            actual = sha(expert / filename)
            expected_expert = (OTHER_NORMALIZER_SHA if filename.startswith("policy_preprocessor_step_3")
                               and name in ("coordination", "receptacle") else expected)
            if actual != expected_expert:
                raise ValueError(f"RoboTwin {name} support changed: {filename}")
        shutil.copy2(prior / filename, output / filename)
    shutil.copytree(prior / "tokenizer", output / "tokenizer")
    config = json.loads((prior / "config.json").read_text())
    if config.get("use_peft") is not False:
        raise ValueError("RoboTwin prior is not a dense deployment checkpoint")
    config["pretrained_path"] = str(output)
    (output / "config.json").write_text(json.dumps(config, indent=4) + "\n")


def main() -> None:
    which = os.environ.get("ROBOTWIN_TCR_PASS")
    if which not in {"A", "B"}:
        raise ValueError("ROBOTWIN_TCR_PASS must be A or B")
    prior = arg_path("prior-model")
    if which == "A" and prior != SOUP.resolve():
        raise ValueError("Pass A must start at the frozen Soup model")
    if which == "B" and prior == SOUP.resolve():
        raise ValueError("Pass B must start at same-repeat pass A")
    # The generic solver imports this contract from inside main().
    sys.modules["pi05_tcr_e_dense_contract"] = contract
    materializer.validate_second_round_config = contract.validate_second_round_config
    materializer.validate_second_round_trace = contract.validate_second_round_trace
    materializer.SECOND_ROUND_MASSES = {"uniform_three": "none"}
    table3_contract.validate_rows = (
        lambda metrics, names, variant: contract.validate_realized_rows(metrics)
    )
    materializer.copy_support_files = (
        lambda experts, output: retained_support(experts, output, prior)
    )
    completed = False
    try:
        materializer.main()
        completed = True
    finally:
        import torch
        report = {
            "schema": "robotwin_local_cuda_peak_v1",
            "original_materializer": str(VLA / "experiments/claude-20260917/materialize_second_round_v3.py"),
            "solver_completed": completed,
            "cuda_available": torch.cuda.is_available(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if report["cuda_available"]:
            report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
            report["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            report["exit_free_mib"] = free_bytes / 1024**2
            report["device_total_mib"] = total_bytes / 1024**2
        target = os.environ.get("ROBOTWIN_TCR_PEAK_PATH")
        if target:
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, path)


if __name__ == "__main__":
    main()
