"""Registered three-arm control contract; no CUDA imports or historical edits."""
import copy
from collections import defaultdict
from pathlib import Path

SCHEMA = "tcr_pass2_controls_20260919_v1"
ARMS = ("reuse", "alternate", "soup")
NAMES = ("spatial", "object", "goal", "long")
ROWS = 1331600


def attach_request_slots(manifest):
    """Add deterministic request slots without reordering/altering stored tensors."""
    out = copy.deepcopy(manifest)
    if out.get("sample_count") != 150 or len(out.get("samples", [])) != 150:
        raise ValueError("Expected exactly150 stored states")
    grouped = defaultdict(lambda:defaultdict(list))
    for index, sample in enumerate(out["samples"]):
        if sample["index"] != index or sample["vision_count"] != 3:
            raise ValueError("Stored state index or camera count differs")
        grouped[sample["prompt_signature"]][sample["request_index"]].append(sample)
    if len(grouped) != 10 or any(len(requests) != 5 for requests in grouped.values()):
        raise ValueError("Expected10 prompts with5 requests each")
    for requests in grouped.values():
        for slot, request in enumerate(sorted(requests)):
            samples = requests[request]
            if sorted(s["flow_index"] for s in samples) != [0, 5, 9]:
                raise ValueError("Missing native flow triple")
            for sample in samples:
                if "selected_request_slot" in sample and sample["selected_request_slot"] != slot:
                    raise ValueError("Existing slot differs from deterministic ordering")
                sample["selected_request_slot"] = slot
    return out


def validate_config(config):
    if (config.get("schema") != SCHEMA or config.get("arm") not in ARMS or
            "variant" in config or config.get("repeat") != 1 or
            config.get("row_cap_per_request_module") != 16 or
            config.get("expert_masses") != "uniform_quarter" or
            config.get("merged_slots") != [] or config.get("expected_realized_rows") != ROWS):
        raise ValueError("Unregistered recipe")
    for key in ("start_point", "start_point_manifest", "start_point_sha256",
                "numeric_ridge_source", "numeric_ridge_source_sha256", "base_model", "bank_path"):
        if not config.get(key):
            raise ValueError(f"Missing identity: {key}")
    if set(config.get("calibration", {})) != set(NAMES):
        raise ValueError("Wrong calibration expert set")
    expected = "fresh_repeat01" if config["arm"] == "reuse" else "table3_across"
    if config.get("calibration_family") != expected:
        raise ValueError("Calibration family differs from assigned arm")
    if (config["arm"] == "soup") != (config.get("initialization") == "soup"):
        raise ValueError("Incorrect initialization label")
    for item in config["calibration"].values():
        if not all(item.get(k) for k in ("tensor", "manifest", "tensor_sha256", "manifest_sha256")):
            raise ValueError("Missing calibration identity")


def validate_cli(args, config):
    validate_config(config)
    for key, expected in (("prior_model", config["start_point"]),
                          ("base_model", config["base_model"]),
                          ("dense_expert_bank", config["bank_path"])):
        if Path(getattr(args, key)).resolve() != Path(expected).resolve():
            raise ValueError(f"CLI identity differs: {key}")
    for values, key in ((args.calibration, "tensor"), (args.manifest, "manifest")):
        actual = [value.split("=", 1) for value in values]
        expected = {n:str(Path(v[key]).resolve()) for n,v in config["calibration"].items()}
        if len(actual) != 4 or {n:str(Path(p).resolve()) for n,p in actual} != expected:
            raise ValueError("CLI calibration bindings differ")
