"""CPU-only identity and recipe checks for the Table-1 TCR-E dense runner."""
from collections import Counter
import hashlib
import json
from pathlib import Path

NAMES = ("spatial", "object", "goal", "long")
RECIPE = dict(ridge_ratio=0.05, ridge_scale="feature_energy", max_correction_ratio=3.0,
              max_rows_per_sample=10, expert_loss_normalization="prior",
              expert_loss_normalization_power=1.0, expert_aggregation="mean",
              objective_grouping="expert", replay_prefix="merged",
              action_block_solver="independent_linear", require_objective_improvement=False,
              freeze_prefix_from_prior=False, freeze_action_interface_from_prior=False,
              max_states_per_calibration_source=0, expert_weight_plan=None)

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def validate_recipe(args):
    for key, expected in RECIPE.items():
        if getattr(args, key) != expected:
            raise ValueError(f"TCR-E recipe drift: {key} must be {expected!r}")
    if args.prior_model is None:
        raise ValueError("TCR-E requires the frozen dense uniform-soup prior")

def validate_dense_bank(path, experts=None, *, verify_weights=False):
    bank = json.loads(Path(path).read_text())
    if bank.get("kind") != "iclr2027_table1_peft_safe_dense_expert_bank" or bank.get("status") != "passed_parameter_exact":
        raise ValueError("Not an accepted PEFT-safe dense bank")
    if bank.get("deployment_policy") != "dense_only_no_unmerged_adapter_fallback":
        raise ValueError("Dense-only bank semantics required")
    rows = bank["experts"]
    if [row["name"] for row in rows] != list(NAMES):
        raise ValueError("Expert order or membership differs")
    result = {}
    for row in rows:
        name = row["name"]
        dense = row["dense_checkpoint"]
        if experts is not None and Path(experts[name]).resolve() != Path(row["source_adapter"]["path"]).resolve():
            raise ValueError(f"{name}: adapter provenance differs")
        for field in ("manifest", "verification"):
            identity = dense[field]
            if digest(identity["path"]) != identity["sha256"]:
                raise ValueError(f"{name}: {field} identity differs")
        model = Path(dense["path"]) / "model.safetensors"
        if not model.is_file():
            raise ValueError(f"{name}: dense checkpoint absent")
        if verify_weights and digest(model) != dense["model_sha256"]:
            raise ValueError(f"{name}: dense checkpoint digest differs")
        result[name] = {"path": dense["path"], "model_sha256": dense["model_sha256"]}
    return result

def validate_trace(manifest, dense_path, name):
    if manifest.get("task") != name or Path(manifest["calibration_policy"]).resolve() != Path(dense_path).resolve():
        raise ValueError("Trace does not identify this frozen dense expert")
    required = dict(sample_count=150, prompt_count=10, requests_per_episode=5,
                    episode_aware=True, flow_indices=[0, 5, 9], request_mode="reservoir",
                    method="pi05_full_vision_language_action_block_regmeanpp_replay_calibration")
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Incomplete or mismatched execution trace: {key}")
    samples = manifest.get("samples", [])
    if len(samples) != 150 or set(manifest.get("prompt_sample_counts", {}).values()) != {15}:
        raise ValueError("Trace must have 15 call states per task")
    grouped = {}
    for sample in samples:
        key = (sample["prompt_signature"], sample["request_index"])
        grouped.setdefault(key, []).append(sample["flow_index"])
        if sample.get("vision_count") != 3:
            raise ValueError("Trace must retain all three vision inputs")
    if len(grouped) != 50 or any(sorted(v) != [0, 5, 9] for v in grouped.values()):
        raise ValueError("Incomplete request-flow groups")
    if set(Counter(k[0] for k in grouped).values()) != {5}:
        raise ValueError("Each task needs exactly five requests")

def validate_realized_rows(metrics):
    classes = Counter()
    total = 0
    for name, entry in metrics.items():
        expected = 150 if name in ("model.time_mlp_in", "model.time_mlp_out") else 4500 if ".vision_tower." in name else 1500
        if entry.get("rows_by_expert") != {expert: expected for expert in NAMES}:
            raise ValueError(f"Per-module realized-row contract differs: {name}")
        classes[expected] += 1
        total += 4 * expected
    if classes != {150: 2, 1500: 254, 4500: 162} or total != 4441200:
        raise ValueError("Full 418-module realized-row contract not met")
    return total
