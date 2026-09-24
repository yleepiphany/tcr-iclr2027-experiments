"""Frozen contract for the mainline formal batch (13h plan rev3).

Claude owns this code; Codex owns the local capture lanes and the repeat-01 formal
suites. This module is the single place the frozen recipe and the arm matrix are
written down, so a runner cannot quietly drift from either.

THE FROZEN PRIMARY RECIPE IS ONE-PASS N0
----------------------------------------
Chosen before the formal batch and **not** revisable by any score inside it. In
particular the expert2 / mixed2 pass-2 results do not change it: those are secondary
variants and must never be relabelled TCR-E/TCR-O.

The matrix is three arms of three independent acquisitions/builds each, plus an audited
FeatCal baseline:

    N0 Full      fresh expert execution, covariance balance ON     3 builds
    S0 control   the exact paired N0 cache, balance OFF            3 builds
    Demo control demo observations + expert native generation, ON  3 builds
    FeatCal      registered checkpoint; reuse after compatibility audit

WHAT THE CONTRASTS DO AND DO NOT ISOLATE
----------------------------------------
Balance-on recomputes the coefficients from each arm's *own* features by the same
frozen rule. So **Full - Demo is the input-source effect under this algorithm**, not a
comparison at identical realised objective coefficients; the coefficients are recorded
per arm and no claim is made that the source is isolated from that induced metric
change. **Full - S0** keeps the input cache and the ridges fixed and switches only the
metric, which is the intended clean ablation.

Covariance balancing is FeatCal-inspired. No FeatCal checkpoint is used to build any
TCR arm.
"""
from __future__ import annotations

import json
from pathlib import Path

WORK = Path(__file__).resolve().parents[3]

SCHEMA = "claude_formal_mainline_v1"

# The frozen reference build. Every formal solve reproduces this recipe on fresh inputs;
# none of them may start from it.
N0_REFERENCE = (WORK / "vla-merge-runtime/experiments/claude-covariance-20260919"
                / "checkpoints/n0")
N0_REFERENCE_SHA = "32810add4b3ba03e9136811bf85c8f34280a2c3bcccd2e322524f561dc0b5ff1"

# Soup is the initialisation AND the ridge centre for every formal build, exactly as in
# the registered covariance study's first pass.
SOUP = WORK / ("vla-merge-runtime/experiments/iclr2027-table1-20260910/libero/model-soups"
               "/repeat-shared/merge/attempt-02-peft-safe-v2/pretrained_model")
RIDGE_MANIFEST = WORK / ("vla-merge-runtime/experiments/tcr-unified-night-20260919/models"
                         "/tcre-r1/block_regmeanpp_manifest.json")
BANK_SHA = "d61a5f9e56bb0f76d8186a33dc26fe283cf8cfab220e903a460f92e4507f209b"

# The start point and the ridge source are DIFFERENT checkpoints, and the solver that
# produced N0 refuses a config in which they coincide. An earlier version of this file
# inherited the pass-2 rule "start point and ridge centre are both N0" and required them
# to be equal, which is the exact conflation this study's schema exists to prevent.
SOUP_SHA = "a92aacc43146dc41663c0057f2999bd90252fb3fb316ef963f165be67e1be01a"
RIDGE_MANIFEST_SHA = "d680f11849e197535f3384f4a02c802252c51f7797adf92a1602a34195dcda00"

# The formal builds reproduce the registered N0 recipe on fresh inputs, so they go
# through the solver that produced N0 and speak its schema. Our own per-build identity
# (acquisition index, seeds, input source) is carried in a sidecar, because that schema
# has no field for it and inventing one would not be validated by anything.
SOLVER_SCHEMA = "claude_covariance_study_v1"
SOLVER_ARM = {"full": "n0", "s0": "s0", "demo": "n0"}

EXPECTED_ROWS = 1331600
MODULE_COUNT = 418
MODIFIED_TENSORS = 422
DUAL_MODULES = 414
ROW_CAP = 16
LOCAL_TEACHER = ("model.action_in_proj", "model.action_out_proj",
                 "model.time_mlp_in", "model.time_mlp_out")

# Registered acquisition identity. init 0/1/2 with the matching seed triples; the
# collision audit over 41 existing calibration manifests found no reuse of these.
REPEATS = (1, 2, 3)
START_SEEDS = {1: 291001, 2: 291002, 3: 291003}
FLOW_SEEDS = {1: 292001, 2: 292002, 3: 292003}
INIT_STATE_OFFSETS = {1: 0, 2: 1, 3: 2}

# Demo episode rank r corresponds to repeat r. The plan requires this choice to be
# adapted and tested against the actual restored dataset rather than assumed.
DEMO_EPISODE_RANK = {1: 0, 2: 1, 3: 2}

#: arm -> (covariance_balance, input source)
ARMS = {
    "full": (True, "expert_execution"),
    "s0":   (False, "expert_execution"),
    "demo": (True, "demo_observation_expert_generation"),
}
PRIMARY_ARM = "full"

# ---------------------------------------------------------------- ownership by stage
#
# Ownership is per (stage, repeat), NOT per repeat. An earlier version of this file
# split whole repeats and made this host refuse repeat 01 outright, which would have
# stranded three solves and four demo builds that are in fact ours to produce.
#
# Build stages are cross-host safe: they write into our own run directory and are
# identified by content hashes. Evaluation is the stage where a duplicate cross-host
# dispatch actually costs something, and repeat 01's evaluation runs on Codex's host
# against its local procedural bank.
STAGES = ("solve", "demo_build", "evaluate")
CLAUDE_REPEATS = REPEATS          # every solve and demo build is ours
CODEX_REPEATS = (1,)              # repeat 01 EVALUATION only
EVALUATE_REPEATS = (2, 3)         # what this host may dispatch to GPUs

# ------------------------------------------------------------- formal evaluation path
#
# The formal entrypoint is the original main-table evaluator wrapped for resource
# accounting only. Two substitutions are explicitly forbidden:
#
#   * `eval_pi05_expanded_development.py` — the D2 development wrapper. It reads init
#     states off the constructed environment for a stock offset sweep. Development
#     scores were never the main-table procedure and must not stand in for it.
#   * any stock `--env.init_states` offset scheme in place of the procedural bank.
#
# The D2 dispatchers also export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1. That override is
# NOT inherited here; the formal numeric path is whatever the main table ran.
FORMAL_EVALUATOR = "eval_tcr_10k_formal_bounded.py"
FORBIDDEN_EVALUATORS = ("eval_pi05_expanded_development.py",
                        "eval_pi05_libero_with_init_offset.py")
FORBIDDEN_EVAL_ENV = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                      "PI05_LIBERO_INIT_STATE_OFFSET",
                      "PI05_LIBERO_INIT_STATE_COUNT")
PROCEDURAL_BANK = ("vla-merge-runtime/experiments/iclr2027-table1-20260910"
                   "/reset-banks/libero-procedural-clean-v1")
#: repeat -> (selection file, its registered sha256, the eval seed it pins)
PROCEDURAL_SELECTIONS = {
    1: ("selections/repeat-01.json", "39d8f7e566e69c09", 274001),
    2: ("selections/repeat-02.json", "401ec2256ce5edc1", 274002),
    3: ("selections/repeat-03.json", "3d2963b6a5452460", 274003),
}
EVAL_SEEDS = {r: v[2] for r, v in PROCEDURAL_SELECTIONS.items()}
EVAL_EPISODES_PER_TASK = 10       # 10 tasks x 10 = 100 episodes per suite

FORMAL_EPISODES_PER_SUITE = 100
FORMAL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

AUTHORISED_GPUS = (1, 2, 4, 5, 6, 7)
MAX_EVAL_WORKERS = 12
SOLVES_PER_CARD = 1
RESERVE_MIB = 12 * 1024


def validate_config(config: dict, bank_sha: str) -> tuple[str, int]:
    """Fail closed on every identity this batch fixes."""
    if config.get("schema") != SCHEMA:
        raise ValueError("Not a mainline formal configuration")
    arm = config.get("arm")
    if arm not in ARMS:
        raise ValueError(f"Unknown formal arm {arm!r}; registered: {sorted(ARMS)}")
    repeat = config.get("repeat")
    if repeat not in REPEATS:
        raise ValueError(f"Unknown repeat {repeat!r}; registered: {list(REPEATS)}")
    balance, source = ARMS[arm]
    if config.get("covariance_balance") is not balance:
        raise ValueError(f"Arm {arm} is registered with balance={balance}")
    if config.get("input_source") != source:
        raise ValueError(f"Arm {arm} is registered with input source {source!r}")
    if config.get("teacher_alpha") != 0.0:
        raise ValueError("The frozen recipe is alpha=0; no teacher interpolation here")
    if config.get("start_point_sha256") != SOUP_SHA:
        raise ValueError("The initialisation must be the registered Soup checkpoint")
    if config.get("ridge_source_sha256") != RIDGE_MANIFEST_SHA:
        raise ValueError("The ridges must come from the registered tcre-r1 manifest")
    if config.get("start_point_sha256") == config.get("ridge_source_sha256"):
        raise ValueError("Start point and ridge source are distinct in this recipe")
    if config.get("row_cap_per_request_module") != ROW_CAP:
        raise ValueError(f"The frozen recipe fixes cap {ROW_CAP}")
    if config.get("expected_realized_rows") != EXPECTED_ROWS:
        raise ValueError(f"The frozen recipe fixes {EXPECTED_ROWS} rows")
    if config.get("dense_expert_bank_sha256") != bank_sha or bank_sha != BANK_SHA:
        raise ValueError("Expert bank differs from the registered one")
    if config.get("start_seed") != START_SEEDS[repeat]:
        raise ValueError(f"Repeat {repeat} is registered at start seed {START_SEEDS[repeat]}")
    if config.get("flow_seed") != FLOW_SEEDS[repeat]:
        raise ValueError(f"Repeat {repeat} is registered at flow seed {FLOW_SEEDS[repeat]}")
    if config.get("init_state_offset") != INIT_STATE_OFFSETS[repeat]:
        raise ValueError(f"Repeat {repeat} is registered at init {INIT_STATE_OFFSETS[repeat]}")
    if config.get("starts_from_n0"):
        raise ValueError("Formal builds reproduce the N0 recipe on fresh inputs; "
                         "they must not start from the N0 checkpoint")
    if tuple(config.get("local_teacher_modules") or ()) != LOCAL_TEACHER:
        raise ValueError("The four local-teacher modules must be declared verbatim")
    return arm, repeat


def owner_of(repeat: int, stage: str = "evaluate") -> str:
    """Who dispatches a given stage of a repeat.

    Defaults to the evaluation stage because that is the one with a real cross-host
    hazard; build stages are ours for every repeat.
    """
    if repeat not in REPEATS:
        raise ValueError(f"Repeat {repeat} has no registered owner")
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; registered: {list(STAGES)}")
    if stage in ("solve", "demo_build"):
        return "claude-remote"
    return "claude-remote" if repeat in EVALUATE_REPEATS else "codex-local"


def solver_config(arm: str, repeat: int, caches: dict) -> dict:
    """The config the N0 solver validates, in ITS schema.

    `repeat: 1` here is the solver's single-repeat *row budget*, which every formal build
    satisfies: one acquisition, cap 16, 1331600 realised rows. It is not the acquisition
    index - that is `formal_repeat` in the sidecar, and conflating the two would let a
    build claim an acquisition it did not use.
    """
    if arm not in SOLVER_ARM:
        raise ValueError(f"Unknown formal arm {arm!r}")
    if repeat not in REPEATS:
        raise ValueError(f"Unknown repeat {repeat!r}")
    balance, source = ARMS[arm]
    return {
        "schema": SOLVER_SCHEMA, "arm": SOLVER_ARM[arm],
        "teacher_alpha": 0.0, "covariance_balance": balance,
        "repeat": 1, "row_cap_per_request_module": ROW_CAP,
        "expected_realized_rows": EXPECTED_ROWS,
        "expert_masses": "covariance_2x2", "merged_slots": [],
        "calibration": f"formal mainline {arm} repeat {repeat:02d}: {source}",
        "start_point": str(SOUP), "start_point_sha256": SOUP_SHA,
        "start_point_role": "initialisation and regularisation anchor (Soup)",
        "ridge_source_manifest": str(RIDGE_MANIFEST),
        "ridge_source_sha256": RIDGE_MANIFEST_SHA,
        "ridge_source_role": ("418 numeric ridges from the fresh tcre-r1 build; "
                              "a ridge source only, NOT the initialisation"),
        "dense_expert_bank_sha256": BANK_SHA,
        "dual_propagation_modules": DUAL_MODULES,
        "local_teacher_modules": list(LOCAL_TEACHER),
        "not_a_table3_variant": True,
        "not_a_second_round_contract": ("start point and ridge source are distinct "
                                        "checkpoints in this recipe"),
        "caches": caches,
    }


def formal_sidecar(arm: str, repeat: int, caches: dict) -> dict:
    """Our identity for one build: what the solver's schema cannot express."""
    balance, source = ARMS[arm]
    config = {
        "schema": SCHEMA, "arm": arm, "repeat": repeat,
        "covariance_balance": balance, "input_source": source, "teacher_alpha": 0.0,
        "solver_schema": SOLVER_SCHEMA, "solver_arm": SOLVER_ARM[arm],
        "solver_repeat_field_is_the_row_budget_not_the_acquisition": True,
        "start_point_sha256": SOUP_SHA,
        "ridge_source_sha256": RIDGE_MANIFEST_SHA,
        "row_cap_per_request_module": ROW_CAP,
        "expected_realized_rows": EXPECTED_ROWS,
        "dense_expert_bank_sha256": BANK_SHA,
        "start_seed": START_SEEDS[repeat], "flow_seed": FLOW_SEEDS[repeat],
        "init_state_offset": INIT_STATE_OFFSETS[repeat],
        "demo_episode_rank": DEMO_EPISODE_RANK[repeat] if arm == "demo" else None,
        "starts_from_n0": False,
        "local_teacher_modules": list(LOCAL_TEACHER),
        "caches": caches,
    }
    validate_config(config, BANK_SHA)
    return config


def selection_for(repeat: int, work: Path | None = None) -> dict:
    """The registered procedural selection for a repeat, verified by hash.

    A selection pins the reset states AND the eval seed. Loading the wrong repeat's
    selection would run a different draw under the right-looking label.
    """
    import hashlib
    root = (work or WORK) / PROCEDURAL_BANK
    relative, prefix, seed = PROCEDURAL_SELECTIONS[repeat]
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"registered selection missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(prefix):
        raise ValueError(f"selection {path} is sha {digest[:16]}, registered {prefix}")
    payload = json.loads(path.read_text())
    if payload.get("repeat_id") != f"repeat-{repeat:02d}":
        raise ValueError(f"selection says repeat {payload.get('repeat_id')!r}")
    if payload.get("eval_seed") != seed:
        raise ValueError(f"selection eval seed {payload.get('eval_seed')} != {seed}")
    return {"bank": str(root), "selection": str(path), "sha256": digest,
            "eval_seed": seed, "tasks": len(payload.get("tasks") or {})}


def check_eval_command(command: list, environment: dict) -> None:
    """Refuse a development evaluator or a D2 override standing in for the formal path."""
    joined = " ".join(str(part) for part in command)
    for banned in FORBIDDEN_EVALUATORS:
        if banned in joined:
            raise ValueError(f"{banned} is a development evaluator; formal runs "
                             f"{FORMAL_EVALUATOR} over the procedural bank")
    if FORMAL_EVALUATOR not in joined:
        raise ValueError(f"formal evaluation must invoke {FORMAL_EVALUATOR}")
    if "--procedural-bank=" not in joined or "--procedural-selection=" not in joined:
        raise ValueError("formal evaluation requires the registered procedural bank "
                         "and selection; a stock offset sweep is not a substitute")
    leaked = sorted(k for k in FORBIDDEN_EVAL_ENV if k in environment)
    if leaked:
        raise ValueError(f"development environment leaked into formal evaluation: {leaked}")


def expected_request_indices(total_requests: int) -> list[int]:
    """floor((N-1)*k/4) for k=0..4 over the FULL request history.

    Verified against the first-pass across cache. Two earlier captures in the pass-2
    study passed every other check while selecting the wrong requests, so this is
    recomputed from `prompt_seen_call_counts` rather than trusted from a manifest label.
    """
    if total_requests < 5:
        raise ValueError("Fewer than five distinct requests; preserve the failure, "
                         "do not select by success")
    return [(total_requests - 1) * slot // 4 for slot in range(5)]
