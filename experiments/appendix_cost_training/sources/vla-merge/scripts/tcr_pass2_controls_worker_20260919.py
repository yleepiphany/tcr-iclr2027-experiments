"""Isolated new-run adapter. Historical solvers/evaluation sources remain read-only."""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

WORK = Path(__file__).resolve().parents[2]
SCRIPTS = WORK / "vla-merge/scripts"
SOLVER = WORK / "vla-merge/experiments/claude-20260917/materialize_second_round_v3.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--mode", choices=("solve", "development"), required=True)
    parser.add_argument("--resources", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.mode == "development":
        from tcr_night_worker_20260919 import main as evaluate
        return evaluate()
    from tcr_night_worker_20260919 import stop_group
    signal.signal(signal.SIGTERM, stop_group)
    signal.signal(signal.SIGINT, stop_group)
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != args.parent:
        raise RuntimeError("Parent-death identity guard failed")
    import libero.libero as inner
    inner._assets_path_cache = str(WORK / ".datasets/LIBERO/20260919/runtime/assets")
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.70, 0)
    from run_tcr_night_20260919 import read, sha
    from run_pi05_generation_path_probe import memory_free
    from tcr_pass2_controls_contract_20260919 import SCHEMA, ARMS, validate_config, validate_cli
    spec = importlib.util.spec_from_file_location("registered_pass2_solver", SOLVER)
    solver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = solver
    spec.loader.exec_module(solver)
    # Only contract dispatch changes. Numeric solve, captures and row selection are
    # the exact imported v3 source, including its fixed per-request cap handling.
    solver.SECOND_ROUND_SCHEMA = SCHEMA
    solver.SECOND_ROUND_ARMS = ARMS
    solver.validate_second_round_config = validate_config
    native_parse = solver.parse_args
    def parse():
        parsed = native_parse()
        config = read(parsed.ablation_config)
        validate_cli(parsed, config)
        if sha(Path(config["start_point"]) / "model.safetensors") != config["start_point_sha256"]:
            raise ValueError("Starting weights changed")
        if sha(config["numeric_ridge_source"]) != config["numeric_ridge_source_sha256"]:
            raise ValueError("Numeric ridge source changed")
        reference = read(config["start_point_manifest"])
        source = read(config["numeric_ridge_source"])
        if reference.get("schema") != "explicit_start_and_separate_numeric_ridge_reference_v1":
            raise ValueError("Not the declared separate start/ridge reference")
        if reference["modules"] != {k:{"ridge":v["ridge"]} for k,v in source["modules"].items()}:
            raise ValueError("Frozen numeric ridges changed")
        for item in config["calibration"].values():
            if sha(item["manifest"]) != item["manifest_sha256"] or sha(item["tensor"]) != item["tensor_sha256"]:
                raise ValueError("Calibration changed")
        return parsed
    solver.parse_args = parse
    native_solve = solver.solve_weight_multi
    calls = 0
    def guarded(*a, **kw):
        nonlocal calls
        if memory_free(int(os.environ["ITERATION_PHYSICAL_GPU"])) < 12*1024:
            raise RuntimeError("GPU free reserve below12GiB")
        result = native_solve(*a, **kw)
        calls += 1
        return result
    solver.solve_weight_multi = guarded
    begin, complete = time.monotonic(), False
    try:
        sys.argv = [str(SOLVER), *remaining]
        solver.main()
        complete = True
    finally:
        args.resources.parent.mkdir(parents=True, exist_ok=True)
        with args.resources.open("x") as stream:
            json.dump(dict(completed=complete, mode="solve", calls=calls,
                           wall_seconds=time.monotonic()-begin,
                           peak_allocator_gib=torch.cuda.max_memory_allocated()/2**30,
                           allocator_fraction=.70, duty_sleep=0, changes_policy_rng=False,
                           parent_death_guard=True, runtime_config=os.environ["LIBERO_CONFIG_PATH"],
                           numeric_solver=str(SOLVER), numeric_solver_sha256=sha(SOLVER)), stream, indent=2)


if __name__ == "__main__":
    main()
