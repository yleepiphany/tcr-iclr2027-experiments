#!/usr/bin/env python3
"""Frozen, resumable Fast-WAM own-suite experts evaluation for ICLR 2027 v2.

Prepare validates the existing procedural reset bank and checkpoint identities.
Wait admits only a genuinely empty GPU and holds the shared GPU lease.  Run
writes an episode receipt before attempting the next episode; audit accepts a
suite/repeat only when all 100 expected episodes and their hashes are present.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path("/mnt/workspace/Wilson/parameter-fusion")
FAST = ROOT / "vla-merge_table4/Fast-WAM"
SOURCE = FAST / "source"
RUNTIME = ROOT / "vla-merge-runtime"
BANK = RUNTIME / "experiments/iclr2027-table1-20260910/reset-banks/libero-procedural-clean-v1"
SOUP_MANIFEST = RUNTIME / "experiments/openvla-fastwam-diagnostics-20260923/fastwam-soup/manifest.json"
OUT = RUNTIME / "experiments/iclr2027-fastwam-experts-formal-v2-20260924"
SUITES = {"spatial": "libero_spatial", "object": "libero_object", "goal": "libero_goal", "long": "libero_10"}
REPEATS = ("repeat-01", "repeat-02", "repeat-03")
EXPECTED_STEPS = {"spatial": 12000, "object": 10000, "goal": 10000, "long": 9000}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(data)
    return h.hexdigest()


def state_sha(state) -> str:
    """Match vla_merge.libero_procedural_bank.sha256_state exactly."""
    import numpy as np

    value = np.ascontiguousarray(state)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def selected_states(bank: dict, selection: dict, suite: str, task_id: int):
    import numpy as np

    key = f"{suite}/{task_id:02d}"
    task = bank["tasks"][key]
    indices = selection["tasks"][key]
    if len(indices) != 10 or len(set(indices)) != 10:
        raise ValueError(f"Expected ten unique resets: {key}")
    states_path = BANK / task["states_file"]
    if sha(states_path) != task["states_file_sha256"]:
        raise ValueError(f"Reset bank file changed: {states_path}")
    states = np.load(states_path, allow_pickle=False)
    if states.dtype.str != task["state_dtype"] or list(states.shape[1:]) != task["state_shape"]:
        raise ValueError(f"Reset array shape/dtype changed: {key}")
    selected = []
    for index in indices:
        if not 0 <= index < len(states):
            raise ValueError(f"Reset index out of bounds: {key}/{index}")
        expected = task["states"][index]["raw_sha256"]
        actual = state_sha(states[index])
        if actual != expected:
            raise ValueError(f"Reset bytes changed: {key}/{index}")
        selected.append((index, expected, states[index]))
    return selected


def prepare() -> None:
    if (OUT / "contract.json").exists():
        raise FileExistsError("Frozen contract exists; inspect it instead of overwriting")
    source = read_json(SOUP_MANIFEST)
    if source.get("complete") is not True or source.get("source_steps") != EXPECTED_STEPS:
        raise ValueError("Four-source checkpoint manifest is incomplete or differs")
    bank_path = BANK / "manifest.json"
    bank_sha = sha(bank_path)
    bank = read_json(bank_path)
    if bank.get("bank_id") != "procedural-clean-v1":
        raise ValueError("Wrong procedural reset bank")
    checkpoints, normalizers = {}, {}
    for expert in SUITES:
        checkpoint = source["sources"][expert]
        path = Path(checkpoint["path"])
        stat = path.stat()
        if stat.st_size != checkpoint["size"] or stat.st_mtime_ns != checkpoint["mtime_ns"]:
            raise ValueError(f"Checkpoint stat changed: {path}")
        if path.name != f"step_{EXPECTED_STEPS[expert]:06d}.pt":
            raise ValueError(f"Wrong checkpoint step: {path}")
        checkpoints[expert] = checkpoint
        normalizer = source["suite_normalizers"][expert]
        stats_path = Path(normalizer["path"])
        if sha(stats_path) != normalizer["sha256"]:
            raise ValueError(f"Normalizer changed: {stats_path}")
        normalizers[expert] = normalizer
    jobs = []
    all_hashes = {}
    for repeat in REPEATS:
        selection_path = BANK / "selections" / f"{repeat}.json"
        selection = read_json(selection_path)
        if selection["repeat_id"] != repeat or selection["bank_manifest_sha256"] != bank_sha:
            raise ValueError(f"Selection identity changed: {repeat}")
        if selection["eval_seed"] not in (274001, 274002, 274003):
            raise ValueError(f"Unexpected evaluation seed: {repeat}")
        for expert, suite in SUITES.items():
            for task_id in range(10):
                resets = selected_states(bank, selection, suite, task_id)
                key = f"{suite}/{task_id:02d}"
                all_hashes.setdefault(key, set()).update(h for _, h, _ in resets)
            jobs.append({"id": f"experts-{expert}-{repeat}", "expert": expert, "suite": suite,
                         "repeat": repeat, "seed": selection["eval_seed"], "episodes": 100,
                         "selection_path": str(selection_path), "selection_sha256": sha(selection_path),
                         "checkpoint": checkpoints[expert], "normalizer": normalizers[expert]})
    for key, hashes in all_hashes.items():
        if len(hashes) != 30:
            raise ValueError(f"Repeated reset across formal repeats: {key}")
    contract = {"schema": "fastwam_experts_procedural_formal_v2", "method": "Experts",
                "source_manifest": str(SOUP_MANIFEST), "source_manifest_sha256": sha(SOUP_MANIFEST),
                "bank": str(bank_path), "bank_sha256": bank_sha,
                "evaluator": str(Path(__file__).resolve()), "evaluator_sha256": sha(Path(__file__).resolve()),
                "native_evaluator": str(SOURCE / "experiments/libero/eval_libero_single.py"),
                "native_evaluator_sha256": sha(SOURCE / "experiments/libero/eval_libero_single.py"),
                "native_config": str(SOURCE / "configs/sim_libero.yaml"),
                "native_config_sha256": sha(SOURCE / "configs/sim_libero.yaml"),
                "jobs": jobs, "expected_jobs": 12, "expected_episodes": 1200,
                "episode_seed_formula": "eval_seed + ordinal_in_task; ordinal=0..9",
                "note": "No stock-reset development or seed1000 checkpoint sweep episodes are reused."}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "contract.json", contract)
    print(json.dumps({"contract": str(OUT / "contract.json"), "jobs": len(jobs),
                      "episodes": sum(job["episodes"] for job in jobs)}, sort_keys=True))


def load_job(job_id: str):
    contract_path = OUT / "contract.json"
    contract = read_json(contract_path)
    if contract.get("schema") != "fastwam_experts_procedural_formal_v2" or len(contract["jobs"]) != 12:
        raise ValueError("Formal contract incomplete")
    jobs = [job for job in contract["jobs"] if job["id"] == job_id]
    if len(jobs) != 1:
        raise ValueError(f"Unknown or duplicate job: {job_id}")
    return contract, sha(contract_path), jobs[0]


def expected_rows(job: dict):
    bank = read_json(BANK / "manifest.json")
    selection_path = Path(job["selection_path"])
    if sha(selection_path) != job["selection_sha256"]:
        raise ValueError("Selection changed")
    selection = read_json(selection_path)
    expected = []
    for task_id in range(10):
        for index, reset_sha, state in selected_states(bank, selection, job["suite"], task_id):
            expected.append((task_id, index, reset_sha, state))
    if len(expected) != 100:
        raise ValueError("Job is not 100 episodes")
    return expected


def rows_from(path: Path, job: dict, contract_sha: str, expected: list):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = [json.loads(line) for line in lines if line]
    if len(rows) > len(expected):
        raise ValueError("Too many episode rows")
    for pos, row in enumerate(rows):
        task_id, index, reset_sha, _ = expected[pos]
        ordinal = pos % 10
        if (row.get("job_id") != job["id"] or row.get("contract_sha256") != contract_sha
                or row.get("checkpoint_sha256") != job["checkpoint"]["sha256"]
                or row.get("selection_sha256") != job["selection_sha256"]
                or row.get("suite") != job["suite"] or row.get("repeat") != job["repeat"]
                or row.get("task_id") != task_id or row.get("reset_index") != index
                or row.get("reset_sha256") != reset_sha or row.get("seed") != job["seed"]
                or row.get("episode_ordinal") != ordinal
                or row.get("episode_seed") != job["seed"] + ordinal
                or type(row.get("success")) is not bool):
            raise ValueError(f"Episode receipt differs at position {pos}")
    return rows


def audit(job_id: str) -> dict:
    _, contract_sha, job = load_job(job_id)
    expected = expected_rows(job)
    directory = OUT / "results" / job_id
    rows = rows_from(directory / "episodes.jsonl", job, contract_sha, expected)
    complete_path = directory / "complete.json"
    if len(rows) != 100 or not complete_path.is_file():
        return {"status": "incomplete", "job_id": job_id, "episodes": len(rows)}
    complete = read_json(complete_path)
    if (complete.get("status") != "complete" or complete.get("job_id") != job_id
            or complete.get("contract_sha256") != contract_sha
            or complete.get("checkpoint_sha256") != job["checkpoint"]["sha256"]
            or complete.get("episodes_sha256") != sha(directory / "episodes.jsonl")
            or complete.get("episodes") != 100
            or complete.get("successes") != sum(int(r["success"]) for r in rows)):
        raise ValueError("Complete receipt does not match episodes")
    return {"status": "complete", "job_id": job_id, "episodes": 100,
            "successes": complete["successes"], "checkpoint_sha256": complete["checkpoint_sha256"]}


def run(job_id: str, physical_gpu: int) -> None:
    contract, contract_sha, job = load_job(job_id)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("Physical GPU mapping must be explicit")
    if os.environ.get("FASTWAM_LEASE_GPU") != str(physical_gpu):
        raise RuntimeError("Worker requires inherited shared GPU lease")
    if sha(Path(contract["bank"])) != contract["bank_sha256"]:
        raise ValueError("Reset bank manifest changed")
    if (sha(Path(contract["evaluator"])) != contract["evaluator_sha256"]
            or sha(Path(contract["native_evaluator"])) != contract["native_evaluator_sha256"]
            or sha(Path(contract["native_config"])) != contract["native_config_sha256"]):
        raise ValueError("Evaluator or native configuration changed")
    checkpoint = Path(job["checkpoint"]["path"])
    stat = checkpoint.stat()
    if (stat.st_size, stat.st_mtime_ns) != (job["checkpoint"]["size"], job["checkpoint"]["mtime_ns"]):
        raise ValueError("Checkpoint stat differs from SHA-bound contract")
    stats_path = Path(job["normalizer"]["path"])
    if sha(stats_path) != job["normalizer"]["sha256"]:
        raise ValueError("Normalizer hash changed")
    expected = expected_rows(job)
    directory = OUT / "results" / job_id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if audit(job_id)["status"] == "complete":
            return
        rows = rows_from(directory / "episodes.jsonl", job, contract_sha, expected)
        started = directory / "started.json"
        if started.exists():
            prior = read_json(started)
            if prior.get("contract_sha256") != contract_sha or prior.get("checkpoint_sha256") != job["checkpoint"]["sha256"]:
                raise ValueError("Prior attempt used another contract or checkpoint")
        else:
            write_json(started, {"job_id": job_id, "contract_sha256": contract_sha,
                                 "checkpoint_sha256": job["checkpoint"]["sha256"],
                                 "normalizer_sha256": job["normalizer"]["sha256"],
                                 "selection_sha256": job["selection_sha256"],
                                 "host": socket.gethostname(), "physical_gpu": physical_gpu,
                                 "started_at": time.time()})
        os.environ["LIBERO_CONFIG_PATH"] = str(FAST.parent / "DreamZero/run/libero_config")
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / ".datasets/FastWAM/models"))
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(physical_gpu)
        sys.path[:0] = [str(FAST / ".python-packages"), str(SOURCE / "src"), str(SOURCE),
                        str(SOURCE / "experiments/libero"), str(RUNTIME / "references/LIBERO-MergeVLA")]
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from libero.libero import benchmark
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        import eval_libero_single as native

        torch.backends.cuda.matmul.allow_tf32 = False
        native._install_robosuite_mujoco_compatibility()
        with initialize_config_dir(version_base="1.3", config_dir=str(SOURCE / "configs")):
            cfg = compose(config_name="sim_libero.yaml", overrides=[
                "task=libero_uncond_2cam224_1e-4", f"EVALUATION.task_suite_name={job['suite']}",
                f"seed={job['seed']}"])
        cfg.EVALUATION.num_trials = 10
        if bool(cfg.EVALUATION.visualize_future_video) or int(cfg.EVALUATION.num_inference_steps) != 10:
            raise ValueError("Native action-only evaluation configuration drift")
        model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
        weights = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
        model.mot.load_state_dict(weights["mot"], strict=True)
        if model.proprio_encoder is None or "proprio_encoder" not in weights:
            raise ValueError("Expert checkpoint lacks proprio encoder")
        model.proprio_encoder.load_state_dict(weights["proprio_encoder"], strict=True)
        del weights
        model = model.to("cuda:0").eval()
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
        suite = benchmark.get_benchmark_dict()[job["suite"]]()
        episode_path = directory / "episodes.jsonl"
        with episode_path.open("a") as stream:
            for task_id in range(10):
                first = task_id * 10
                pending = [(ordinal, item) for ordinal, item in enumerate(expected[first:first + 10])
                           if first + ordinal >= len(rows)]
                if not pending:
                    continue
                cfg.EVALUATION.task_id = task_id
                task = suite.get_task(task_id)
                env, description = native.get_libero_env(task, native.LIBERO_ENV_RESOLUTION, job["seed"])
                try:
                    for ordinal, (_, index, reset_sha, state) in pending:
                        episode_seed = job["seed"] + ordinal
                        native.set_global_seed(episode_seed, get_worker_init_fn=False)
                        env.seed(episode_seed)
                        success, replay_images, future_clips, _ = native.run_single_episode(
                            env=env, initial_state=state, task_description=description,
                            model=model, processor=processor, cfg=cfg, episode_idx=index,
                            action_horizon=int(cfg.data.train.num_frames) - 1,
                            input_w=int(cfg.data.train.video_size[1]),
                            input_h=int(cfg.data.train.video_size[0]), model_device="cuda:0")
                        del replay_images, future_clips
                        row = {"job_id": job_id, "contract_sha256": contract_sha,
                               "checkpoint_sha256": job["checkpoint"]["sha256"],
                               "selection_sha256": job["selection_sha256"], "suite": job["suite"],
                               "repeat": job["repeat"], "task_id": task_id,
                               "reset_index": index, "reset_sha256": reset_sha,
                               "seed": job["seed"], "episode_ordinal": ordinal,
                               "episode_seed": episode_seed, "success": bool(success)}
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        rows.append(row)
                finally:
                    env.close()
        if len(rows) != 100:
            raise ValueError("Incomplete job")
        write_json(directory / "complete.json", {"status": "complete", "job_id": job_id,
                    "contract_sha256": contract_sha, "checkpoint_sha256": job["checkpoint"]["sha256"],
                    "episodes": 100, "successes": sum(int(r["success"]) for r in rows),
                    "episodes_sha256": sha(episode_path), "host": socket.gethostname(),
                    "physical_gpu": physical_gpu, "finished_at": time.time()})


def gpu_snapshot(index: int):
    values = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                                      "--format=csv,noheader,nounits"], text=True)
    mapping = {}
    for line in values.splitlines():
        fields = [x.strip() for x in line.split(",")]
        mapping[int(fields[0])] = (fields[1], int(fields[2]))
    uuid, used = mapping[index]
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                    "--format=csv,noheader"], text=True)
    pids = [line.split(",", 1)[1].strip() for line in apps.splitlines()
            if line.split(",", 1)[0].strip() == uuid]
    return used, pids


def run_claimed(job_id: str, physical_gpu: int) -> None:
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(physical_gpu), FASTWAM_LEASE_GPU=str(physical_gpu),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false")
    directory = OUT / "results" / job_id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "attempts.log").open("a") as log:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "run",
                               "--job-id", job_id, "--gpu", str(physical_gpu)],
                              env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode == 0 and audit(job_id)["status"] == "complete":
        return
    write_json(directory / "last-attempt.json", {"returncode": proc.returncode,
               "job_id": job_id, "time": time.time(), "gpu": physical_gpu})
    raise RuntimeError(f"Formal job failed; inspect {directory / 'attempts.log'}")


def wait_any(job_id: str, physical_gpus: list[int], poll_seconds: int) -> None:
    load_job(job_id)
    if not physical_gpus or len(set(physical_gpus)) != len(physical_gpus):
        raise ValueError("GPU candidates must be nonempty and unique")
    lease_dir = RUNTIME / "resource-leases" / socket.gethostname()
    lease_dir.mkdir(parents=True, exist_ok=True)
    while True:
        if audit(job_id)["status"] == "complete":
            return
        for physical_gpu in physical_gpus:
            used, pids = gpu_snapshot(physical_gpu)
            if used > 512 or pids:
                continue
            lease = lease_dir / f"gpu-{physical_gpu}.lock"
            with lease.open("a+") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                used, pids = gpu_snapshot(physical_gpu)
                if used > 512 or pids:
                    continue
                run_claimed(job_id, physical_gpu)
                return
        time.sleep(poll_seconds)


def wait(job_id: str, physical_gpu: int, poll_seconds: int) -> None:
    wait_any(job_id, [physical_gpu], poll_seconds)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    for cmd in ("audit", "run", "wait"):
        c = sub.add_parser(cmd)
        c.add_argument("--job-id", required=True)
        if cmd != "audit":
            c.add_argument("--gpu", type=int, choices=range(8), required=True)
        if cmd == "wait":
            c.add_argument("--poll-seconds", type=int, default=30)
    lane = sub.add_parser("lane")
    lane.add_argument("--lane", type=int, choices=range(4), required=True)
    lane.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                      help="Comma-separated physical GPU candidates on this node")
    lane.add_argument("--poll-seconds", type=int, default=30)
    args = p.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "audit":
        print(json.dumps(audit(args.job_id), sort_keys=True))
    elif args.command == "run":
        run(args.job_id, args.gpu)
        print(json.dumps(audit(args.job_id), sort_keys=True))
    elif args.command == "wait":
        wait(args.job_id, args.gpu, args.poll_seconds)
    else:
        contract = read_json(OUT / "contract.json")
        gpus = [int(raw) for raw in args.gpus.split(",")]
        if any(gpu not in range(8) for gpu in gpus):
            raise ValueError("GPU candidate index outside [0, 8)")
        for job in contract["jobs"][args.lane::4]:
            wait_any(job["id"], gpus, args.poll_seconds)


if __name__ == "__main__":
    main()
