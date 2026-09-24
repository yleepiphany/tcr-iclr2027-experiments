"""Strict CPU acceptance of a frozen Fast-WAM TCR evaluation task."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(bank_path: Path, output: Path, job_id: str,
          checkpoint_sha256: str) -> dict:
    bank = json.loads(bank_path.read_text())
    if bank.get("schema") != "fastwam_tcr_eval_bank_v1":
        raise ValueError("Wrong evaluation bank")
    jobs = [j for j in bank["jobs"] if j["id"] == job_id]
    if len(jobs) != 1:
        raise ValueError("Job absent or duplicated in bank")
    job = jobs[0]
    started = json.loads((output / "started.json").read_text())
    completed = json.loads((output / "complete.json").read_text())
    if (started.get("job_id") != job_id or started.get("bank_sha256") != sha(bank_path)
            or started.get("checkpoint_sha256") != checkpoint_sha256
            or started.get("stats_sha256") != job["stats_sha256"]
            or completed.get("complete") is not True
            or completed.get("job_id") != job_id
            or completed.get("checkpoint_sha256") != checkpoint_sha256):
        raise ValueError("Start/complete identity differs")
    episodes_path = output / "episodes.jsonl"
    if completed.get("episodes_sha256") != sha(episodes_path):
        raise ValueError("Episode record digest differs")
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
    if (len(rows) != len(job["stock_indices"])
            or completed.get("episodes") != len(rows)):
        raise ValueError("Incomplete episode count")
    expected = list(zip(job["stock_indices"], job["reset_sha256"]))
    for row, (index, reset_sha) in zip(rows, expected):
        if (row.get("job_id") != job_id or row.get("stage") != job["stage"]
                or row.get("suite") != job["suite"]
                or row.get("task_id") != job["task_id"]
                or row.get("stock_state_index") != index
                or row.get("state_sha256") != reset_sha
                or row.get("seed") != job["seed"]
                or type(row.get("success")) is not bool):
            raise ValueError(f"Episode identity differs at stock index {index}")
    return {"accepted": True, "job_id": job_id, "stage": job["stage"],
            "episode_count": len(rows), "episodes_sha256": sha(episodes_path),
            "checkpoint_sha256": checkpoint_sha256}
