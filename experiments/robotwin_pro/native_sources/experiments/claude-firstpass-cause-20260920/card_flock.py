#!/usr/bin/env python3
"""Card reservation by flock, keyed by host and GPU UUID (Codex C118).

Why this replaces the previous slot lease
-----------------------------------------
The earlier lease was a file whose holder was checked by reading a recorded pid and
kernel start time, and a stale holder was reclaimed by UNLINKING the file. That has two
defects Codex named: the read-then-unlink window is a race, and a crashed holder leaves
a file that somebody has to decide is dead.

`flock` has neither property. The kernel releases the lock when the holding file
description is closed, including on abrupt death, so there is no staleness to detect and
nothing to reclaim. The lock FILE is created once and **never unlinked**, so two
processes can never end up flocking two different inodes by the same name - which is the
failure that makes unlink-based schemes silently admit two holders.

Keys are `hostname/<gpu-uuid>`, not a card index: an index means different hardware on a
different host, and these directories are on shared storage.

Budget
------
* one worker per card, for builds and evaluations alike;
* at most two concurrent builds host-wide, enforced by two separate build-slot locks.

Nothing here signals anything, and a lock that cannot be taken means the caller waits or
declines - never that somebody else's job should be removed.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

WORK = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = WORK / "vla-merge-runtime/experiments/claude-card-flocks"
BUILD_SLOTS = 2
# The user explicitly opened all eight cards for this experiment queue on
# 2026-09-20.  Stable UUID-keyed flocks and one-worker-per-card still apply.
REGISTERED_GPUS = tuple(range(8))
FORBIDDEN_GPUS = ()


def gpu_uuids() -> dict[int, str]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    table = {}
    for line in out.strip().splitlines():
        index, uuid = [p.strip() for p in line.split(",")]
        table[int(index)] = uuid
    return table


def free_mib(gpu: int) -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"], text=True)
    return float(out.strip().splitlines()[0])


class Held:
    """An open, flocked file description. Closing it releases the lock; the file stays."""

    def __init__(self, path: Path, handle: int, label: str):
        self.path = path
        self.handle = handle
        self.label = label
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
        finally:
            os.close(self.handle)
        # The file is deliberately NOT unlinked.

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


def _try_lock(path: Path, label: str, note: dict) -> Held | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(handle)
        return None
    # Informational only: the lock is the flock, not this text.
    os.ftruncate(handle, 0)
    os.lseek(handle, 0, os.SEEK_SET)
    os.write(handle, json.dumps({**note, "pid": os.getpid(),
                                 "hostname": os.uname().nodename,
                                 "taken_unix": time.time()}).encode())
    os.fsync(handle)
    return Held(path, handle, label)


def card_path(gpu: int, uuid: str, root: Path | None = None) -> Path:
    return (root or DEFAULT_ROOT) / os.uname().nodename / f"gpu-{gpu}-{uuid}.lock"


def take_card(gpu: int, uuid: str, job: str, stage: str,
              root: Path | None = None) -> Held | None:
    """One worker per card. Returns None immediately if the card is taken."""
    if gpu in FORBIDDEN_GPUS:
        raise ValueError(f"GPU {gpu} is forbidden for this queue")
    if gpu not in REGISTERED_GPUS:
        raise ValueError(f"GPU {gpu} is not registered for this host")
    return _try_lock(card_path(gpu, uuid, root), f"gpu-{gpu}",
                     {"job": job, "stage": stage, "gpu": gpu, "gpu_uuid": uuid})


def take_build_slot(job: str, root: Path | None = None) -> Held | None:
    """At most two builds host-wide, whatever they are running on."""
    base = (root or DEFAULT_ROOT) / os.uname().nodename
    for index in range(BUILD_SLOTS):
        held = _try_lock(base / f"build-slot-{index}.lock", f"build-slot-{index}",
                         {"job": job, "stage": "solve"})
        if held is not None:
            return held
    return None


def occupancy(root: Path | None = None) -> dict:
    """Who holds what, by probing the locks read-only. Never unlinks."""
    base = (root or DEFAULT_ROOT) / os.uname().nodename
    report = {}
    if not base.exists():
        return report
    for path in sorted(base.glob("*.lock")):
        handle = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            held = True
        else:
            held = False
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            try:
                note = json.loads(path.read_text() or "{}")
            except json.JSONDecodeError:
                note = {}
            os.close(handle)
        report[path.name] = {"held": held, "note": note}
    return report
