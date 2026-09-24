"""Process-safety primitives for this study's runners (Codex C026.1 / C026.2).

The original runners used a bare `subprocess.run` with no process-group registration, no
signal handling and no parent-death guard, and a failing child only returned from its own
worker thread without blocking the batch.  So "a batch killed by a signal stops and is
never restarted" was not actually implemented.

This module provides the three pieces that were missing, deliberately narrow:

* `validate_gpus` - fail closed unless the requested GPUs are a duplicate-free subset of
  the authorised set.  A typo must not silently place a worker on someone else's card.
* `Child` - a subprocess launched in **its own process group**, with `PR_SET_PDEATHSIG`
  so it dies if the supervisor does.  Its own group id is recorded, so the supervisor can
  signal exactly that child and its descendants and nothing else.  No `pkill`, no
  name matching, no cross-host action.
* `BatchStop` - a stop latch wired to SIGTERM/SIGINT.  Once set, no further job starts,
  running children are terminated, and nothing is ever restarted.

`classify_exit` names the difference that matters for the protocol: a child killed by a
signal is not the same event as a child that failed on its own, and neither is retried.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from guard_batch_v3 import is_running, start_time  # shared identity primitives

CLOCK_TICKS = os.sysconf("SC_CLK_TCK")


def uptime_ticks() -> int:
    """Current time since boot in the same units as /proc/<pid>/stat field 22."""
    return int(float(Path("/proc/uptime").read_text().split()[0]) * CLOCK_TICKS)

PR_SET_PDEATHSIG = 1
AUTHORISED_GPUS = (1, 2, 4, 5, 6, 7)
# Shell convention: 128 + signal number.
SIGNAL_EXIT_CODES = {137: "SIGKILL", 143: "SIGTERM", 130: "SIGINT", 139: "SIGSEGV"}


def validate_gpus(requested, allowed=AUTHORISED_GPUS):
    """Duplicate-free, non-empty subset of the authorised GPUs, or raise."""
    if isinstance(requested, str):
        requested = [part.strip() for part in requested.split(",") if part.strip()]
    try:
        gpus = [int(value) for value in requested]
    except (TypeError, ValueError) as error:
        raise ValueError(f"GPU list is not integral: {requested!r}") from error
    if not gpus:
        raise ValueError("No GPUs requested")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"Duplicate GPUs requested: {gpus}")
    outside = sorted(set(gpus) - set(allowed))
    if outside:
        raise ValueError(
            f"GPUs {outside} are not in the authorised set {sorted(allowed)}; "
            f"refusing to start work on a card this study was not given")
    return gpus


def classify_exit(returncode: int) -> dict:
    """Distinguish signal death from an ordinary failure. Neither is ever retried."""
    if returncode == 0:
        return {"returncode": 0, "outcome": "success", "killed_by_signal": False}
    if returncode < 0:
        name = signal.Signals(-returncode).name
        return {"returncode": returncode, "outcome": "killed", "killed_by_signal": True,
                "signal": name}
    if returncode in SIGNAL_EXIT_CODES:
        return {"returncode": returncode, "outcome": "killed", "killed_by_signal": True,
                "signal": SIGNAL_EXIT_CODES[returncode]}
    return {"returncode": returncode, "outcome": "failed", "killed_by_signal": False}


def _child_preexec(expected_parent: int):
    """Own process group + die with the supervisor. Runs in the forked child.

    The re-check after `prctl` closes a real race (Codex C035): if the supervisor dies
    between `fork` and the `prctl` call, the death signal is never delivered and the child
    would run on as an orphan holding a GPU.  Comparing the parent pid afterwards catches
    exactly that window; the child exits rather than starting work.
    """
    os.setpgid(0, 0)
    if ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError("Cannot install the parent-death guard")
    if os.getppid() != expected_parent:
        os._exit(143)  # supervisor already gone; never start work


class Child:
    """One subprocess this batch owns, isolated in its own process group."""

    def __init__(self, command, env, log_path: Path, cwd=None):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.command = list(command)
        self.log_path = log_path
        self._stream = log_path.open("w")
        supervisor = os.getpid()
        self.process = subprocess.Popen(
            self.command, cwd=cwd, env=env, stdout=self._stream,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: _child_preexec(supervisor))
        self.pid = self.process.pid
        self.started_unix = time.time()
        # Pins this child's identity for every later signal decision.
        self.start_time_at_spawn = start_time(self.pid)
        self._snapshot: dict[int, int] = {}
        self._leader_gone_ticks: int | None = None
        # After setpgid the group id equals the child's pid; read it back rather than
        # assuming, so a signal is only ever sent to a group we actually confirmed.
        for _ in range(100):
            try:
                self.pgid = os.getpgid(self.pid)
                break
            except ProcessLookupError:
                time.sleep(0.01)
        else:
            self.pgid = self.pid
        if self.pgid == os.getpgid(0):
            raise RuntimeError("Child shares the supervisor's process group; refusing to "
                               "run, because stopping it would also signal the supervisor")

    def identity(self) -> dict:
        return {"pid": self.pid, "pgid": self.pgid, "started_unix": self.started_unix,
                "log": str(self.log_path), "command": self.command}

    def wait(self, poll: float = 2.0) -> int:
        """Wait for the leader, refreshing group membership while it lives.

        The refresh is what makes post-mortem cleanup possible at all: descendants the
        leader spawned are only identifiable as ours while the leader is still alive to
        anchor the group. Without it, a leader that exits on its own leaves orphans that
        can never be safely attributed.
        """
        while True:
            # Snapshot *before* testing for exit: a leader that dies immediately would
            # otherwise never be observed alive, and its descendants could never be
            # attributed to us afterwards.
            self._snapshot = self.group_members()
            code = self.process.poll()
            if code is not None:
                break
            try:
                self.process.wait(timeout=poll)
            except subprocess.TimeoutExpired:
                continue
        self._leader_gone_ticks = uptime_ticks()
        self._stream.close()
        return code

    def group_members(self) -> dict[int, int]:
        """Live pids in this child's group that could actually be its descendants.

        Anything that started *before* our leader cannot have been forked by it, so it is
        excluded no matter what the group id says. That single filter is what stopped an
        earlier version from signalling the test runner when it was pointed at a group
        that was not ours.
        """
        members = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                if os.getpgid(pid) != self.pgid:
                    continue
            except (ProcessLookupError, PermissionError):
                continue
            stamp = start_time(pid)
            if stamp is None or pid == os.getpid():
                continue
            if self.start_time_at_spawn is not None and stamp < self.start_time_at_spawn:
                continue  # predates our leader; cannot be its descendant
            members[pid] = stamp
        return members

    def stop(self, grace: float = 10.0) -> dict:
        """SIGTERM then SIGKILL, scoped so a recycled pgid can never be hit.

        Codex C039: the previous version unconditionally `killpg`-ed a historical group
        id. A process group id is reusable once every member has exited, so on a long
        queue that call could land on an unrelated program.

        The rule used here:

        * while the group **leader** still holds its registered identity, the group is
          provably ours (a pgid cannot be recycled while its leader lives), so `killpg`
          is safe and reaches descendants the leader spawned;
        * once the leader is gone, only members captured in a snapshot taken while it was
          alive are signalled, each re-verified by pid *and* kernel start time. Anything
          that appeared afterwards is treated as not ours and left alone.
        """
        events = []
        leader_alive = is_running(self.pid, self.start_time_at_spawn)
        if leader_alive:
            # Snapshot while the group is provably ours.
            snapshot = self.group_members()
        else:
            # Once the leader is gone the *only* processes we may signal are ones we
            # positively observed in this group while the leader was still alive.
            #
            # An earlier attempt here admitted any current group member that predated the
            # leader's death, reasoning that a pgid cannot be recycled until the group
            # empties. That reasoning is sound about recycling but says nothing about
            # ownership, and it is dangerous: pointed at a group that was never ours it
            # selects strangers. In testing it selected the test runner's own group and
            # SIGTERM-ed pytest. Membership is not ownership; only prior observation is.
            snapshot = {pid: stamp for pid, stamp in self._snapshot.items()
                        if is_running(pid, stamp)}
        self._snapshot = snapshot
        for name, sig, pause in (("SIGTERM", signal.SIGTERM, grace),
                                 ("SIGKILL", signal.SIGKILL, 2.0)):
            if is_running(self.pid, self.start_time_at_spawn):
                try:
                    os.killpg(self.pgid, sig)
                    events.append({"signal": name, "scope": "group (leader verified)"})
                except ProcessLookupError:
                    pass
            else:
                targets = {pid: stamp for pid, stamp in snapshot.items()
                           if is_running(pid, stamp)}
                if not targets:
                    break
                for pid in sorted(targets):
                    if pid == os.getpid() or pid == os.getpgid(0):
                        continue  # never signal the supervisor process or its own group
                    try:
                        os.kill(pid, sig)
                        events.append({"signal": name, "scope": "verified pid", "pid": pid})
                    except ProcessLookupError:
                        continue
            deadline = time.monotonic() + pause
            while time.monotonic() < deadline:
                if not any(is_running(pid, stamp) for pid, stamp in snapshot.items()) \
                        and not is_running(self.pid, self.start_time_at_spawn):
                    break
                time.sleep(0.1)
        return {"signalled": events, **self.identity()}


class BatchStop:
    """Latch that blocks new work and terminates registered children exactly once."""

    def __init__(self, report_path: Path | None = None):
        self._lock = threading.Lock()
        self._children: list[Child] = []
        self.reason: str | None = None
        self.report_path = report_path
        self.events: list[dict] = []

    @property
    def stopped(self) -> bool:
        return self.reason is not None

    def register(self, child: Child) -> bool:
        """Register a child, or stop it if the batch has already been cancelled.

        Closes the cancellation race Codex reproduced (C035): a worker that checked
        `stopped`, then spawned, could register after `trip()` had already walked the
        child list, leaving a live child nobody would ever stop.  The check and the
        append happen under one lock; if the latch is already set, the caller's brand-new
        child is stopped *outside* the lock, so a blocking stop can never deadlock
        against a signal handler re-entering `trip`.

        Returns True if registered, False if the batch was already stopped and the child
        has been terminated.
        """
        with self._lock:
            if self.reason is None:
                self._children.append(child)
                return True
        self.events.append(child.stop())
        return False

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda number, _frame: self.trip(
                f"supervisor received {signal.Signals(number).name}"))

    def trip(self, reason: str) -> None:
        with self._lock:
            if self.reason is not None:
                return
            self.reason = reason
            children = list(self._children)
        for child in children:
            self.events.append(child.stop())
        if self.report_path is not None:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_path.write_text(json.dumps({
                "hostname": os.uname().nodename, "reason": reason,
                "stopped_unix": time.time(), "stopped_children": self.events,
                "restarted": False,
                "policy": "batch stopped; no automatic restart, no parameter change, "
                          "partial receipts preserved, nothing outside this batch "
                          "signalled"}, indent=2) + "\n")
