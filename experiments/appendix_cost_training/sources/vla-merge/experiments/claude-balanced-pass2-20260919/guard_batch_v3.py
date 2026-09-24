#!/usr/bin/env python3
"""Identity-scoped batch guard, v3 — real stop-on-kill.

v2 detected a signal-killed worker and stopped that batch's *children*, but it excluded
the supervisor from `owned_members` and never signalled it, so an old-style runner kept
dispatching the next job. Detection is not cessation; Codex reproduced the gap with a
mock probe (C038). v3 fixes the ordering and the identity checks:

1. **Stop dispatch first.** On a kill, the *supervisor* is signalled before anything else,
   because it is the thing that would otherwise start the next job. Children are cleaned
   up afterwards.
2. **Identity re-verified immediately before every signal.** A pid is never trusted on its
   own. Each target must still match the kernel start time captured at registration, its
   full argv must still name this study's runner, and that argv must contain this batch's
   own `--run` / `--output_dir` path. A recycled pid or pgid therefore cannot be signalled
   by mistake — the guard would rather do nothing than hit an unrelated process.
3. **Never signals a process group.** Only individually verified pids, so a reused pgid
   can never take innocent processes with it.
4. **Completion requires unique (arm, offset) coverage**, not merely a record count, so a
   summary with duplicate rows cannot pass as a finished batch.
5. **Never restarts anything.** Ever.

Scope: one supervisor and its descendants on this host. Nothing else is ever touched — no
other team's queue, no other host, no name-based scanning, no `pkill`, no `killpg`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

OWNED = ("run_covariance_study.py", "run_covariance_development.py",
         "materialize_covariance_alpha.py", "eval_pi05_expanded_development.py")
SIGNAL_EXIT_CODES = {137: "SIGKILL", 143: "SIGTERM", 130: "SIGINT", 139: "SIGSEGV"}


def _stat_fields(pid: int) -> list[bytes] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    # comm can contain spaces and parentheses, so split after the final ')'.
    return stat[stat.rindex(b")") + 2:].split()


def start_time(pid: int) -> int | None:
    fields = _stat_fields(pid)
    return None if fields is None else int(fields[19])


def is_zombie(pid: int) -> bool:
    """A terminated-but-unreaped process still has a /proc entry and a start time.

    Without this, a supervisor we have just killed looks alive for as long as its parent
    has not reaped it, and the guard would report `supervisor_stopped=False` for a batch
    it did in fact stop. A zombie dispatches nothing, so it counts as stopped.
    """
    fields = _stat_fields(pid)
    return fields is not None and fields[0] == b"Z"


def is_running(pid: int, expected_start: int | None) -> bool:
    """Alive, same identity, and not a zombie."""
    observed = start_time(pid)
    if observed is None:
        return False
    if expected_start is not None and observed != expected_start:
        return False
    return not is_zombie(pid)


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", "replace").strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return ""


def process_group(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None


def parent_of(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return int(stat[stat.rindex(b")") + 2:].split()[1])


def is_runner(command: str) -> bool:
    """A python process whose argv actually names one of our runners."""
    parts = command.split()
    if not parts or not Path(parts[0]).name.startswith("python"):
        return False
    if "-c" in parts[1:3]:
        return False
    return any(Path(part).name in OWNED for part in parts[1:])


def belongs_to_batch(command: str, run_path: str) -> bool:
    """The argv must reference this batch's own run directory.

    `is_runner` only says "a runner of ours"; two batches of the same study would both
    pass it. Binding to the run path is what makes the guard batch-scoped rather than
    study-scoped, so a second concurrent batch is never collateral damage.
    """
    return is_runner(command) and run_path in command


def verify_identity(pid: int, expected_start: int | None, run_path: str) -> dict:
    """Read-only verdict used immediately before any signal."""
    observed = start_time(pid)
    if observed is None:
        return {"pid": pid, "ok": False, "reason": "process is gone"}
    if expected_start is not None and observed != expected_start:
        return {"pid": pid, "ok": False, "observed_start_time": observed,
                "reason": "start time differs; pid was recycled by an unrelated process"}
    if is_zombie(pid):
        return {"pid": pid, "ok": False, "reason": "process already exited (zombie)"}
    command = cmdline(pid)
    if not belongs_to_batch(command, run_path):
        return {"pid": pid, "ok": False, "cmdline": command[:200],
                "reason": "argv does not name this study's runner bound to this run path"}
    return {"pid": pid, "ok": True, "start_time": observed, "cmdline": command[:200]}


def descendants_of(supervisor: int, run_path: str, expected: dict[int, int]) -> list[dict]:
    """Live processes that are verifiably this batch's own descendants.

    A process qualifies only if its argv names one of our runners *and* this run path, and
    it is either a registered child (start time must match) or its parent is the
    supervisor. Group membership alone is never sufficient.
    """
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == supervisor:
            continue
        command = cmdline(pid)
        if not belongs_to_batch(command, run_path):
            continue
        if pid in expected:
            if start_time(pid) != expected[pid]:
                continue  # recycled pid; not ours
        elif parent_of(pid) != supervisor:
            continue
        if is_zombie(pid):
            continue  # already exited, nothing to stop
        found.append({"pid": pid, "start_time": start_time(pid), "cmdline": command[:300]})
    return found


def signal_verified(pid: int, expected_start: int | None, run_path: str,
                    sig: signal.Signals) -> dict:
    """Signal one pid, re-verifying identity in the same breath."""
    verdict = verify_identity(pid, expected_start, run_path)
    if not verdict["ok"]:
        return {"signalled": False, "signal": sig.name, **verdict}
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return {"signalled": False, "signal": sig.name, "pid": pid,
                "reason": "exited between verification and signal"}
    return {"signalled": True, "signal": sig.name, "pid": pid}


def stop_batch(record: dict, reason: str, detail=None) -> dict:
    """Stop dispatch first, then clean up verified descendants. Never restarts."""
    supervisor = record["supervisor_pid"]
    run_path = record["run"]
    expected = {c["pid"]: c["start_time"] for c in record.get("children_at_registration", [])}
    events = []

    # 0. Snapshot the descendants BEFORE touching the supervisor.
    #    Killing the supervisor reparents its children to init, after which parentage can
    #    no longer identify them - and those reparented children are precisely the orphans
    #    this guard exists to clean up. Identity is still re-verified per pid at signal
    #    time (start time + argv + run path), so the snapshot can never widen the blast
    #    radius; it only preserves knowledge that the kill would otherwise destroy.
    snapshot = descendants_of(supervisor, run_path, expected)
    snapshot_starts = {entry["pid"]: entry["start_time"] for entry in snapshot}

    # 1. Dispatch first. Until the supervisor stops, killing children achieves nothing.
    supervisor_signalled = False
    for sig in (signal.SIGTERM, signal.SIGKILL):
        outcome = signal_verified(supervisor, record["supervisor_start_time"], run_path, sig)
        supervisor_signalled = supervisor_signalled or outcome.get("signalled", False)
        events.append({"stage": "supervisor", **outcome})
        deadline = time.monotonic() + (10.0 if sig is signal.SIGTERM else 3.0)
        while time.monotonic() < deadline:
            if not is_running(supervisor, record["supervisor_start_time"]):
                break
            time.sleep(0.2)
        if not is_running(supervisor, record["supervisor_start_time"]):
            break
    supervisor_stopped = not is_running(supervisor, record["supervisor_start_time"])

    # 2. Then the descendants: the snapshot plus anything still discoverable, each pid
    #    verified individually. No process group is ever signalled.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        targets = dict(snapshot_starts)
        for entry in descendants_of(supervisor, run_path, expected):
            targets[entry["pid"]] = entry["start_time"]
        live = {pid: start for pid, start in targets.items() if is_running(pid, start)}
        if not live:
            break
        for pid, start in sorted(live.items()):
            events.append({"stage": "descendant",
                           **signal_verified(pid, start, run_path, sig)})
        time.sleep(10.0 if sig is signal.SIGTERM else 3.0)
    remaining = [{"pid": pid, "start_time": start}
                 for pid, start in snapshot_starts.items() if is_running(pid, start)]
    remaining += [e for e in descendants_of(supervisor, run_path, expected)
                  if e["pid"] not in snapshot_starts]


    return {"reason": reason, "detail": detail, "stopped_unix": time.time(),
            # Two distinct facts, kept apart on purpose: whether the registered identity is
            # still running, and whether *we* signalled it. If a pid was recycled the
            # registered process is gone but we deliberately signalled nothing, and
            # conflating the two would let the guard claim credit for a stop it did not
            # perform - or, worse, hide that it declined to act.
            "supervisor_stopped": supervisor_stopped,
            "supervisor_signalled": supervisor_signalled,
            "remaining_descendants": remaining,
            "events": events, "restarted": False,
            "policy": ("dispatch stopped before cleanup; every signal re-verified against "
                       "kernel start time, argv and this run path; no process group was "
                       "signalled; nothing restarted and no parameter changed")}


def batch_completed(summary_path: Path, arms: list[str], offsets: list[int]) -> dict:
    """Completion by unique (arm, offset) coverage, not by a record count."""
    if not summary_path.exists():
        return {"completed": False, "why": "expected summary file does not exist"}
    try:
        summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        return {"completed": False, "why": f"summary unreadable: {error}"}
    records = summary.get("records")
    if records is None:
        return {"completed": False, "why": "summary has no evaluation records"}
    seen = {(r.get("arm"), r.get("offset")) for r in records}
    if len(seen) != len(records):
        return {"completed": False, "why": "summary contains duplicate (arm, offset) rows"}
    wanted = {(arm, offset) for arm in arms for offset in offsets}
    if seen != wanted:
        return {"completed": False,
                "why": f"coverage differs; missing {sorted(wanted - seen)[:3]}"}
    if summary.get("status") not in ("complete", "done"):
        return {"completed": False, "why": f"status is {summary.get('status')!r}"}
    return {"completed": True, "why": "unique (arm, offset) coverage complete"}


def killed_jobs(run: Path) -> list[dict]:
    killed = []
    for path in sorted(run.glob("*/offset-*/exit.json")):
        try:
            code = json.loads(path.read_text()).get("return_code")
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(code, int) or code == 0:
            continue
        if code < 0 or code in SIGNAL_EXIT_CODES:
            killed.append({"job": str(path.parent.relative_to(run)), "return_code": code})
    return killed


def register(args) -> dict:
    stamp = start_time(args.supervisor)
    if stamp is None:
        raise SystemExit(f"supervisor {args.supervisor} is not running")
    command = cmdline(args.supervisor)
    run_path = str(args.run.resolve())
    if not belongs_to_batch(command, run_path):
        raise SystemExit(
            f"supervisor {args.supervisor} is not a python runner of this study bound to "
            f"{run_path}; refusing to register (a shell wrapper would be a dangerous "
            f"false match): {command}")
    expected = {}
    children = []
    for candidate in descendants_of(args.supervisor, run_path, expected):
        children.append(candidate)
    record = {"hostname": os.uname().nodename, "guard_version": 3,
              "supervisor_pid": args.supervisor, "supervisor_start_time": stamp,
              "supervisor_cmdline": command,
              "process_group": process_group(args.supervisor),
              "guard_pid": os.getpid(), "registered_unix": time.time(),
              "run": run_path, "arms": args.arm_list, "offsets": args.offset_list,
              "completion_summary": str(args.summary_file),
              "children_at_registration": children,
              "policy": ("stop-on-kill: supervisor dispatch is stopped first, then "
                         "individually verified descendants; never restarts; never "
                         "signals a process group; never touches another batch or host")}
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(args):
    args.arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
    args.offset_list = [int(o) for o in args.offsets.split(",") if o.strip()]
    record = register(args)
    print(json.dumps({"registered": record["supervisor_pid"], "guard_version": 3,
                      "children": len(record["children_at_registration"]),
                      "already_complete": batch_completed(
                          args.summary_file, args.arm_list, args.offset_list)}), flush=True)
    if args.register_only:
        return
    while True:
        if args.stop_file.exists():
            print(json.dumps({"guard": "released by stop file"}), flush=True)
            return
        killed = killed_jobs(args.run)
        if killed:
            report = stop_batch(record, "a job in this batch was killed by a signal", killed)
            args.stop_report.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
            return
        if not is_running(record["supervisor_pid"], record["supervisor_start_time"]):
            time.sleep(args.interval)
            verdict = batch_completed(args.summary_file, args.arm_list, args.offset_list)
            if verdict["completed"]:
                print(json.dumps({"guard": "batch finished normally", **verdict}), flush=True)
                return
            report = stop_batch(record, "supervisor died without a complete summary", verdict)
            args.stop_report.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor", type=int, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--offsets", default=",".join(str(o) for o in range(30, 40)))
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--stop-report", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--register-only", action="store_true")
    main(parser.parse_args())
