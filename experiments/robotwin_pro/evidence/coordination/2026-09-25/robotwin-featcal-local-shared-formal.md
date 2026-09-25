# CLAIM: RoboTwin FeatCal formal evaluation on local GPUs 5 and 7

User explicitly requested concurrently using local GPU capacity (2026-09-25).
This queue claims the nine FeatCal M3 formal group/repeat jobs. Do not launch a
duplicate remote FeatCal evaluation for these IDs.

- Runtime: `vla-merge-runtime/experiments/robotwin-featcal-local-20260925/formal-v1`.
- Code: `vla-merge/experiments/robotwin-tcr-local-codex-20260924/featcal_shared_formal.py`.
- Existing completed FeatCal M3 TF32-v2 model: SHA256
  `d2b6d99abcea01877045b98c6e704b552a0dee804c70bebdbb1519a0b71e0228`.
- Same Experts/Soup/TIES/RegMean++ nine-job panel: three groups × three repeats,
  ten tasks × six episodes per job = 540 episodes. No new fusion or training.
- Full model bytes, sidecars and frozen formal panel were checked before preparation.
  Existing native receipt validator is reused with explicit FeatCal constants;
  resets, horizons, 14D action normalization and all raw outcomes remain checked.
- CPU tests: all 540 keys, duplicate-seed rejection, group/repeat coverage, both
  known PRO supervisor identities and invalid start-time rejection passed.

## Why the boards looked empty

PRO Soup/TIES already own GPUs 5/7. Their short ten-episode workers repeatedly
initialize the environment and load the model on CPU before CUDA allocation.
At the screenshot time both workers were alive, CPU-active, and in model setup.
The idle GPU snapshot therefore was not an absent job. Separately, PRO static
baselines and RegMean++ RoboTwin queues are bound to the other host (1016).

## Safe shared admission

The added queue runs at most one FeatCal worker per card, two total. It binds the
exact host/UUID and known PRO supervisor PID/start-time/command. Shared admission
requires at least 48 GiB free; the added worker allocator is capped at 35% of one
80GB GPU, with a 12 GiB runtime free-memory floor. On reserve violation only the
added worker is stopped; never the original PRO/TCR workers. No automatic retries.

The existing PRO host/index leases are deliberately not stolen/unlocked. The new
worker holds the UUID lock, legacy compatibility lock and dedicated shared-capacity
lock continuously via inherited descriptors. If the known PRO owner is gone, a new
job instead requires the ordinary exclusive index lease. Failures stop new dispatch
and preserve all evidence. Supervisor SIGTERM drains already-running children.

Preparation alone is not evidence of execution. Use `launches/`, `state.json`,
per-job `STARTED.json`, and `ACCEPTED.json` for actual status. Do not report a result
until all nine jobs pass and their complete statistics are independently reviewed.

## Actual dispatch

At 2026-09-25 12:44 UTC, supervisor PID 2147115 dispatched coordination-repeat-01
to GPU 5 (PID 2147127) and receptacle-repeat-01 to GPU 7 (PID 2147156).
Both workers passed model hashing, inherited-lease validation and pre-CUDA memory
admission, wrote their STARTED receipts, and entered native model initialization.
Seven jobs remain queued. No episode completion is claimed at this handoff update.
The existing PRO workers remain untouched; CPU model initialization can temporarily
show negligible GPU memory usage even after a formal worker has been dispatched.
