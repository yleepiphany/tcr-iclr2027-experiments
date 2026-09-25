# RoboTwin RegMean++ M3 formal 540-episode queue

Status: **CPU frozen, not started, external formal permit required.** Fifteen CPU tests and the real frozen-plan validation passed without GPU queries or CUDA initialization. No native evaluation job has run from this queue.

## Identities and scope

- Plan SHA256: `b59c96eb520874e1083bc4bf30e6027773a8135e7b44075a6f7ad04281acd60d`
- Accepted model SHA256: `87268d409201894aee6c903dc9ade71f9a5ac591654f7e55dc7d0da4c3b00b5e`
- MODEL-ACCEPTED SHA256: `ff00d95bbb372376dd1effe9d5bf0f43b33d018235006205c5936d1afe074f81`
- Original Experts protocol SHA256: `ed47fc5340cfb53a4a1c497a08ef3aca984c0ad50182748ac3df3ca7e060d0de`
- Queue source SHA256: `3a3a6f7f045274d4430f2b0c85ce779225f6e97daa6b43ab063bdfee9cdb6930`
- Worker SHA256: `f14deead51db9c31f9770e1a15622f686b28e4bb0fa0d791e81ffb3e03b48232`
- Contract SHA256: `c028c6cf46e1c0c06058bb329767222e1cab303f4a91e9c8734f5e853d248ea2`
- Receipt validator SHA256: `631876af1ddbb117c6b59079a16c5d2bfa5f846d4b24b792b969caf96966ba6a`

Code directory: `/mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/robotwin-regmeanpp-m3-codex-20260925/formal_540_v1/`

Plan/output: `/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/formal-540-v1/plan.json`

The new `deployment/pretrained_model` symlink points to the existing accepted `cpu-plan-v1/checkpoint`; weights are not copied or modified. CPU preparation recomputed the complete 9,354,050,752-byte model SHA. All model sidecars, tokenizer files, normalizers, accepted-model receipt, source protocol, task/reset files and native source files are SHA-bound.

One fixed three-expert merged checkpoint is evaluated on three groups × three reset repeats. Each of the nine jobs contains ten tasks × six seeds = 60 episodes; total 540. This is three **evaluation** repeats of one checkpoint, not three independent model constructions. The original receptacle repeat-2 task-27 amendment uses seed 877886121 in place of invalid 437698954. Seed derivation is checked against the original namespace and explicit amendment.

## Original native engine

The worker calls the unchanged `run_iclr2027_robotwin_checkpoint_development.simulator_audit(runtime, output, False)`. Runtime data are formed exactly like the accepted dense TIES/Soups queue: the same parent environment/robot configuration, original task metadata, sorted complete task/seed panel, `demo_clean`, joint actions, native registered horizons and `plateau_eligible=false`. Only the checkpoint deployment path and method-purpose label identify the new model. Runtime files retain their original parent provenance; active policy identity is separately bound to the accepted RegMean++ model, never to those parent expert checkpoints.

Each job freezes all 60 original Experts initial-observation fingerprints as an independent reset check. The baseline outcome values are not used to select seeds or modify the merged policy. No simulator parameter, action transform, horizon, observation processing, seed or model weight is tuned in this runner.

## Resource ownership and failure behavior

An external single-use permit selects a nonempty subset of 1016 GPUs 1/4/5/6, with exactly matching UUID entries. One worker runs per allowed card. Before admission, a card must have no compute PIDs, utilization zero, no more than 64 MiB used and at least 40 GiB free. The worker allocator fraction is .50.

The queue acquires and retains **all three compatible card locks**:

1. `claude-card-flocks/HOST/gpu-INDEX-UUID.lock`;
2. `resource-leases/HOST/gpu-INDEX.lock`;
3. legacy `resource-leases/HOST-gpu-INDEX.lock`.

It passes the same open-file descriptions through `pass_fds`. The worker validates descriptor/path inode identities and exclusive ownership, repeats full model/source/input verification and a fresh empty-card check, then writes its single-use `STARTED.json` before CUDA work. Parent cleanup closes only its descriptor copies and never unlocks the inherited descriptions. No build-slot lock is consumed by these evaluation jobs.

There is exactly one attempt per job. Existing output or attempt markers refuse admission. On any failure, new jobs stop being scheduled, already-running children retain their leases and finish naturally, and partial/accepted outputs remain preserved. There are no process signals, automatic retries, seed resampling, partial-row recycling, or score-based stopping rules. The queue itself is single-use and cannot silently restart after a supervisor failure. Candidate-card polling is every 15 seconds, with a 48-hour scheduling deadline.

## Per-job acceptance and aggregation

Every job writes original native episode JSON and JSONL traces into its new `attempt-01` directory. `ACCEPTED.json` requires:

- Exactly the frozen 60 `(task_index, task, seed)` rows, ten tasks × six episodes; strict Boolean outcomes.
- Native registered horizons and positive steps; an unsuccessful episode must reach its full horizon.
- Matching reset trace, three camera identities, state shape `[1,14]`, and initial-observation SHA equal to the frozen Experts reference.
- Fourteen finite values in normalized, postprocessed and expected-executed first-action arrays. The actual checkpoint action quantiles must reconstruct the native postprocessed action within `1e-7`.
- Raw success totals and task rates consistent with the unchanged native `simulator.json` summary.
- Exact model SHA and unchanged sidecars after the job, plus SHA/byte bindings for every episode JSON, trace, job manifest and native summary.

All nine accepted receipts are required for `AGGREGATE.json`. Each repeat averages the equally weighted thirty tasks (180 episodes), then the overall mean and sample standard deviation are computed over three repeats. Group results use the same three-repeat sample standard deviation. Zero successes are valid outcomes if all execution checks pass; there is no result threshold or tuning loop.

## External permit and launch

`PERMIT-TEMPLATE-NOT-AUTHORIZED.json` was generated with `allowed=false`; validation confirms it is rejected. It is not a GPU authorization. An external permit must bind the exact plan/run/host/model/accepted-model identities, all nine job IDs, 540 episodes and one attempt per job. If choosing fewer cards, its `gpu_uuids` mapping must contain exactly that selected subset.

After review and external permit issuance, launch:

```sh
PYTHONDONTWRITEBYTECODE=1 \
 /mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/envs/iclr2027-robotwin2-py312-mplib-curobo-v3/bin/python \
 /mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/robotwin-regmeanpp-m3-codex-20260925/formal_540_v1/run_formal.py run \
 --plan /mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/formal-540-v1/plan.json \
 --permit /absolute/path/to/externally-issued-formal-permit.json --execute
```

`state.json` reports pending/active/accepted/failed jobs. Each job preserves handoff, child PID/start ticks, exit, native log, raw episode traces and acceptance/failure records. This code does not edit old TIES/PRO queues or any training process.

CPU evidence is recorded in `CPU-TEST-RECEIPT.json` and runtime `CPU-PREPARED.json`. Tests cover native runtime argument construction, exact real 540-seed/amendment panel, resource gates, all three lock schemes, inherited ownership helpers, reset/action/horizon negative cases, duplicate-attempt refusal, permit identities, repeat statistics, native full-episode call scope and CUDA remaining uninitialized.
