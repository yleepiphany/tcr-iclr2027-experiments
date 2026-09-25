# One-shot native-smoke waiter

Status: **CPU PREPARED; external permit required; not started.** This directory is separate from the materializer's frozen root `*.py` files and leaves the original CPU plan unchanged.

- Supervisor code SHA256: `95e84d136c8d94569b4e1f8910ae2f89f7c6f9b9bcaf7560e73bbeb200bc4986`
- Supervisor plan SHA256: `99786198dac0e61cb594714c0854675b04cca03a3a2c6acf573cc08ef847538d`
- Original M3 plan SHA256: `bb25730b256669eecd1180026f1339cbb6b52d16342d9e65750e521e1b39ad0b`

Code: `/mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/robotwin-regmeanpp-m3-codex-20260925/native_smoke_supervisor_v1/supervisor.py`

Frozen supervisor plan/output: `/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/native-smoke-supervisor-v1/plan.json`

An externally provided permit chooses exactly one of 1016 GPUs 1/4/5/6 and binds that GPU's UUID, the host, original plan SHA, original run path and `stage=smoke`. The supervisor never generates a permit. It polls every 15 seconds for up to 24 hours, requiring at least 70 GiB free, at most 64 MiB used, zero utilization and no compute PIDs. UUID, legacy host/index and build-slot leases must all be available.

The supervisor performs an availability probe and releases the probe leases before creating the child. The original guarded child runner then independently acquires all three leases, repeats its fresh card checks, and retains them during the GPU smoke. This avoids passing guessed or duplicated ownership across processes. A race that causes child admission to fail is preserved as a single failure; it is not retried. The child itself rehashes the four model files and checks all original frozen inputs before GPU admission.

Only one child command can be launched. It is always `run_regmeanpp_m3.py --stage smoke`; this supervisor contains no materialization or evaluation transition. If the smoke passes, the supervisor independently checks the three groups, physical t=1/replica0, all 418 bitwise sampled-input matches, bitwise velocity agreement and the three candidate-block parity checks. It writes `COMPLETE.json` with the native smoke receipt SHA. Failure writes `FAILED.json` and stops. No process is signaled and no lease file is unlinked.

The external permit schema is the original runner's `robotwin_regmeanpp_m3_single_use_execution_permit_v1`, with `allowed=true`, `stage=smoke`, `plan_sha256`, `run`, `gpu`, `gpu_uuid` and `host` matching the frozen original plan. A permit has **not** been created in this directory. The supervisor plan is also not permission to run.

After separate review and external permit issuance, the launch command is:

```sh
PYTHONDONTWRITEBYTECODE=1 \
  /mnt/workspace/Wilson/parameter-fusion/pi05_lora_finetune_v2_20260826/.venv/bin/python \
  /mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/robotwin-regmeanpp-m3-codex-20260925/native_smoke_supervisor_v1/supervisor.py run \
  --plan /mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/native-smoke-supervisor-v1/plan.json \
  --permit /absolute/path/to/externally-issued-smoke-permit.json --execute
```

`STARTED.json` makes the supervisor attempt single-use. `CHILD-STARTED.json` records command, PID and Linux start ticks; `CHILD-EXIT.json` records its exit code; `native-smoke.log` holds child output. An existing `STARTED.json` or original smoke attempt prevents relaunch. It does not resume a partial smoke or recycle a failed attempt.

CPU evidence: 10 synthetic supervisor tests passed, covering busy cards, held leases, second-check contention, UUID drift, timeout, external-claim drift, smoke-only command construction, release of probe locks, and strict parity receipt acceptance. Local tests made zero GPU queries, launched zero children and sent zero signals. Remote frozen-plan validation also passed without GPU access. See `CPU-TEST-RECEIPT.json`.
