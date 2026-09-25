# One-shot smoke recovery with continuous lease ownership

Status: **CPU PASS; not started; requires a new external permit.** The original M3 plan and all five root implementation files remain unchanged. This version changes resource handoff only and calls the original frozen `materialize_regmeanpp_m3.smoke` function.

The inspected v1 child PID 3073204 exited 1 at original `run_regmeanpp_m3.py:44`, with `BLOCKED_RESOURCE_NOT_EMPTY_OR_LOW_MEMORY`. The failure occurred before the CUDA allocator call, model initialization and original core permit consumption. `smoke-PERMIT-CONSUMED.json`, `smoke-FAILED.json` and `native-smoke.json` were all absent. The v2 plan freezes the six original v1 evidence files by SHA; none is overwritten.

## Frozen identities

- Supervisor: `ed9425afd5ce3bd784be35de776df8c71f4f07f3e93eefdfbf8f0f8a4b0cd57b`
- Child resource guard: `4d78c39fea7553a16d99953258bc087425343c8e1e2331efdba905ad72601ad8`
- CPU tests: `166ea10ffcf7bcf31de10edf7ba8952ed02619231b9fa20f8ba86cbe073e0200`
- V2 supervisor plan: `5273959bf35c2627bbde5a271841e31304757cf11fcae626da1afa04f1cdf42f`
- Unchanged original M3 plan: `bb25730b256669eecd1180026f1339cbb6b52d16342d9e65750e521e1b39ad0b`

Code root: `/mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/robotwin-regmeanpp-m3-codex-20260925/native_smoke_supervisor_v2/`

Supervisor plan/output: `/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/native-smoke-supervisor-v2/plan.json`

## Lease handoff

The supervisor waits for the externally authorized 1016 GPU from 1/4/5/6: no compute processes, at least 70 GiB free, at most 64 MiB used, and zero utilization. It acquires UUID, legacy host/index and build-slot flocks and repeats the card check. **It retains all three locks through child creation, the full model-file hashing pass and the entire native smoke.**

`Popen(pass_fds=...)` gives the child the same open-file descriptions. The child checks each descriptor's device/inode against the approved lock path, proves that a separate file description cannot acquire the lock, and proves that the inherited descriptor owns it. Arbitrary lock paths, lost locks and same-inode descriptors belonging to another open-file description are rejected. The parent never calls `LOCK_UN` on these descriptors: doing that would unlock the child's inherited description too. Parent cleanup only closes its own copies; the child continues owning the locks even if the parent exits. Kernel ownership ends after the final copy closes. The separately owned supervisor lock is not inherited.

Under those retained locks, the child verifies all frozen original sources, inputs and four complete model-file hashes, then repeats the fresh card check. It atomically writes original `CORE/smoke-PERMIT-CONSUMED.json` before initializing CUDA. Any existing consumed marker, `native-smoke.json` or `smoke-FAILED.json` refuses admission. The original smoke function and original output schema are unchanged. Failure after GPU admission is recorded in the original `smoke-FAILED.json` plus the independent supervisor failure receipt; it is never automatically retried. The original core consume marker remains definitive if failure occurs immediately after claiming admission.

There is at most one child launch. No model materialization, simulator rollout, formal evaluation or process signal is available from this supervisor. Success checks the original parity receipt and writes `COMPLETE.json`; failure preserves logs and stops. Polling is every 15 seconds with a 24-hour maximum wait.

## External permit and execution

The **new external permit** must contain the original `robotwin_regmeanpp_m3_single_use_execution_permit_v1` fields (allowed, smoke stage, original plan SHA, original core run, host, selected GPU and UUID), and additionally:

```json
{"supervisor_plan_sha256":"5273959bf35c2627bbde5a271841e31304757cf11fcae626da1afa04f1cdf42f"}
```

This fragment is documentation, not a valid permit. No permit has been created. V1's permit lacks the new binding and is rejected.

After review and external permit issuance, the command is:

```sh
PYTHONDONTWRITEBYTECODE=1 \
 /mnt/workspace/Wilson/parameter-fusion/pi05_lora_finetune_v2_20260826/.venv/bin/python \
 /mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/robotwin-regmeanpp-m3-codex-20260925/native_smoke_supervisor_v2/supervisor.py run \
 --plan /mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/native-smoke-supervisor-v2/plan.json \
 --permit /absolute/path/to/new-externally-issued-permit.json --execute
```

## CPU evidence

Thirteen tests passed on the remote 1016 Linux host without GPU queries or native smoke launch. A real CPU subprocess inherited all three lock descriptors; competing opens were blocked both before and after the parent closed its copies, and became available only after the child exited. Other tests cover lost leases, inode tampering, a different open-file description of the same inode, partial-acquisition cleanup, second-check contention, timeout, repeated core admission refusal and smoke-only delegation. Remote frozen-plan validation passed, with `STARTED=false` and no core GPU-attempt markers.

The resource shim does not establish GPU parity. The native smoke still needs to execute once after separate authorization and resource admission. See `CPU-TEST-RECEIPT.json` for the executed command and evidence scope.
