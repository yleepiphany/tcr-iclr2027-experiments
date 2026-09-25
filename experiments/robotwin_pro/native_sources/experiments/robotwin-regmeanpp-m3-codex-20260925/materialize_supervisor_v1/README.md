# Single-use M3 materialization successor

Status: **CPU PASS; prepared only; no permit or GPU launch.** The original M3 plan, static bank, accepted three experts, Table1 kernel and 67-stage graph implementation remain unchanged. This successor requires the accepted native smoke and a separately authorized materialize permit.

## Frozen identities

- Original M3 plan: `bb25730b256669eecd1180026f1339cbb6b52d16342d9e65750e521e1b39ad0b`
- Accepted smoke: `454a48b6e639020a0ec93604831cc2bb1d001cb076f29800c168197b2ca85e91`
- New materialize supervisor plan: `aa0870c2be8d3a1fb662c2f697aeddfb076af7bff2d1359c015dac494c86d683`
- Supervisor source: `9702719163d198f1b216474bb4dcd5c3d53e8365633bea3f0e5c63daa5df0e8d`
- Child guard: `6d54f076831548762f49dde2c4588427fcce2aba81d174870fa2fbb21ab6b0d6`
- Model acceptance: `4754b83371e38d886c73d63d15da6b3ee6d9cfe598b95f3966d7117fcd86fe56`

Code: `/mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/robotwin-regmeanpp-m3-codex-20260925/materialize_supervisor_v1/`

Plan/output: `/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/materialize-supervisor-v1/plan.json`

Original core output remains `/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/cpu-plan-v1/checkpoint`. The original plan is an immutable CPU-preparation snapshot; its separate native smoke receipt establishes the passed GPU parity gate.

## Resource and single-use behavior

Only 1016 GPUs 1/4/5/6 are allowed. An external permit chooses one GPU and binds its UUID. Admission requires no compute process, zero utilization, at least 70 GiB free and no more than 64 MiB used. The same UUID, host/index and build-slot flocks are retained continuously, using the v2 handoff that passed a real Linux inherited-FD competition test and the successful native GPU smoke.

The child fully hashes the four original model files while holding all leases and verifies the exact accepted smoke SHA and three-group parity receipt. After the final empty-card check, it exclusively creates original `materialize-PERMIT-CONSUMED.json` before initializing CUDA. Existing build progress, checkpoint, completion, failure or consumed markers refuse admission; a partial build is never silently resumed or retried.

The allocator cap is .70. A resource-only wrapper checks the physical card's UUID and minimum 12 GiB runtime free memory at static input batch boundaries, throttled to once per five seconds; it delegates every batch unchanged to the original loader. No sampling, numeric solver, stage order, bias rule or data source is changed. Below the floor, only this child fails with evidence. It never signals any process. The original loader is restored before model acceptance.

After the one child completes or fails, no materialization retry or formal evaluation is launched. Parent cleanup closes its own inherited file-descriptor copies without `LOCK_UN`, so child lease ownership survives an unexpected parent exit. Child process exit closes the remaining descriptors.

## Model acceptance

After original materialization/export, the gate checks:

1. The full saved model SHA against the original build-complete receipt and model manifest, plus the original core plan SHA.
2. Exact 418 module identities, dimensions, three-expert row quotas totaling 1,776,300, alpha .3, smooth filter, mean interface biases and absence of clipping/caps/teacher-output targets. The smoothed normal-equation residual must be finite and at most 1e-7; the original centered-equation residual remains report-only.
3. A native `PI05Policy.from_pretrained` reload of the **exported** checkpoint, without mean reinitialization. All 813 loaded tensors must match the serialized tensor's shape, dtype and values exactly and be finite.
4. One accepted replica-0 static observation per group through the reloaded native graph: finite velocity of shape `[1,50,32]`, with output SHA and maximum absolute value reported. These are deployment integrity checks, not success measurements or performance-based parameter selection.
5. The model SHA again after reload.

Only then is `MODEL-ACCEPTED.json` written with model, manifest, original build and smoke SHA bindings. `COMPLETE.json` references that receipt. The original `BUILD-COMPLETE.json` remains unchanged; the separate acceptance receipt discharges its stated model-reload requirement.

## External permit

No permit has been created. The new permit must use the original single-use execution schema, with `stage="materialize"`, original core run/plan SHA, selected GPU/UUID and host. It must additionally bind:

```json
{
  "supervisor_plan_sha256": "aa0870c2be8d3a1fb662c2f697aeddfb076af7bff2d1359c015dac494c86d683",
  "native_smoke_sha256": "454a48b6e639020a0ec93604831cc2bb1d001cb076f29800c168197b2ca85e91"
}
```

This fragment is documentation, not a usable permit. After external review and permit issuance:

```sh
PYTHONDONTWRITEBYTECODE=1 \
 /mnt/workspace/Wilson/parameter-fusion/pi05_lora_finetune_v2_20260826/.venv/bin/python \
 /mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/robotwin-regmeanpp-m3-codex-20260925/materialize_supervisor_v1/supervisor.py run \
 --plan /mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/materialize-supervisor-v1/plan.json \
 --permit /absolute/path/to/externally-issued-materialize-permit.json --execute
```

## Formal evaluation binding and remaining step

The plan freezes the original Experts formal-v2 protocol SHA `ed47fc5340cfb53a4a1c497a08ef3aca984c0ad50182748ac3df3ca7e060d0de`, all nine group/repeat job manifests, all three reset banks, and their exact tasks/seeds. This is 540 episodes: three groups × ten tasks × six trials × three repeats. The receptacle repeat-2 task-27 seed replacement 877886121 is preserved; invalid old seed 437698954 is excluded.

A reusable dense native engine exists in `scripts/run_iclr2027_robotwin_checkpoint_development.py` (`simulator_audit`), with a working 9-job dense queue reference in `scripts/watch_robotwin_model_soups_formal_v1.py`. That queue currently restricts methods and paths to Model Soups/TIES, so it is **not** directly a RegMean++ queue. The next step is an isolated RegMean++ clone bound to the newly produced `MODEL-ACCEPTED.json` and model SHA, with a `pretrained_model` deployment path view for the existing engine. This plan records `model_sha256=null` and blocks formal launch until that binding exists. No old model, active queue or formal output is reused.

## CPU evidence

Thirteen new tests passed on 1016 Linux, without CUDA initialization, GPU queries or native child launch. They cover complete scope/row/residual acceptance, rejection of altered model/input/solver identities, distinct materialize permission, refusal of partial or repeated builds, runtime memory/UUID guards, the real frozen nine-job/540-episode protocol and amended seed, and exclusive use of the original materialize entrypoint. Existing v2 continuous lease code is reused unchanged and frozen by SHA. See `CPU-TEST-RECEIPT.json`.
