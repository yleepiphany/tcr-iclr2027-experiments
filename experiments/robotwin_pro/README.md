# Unfinished RoboTwin and LIBERO-PRO experiments

This directory contains exact source snapshots, small configuration/provenance records, and a fail-closed release preflight. It does **not** claim completed evaluations or ship robot hardware experiments, datasets, environments, action caches, or model weights.

## Quick checks

From the public repository root, using Python 3.10 or later:

```bash
python experiments/robotwin_pro/preflight.py --mode cpu
python -m unittest discover -s experiments/robotwin_pro/tests -v
python experiments/robotwin_pro/smoke.py --workspace-root /path/to/original/workspace --mode cpu
```

`--workspace-root` must contain the original `vla-merge/`, `vla-merge-runtime/`, and `pi05_lora_finetune_v2_20260826/` directories. It is optional for package-only verification. A missing native workspace is reported separately from successful source-integrity checks.

The CPU native smoke executes only three allowlisted native `--help` paths: PRO evaluation, RoboTwin TIES evaluation, and RoboTwin TCR evaluation. CUDA is hidden, Python bytecode writes are disabled, CPU thread counts are one, network model loading is offline, and each child has a 45-second limit. This verifies native imports/CLI, **not model loading, simulation, or a rollout**. The CPU mode neither claims GPU resources nor signals an existing process. A later source inspection found an existing bounded native RoboTwin `--smoke` path; GPU mode now wraps that exact entry, as described below. It never launches a formal panel.

Select a target to enforce its blocked status in the exit code:

```bash
python experiments/robotwin_pro/preflight.py --method libero_pro/regmean_pp
python experiments/robotwin_pro/preflight.py --method libero_pro/featcal
```

Both commands deliberately exit **2 / BLOCKED**. This is expected behavior, not a missing check to bypass. Exit 0 is PASS for the stated check scope; it never means that unfinished benchmark results exist.

## Archived 2026-09-24 identities and blockers

| Target | Snapshot state | Native entrypoints |
|---|---|---|
| LIBERO-PRO Model Soups | Unfinished; historical terminal reports 120/4,800 accepted episodes, stopped | `prepare_soup.py`, `run_soup_formal.py` |
| LIBERO-PRO TIES | Unfinished; historical terminal reports 40/4,800 accepted episodes, stopped | `prepare_pro_baseline_selection_v1.py`, `run_baseline_formal.py` |
| LIBERO-PRO RegMean++ | **BLOCKED: archived selection and runner bind stale checkpoint** | Same PRO prepare/formal scripts; do not launch them for this method |
| LIBERO-PRO FeatCal | **BLOCKED: archived selection and runner bind stale checkpoint** | Same PRO prepare/formal scripts; do not launch them for this method |
| RoboTwin TIES | Native materialization/formal entries exist; no completed result claimed here | `materialize_robotwin_three_expert_ties_formal_v1.py`, `watch_robotwin_model_soups_formal_v1.py` with `ROBOTWIN_FORMAL_MERGE_METHOD=ties` |
| RoboTwin TCR | Latest local pass-B recovery/build completed; formal panel unfinished at snapshot | `run_tcr_builds_local_v4.py`, `run_materializer_local_v4.py`, `run_tcr_formal_local_v4.py` |
| RoboTwin RegMean++ | **BLOCKED: no accepted RoboTwin-native activation/solver/checkpoint/evaluation contract found** | No substituted LIBERO implementation |
| RoboTwin FeatCal | **BLOCKED: no accepted RoboTwin-native activation/solver/checkpoint/evaluation contract found** | No substituted LIBERO implementation |

All names/paths and exact hashes are in `config/catalog.json` and `PROVENANCE.json`. Current LIBERO main-table checkpoint identities are:

| Method | Current SHA256 | Stale SHA256 still bound by archived PRO source |
|---|---|---|
| RegMean++ with disclosed spectral smoothing | `fea99a8c09f8ca16bf8a2889bac659aba86bcbf831ef422e8c0a30115bccffb2` | `38745ba812c2f4ac4327ab40719e0ce8da017873c0a1765608f65dd926cd4ab3` |
| FeatCal static observations / initial noise | `b0da07eda2a9ab4485f555f694d31afe08c40a39d6e3bfd44e9b9d8d006b0114` | `e43de109844431c02e316e57f701d7c06a9a3c8feaa3c41c9f391281f8197efa` |

The preflight rejects stale identities even when the stale manifest's own hash is correct. Updating a JSON model hash alone cannot unblock the archived runner. Regenerate **new immutable** PRO selections with current weights, preserve all three original repeat/reset/task/perturbation bindings, verify checkpoint and preprocessing identity, and adapt the current-checkpoint runner before updating this release's blocked catalog state. Do not modify historical selections in place or relabel old outputs as current results.

That current-model follow-up was prepared on 2026-09-25. The immutable `selection-static-v2` and `formal-static-v2` plans bind the Table 1 RegMean++ and FeatCal model SHA values above. Neither old plan was launched. The [1016 transfer runner](native_sources/experiments/pi05-pro-static1016-codex-20260925/run_transfer.py) and [frozen plan](evidence/PRO-STATIC1016-PLAN.json) move only each job's output directory, keep its model/selection/reset/seed/evaluator arguments intact, and use host/UUID plus host/index GPU leases on cards 1, 4, 5 and 6. Its source-owner fence and transfer markers prevent the stale local lane from starting concurrently. CPU guard tests and exact-plan validation passed; the first RegMean++ and FeatCal native 10-episode jobs both passed raw-outcome and receipt auditing. The [launch receipt](evidence/PRO-STATIC1016-LAUNCH.json) records the supervisor identity. **The 960-job panel remains in progress; this release does not claim final PRO scores.** The archived catalog and its stale-selection rejection tests remain as evidence of why the original runner must not be used.

RoboTwin static-baseline preparation advanced separately on 2026-09-25. The [CPU readiness plan](evidence/ROBOTWIN-STATIC-CPU-READINESS-PLAN.json) and [bank manifest](evidence/ROBOTWIN-STATIC-BANK-MANIFEST.json) bind 150 calibration observations and 450 independent initial-noise states from the existing calibration split, without action labels or TCR execution caches. The three tensor banks remain external assets. The [isolated source](native_sources/vla-merge-runtime/experiments/robotwin-static-baselines-readiness-20260925/source-v1/README.md) provides an M=3 FeatCal graph driver; the [CPU plan](evidence/ROBOTWIN-FEATCAL-M3-CPU-PLAN.json), [kernel check](evidence/ROBOTWIN-M3-KERNEL-CPU-CHECK.json), and [CPU test receipt](evidence/ROBOTWIN-M3-CPU-FINAL-TESTS.json) record 47 stages, 418 adapted weights and independent numerical checks. **That CPU preparation produced no merged checkpoint or formal result.** RegMean++ subsequently passed guarded native GPU parity, as recorded below; its scored model is still unfinished.

The [RoboTwin RegMean++ M=3 port](native_sources/experiments/robotwin-regmeanpp-m3-codex-20260925/README.md) supplies the distinct 67-stage merged-prefix graph materializer and a [frozen CPU plan](evidence/ROBOTWIN-REGMEANPP-M3-CPU-PLAN.json). Ten CPU tests check three-expert row budgets, block input identity, atomic updates and the unchanged spectral-smoothing kernel. Separate [FeatCal](native_sources/vla-merge-runtime/experiments/robotwin-static-baselines-readiness-20260925/safe-successor-v1/README.md) and [RegMean++ v2](native_sources/experiments/robotwin-regmeanpp-m3-codex-20260925/native_smoke_supervisor_v2/README.md) one-shot supervisors were prepared to attempt only native parity smokes on an empty, leased GPU. Their launch receipts preserve that sequence. RegMean++ v2 has now passed; a smoke remains distinct from a model build or a scored RoboTwin episode.

### RegMean++ smoke resource race: v1 preserved, v2 isolated

The [v1 failure/competition record](evidence/coordination/2026-09-25/robotwin-regmeanpp-smoke-admission-race-20260925.json) and its six SHA-bound original evidence files are preserved. V1 probed and released its leases before the child finished full model-file hashing; another legitimate TIES job acquired the card in that gap. The v1 child then exited with `BLOCKED_RESOURCE_NOT_EMPTY_OR_LOW_MEMORY` at `run_regmeanpp_m3.py:44`, **before CUDA initialization, model forward, or original smoke permit consumption**. No native-smoke result or formal episode exists from v1. This is a resource-admission failure, not a zero score or a failed numerical parity test.

The [v2 plan](evidence/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/native-smoke-supervisor-v2/plan.json) (`5273959b…`) changes resource handoff only. The supervisor retains UUID, host/index and build-slot flocks while the child inherits the same open-file descriptions through `pass_fds`; the child validates their paths, inode identities and actual ownership before CUDA. Parent cleanup closes only its own copies and never unlocks the inherited descriptions. The original M3 source/plan and original `smoke` function remain unchanged. V2 allows at most one smoke child, requires a fresh supervisor-bound external permit, and has no materialization, formal evaluation, automatic retry or process-signalling action.

At the **2026-09-25 04:10 UTC snapshot**, v2 supervisor PID `3129312` was started and waiting for its approved idle card; no child-start or completion receipt was present. Native parity and formal performance had not been measured at that point. The byte-preserved native README and CPU receipt record the earlier pre-launch preparation state; the later launch/STARTED receipts establish the waiting supervisor. Thirteen original resource-guard CPU tests also passed locally with CUDA hidden, including a real inherited-lock subprocess; no GPU query or native smoke was run while packaging.

### RegMean++ native parity passed; materialization waiting

The later [native smoke receipt](evidence/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/cpu-plan-v1/native-smoke.json) and [v2 completion](evidence/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/native-smoke-supervisor-v2/COMPLETE.json) establish **PASS at 2026-09-25 04:54 UTC**. Child PID `3229434` exited 0. All three groups matched the original native velocity and all 418 sampled linear inputs bitwise, and candidate vision0/language0/action0 captures matched. Peak CUDA allocation was 9,281.83 MiB. The smoke used three distinct static observations, replica 0 at physical t=1, and **zero formal episodes**. Its SHA256 is `454a48b6e639020a0ec93604831cc2bb1d001cb076f29800c168197b2ca85e91`.

The next [materialize supervisor](native_sources/experiments/robotwin-regmeanpp-m3-codex-20260925/materialize_supervisor_v1/README.md), [13-test CPU receipt](native_sources/experiments/robotwin-regmeanpp-m3-codex-20260925/materialize_supervisor_v1/CPU-TEST-RECEIPT.json), and [frozen plan](evidence/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/materialize-supervisor-v1/plan.json) preserve the original M3 plan and numerical method. The plan SHA256 is `aa0870c2be8d3a1fb662c2f697aeddfb076af7bff2d1359c015dac494c86d683`. It retains UUID, host/index and build-slot leases through `pass_fds`, requires an empty card with at least 70 GiB free, applies a .70 allocator cap, and stops its own work if runtime free memory falls below 12 GiB. It requires a distinct `stage=materialize` permit, allows one build attempt, and never launches formal evaluation automatically.

At the **2026-09-25 05:18:35 UTC read-only snapshot**, [launch](evidence/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/materialize-supervisor-v1/launch-receipt.json) and [STARTED](evidence/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/materialize-supervisor-v1/STARTED.json) identify live supervisor PID `3270866` (kernel start time `490688995`), waiting for its approved GPU 6. No materialization child, accepted model, build completion or formal result existed. **Native smoke is PASS; model materialization and the 540-episode evaluation are unfinished.** The [snapshot inventory](evidence/ROBOTWIN-REGMEANPP-MATERIALIZE-SNAPSHOT.json) records exact retrieval scope; it is not a live status feed. Byte-preserved source README/CPU receipts describe their earlier preparation state.

After export, the queued code requires full model SHA/manifest binding, exact 418-module row and solver checks, a native policy reload with all 813 saved/loaded tensors bitwise equal, and finite native outputs for the three static observations before producing `MODEL-ACCEPTED.json`. These checks have been prepared, **not completed for a merged model**. The plan also freezes all nine formal job manifests and three reset banks (540 episodes, including the receptacle repeat-2 task-27 amended seed 877886121). Formal launch remains blocked until an isolated RegMean++ dense runner binds those jobs to the new accepted model SHA; the archived TIES/Soups queue is only a reference engine.

The [materialize permit snapshot](evidence/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/materialize-supervisor-v1/MATERIALIZE-EXECUTION-PERMIT.json) is included as explicitly requested audit evidence. It is bound to the historical host, GPU UUID, original plan, accepted smoke and single-use supervisor plan; it is **not portable launch authorization**. Release preflight parses source and verifies hashes only, and does not execute this permit or any queue. No model weights were copied.

RoboTwin TIES subsequently completed its full nine-job, 540-episode panel. The [independent raw audit](evidence/ROBOTWIN-TIES-540-INDEPENDENT-AUDIT.json) checks every episode and trace identity against the frozen model, task/seed/reset and action normalizer. Its three-repeat overall result is **5.37 ± 1.79%** (sample standard deviation); Coordination, Receptacle and Precision are 2.22 ± 0.96%, 9.44 ± 2.55% and 4.44 ± 2.55%. These values have been added to the paper; the older status table above remains the dated package snapshot.

## Scope and execution contracts

LIBERO-PRO is 3 repeats × 4 dimensions (`object`, `swap`, `semantic`, `task`) × 4 suites × 10 tasks × 10 states = 480 jobs / 4,800 episodes **per method**. Changed-goal `task` results are distinct from same-task robustness. The packaged source fixes the PRO checkout and asset-tree identities. Large reset/scene assets are external, recorded by references rather than replaced with invented data.

RoboTwin uses three 15k-step specialists (`coordination`, `receptacle`, `precision`), 30 tasks, three formal repeats, six episodes per task/repeat = 540 episodes per method. TCR has three separate A/B constructions; latest recovery quotas are 3,330,900 / 5,328,900 rows. The TIES density 0.3 / alpha 0.9 transfers from the main-table recipe without a RoboTwin search. These are distinct from RegMean/RegMean++ Gram alpha settings.

Native source entrypoints retain historical workspace/host/GPU/output constants because their bytes are preserved for audit. They are **reference snapshots**, not portable launch commands. The release smoke does not launch their `run`, `worker`, `build`, or `--execute` modes. Those modes can address historical output paths and persistent queues; use a separately frozen, resource-admitted new run on a reconstructed native workspace. Missing calibration/adapter contracts remain blocked.

`PROVENANCE.json` records each source path, SHA256 and size. The original snapshot retains its retrieval time; the five 1016 transfer additions have their own date. `native_sources/` mirrors the original project-relative layout. `evidence/` contains identity, plan, and terminal/state records; mutable state files are historical snapshots, never live status. Normal Git contains no weights or `.safetensors` binaries. The release root manages heavyweight assets separately.

The v2 resource-race addition preserves exact source paths and retrieval timestamps for its supervisor, child guard, tests, README, plan, launch records and original v1 failure chain. The later materialize addition includes its explicitly requested, SHA-bound permit snapshot as audit data. No model weights were added and no native queue was run by packaging.

The native dependencies include the project PI0.5/LeRobot environment, RoboTwin2 simulator with MPLib/CuRobo, LIBERO-PRO, MuJoCo/EGL, model/tokenizer/normalizers, and reset/asset manifests. They are not implicitly downloaded by this package. Native code stores their exact environment paths and dataset identities.

## Validation evidence

`evidence/PACKAGE-PREFLIGHT.json` records the original 97-file snapshot; the live provenance preflight also checks later additions. Eleven local package tests cover provenance integrity, stale checkpoint rejections, reset coverage, forbidden cross-benchmark substitution, the v2 smoke recovery, and the new smoke/permit/launch/materialize/formal-plan bindings. [The materialize package check](evidence/ROBOTWIN-REGMEANPP-MATERIALIZE-PACKAGE-CHECK.json) records package preflight, tests, doctor and diff checks; the 13 native materialize CPU tests are separately preserved as their original remote receipt, without being mislabeled as a GPU or completed-model test. [The v2 package check receipt](evidence/ROBOTWIN-REGMEANPP-SMOKE-V2-PACKAGE-CHECK.json) records all 13 original CPU lock/competition tests rerun locally. `evidence/NATIVE-CPU-SMOKE.json` records PASS for all three original native import/CLI checks. `evidence/GPU-SMOKE-BLOCKED.json` is the initial, superseded no-plan check. `evidence/GPU-RESOURCE-CONTENDED.json` records the first refused GPU 5 admission. After the main coordinator explicitly freed the card, `evidence/NATIVE-GPU-SMOKE.json` records PASS for one real native RoboTwin expert episode. Packaging signaled no existing process.


## Bounded native GPU smoke

An existing entrypoint `scripts/run_iclr2027_robotwin_checkpoint_development.py run --mode simulator --smoke` selects exactly the first task and first seed, running **one episode at that task's native horizon**. `gpu_smoke.py` reuses it unchanged with the already validated coordination 15k development parent. It tests the native expert/simulator interface, not the missing RoboTwin RegMean++ or FeatCal adapters and not a merged-policy score.

```bash
python experiments/robotwin_pro/smoke.py --workspace-root /mnt/workspace/Wilson/parameter-fusion --mode gpu --gpu 5 --expected-gpu-uuid GPU-4ed28198-b742-ee3c-2acb-4183dc944c81 --output /mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/release-robotwin-oneepisode-NEW-ATTEMPT
```

The adapter validates exact source and parent identities, requires a fresh non-formal output, binds host plus GPU UUID, and obtains a nonblocking host/UUID lease. It requires at least 40 GiB free and preserves the native runner's **stricter idle-only rule**: no compute PIDs, at most 64 MiB used, utilization zero. The native runner obtains its original per-index lease and repeats admission before CUDA. This adapter does not silently weaken that rule to share a card and never terminates any process. Resource contention returns BLOCKED immediately without waiting or retrying. A completed episode reports PASS for infrastructure regardless of task success; episode count and `formal_result=false` are checked.

On 2026-09-24, the existing parent passed native `validate` with no GPU. The first admission found GPU 5 at 25,408 MiB used with PID 3146242 and refused execution. The main coordinator later explicitly freed GPU 5. A fresh attempt then passed host/UUID plus native-index leases with 81,153 MiB free and no compute PIDs. The unchanged native `--smoke` completed one `handover_block` episode, seed 1786481984, at its 800-step horizon: exit 0, `smoke_complete`, 11,096 MiB peak CUDA allocation, approximately 659 seconds in the native runner. Task success was false; **PASS means the model/preprocessing/action/simulator path ran correctly, not that the task or an unfinished merging method succeeded**. The card was released before the coordinator resumed its evaluation lanes. See `NATIVE-GPU-SMOKE.json`, `NATIVE-GPU-SMOKE-COMPLETE.json`, and `NATIVE-GPU-SMOKE-EPISODE.json`.
