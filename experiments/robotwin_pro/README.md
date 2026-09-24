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

## Current identities and blockers

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

## Scope and execution contracts

LIBERO-PRO is 3 repeats × 4 dimensions (`object`, `swap`, `semantic`, `task`) × 4 suites × 10 tasks × 10 states = 480 jobs / 4,800 episodes **per method**. Changed-goal `task` results are distinct from same-task robustness. The packaged source fixes the PRO checkout and asset-tree identities. Large reset/scene assets are external, recorded by references rather than replaced with invented data.

RoboTwin uses three 15k-step specialists (`coordination`, `receptacle`, `precision`), 30 tasks, three formal repeats, six episodes per task/repeat = 540 episodes per method. TCR has three separate A/B constructions; latest recovery quotas are 3,330,900 / 5,328,900 rows. The TIES density 0.3 / alpha 0.9 transfers from the main-table recipe without a RoboTwin search. These are distinct from RegMean/RegMean++ Gram alpha settings.

Native source entrypoints retain historical workspace/host/GPU/output constants because their bytes are preserved for audit. They are **reference snapshots**, not portable launch commands. The release smoke does not launch their `run`, `worker`, `build`, or `--execute` modes. Those modes can address historical output paths and persistent queues; use a separately frozen, resource-admitted new run on a reconstructed native workspace. Missing calibration/adapter contracts remain blocked.

`PROVENANCE.json` records each original source path, SHA256, size, and retrieval time. `native_sources/` mirrors the original project-relative layout. `evidence/` contains only small identity, plan, and terminal/state records; mutable state files are historical snapshots, never live status. Normal Git contains no weights or `.safetensors` binaries. The release root manages heavyweight assets separately.

The native dependencies include the project PI0.5/LeRobot environment, RoboTwin2 simulator with MPLib/CuRobo, LIBERO-PRO, MuJoCo/EGL, model/tokenizer/normalizers, and reset/asset manifests. They are not implicitly downloaded by this package. Native code stores their exact environment paths and dataset identities.

## Validation evidence

`evidence/PACKAGE-PREFLIGHT.json` verifies all 97 exact source/config/evidence bindings. Six local CPU tests cover provenance integrity, both stale checkpoint rejections, missing reset coverage, and forbidden cross-benchmark checkpoint substitution. `evidence/NATIVE-CPU-SMOKE.json` records PASS for all three native import/CLI checks on the original workspace (the initial 88-file snapshot; nine additional configuration files were then added and separately included in the 97-file package preflight). `evidence/GPU-SMOKE-BLOCKED.json` is the initial, superseded no-plan check. `evidence/GPU-RESOURCE-CONTENDED.json` records the first refused admission because GPU 5 had another compute process. After the main coordinator explicitly freed the card, `evidence/NATIVE-GPU-SMOKE.json` records PASS for one real native RoboTwin episode. This adapter signaled no existing process.


## Bounded native GPU smoke

An existing entrypoint `scripts/run_iclr2027_robotwin_checkpoint_development.py run --mode simulator --smoke` selects exactly the first task and first seed, running **one episode at that task's native horizon**. `gpu_smoke.py` reuses it unchanged with the already validated coordination 15k development parent. It tests the native expert/simulator interface, not the missing RoboTwin RegMean++ or FeatCal adapters and not a merged-policy score.

```bash
python experiments/robotwin_pro/smoke.py --workspace-root /mnt/workspace/Wilson/parameter-fusion --mode gpu --gpu 5 --expected-gpu-uuid GPU-4ed28198-b742-ee3c-2acb-4183dc944c81 --output /mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/release-robotwin-oneepisode-NEW-ATTEMPT
```

The adapter validates exact source and parent identities, requires a fresh non-formal output, binds host plus GPU UUID, and obtains a nonblocking host/UUID lease. It requires at least 40 GiB free and preserves the native runner's **stricter idle-only rule**: no compute PIDs, at most 64 MiB used, utilization zero. The native runner obtains its original per-index lease and repeats admission before CUDA. This adapter does not silently weaken that rule to share a card and never terminates any process. Resource contention returns BLOCKED immediately without waiting or retrying. A completed episode reports PASS for infrastructure regardless of task success; episode count and `formal_result=false` are checked.

On 2026-09-24, the existing parent passed native `validate` with no GPU. The first admission found GPU 5 at 25,408 MiB used with PID 3146242 and refused execution. The main coordinator later explicitly freed GPU 5. A fresh attempt then passed host/UUID plus native-index leases with 81,153 MiB free and no compute PIDs. The unchanged native `--smoke` completed one `handover_block` episode, seed 1786481984, at its 800-step horizon: exit 0, `smoke_complete`, 11,096 MiB peak CUDA allocation, approximately 659 seconds in the native runner. Task success was false; **PASS means the model/preprocessing/action/simulator path ran correctly, not that the task or an unfinished merging method succeeded**. The card was released before the coordinator resumed its evaluation lanes. See `NATIVE-GPU-SMOKE.json`, `NATIVE-GPU-SMOKE-COMPLETE.json`, and `NATIVE-GPU-SMOKE-EPISODE.json`.
