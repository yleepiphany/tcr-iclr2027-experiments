# Figure 4: fixed TIES, RegMean++, and FeatCal subsets

This snapshot adds the two/three-expert **TIES**, **RegMean++**, and **FeatCal** implementations, frozen small plans, tests, and CPU/launch receipts. `PROVENANCE.json` maps every copied file to its exact original workspace path, size and SHA-256. Files under `sources/` are preserved byte-for-byte. No weights, tensor caches, actions, videos, or live execution permits are bundled.

The parent Figure 4 plan has six new checkpoints and 15 suite jobs/1,500 episodes across the three methods. Its original missing-port flags remain preserved; the later child plans and CPU receipts document the implemented adapters. Source/queue readiness does not mean the six new formal points are complete.

| Method | Verified evidence in this snapshot | Still not a completed Figure 4 result |
|---|---|---|
| TIES | Two CPU builds, strict native CPU weight loads, and one native ten-step finite-action check per model passed. Final model-bound plan is five suite jobs/500 episodes. | Formal queue was launched and waiting in the captured status; no complete new subset success point is claimed. |
| RegMean++ | Two subset CPU plans, eight adapter tests and ten v3 queue tests passed; 37 numerical/capture functions retain identical AST. v3 launch/consumption receipts are included. | CPU checks and a waiting supervisor do not prove model export, native GPU acceptance, or 500-episode completion. |
| FeatCal | M2/M3 frozen plans and 12 original adapter CPU tests; isolated v2 queue and ten CPU queue tests. | Native GPU graph parity, fresh teacher capture, both 47-stage solves, bitwise export/reload and five formal jobs remain execution gates. |

TIES retains global per-expert task-vector trimming (density .3), mass sign election, the global zero-sign fallback, mean disjoint aggregation, and scale .9 over the original 422 tensor segments. It does not use a four-expert merged direction as a subset direction.

RegMean++ retains alpha .3, merged-prefix full-block replay, mean bias, and the disclosed fixed spectral smoothing extension. It uses only selected experts and their original 50 static flow-0 observations each: 1,184,200 rows for two experts and 1,776,300 for three. It adds no Soup-centered ridge, correction cap, or success-based tuning. Spectral smoothing remains an extension, not an exactly equivalent official solve.

FeatCal retains the original exact FP64 kernel, alpha .3, rho 2, lambda .05, covariance epsilon 1e-8, 47 stages/418 weights and a 2 GiB solver workspace. Each selected expert contributes 50 observations and three fixed independent t=1 noises (150 calls), with the original expert ordinals/seeds. M2/M3 use 2,220,600/3,330,900 regression rows. Teacher factors are freshly captured; old four-expert student factors are never reused.

**Use queue v2 for FeatCal.** v1 was owner-stopped before any GPU stage because its generic environment cleanup also removed the original Table 1 construction TF32 override. Its stop/FAILED evidence is preserved and is not a model-performance failure. v2 explicitly sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` for smoke/teachers/solve, bound to the original source plan, while formal evaluation still clears the override. Frozen adapter source, inputs, and mathematical kernels are unchanged.

The three four-expert endpoints reuse the current main-table checkpoints and three-repeat results (36 original suite jobs/3,600 episodes); they are not rebuilt. Two/three-expert points use one fixed subset and repeat-01. The mixed repeat counts and construction/calibration differences are explicit: this figure is descriptive, not an isolated causal test of expert count. Prior main-table results were known; subset rules were fixed before their new outcomes, with no new parameter scan.

## Portable CPU checks

From the release root:

```bash
python3 tools/doctor.py --code-only
python3 experiments/appendix_cost_training/figure4/check.py
python3 -m unittest discover -s experiments/appendix_cost_training/figure4 -p 'test_release.py'
```

The Figure 4 check rehashes all packaged snapshots, verifies frozen plan/readiness bindings, runs an independent NumPy TIES oracle for two/three experts, compares all nested RegMean++ source functions, and exercises the real selected-expert row contract including rejection of contamination and extra ridge. FeatCal checks cover the exact 14-stage/500-episode matrix, selected budgets, original build/formal backend separation, idle-memory boundary and rejection of an unauthorized permit. It never initializes CUDA. The original PyTorch, safetensors, native-loader and GPU-transition tests remain included with their executed CPU receipts; rerunning those full native tests requires the original environment and external assets. A portable metadata/NumPy check is not a substitute for those tests or a rollout result.

## Original execution entrypoints

The byte-preserved source root is `sources/vla-merge/experiments/fig4-baselines-codex-20260925/`:

- `ties_subset_v2.py`: CPU `preflight`, fixed subset `build`, and strict `native-load`.
- `run_ties_cpu_queue.py`: unique low-priority two-build queue; original completed output cannot be retried or overwritten.
- `check_ties_native_actions_cpu.py`: original finite native action check; not a performance evaluation.
- `run_ties_formal_v1.py`: model binding and five-suite formal queue with idle-card admission and resource leases.
- `regmeanpp_subset/prepare_regmeanpp_subset.py`: subset CPU plan/preflight.
- `regmeanpp_subset/queue_regmeanpp_subset_v3.py`: explicit single-use permit, owned-card build/native-acceptance/formal queue. Earlier v1/v2 drafts and failed CPU import evidence remain historical records.
- `featcal_subset_v1/featcal_subset.py`: unchanged M2/M3 API for prepare, native bitwise smoke, selected teacher capture and all 47 solve stages.
- `featcal_queue_v2/queue_featcal.py`: single-use permitted scheduler, parent-held inherited leases, two smokes, five teachers, two solves and five formal suite jobs. Admission is an idle card with at least 48 GiB free and 120 GiB free disk; runtime reserves are 12 GiB GPU and 32 GiB disk. It does not stop unrelated jobs or automatically retry failures.

These scripts deliberately retain original absolute paths, source hashes, hosts and no-retry guards. They are not automatically portable launch commands. Restore the declared native environment/assets and create newly reviewed isolated execution plans before running elsewhere. Historical launch receipts or consumed authorizations grant no permission to restart a live server queue.

`ASSETS.json` declares external files, including the two generated 9,354,050,784-byte TIES models. Root `assets/manifest.json` also records those exact paths/hashes. Their bytes have not been uploaded or included in Git.

The FeatCal 11,003,062-byte and 16,513,266-byte shard-index files are also external SHA-bound metadata. Small plans and 418-module quota manifests are included; large cache tensors and checkpoint bytes are not. Historical launch/permit-consumption records are provenance only and never authorize a new run.
