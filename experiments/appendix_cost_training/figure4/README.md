# Figure 4: fixed TIES, RegMean++, and FeatCal subsets

This snapshot adds the two/three-expert **TIES**, **RegMean++**, and **FeatCal** implementations, frozen small plans, tests, and CPU/launch receipts. `PROVENANCE.json` maps every copied file to its exact original workspace path, size and SHA-256. Files under `sources/` are preserved byte-for-byte. No weights, tensor caches, action tensors or videos are bundled. The already-consumed v3 execution permit is included only as historical provenance with its matching consumption and startup receipts; it cannot authorize another run.

The parent Figure 4 plan has six new checkpoints and 15 suite jobs/1,500 episodes across the three methods. Its original missing-port flags remain preserved; the later child plans and CPU receipts document the implemented adapters. Source/queue readiness does not mean the six new formal points are complete.

| Method | Verified evidence in this snapshot | Still not a completed Figure 4 result |
|---|---|---|
| TIES | Two CPU builds, strict native CPU weight loads, and one native ten-step finite-action check per model passed. Final model-bound plan is five suite jobs/500 episodes. | Formal queue was launched and waiting in the captured status; no complete new subset success point is claimed. |
| RegMean++ | Two subset CPU plans, eight adapter tests and ten v3 queue tests passed; 37 numerical/capture functions retain identical AST. v3 launch/consumption receipts are included. | CPU checks and a waiting supervisor do not prove model export, native GPU acceptance, or 500-episode completion. |
| FeatCal | M2/M3 adapter plans and 12 CPU tests; completed M2 47-stage model/reload plus Spatial 100-episode reuse audit; exact-recovery v3 source, 15 CPU tests, plan and owner-consumed launch receipts. | V2 stopped for a GPU cleanup condition. V3 launched as PID 3267589 to add only the missing 400 episodes and M3 build stages. Launch evidence is not completion of those stages. |

TIES retains global per-expert task-vector trimming (density .3), mass sign election, the global zero-sign fallback, mean disjoint aggregation, and scale .9 over the original 422 tensor segments. It does not use a four-expert merged direction as a subset direction.

RegMean++ retains alpha .3, merged-prefix full-block replay, mean bias, and the disclosed fixed spectral smoothing extension. It uses only selected experts and their original 50 static flow-0 observations each: 1,184,200 rows for two experts and 1,776,300 for three. It adds no Soup-centered ridge, correction cap, or success-based tuning. Spectral smoothing remains an extension, not an exactly equivalent official solve.

FeatCal retains the original exact FP64 kernel, alpha .3, rho 2, lambda .05, covariance epsilon 1e-8, 47 stages/418 weights and a 2 GiB solver workspace. Each selected expert contributes 50 observations and three fixed independent t=1 noises (150 calls), with the original expert ordinals/seeds. M2/M3 use 2,220,600/3,330,900 regression rows. Teacher factors are freshly captured; old four-expert student factors are never reused.

**Use the isolated v3 recovery lineage for FeatCal; do not restart v2.** v1 was stopped before GPU work to restore Table1's construction TF32 override. V2 later completed the M2 model (SHA `e1d88a45…`) and Spatial 100 native episodes, then technically failed while waiting for the entire GPU to clear after a successful child exit. Its cleanup interrupted Goal evaluation and M3 smoke. The original failure and terminal-state snapshot are preserved; this is not a model-performance failure.

V3 inherits the verified M2 checkpoint, 47 prefix barriers/reload evidence and the complete Spatial 100-episode result (88 successes). It does **not rerun Spatial**. Its nine new stages comprise M2 Goal 100 episodes, M3 native smoke, three fresh teachers, the 47-stage solve/reload and M3 Spatial/Object/Goal 300 episodes: **400 new + 100 reused = the original 500**. Old partial Goal data is preserved but never mixed into the new result. Success counts do not determine recovery scope or job order.

V3 first audits and records a successfully exited child's complete artifacts. A subsequently busy GPU only delays another admission; it no longer invalidates those artifacts or terminates other normal workers. Dual/compatibility leases, fixed model identity and idle-card admission remain mandatory. Construction still sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`; formal evaluation clears it. The source, frozen plan, original 15 CPU tests, `CPU-DELIVERY.json`, `RECOVERY.json`, consumed permit and owner launch/start receipts are included by SHA. Plan SHA: `7f4bc975eaca6c8a685bd471eeb90ec59e0124ae52a3363cbfdf6b97b3c5adc0`. The captured owner launch is PID 3267589 / start ticks 490679066; the portable checks below do not start or inspect that server process.

The three four-expert endpoints reuse the current main-table checkpoints and three-repeat results (36 original suite jobs/3,600 episodes); they are not rebuilt. Two/three-expert points use one fixed subset and repeat-01. The mixed repeat counts and construction/calibration differences are explicit: this figure is descriptive, not an isolated causal test of expert count. Prior main-table results were known; subset rules were fixed before their new outcomes, with no new parameter scan.

## Portable CPU checks

From the release root:

```bash
python3 tools/doctor.py --code-only
python3 experiments/appendix_cost_training/figure4/check.py
python3 -m unittest discover -s experiments/appendix_cost_training/figure4 -p 'test_release.py'
```

The Figure 4 check rehashes all packaged snapshots, verifies frozen plan/readiness bindings, runs an independent NumPy TIES oracle for two/three experts, compares all nested RegMean++ source functions, and exercises the real selected-expert row contract including rejection of contamination and extra ridge. FeatCal checks preserve the original 14-stage/500-episode protocol and additionally verify the v3 9-stage/400-new-episode recovery, SHA-bound inherited model, all 100 Spatial boolean outcomes and reset receipt, consumed launch identity, selected budgets, backend separation and resource/permit rejection. They explicitly distinguish the historical CPU evidence from the later owner launch. It never initializes CUDA. The original PyTorch, safetensors, native-loader and GPU-transition tests remain included with their executed CPU receipts; rerunning those full native tests requires the original environment and external assets. A portable metadata/NumPy check is not a substitute for those tests or a rollout result.

## Original execution entrypoints

The byte-preserved source root is `sources/vla-merge/experiments/fig4-baselines-codex-20260925/`:

- `ties_subset_v2.py`: CPU `preflight`, fixed subset `build`, and strict `native-load`.
- `run_ties_cpu_queue.py`: unique low-priority two-build queue; original completed output cannot be retried or overwritten.
- `check_ties_native_actions_cpu.py`: original finite native action check; not a performance evaluation.
- `run_ties_formal_v1.py`: model binding and five-suite formal queue with idle-card admission and resource leases.
- `regmeanpp_subset/prepare_regmeanpp_subset.py`: subset CPU plan/preflight.
- `regmeanpp_subset/queue_regmeanpp_subset_v3.py`: explicit single-use permit, owned-card build/native-acceptance/formal queue. Earlier v1/v2 drafts and failed CPU import evidence remain historical records.
- `featcal_subset_v1/featcal_subset.py`: unchanged M2/M3 API for prepare, native bitwise smoke, selected teacher capture and all 47 solve stages.
- `featcal_queue_v3/queue_featcal.py`: exact recovery from the preserved v2 failure; inherited M2/Spatial acceptance, nine remaining stages and four new formal suite jobs. Successful artifacts are accepted before advisory post-exit GPU checks. Admission still requires an idle card with at least 48 GiB free and 120 GiB free disk; runtime reserves are 12 GiB GPU and 32 GiB disk. No unrelated process is stopped and failed stages are not automatically retried. V2 remains an immutable historical source snapshot.

These scripts deliberately retain original absolute paths, source hashes, hosts and no-retry guards. They are not automatically portable launch commands. Restore the declared native environment/assets and create newly reviewed isolated execution plans before running elsewhere. Historical launch receipts or consumed authorizations grant no permission to restart a live server queue.

`ASSETS.json` declares external files, including the two generated TIES models and the inherited FeatCal M2 model (each 9,354,050,784 bytes). Root `assets/manifest.json` records the original TIES assets; this Figure4 asset manifest additionally binds the inherited FeatCal M2 checkpoint. Their bytes have not been uploaded or included in Git.

The FeatCal 11,003,062-byte and 16,513,266-byte shard-index files are also external SHA-bound metadata. Small plans and 418-module quota manifests are included; large cache tensors and checkpoint bytes are not. Historical launch/permit-consumption records are provenance only and never authorize a new run.
