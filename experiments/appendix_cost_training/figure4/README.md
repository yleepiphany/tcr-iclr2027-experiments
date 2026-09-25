# Figure 4: fixed TIES and RegMean++ subsets

This snapshot adds the two/three-expert **TIES** and **RegMean++** implementations, frozen small plans, tests, and CPU/launch receipts. `PROVENANCE.json` maps every copied file to its exact original workspace path, size and SHA-256. Files under `sources/` are preserved byte-for-byte. No weights, tensor caches, actions, videos, or live execution permits are bundled.

The parent Figure 4 plan has six new checkpoints and 15 suite jobs/1,500 episodes across TIES, RegMean++, and FeatCal. **This release addition implements only TIES and RegMean++**: four new checkpoints and ten suite jobs/1,000 episodes. The parent plan's FeatCal entries describe remaining work and must not be mistaken for an implemented subset port.

| Method | Verified evidence in this snapshot | Still not a completed Figure 4 result |
|---|---|---|
| TIES | Two CPU builds, strict native CPU weight loads, and one native ten-step finite-action check per model passed. Final model-bound plan is five suite jobs/500 episodes. | Formal queue was launched and waiting in the captured status; no complete new subset success point is claimed. |
| RegMean++ | Two subset CPU plans, eight adapter tests and ten v3 queue tests passed; 37 numerical/capture functions retain identical AST. v3 launch/consumption receipts are included. | CPU checks and a waiting supervisor do not prove model export, native GPU acceptance, or 500-episode completion. |

TIES retains global per-expert task-vector trimming (density .3), mass sign election, the global zero-sign fallback, mean disjoint aggregation, and scale .9 over the original 422 tensor segments. It does not use a four-expert merged direction as a subset direction.

RegMean++ retains alpha .3, merged-prefix full-block replay, mean bias, and the disclosed fixed spectral smoothing extension. It uses only selected experts and their original 50 static flow-0 observations each: 1,184,200 rows for two experts and 1,776,300 for three. It adds no Soup-centered ridge, correction cap, or success-based tuning. Spectral smoothing remains an extension, not an exactly equivalent official solve.

The three four-expert endpoints reuse the current main-table checkpoints and three-repeat results (36 original suite jobs/3,600 episodes); they are not rebuilt. Two/three-expert points use one fixed subset and repeat-01. The mixed repeat counts and construction/calibration differences are explicit: this figure is descriptive, not an isolated causal test of expert count. Prior main-table results were known; subset rules were fixed before their new outcomes, with no new parameter scan.

## Portable CPU checks

From the release root:

```bash
python3 tools/doctor.py --code-only
python3 experiments/appendix_cost_training/figure4/check.py
python3 -m unittest discover -s experiments/appendix_cost_training/figure4 -p 'test_release.py'
```

The Figure 4 check rehashes all packaged snapshots, verifies frozen plan/readiness bindings, runs an independent NumPy TIES oracle for two/three experts, compares all nested RegMean++ source functions, and exercises the real selected-expert row contract including rejection of contamination and extra ridge. It never initializes CUDA. The original PyTorch, safetensors, native-loader and GPU-transition tests remain included with their executed CPU receipts; rerunning those full native tests requires the original environment and external assets. A portable metadata/NumPy check is not a substitute for those tests or a rollout result.

## Original execution entrypoints

The byte-preserved source root is `sources/vla-merge/experiments/fig4-baselines-codex-20260925/`:

- `ties_subset_v2.py`: CPU `preflight`, fixed subset `build`, and strict `native-load`.
- `run_ties_cpu_queue.py`: unique low-priority two-build queue; original completed output cannot be retried or overwritten.
- `check_ties_native_actions_cpu.py`: original finite native action check; not a performance evaluation.
- `run_ties_formal_v1.py`: model binding and five-suite formal queue with idle-card admission and resource leases.
- `regmeanpp_subset/prepare_regmeanpp_subset.py`: subset CPU plan/preflight.
- `regmeanpp_subset/queue_regmeanpp_subset_v3.py`: explicit single-use permit, owned-card build/native-acceptance/formal queue. Earlier v1/v2 drafts and failed CPU import evidence remain historical records.

These scripts deliberately retain original absolute paths, source hashes, hosts and no-retry guards. They are not automatically portable launch commands. Restore the declared native environment/assets and create newly reviewed isolated execution plans before running elsewhere. Historical launch receipts or consumed authorizations grant no permission to restart a live server queue.

`ASSETS.json` declares external files, including the two generated 9,354,050,784-byte TIES models. Root `assets/manifest.json` also records those exact paths/hashes. Their bytes have not been uploaded or included in Git.
