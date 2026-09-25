# TIES subset formal evaluation — prepared, not launched

Runtime: `/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/fig4-baselines-codex-20260925/ties-formal-v1`.

Source: `vla-merge/experiments/fig4-baselines-codex-20260925/run_ties_formal_v1.py`.

Frozen template SHA256: `4821f594af7384cb51a5815b44a9a570b99dede98852b149eb34f6caea15d826`.

The exact matrix is Spatial+Goal (two suites, 200 episodes) and Spatial+Object+Goal (three suites, 300 episodes). All jobs use repeat-01, seed 274001, ten tasks per suite, and ten episodes per task. No four-expert jobs are repeated.

The native CPU `load_selection(..., verify_source_files=True)` preflight passed: original source BDDL/init files, bank arrays, selected indices and raw state hashes. The runner carries an isolated byte-identical copy of the existing 1016 native LIBERO configuration; global configuration was not edited. Synthetic audit checks accept exactly 100 Boolean outcomes and reject a missing episode or reset hash mismatch.

CPU-only binding waiter PID **2873148** polls the existing CPU build queue. It waits for both strict native CPU load receipts and `ties-adapter-v2/cpu-queue-01/DONE.json`, then rehashes the two completed models, freezes sidecar/stat identities, and exclusively writes `plan.json` / `PLAN-SHA256.json`. It never invokes the GPU runner. Check `bind-waiter/DONE.json` or `bind-waiter/FAILED.json`.

After final binding and explicit resource coordination, the implemented launch command is:

```bash
<pi05-venv>/bin/python -u run_ties_formal_v1.py run
```

This command has **not** been run. The runner is restricted to host 1016, GPU 1/4/5/6, at most four workers, minimum 78,000 MiB free, and no existing compute application. It takes the UUID card lease, host/index legacy lease, and the historical flat host/index compatibility lease before rechecking admission. Failures stop new dispatch and only terminate this supervisor's own `Popen` children; no retry, foreign process signal, or training action is implemented.

Each subset first receives one fixed-observation finite native `sample_actions` check in a separate child. These two checks use no success criterion and are excluded from the 500 formal episodes. Formal jobs use the unchanged bounded native evaluator, including its allocator fraction 0.20 and duty fraction 0.25. After each complete job, the existing strict original audit verifies all 100 Boolean outcomes, task coverage, actual reset identities, and result/receipt hashes. Partial results are excluded. Final `DONE.json` requires exactly five accepted jobs and 500 episodes.
