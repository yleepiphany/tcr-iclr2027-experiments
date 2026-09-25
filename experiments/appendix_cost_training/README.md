# Appendix, action agreement, cost, and continued-training status

This package covers Tables **8–10**, updated **Table11 RegMean α=.3 action agreement**, **Table14 construction accounting**, the [Figure 4 TIES/RegMean++ subset package](figure4/README.md), and the explicitly blocked **Tables6/15 continued-training design**. Real-robot experiments and model weights are excluded.

## What passed, and what remains incomplete

| Area | Release status | Scientific status |
|---|---|---|
| Tables8–10 cache, budget, recipe | Actual contracts, materializers, queue entrypoints, recorded configs and dependency sources included | Some rows remain unfinished; do not treat source availability or historical table snapshots as completed evidence |
| Table11 RegMean α=.3 | Thin one-checkpoint adapter to the existing 400-request native-action implementation | New checkpoint action metrics have not been measured by this release; no success rerun is used |
| Table14 cost | CPU receipt extractor and historical extraction sources included | Missing wall/memory measurements remain null; no extrapolated compute claims |
| Tables6/15 continued training | Six-arm design template and fail-closed protocol preflight | **BLOCKED:** no reviewed common training budget/configuration exists; no training launcher is invented |
| Figure 4 TIES/RegMean++ subsets | Exact source snapshots, small frozen plans and CPU/native/launch receipts; portable NumPy and contract tests | TIES CPU builds/native checks passed; RegMean++ CPU adapter/queue checks passed. Captured waiting queues do not establish complete formal points. |

`PROVENANCE.json` binds copied sources/configs to their workspace-relative paths, SHA256 and byte counts. The original snapshot records its source Git HEAD and acquisition date; the five A∪B recovery additions are marked with their later creation date. Files under `sources/` are byte-preserved copies. Historical comments describe their original context; they are not new authorizations or current scientific claims. They may reference superseded runs.

`ASSETS.json` lists external artifacts by logical ID and workspace-relative path, with known sizes/hashes. No `.pt`, `.safetensors`, dataset, video, or model archive is stored in regular Git. The parent release may merge these declarations into its root asset manifest.

## CPU preflight / minimal smoke

From this directory:

```bash
python3 smoke.py --workspace-root /path/to/parameter-fusion --mode cpu
```

This really reads and hashes the copied files, parses Python source, imports the original appendix contracts, checks their nested row selection against the actual 418-module scope, and tests cost units, missing-value handling and duplicate guards. It never initializes CUDA or starts training. `status: PASS` describes those CPU checks; the JSON separately reports absent external assets and the blocked training protocol. Evidence from the executed release smoke is in `evidence/`.

The actual row budgets are 666,000 / 1,331,600 / 1,997,200 for the low/base/high second pass, and 5,772,800 for one pass over A∪B. Time projections cannot scale exactly like multi-token modules; the released contract preserves that distinction.

## Tables8–10 entrypoints

```bash
python3 workflow.py list
python3 workflow.py stage --workspace-root /path/to/parameter-fusion
python3 workflow.py plan --workspace-root /path/to/parameter-fusion \
  --workflow table9-row-budget --phase build-prepare
```

`stage` copies source bytes only and refuses to overwrite differing files. Frozen historical JSON/TeX snapshots stay in the release package for provenance; do not overwrite live plans with them. External calibration pools, expert checkpoints, native PI0.5 source/environment, tokenizer, fixed ridge reference and reset bank must be supplied from `ASSETS.json`.

The one-pass A∪B v8 solver completed its model but omitted the top-level `fixed_ridge_reference` manifest field, so the strict artifact gate correctly rejected that build. Its original source, checkpoint, manifest and failed receipt are preserved. For a fresh build, `run_union_v9.py` uses `materialize_union_v9.py`, which fixes only that metadata field, followed by `run_union_eval_fresh_v9.py`. On the original host, `recover_union_metadata_v9.py` independently checks the existing model SHA, all 418 ridge values and the frozen input hashes without changing the checkpoint; `run_union_eval_v9.py` uses that recovery receipt to run the original 400-episode formal protocol. The recovery scripts require the exact original failed build and cannot be used as a general shortcut around a failed solve.

`workflow.py run ... --execute` invokes the original runner; without `--execute` it only prints its command. Original hash, host, no-retry, output-directory and GPU-lease contracts remain enforced. These are reconstruction entrypoints for the original workspace layout, **not a claim that old frozen plans are automatically portable**. For exact replay, mount assets at their recorded layout (`/mnt/workspace/Wilson/parameter-fusion`); migration of embedded absolute paths or a different host needs newly reviewed/frozen plans, rather than bypassing checks. No GPU build or formal evaluation was launched to validate this release.

## Table11 updated ordinary RegMean α=.3

The original multi-candidate runner still names older baseline checkpoints. `action_metrics.py` therefore selects only the new ordinary-RegMean checkpoint with SHA `2d7acd682ec0d242eaa8726338125b65fea282787c508c60073135dcea11f54b`, verifies α=.3 and expert-prefix provenance, and reuses the existing raw held-out bank, teacher actions, native 10-step sampling and strict scorer.

```bash
python3 action_metrics.py plan --workspace-root /path/to/parameter-fusion --run /new/output/run
# In the restored native runtime, after explicit GPU allocation:
python3 action_metrics.py prepare --workspace-root /path/to/parameter-fusion --run /new/output/run --execute
python3 action_metrics.py collect --workspace-root /path/to/parameter-fusion --run /new/output/run --gpu 0 --execute
python3 action_metrics.py score --workspace-root /path/to/parameter-fusion --run /new/output/run --execute
```

Collection requires an explicitly assigned card, checks free memory, takes the existing card lock, and never stops another task. Metrics use dimensions0–5 and the first10 executed actions, exclude gripper/padding, preserve fixed normalizer identity, and aggregate ten requests per task across forty tasks. Zero-norm cosine coverage is reported rather than silently discarded. The original metric implementation's CPU tests passed; the new wrapper has **not** been GPU-executed here. No action MSE/cosine value is invented.

## Table14 cost extraction

```bash
python3 costs.py --manifest /checkpoint/block_regmeanpp_manifest.json \
  --resource /run/resources.json --output /new/cost-report.json
```

Only supplied receipts count. FP64/allocator/board-memory concepts are kept distinct. Worker elapsed seconds are not GPU busy-hours, FLOPs, energy, end-to-end training cost or whole-study elapsed time. Missing telemetry remains null. Sources for the earlier construction/pass2 accounting are preserved under `sources/vla-merge/scripts/`.

## Tables6/15: explicit blocker

```bash
python3 continued_training.py --config configs/continued_training.blocked.json
# Expected exit code: 2; JSON status: BLOCKED
```

The paper lists Base, a development-selected single expert, Soups, RegMean++, FeatCal and TCR, but no frozen shared budgetB, dataset mixture/revision, trainable scope, optimizer/scheduler, batch/accumulation, seeds, evaluation schedule, common thresholdτ, checkpoint bindings, single-expert selection rule, or resume semantics was found. Existing expert/RoboTwin training does not establish this six-arm comparison. The native PI0.5 framework and initialization asset locations are listed, but the template deliberately leaves unknown choices null. There is no fake validated trainer or GPU launch for these tables.
