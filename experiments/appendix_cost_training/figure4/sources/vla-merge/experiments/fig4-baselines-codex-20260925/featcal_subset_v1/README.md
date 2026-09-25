# Figure 4 FeatCal subsets: isolated native adapter

This directory implements only the frozen **spatial + goal** and **spatial + object + goal** builds. It does not modify Table 1 code, input files, checkpoints, or live queues. The four-expert point is read-only reuse of the current Table 1 checkpoint (`b0da07...`) and all its 1200 accepted episodes, mean 68.3333%, sample std 0.76376%. New subset points use repeat-01 only (200 and 300 episodes); this mixed-repeat, differing-budget comparison must be disclosed as specified in the master plan.

`featcal_subset.py` privately imports the original Table 1 helper and exact FP64 kernel. The only method changes are the selected expert list, corresponding accepted Soup initialization, subset shard dimensions, and immutable input/provenance bindings. Alpha=0.3, rho=2, lambda=0.05, covariance eps=1e-8; all 47 stages / 418 linear weights and the original 2 GiB solver workspace cap remain. Four direct biases and all other tensors preserve the selected Soup. Every student stage is recollected using the committed merged prefix and identical teacher row keys; stage commits and reload validation are mandatory.

`subset_inputs.py` provides a read-only view of the original 2026-09-22 observation bank. Each selected expert supplies 50 static demo observations and 3 independent t=1 noises per observation. No action labels, executed actions, rollout filtering, or success filtering is available to the collector. Goal keeps original expert ordinal 2 / RNG base 202609420000 even in the two-expert subset. Legacy flow slots 0/5/9 are noise-replica IDs, never physical times. LIBERO preserves its two valid cameras and masked third image.

| Subset | Observations | Noise calls | Regression rows | Shards (teacher + student) |
|---|---:|---:|---:|---:|
| spatial-goal | 100 | 300 | 2,220,600 | 1,880 |
| spatial-object-goal | 150 | 450 | 3,330,900 | 2,820 |

Both legacy subset bank files have stale `expert_count: 4` metadata. Their actual `expert_bank.experts`, `subset_derivation.expert_names`, and accepted Soup manifests agree on the correct two/three names. The adapter checks those exact ordered names, records both the legacy and effective counts, and leaves old files unchanged.

CPU verification tests compare every generated noise to the original Table 1 helper byte for byte; reject labels, out-of-subset groups, nonunit time, and incorrect masks; exercise the real 47-stage/418-target collectors on a width-two graph; check paired row identities and changed-prefix propagation; reject corrupted row indices; and invoke the actual original stage-zero solver on two/three-expert synthetic shards, including atomic snapshot publication. This validates adapter logic, not native model performance.

## Host commands

Host runtime root: `/mnt/workspace/Wilson/parameter-fusion` (shared by SSH ports 1016/1023). Python: `pi05_lora_finetune_v2_20260826/.venv/bin/python`.

```bash
TASK_WORK=/mnt/workspace/Wilson/parameter-fusion
TASK_SOURCE=$TASK_WORK/vla-merge/experiments/fig4-baselines-codex-20260925/featcal_subset_v1
TASK_RUNTIME=$TASK_WORK/vla-merge-runtime/experiments/fig4-baselines-codex-20260925
cd "$TASK_SOURCE"
CUDA_VISIBLE_DEVICES= "$TASK_WORK/pi05_lora_finetune_v2_20260826/.venv/bin/python" -m unittest -v test_subset_inputs_cpu test_subset_graph_cpu
```

The two prepared plans are under `featcal-adapter-v1/{spatial-goal,spatial-object-goal}/plan.json`. Preparation fully hashes the selected Soup/base/expert model files and checks all original input bytes. It hashes source implementations and deployment sidecars. Later contract checks reject source, input, sidecar, or model stat drift.

The following is a future **authorized resource-owner launch template**, not a command executed during this CPU delivery. Set `TASK_SUBSET` to one of the two frozen names and `TASK_GPU` to an owner-cleared card. No full evaluator is launched by this program.

```bash
TASK_SUBSET=spatial-goal
TASK_GPU=0
CUDA_VISIBLE_DEVICES=$TASK_GPU "$TASK_WORK/pi05_lora_finetune_v2_20260826/.venv/bin/python" "$TASK_SOURCE/featcal_subset.py" \
  --workspace-root "$TASK_WORK" --master-plan "$TASK_RUNTIME/plan.json" \
  --subset "$TASK_SUBSET" --run-root "$TASK_RUNTIME/featcal-adapter-v1/$TASK_SUBSET" \
  --stage smoke --gpu "$TASK_GPU" --authorize-gpu-run
```

After smoke passes, invoke the same command with `--stage teachers --group spatial`, then each other selected group, then `--stage solve`. These are single-card stages; old four-expert parallel workers are deliberately not reused. The solver exports to the exact master-plan target `checkpoints/featcal-$TASK_SUBSET/pretrained_model`. Completion requires all 47 solve barriers and bitwise native reload output equality.

GPU admission requires at least 48 GiB free, default no compute process, both the legacy host/card lease and host/card/UUID lease held nonblocking, a second memory/compute snapshot after acquiring locks, and 120 GiB free disk. Allocator fraction is 0.5; runtime free-memory floor is 12 GiB. A shared card additionally requires explicit `--allow-shared` resource-owner authorization. The program sends no signals to other jobs. It does not reserve a card during CPU preparation.

Remaining GPU gates: native full-joint graph parity on each selected expert and Soup; fresh teacher captures; all 47 student/solve stages; exported checkpoint reload; then the master plan's 5 new native evaluation jobs (500 episodes total). None is claimed complete by CPU tests. Existing teacher factor files are currently not reused: a future optimization would need per-shard original-plan data SHA, row-key/source checks and separate immutable cross-plan receipts; old student factors are never eligible.
