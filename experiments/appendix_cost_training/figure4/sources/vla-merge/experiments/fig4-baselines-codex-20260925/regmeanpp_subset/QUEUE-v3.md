# Figure 4 RegMean++ queue v3 — prepared, not running

The final queue adds two subset builds, two bounded native export checks, and exactly five formal suite jobs (500 episodes). It never rebuilds the four-expert endpoint. The original Figure 4 plan, Table 1 source, subset CPU plans, TIES files, and active PRO queues are unchanged. Older v1/v2 CPU drafts remain preserved and unstarted.

## Reviewable artifacts

Code directory: `/mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/fig4-baselines-codex-20260925/regmeanpp_subset/`

- Entry: `queue_regmeanpp_subset_v3.py`, SHA `9474687fd2210a3a1f969bbfc63f6f08e5f8df8ebb40b51f439dc73799b18c3f`
- Export/native checker: `accept_subset_checkpoint.py`, SHA `95d3afc1eaa228c1c1a8ecd2ccd8cc477b5c41370478f5d101b3edfc5da9fdc6`
- Tests: `test_regmeanpp_queue_v3_cpu.py`, SHA `ae80b1d4e9d487416ddabe612642164bca5925f1cd35c4ba2e8639872d186502`

Queue directory: `/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/fig4-baselines-codex-20260925/regmeanpp-queue-v3/`

- `plan.json`, SHA `c83cfbda79f58f12c322f58c0feac9e08b0109afbe815b005ce753ed7f10e88d`
- `CPU-QUEUE-TEST-RECEIPT.json`: ten CPU tests passed, zero GPU jobs/leases.
- `PERMIT-TEMPLATE-NOT-AUTHORIZED.json`: `allowed=false`; deliberately cannot start work.

## Final start conditions

A coordinating owner must create a separate **allowed, single-use** permit binding the exact queue SHA, parent Figure 4 SHA, host1016 and the four frozen GPU UUIDs, both build IDs, five formal jobs, and 500 episodes. Then the explicit `run --permit PATH` action may start the queue. A preexisting `AUTHORIZATION-CONSUMED.json` prevents any duplicate run; existing checkpoint or formal output directories are refused instead of overwritten.

Before acquiring GPUs, execution rehashes **all** frozen source, model, adapter, replay, observation, selection and bank assets. Each build's original contract also rechecks its selected large files. CPU preparation checked source/plan identities and small files, but did not claim to rehash every large asset again.

The queue only considers host1016 GPUs 1/4/5/6. Admission requires at least **71,680 MiB (70 GiB)** free, at most 64 MiB used, zero utilization and no compute PIDs. It takes the existing host/UUID `card_flock` and host/index resource lease, plus a host build slot for a construction, then rechecks the card immediately before launching. Busy PRO cards are skipped; after explicit authorization the queue can wait unattended until PRO releases them. **No watcher or GPU process has been started yet.**

After a build child exits, v3 retains the exact same two GPU leases and build slot while waiting up to **60 seconds**, polling every **2 seconds**, for its CUDA context to disappear. An NVML entry may name only the already-reaped child; another PID, reuse of that PID by a live process, or a changed GPU UUID fails closed. This transition never unlocks/reclaims a card or sends a signal. Its measured checks are written to `build-to-accept-resource-transition.json`.

The export checker reopens all 813 tensors, checks finite values and the actual model SHA, validates the selected expert set/418-module row budgets/smoothing metadata, and loads the saved model for **one native ten-step generation request**. The request is the parent plan's fixed preprocessed Spatial observation and exact initial noise, used only for shape/finite/reload acceptance, not calibration or a success gate. No action labels or score-based choices are introduced.

Only after export acceptance may that model's two or three fixed suites run. Commands are byte-for-byte the parent Figure 4 command templates: repeat-01, seed 274001, ten tasks × ten episodes per suite. Results are audited against exact task names, reset indices, raw reset hashes, evaluator SHA and boolean outcomes. No subset hyperparameter or success threshold is tuned. Queue completion records both builds and five accepted results.

Runtime reserve violations affect only the child process group created by this queue. Other jobs and training are never terminated. A stop request prevents new jobs and drains existing children under their leases. Any engineering failure preserves outputs and ends without automatic retry; recovering an already-created model requires a new explicit recovery plan.

CPU-only preflight command:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/Wilson/parameter-fusion/pi05_lora_finetune_v2_20260826/.venv/bin/python /mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/fig4-baselines-codex-20260925/regmeanpp_subset/queue_regmeanpp_subset_v3.py preflight
```

Future execution entry, after owner approval and without changing the frozen plan:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/Wilson/parameter-fusion/pi05_lora_finetune_v2_20260826/.venv/bin/python /mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/fig4-baselines-codex-20260925/regmeanpp_subset/queue_regmeanpp_subset_v3.py run --permit /absolute/path/to/approved-single-use-permit.json
```
