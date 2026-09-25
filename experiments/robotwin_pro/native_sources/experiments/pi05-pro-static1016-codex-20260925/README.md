# Exact PRO static-v2 transfer to host1016

Prepared only. No GPU job has been started by the author of this package, and source plans/results remain unchanged.

- Code: `/mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/pi05-pro-static1016-codex-20260925/run_transfer.py`
- Frozen plan: `/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/pi05-pro-static1016-codex-20260925/attempt-01/plan.json`
- Plan SHA256: `8cffd8dbf8fd425f044d29f32087f6fa77b20b72da717462a265f9d260dd2b3e`
- Runner SHA256: `8dc34b7e7a9dd34b56226c0da508bbb942b5505c82ed6cca1d20299f5c8cead7`

The two unstarted source plans each contain 480 jobs / 4,800 episodes. The target interleaves both methods, producing 960 jobs / 9,600 episodes, with at most one worker on each of host1016 GPUs 1, 4, 5, and 6. Each native command changes **exactly one `--output_dir` token**; checkpoint, selection, seed, task, episode count, policy settings and evaluator are unchanged. Existing model SHA identities remain `fea99a8c...ffb2` and `b0da07ed...0114`.

`prepare` rehashed both model files and all selection/model proofs. `validate` reconstructs the complete target plan and compares every field. Five CPU tests pass: output-only translation; missing/duplicate output refusal; refusal of any started/result source; idle gate refusal for busy/unknown cards; method output separation. Native `nvidia-smi -i N --query-compute-apps=pid` was compared against global UUID buckets on GPUs 0, 1, 2, 4, 5, and 6: all matched. The busy 0/2 cards were correctly distinguished from empty 1/4/5/6.

## Final ownership handoff and execution

The coordinating agent must invoke `arm` only after its final source-owner review. `arm` checks exact source plan and CLAIM hashes, requires no source started/state/terminal/results/launch/exit files, takes the shared original baseline ownership fence, and writes new transfer sidecars plus target `ARMED.json`. It does not alter old plans or results. `run` repeats the checks and retains the same ownership fence throughout execution, preventing the original static-v2 runner from starting concurrently.

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/envs/iclr2027-libero-pro-py312-v1/bin/python /mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/pi05-pro-static1016-codex-20260925/run_transfer.py validate
# Only the coordinating agent performs these after final review:
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/envs/iclr2027-libero-pro-py312-v1/bin/python /mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/pi05-pro-static1016-codex-20260925/run_transfer.py arm
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/envs/iclr2027-libero-pro-py312-v1/bin/python /mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/pi05-pro-static1016-codex-20260925/run_transfer.py run
```

Each worker holds both the existing host/UUID `card_flock` and host/index resource lease. Admission requires the exact GPU UUID, at least 40 GiB free, at most 64 MiB used, zero utilization and no compute PID; busy cards are skipped. An in-run 12 GiB resource floor can stop only the worker process group created by this runner. SIGTERM to the supervisor stops new scheduling and drains existing workers. Unexpected supervisor errors retain ownership until its already-created children finish. No other process is killed, no training is launched, and no automatic retry or overwrite is allowed.

Every completed job is verified with the unchanged original static-v2 `verify_job`: model, method, selection, reset hashes, task identity, ten boolean episode outcomes, raw receipt hash and `eval_info` hash. Logs, launches and exits are separated by method under the new root. Original Soup/TIES resume queues on host1023 are not modified.
