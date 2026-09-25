# Figure 4 RegMean++ subset adapter: CPU stage

CPU preparation and eight tests passed. No CUDA context, GPU work, model export, rollout, or change to the Figure 4 parent plan or running PRO queues occurred.

The prefix is **merged-prefix full-block replay**, as frozen by Figure 4 and the main-table RegMean++ endpoint. Expert-prefix would be ordinary RegMean and is explicitly rejected. The earlier delegation wording was corrected by the root coordinator before implementation.

| Subset | Selected experts | Static observations | Regression rows | New evaluation scope |
|---|---|---:|---:|---:|
| spatial-goal | Spatial, Goal | 100 | 1,184,200 | repeat-01, 200 episodes |
| spatial-object-goal | Spatial, Object, Goal | 150 | 1,776,300 | repeat-01, 300 episodes |

Every selected expert retains its original 50 observation / flow-0 inputs and 592,100 row contribution: 50 rows in each of two time modules, 800 rows in each of 254 other modules, and 2,400 rows in each of 162 vision modules. All 418 weight modules and 422 adapted tensors remain in scope. Selected dense expert weights and biases are averaged over the selected set only. The four-expert model and Soup are not subset initializations.

The original smoothing kernel and adapter are reused byte-for-byte. Alpha .3, the input-width × FP32 epsilon × Gram spectral-scale threshold rule, power seed/iteration budget, complex128/FP64 solution, and arithmetic mean bias are unchanged. Thirty-seven source functions, including every nested numerical, capture, mean-bias and replay helper, have identical AST. Only `main` and the input loader's binding import differ.

## Files and frozen plans

Remote code directory:
`/mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/fig4-baselines-codex-20260925/regmeanpp_subset/`

- `materialize_regmeanpp_subset.py`: `2ef3f76e312b1aab7eb0290ddaceb84ac6bc92d707e2ebd5009af98680b3557c`
- `fig4_regmeanpp_contract.py`: `252140bb33973f4ad3e01f84427760591a9cf71f7224e5353acda8b6d700bd46`
- `prepare_regmeanpp_subset.py`: `2136e938bb3035e73a8e4e5b06f155cb2c140eb5baac5924bc7f810fe313099d`

Remote result/plan root:
`/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/fig4-baselines-codex-20260925/regmeanpp-subset-cpu-v1/`

- `spatial-goal/plan.json`: `d52e39fcf4b8f5469c5a1290d4ab589739c73bb0e95e299c94538df520860cbd`
- `spatial-object-goal/plan.json`: `8ba2f74c51409e0c061e69085407eef2032a705bd8b47139bac9215aec6fe19f`
- `CPU-ADAPTER-RECEIPT.json`: includes exact child command templates.
- `CPU-TEST-RECEIPT.json`: test outcome and all code hashes.

Parent Figure 4 plan SHA remains `62ff0891df02a75d20a8b724f70f93a8373f75852a7385c88034993c326d7875`.

## Required assets and remaining launch gate

Both subset plans explicitly bind the original common base, selected 10k LoRA adapters, their accepted dense-equivalent checkpoint files and manifest/verification proofs, selected static `demo-A/repeat-01/{expert}/replay.{json,safetensors}`, original fixed smoothing kernel, and the subset projection bank. All assets are present and their metadata identities match. Large replay/weight hashes are inherited from the frozen parent at CPU preparation; the execution contract rehashes selected replay files and dense model bytes before use.

GPU dispatch is deliberately still blocked by the parent CPU-only plan. A separate execution permit must use schema `fig4_regmeanpp_execution_permit_v1`, `allowed=true`, the exact `job_id`, `build_plan_sha256`, and `fig4_plan_sha256`. Its file path is supplied through `FIG4_REGMEANPP_EXECUTION_PERMIT`. This names the authorized model construction; **it is not itself a GPU reservation**.

The future coordinating launcher must first obtain an independently approved host/card allocation, fresh no-training/no-compute checks, and both existing host/UUID card flock and host/index resource lease, plus the host build-slot lock. Use an idle A100 with at least 70 GiB free, no existing compute PID, and a fresh checkpoint output. Keep the existing `.70` allocator limit. Do not use the currently occupied PRO cards or relabel the CPU plan as an execution permit. Native export/reload checks and the frozen subset evaluations remain unperformed.

For CPU-only revalidation:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/Wilson/parameter-fusion/pi05_lora_finetune_v2_20260826/.venv/bin/python /mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/fig4-baselines-codex-20260925/regmeanpp_subset/prepare_regmeanpp_subset.py preflight
```

Tests reject excluded experts, expert-prefix, budget changes and absent execution permits; they verify selected dense-bank projection, selected two/three-expert mean semantics and no excluded-weight influence. They do not substitute for a full-model GPU smoke. The four-expert Figure 4 endpoint remains the already-audited Table 1 result; it must not be rebuilt or rerun.
