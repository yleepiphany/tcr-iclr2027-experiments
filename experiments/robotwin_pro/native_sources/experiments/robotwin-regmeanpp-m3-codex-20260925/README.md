# RoboTwin three-expert RegMean++ native graph port

Status: **CPU PASS; BLOCKED_NATIVE_GPU_PARITY**. No GPU stage or formal evaluation has been started. No execution permit has been created. This isolated port does not modify Table1, the FeatCal M3 port, PRO queues, or training.

Frozen plan: `/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/cpu-plan-v1/plan.json`

Plan SHA256: `bb25730b256669eecd1180026f1339cbb6b52d16342d9e65750e521e1b39ad0b`

Code: `/mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/robotwin-regmeanpp-m3-codex-20260925/`

## Mathematical and graph contract

The accepted RoboTwin dense experts are coordination, receptacle and precision, in that order. All four model files (base plus these three experts) were fully hashed during CPU preparation. Shapes of all 813 tensors match; the calibrated scope is 418 weights plus four arithmetic-mean interface biases. Unadapted output tensors remain the common base. Initialization of adapted tensors is the arithmetic mean of the accepted dense experts, as in the Table1 RegMean++ materializer.

The static bank is the accepted 150 demonstration observations, 50 per expert (10 tasks × 5 requests), using only independent Gaussian noise replica 0 at physical t=1. No demonstration action label, reward, success filter, expert rollout, or TCR execution feature is read. All three bank tensor files and the selection plan are frozen by SHA.

The 67 atomic stages follow the Table1 order: 27 vision blocks, 18 language blocks, action input projection, time MLP input, time MLP output, 18 action blocks, then the action head. For each complete block, its current expert weights are installed together, all selected internal inputs are collected, and those temporary weights are restored. The preceding blocks remain merged. All three experts' inputs are collected before solving; all block solutions are validated before any parameter is committed. This is the RegMean++ merged-prefix graph, including expert-specific *within-block* activations.

The native PI0.5 prefix is recomputed from static raw input for every stage. Language prefix tokens cannot attend to action suffix tokens under the native mask; therefore language-first then action is the same valid topological ordering used by Table1's cached prefix path. No old `ReplayState` cache is relabeled as a new static observation.

For every linear module, let `H_i = .3 X_i^T X_i + .7 diag(X_i^T X_i)`, `A = sum_i H_i`, `Wbar = mean_i W_i`, and `R = sum_i H_i (W_i-Wbar)^T`. The unmodified Table1 kernel computes

`W = Wbar + Re[(A + i*tau*I)^(-1) R]^T`,

where `tau = input_width * eps(float32) * deterministic_power_lambda_max(A)` with the frozen seed 20260922 and maximum 64 power iterations. Dense complex128 and complex128 Woodbury implement the same filter. There is no Soup-centered ridge, weight clipping, trust cap, nonzero energy floor, teacher output regression, or success-dependent setting. This is the disclosed spectral-smoothing extension, not unmodified official RegMean++.

The exact Table1 kernel SHA is `55b06cc4b6a43c029ade1a5471e3ca1259eb5ec1234420a454333b98411ee3fc`. Its original-equation residual and smoothed-equation residual remain distinct. Only the latter is the numerical solve criterion (`<=1e-7`).

## Fixed row budget

| Module family | Modules | Rows per expert per module | M=3 total rows |
|---|---:|---:|---:|
| Vision | 162 | 2,400 | 1,166,400 |
| Language, action blocks, action input/head | 254 | 800 | 609,600 |
| Time MLP | 2 | 50 | 300 |
| Total | 418 | | 1,776,300 |

Sampling is the original `linspace(...).round()` rule with at most 16 rows **per observation**, separately per camera. B=5 never becomes a single 16-row sample. Row order inside a Gram can differ from the B=1 cached implementation, while the selected per-observation rows and mathematical objective are unchanged. The native graph port uses 150 distinct static observations; it reruns them through prefixes across 67 stages (2,010 task batches, up to 10,050 observation passes), rather than claiming only 150 total forward executions. At most one block's inputs are retained. Total factor volume across stages is 14,738,073,600 bytes, not a required simultaneous allocation.

Preprocessor normalizers are compared by keys and flattened tensor values. Accepted coordination/receptacle and precision files have different byte layouts but equal semantics; export uses coordination support files. All deployment sidecars are frozen by SHA.

## CPU evidence

`CPU-PREPARED.json` records 150 validated observations, all four model-file hashes, 1,672 adapted matrix shape checks, and no CUDA initialization. `CPU-TEST-RECEIPT.json` records 10 passing tests:

- 67-stage scope, 418 weights, exact row budget, and B=5 versus original per-observation sampling.
- Two complete nonlinear blocks: candidate internal inputs match an independent expression, the earlier merged boundary stays fixed, temporary candidate weights restore, and early-stop/full-forward captures agree.
- Atomic block update rejects missing or nonfinite outputs before changing weights.
- Dense and Woodbury results each match independent complex direct-solve and Hermitian-eigenvalue oracles, for ordinary and weak/rank-deficient matrices.
- Identical-expert exact recovery, resource-admission negatives, permit identity negatives, and CUDA remaining uninitialized.

Executed CPU command:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /mnt/workspace/Wilson/parameter-fusion/pi05_lora_finetune_v2_20260826/.venv/bin/python \
  /mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/robotwin-regmeanpp-m3-codex-20260925/test_regmeanpp_m3_cpu.py
```

## Future native GPU gate (not executed)

Every GPU invocation needs a fresh external single-use permit bound to the exact plan SHA, absolute run path, stage, host, GPU index and UUID. `run_regmeanpp_m3.py` does not create permits. It only admits the frozen 1016 host and GPUs 1/4/5/6 after checking **at least 70 GiB free, at most 64 MiB used, zero GPU utilization and no compute PIDs**. It holds both UUID and legacy host/index flocks plus a build-slot lock, rechecks the card after locking, then sets the allocator fraction to .70. Busy cards fail closed; no process is signaled and no lease file is unlinked.

The future native smoke uses one static replica-0 observation per group at the mean-initialized model. It requires bitwise agreement of native fixed-time velocity and all 418 sampled linear inputs with the original `model.forward` interface supplied fixed synthetic flow inputs. The oracle's zero dummy actions are only interface placeholders: the flow constructor is replaced before interpolation and no dataset action is read. Candidate vision0, language0 and action0 sampled inputs must also agree between early-stop and full-forward collection. This tests 3 distinct static observations and 24 bounded forward attempts, with zero simulator episodes.

Only a `PASS` smoke bound to this exact plan enables a separately permitted `--stage materialize`. There is no auto-launch or formal-evaluation path in this package. The final build reopens all 813 saved tensors, records the model SHA and residual metadata, and still requires model-class reload / deployment acceptance before formal evaluation. Native parity, device memory fit, saved-checkpoint native behavior and task success are not established by these CPU tests.

The guarded future command shape is:

```sh
PYTHONDONTWRITEBYTECODE=1 \
  /mnt/workspace/Wilson/parameter-fusion/pi05_lora_finetune_v2_20260826/.venv/bin/python \
  /mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/robotwin-regmeanpp-m3-codex-20260925/run_regmeanpp_m3.py \
  --stage smoke --run /mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/robotwin-regmeanpp-m3-codex-20260925/cpu-plan-v1 \
  --gpu INDEX --execute --permit /absolute/path/to/externally-authorized-permit.json
```

`INDEX` must identify an actually empty approved card. The presence of this command or plan is not GPU execution authorization.
