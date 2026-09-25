# RoboTwin static baselines — CPU preparation, 2026-09-25

The paper currently retains RoboTwin RegMean++ and FeatCal rows in Table 5 and
the group breakdown. This directory adds an isolated shared static observation
bank and an M=3 FeatCal driver. **No GPU stage, merged model or formal evaluation
has been executed by this work.** Existing TIES/TCR queues and training are intact.

## Inputs and choices

The existing three 15k experts are Coordination, Receptacle and Precision. Their
813-tensor checkpoints, 418 adapted Linear weights, three RGB cameras, 14 physical
joint actions, 50-token action chunk and 32-dimensional padded action interface
match the existing PI0.5 graph. The normalizer files have packaging/hash
differences, but their flattened tensor values were independently checked equal.

Reusing TCR rollout/late-flow/prefix caches would change the requested Table-1
static baseline. Instead, `prepare_readiness.py` chooses one episode per task from
the existing content-hash **calibration split, rank 45**, separate from the 40
training and five development episodes. Five deterministic frame positions use
the valid 50-frame context range. This produces 150 static observations across
30 tasks. The split choice is explicitly dataset-specific; it is not asserted
to be the same episode bank as the earlier LIBERO study.

`collect_static_bank_cpu.py` reads only six parquet columns: observation state,
timestamp, frame index, episode index, row index and instruction-task index.
It does not read action/reward/success columns. Native PyAV decoding and the
saved Soup PI0.5 preprocessor produce normalized images, tokens and masks.
Three independent standard Gaussian `[50,32]` latents are stored per observation,
all at native generation time `t=1`. RegMean++ uses replica 0 (150 calls);
FeatCal uses all three (450 calls). There are no simulator rollouts or expert
feature statistics in this bank.

The existing Table-5 Soup is the equal task-delta, rank-192 PEFT-safe dense model.
Its exact numerical construction is preserved; it is not newly claimed to be
byte-identical to averaging already-rounded dense expert checkpoints.

## Artifacts

Runtime root:
`/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/robotwin-static-baselines-readiness-20260925`

- `cpu-readiness-plan.json`: frozen model, split, task and candidate input selection.
- `static-bank-v1/bank.json`: complete shared processed-input bank and source hashes.
- `static-bank-v1/{coordination,receptacle,precision}.safetensors`: each 91,429,158 bytes.
- `m3-kernel-cpu-check.json`: independent M=3 matrix-oracle comparisons.
- `source-v1/`: this isolated executable source, copied without changing active code.
- `featcal-m3-cpu-plan-v1/`: CPU-prepared FeatCal graph/index/implementation plan.

The shared bank hash is
`4c04affedb0f85622df565f77e6b03f8d67259ffa1b0fa3f5822ec5e08fa3bf5`.
The bank is about 274 MB; its status is `CPU_BANK_COMPLETE_GPU_UNVERIFIED`.

## FeatCal M=3 graph adapter

`featcal_m3.py` privately loads the existing Table-1 cache and solve modules.
It binds the three RoboTwin expert identities, bank, Soup and base without
editing the original files or their process-global live instances. The original
47-stage order, teacher snapshot hooks, exact teacher/student row pairing,
interpolation alpha=.3, Soup/base anchor rho=2, lambda=.05, epsilon=1e-8, FP64
kernel and atomic prefix barriers are retained. All 418 weights are calibrated;
four direct biases and all other tensors preserve the initial Soup.

The M=3 index has **450 calls, 2,820 shards (1,410 teacher and 1,410 student), and
3,330,900 regression rows**. The original LIBERO `_spec_map` hard-coded 3,760;
this adapter replaces only that identity mapping with the exact M=3 Cartesian
role/group/task/stage set. It does not relax duplicate/missing-row checks.

The matrix kernels already support arbitrary expert count. CPU full-rank and
rank-deficient M=3 examples matched independent references: RegMean++ spectral
smoothing had max absolute error 0 after FP32 output; FeatCal normal-equation
errors were below 9e-16. The RegMean++ recipe remains the **disclosed spectral
smoothing extension**, not an unmodified official solver. A graph-level RegMean++
M=3 materializer is still missing and must not be replaced by TCR's ridge/cap solve.

## CPU verification

Using the existing PI0.5 Python environment and this source directory:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /mnt/workspace/Wilson/parameter-fusion/pi05_lora_finetune_v2_20260826/.venv/bin/python \
  -m unittest -v test_readiness.py test_static_bank.py test_featcal_m3_graph_cpu.py
```

The tests cover selection disjointness, absence of label fields, independent
noise, three actual cameras, shape/finite checks, the full 47-stage/418-name hook
order on a width-2 CPU graph, teacher/student row pairing, propagation after a
prefix update, corrupted-row rejection, and the actual original solve-step
consuming M=3 shards followed by an atomic checkpoint barrier. The tiny graph is
**not** a native PI0.5 inference test. CUDA remains uninitialized.

The CPU collector narrowly hides optional TransformerEngine package discovery
in its own process because that optional PEFT import otherwise starts a
CUDA-only Triton autotuner. No installed library or policy code is modified.

## Future GPU admission and acceptance

The proposed single-card admission is **at least 48 GiB free**, a 50% PyTorch
allocator cap, both existing legacy and GPU-UUID leases, and checks for at least
12 GiB free before/after kernels and between collection calls. These are proposed
limits, not measured RoboTwin FeatCal peak-memory results. An occupied device
requires the explicit `--allow-shared` flag after owner/resource coordination;
no code here signals another process. Do not use an active TIES/TCR/training card.

GPU stages additionally require `--authorize-gpu-run`, `--gpu` and matching
`CUDA_VISIBLE_DEVICES`. They are not invoked by CPU preparation:

1. `--stage smoke`: native full-joint versus explicit static-input forward must
   be bitwise equal for Soup and all three experts; verify the full hook trace.
2. `--stage teachers --group GROUP`: collect all ten tasks of each expert, keeping
   exact static input and row identities. All three groups must finish.
3. `--stage solve`: each of 47 steps recollects the student through its current
   prefix, invokes the original FeatCal kernel, and commits all step weights
   atomically. Resume requires valid shard and prefix hashes.
4. Export all 813 tensors, preserving non-target Soup values; load the exported
   policy and require bitwise matching static-input output before `complete.json`.
5. Only then freeze a new method-specific RoboTwin evaluation using the existing
   9 group/repeat jobs, 540 episodes, reset identities and native evaluator.

The first four stages implement construction only, with zero formal episodes.
The final evaluation protocol is described but **not launched or claimed complete**.
Native GPU verification, full construction, and a method-specific 540-episode
manifest remain outstanding. The existing expert bank and formal result rows
must not be rerun as part of this preparation.
The inherited per-step `formal_cache` field means complete row quotas, not a
formal success-rate result; the new outer plan/checkpoint explicitly records
zero formal episodes.

Expected work is one teacher pass and 47 student replay stages over the 450
static calls. Disk/memory estimates are recorded in the shard index; kernel
workspace remains capped at 2 GiB per solve. Actual full-model peak and wall time
must be measured at the native GPU gate before scheduling a long build.
