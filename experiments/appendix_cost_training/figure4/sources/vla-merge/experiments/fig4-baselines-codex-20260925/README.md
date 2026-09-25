# Figure 4 readiness and TIES CPU build

The immutable parent CPU plan is deployed under:

`/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/fig4-baselines-codex-20260925/plan.json`

Parent SHA256: `62ff0891df02a75d20a8b724f70f93a8373f75852a7385c88034993c326d7875`.

The plan contains six new two/three-expert checkpoints and 15 native suite evaluations (1,500 episodes). The three four-expert endpoints reuse 36 original complete Table 1 jobs (3,600 episodes). Two/three-expert points use one fixed repeat; four-expert points use main-table three-repeat mean and sample standard deviation. This is a descriptive mixed-protocol figure, not an isolated causal test of expert count.

`four-expert-reuse.json` independently checks all original Boolean episodes, task/reset indices and raw state hashes, bank, selection, seed, native entrypoint, result hashes, and launch/checkpoint SHA bindings. Existing large checkpoint bytes were not rehashed by this planning audit; their original immutable accepted identities are retained explicitly. No four-expert rollout is repeated.

## TIES implementation

Current adapter: `ties_subset_v2.py`, plan `ties-adapter-v2/plan.json`, SHA256 `e9b38b1e7793b31876c7c7804e0bc446e6287ec91f34d33444eca818432c407b`.

It calls the unchanged main-table `ta_ties_core.ties_direction` on only the selected original expert memmaps: density 0.3, global per-expert radix threshold, mass sign election, global zero-sign fallback, mean disjoint aggregation, and global scale 0.9. It preserves the original 422 adapted tensor segments / 2,706,471,968 coordinates and writes `base + 0.9 * direction` in FP64 before the original output dtype cast. Unadapted tensors remain exact base bytes. No four-expert merged direction is reused.

CPU tests passed for an independent two/three-expert global sort oracle and exact F32/BF16 safetensors export with base/untouched tensor preservation. Native `PI05Config`/`PI05Policy` imports, fixed action window, and full 422-tensor layout preflight passed. New model weights were not loaded during preflight because the builds had not finished.

The old v1 CPU import failure is preserved in `ties-adapter-v1/CPU-PREFLIGHT-FAILED.json`. PEFT eagerly discovered an optional system Transformer Engine installation requiring a visible CUDA driver. v2 uses the repository's existing CPU-only scoped discovery suppression; no shared dependency file, mathematical kernel, or GPU evaluator was modified.

## Running queue — do not duplicate

On host port 1016, supervisor PID **2863258**, kernel start ticks **489274316**, started `2026-09-25T01:17:41Z`. Initial build children: **2863259** (Spatial+Goal), **2863260** (Spatial+Object+Goal).

The process has nice 19, idle-class I/O, empty `CUDA_VISIBLE_DEVICES`, and two threads per child. Admission was 892 GiB available RAM, 86 CPUs, load 43; at most two CPU children run. The queue uses a unique directory and will not overwrite/retry existing builds. It can terminate only its own `Popen` children if a failure or host-memory reserve violation occurs.

Queue root: `ties-adapter-v2/cpu-queue-01/`; inspect `STATE.json`, per-subset build/native-load logs, `DONE.json`, or `FAILED.json`. Each completed build automatically runs strict native CPU weight loading and finite-state checks. No formal evaluation is launched.

CPU commands implemented for a future explicitly isolated run (do not invoke while this queue owns these outputs):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  <pi05-venv>/bin/python ties_subset_v2.py preflight

# The active queue already owns both build and native-load commands:
<pi05-venv>/bin/python ties_subset_v2.py build --subset spatial-goal
<pi05-venv>/bin/python ties_subset_v2.py native-load --subset spatial-goal
```

Before formal evaluation, require the completed checkpoint SHA/sidecar binding and one finite native action check on CPU or a newly allocated GPU. The frozen parent plan already lists the exact 15 formal command templates and reset selection, but it is not a GPU dispatcher.

RegMean++ and FeatCal subset adapters remain unimplemented. Their exact hardcoded expert-list, source-contract, mean-initialization, row-budget, and teacher-shard provenance gaps are recorded in the parent plan. The existing selected-expert static inputs and 1,410 FeatCal teacher shards are present; teacher data must be independently hashed before any cross-plan reuse, and old four-expert student shards cannot be reused.
