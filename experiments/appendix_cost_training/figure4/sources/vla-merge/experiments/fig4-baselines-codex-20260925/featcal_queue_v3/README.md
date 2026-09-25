# Figure4 FeatCal exact recovery v3

This is a new isolated queue. Historical v2 source, failure records, model, caches and formal outputs remain read-only. No method, input, noise seed, subset, regression quota, model initialization, solver or evaluation protocol is changed.

The v2 supervisor accepted the complete two-expert model but failed after the Spatial evaluator had already completed, while requiring the entire GPU to become idle within 60 seconds. That resource check ran before artifact acceptance and triggered cleanup of the still-running Goal evaluator and three-expert smoke. Its terminal `state.json` was not refreshed. All four recorded v2 supervisor/child process identities were gone before recovery preparation.

## Exact reuse and remaining work

V3 inherits the two-expert smoke, two teachers and solve, including the full 47-stage prefix chain and bitwise reload receipt. Preparation independently rechecked all snapshot/tensor hashes, 418 target coverage, all 813 exported tensors for finiteness, full exported model SHA, and sidecars. Model SHA: `e1d88a4570a31d742dba600abed033f6cd1a97c30670961cb035750bcfad745f`.

Spatial's original 100 native episodes are reused after independent reset-bank, seed, 10×10 boolean outcome, completed bounded-evaluator and model-launch validation. The recorded result is 88/100; success is not used to select or schedule work. Eval SHA: `c59eef3fa51bcc090bb19beeb1fdf5a2289a856bbe5de1080727c84eb686c266`.

V2 has no individual Spatial exit receipt because it failed before writing that receipt. Its traceback reached the post-exit clearance check after the source's zero-exit guard. This evidence limitation is retained explicitly in `RECOVERY.json`; the complete native/bounded/reset artifacts are independently verified rather than fabricating a historical exit receipt.

Nine new stages remain, in fixed dependency order:

- Two-expert Goal: one new full 100-episode job. Interrupted old Goal data is preserved and never mixed into the new result.
- Three-expert native smoke, three fresh teacher captures, and 47-stage solve/reload.
- Three-expert Spatial, Object and Goal: three full 100-episode jobs.

Thus 400 new formal episodes plus 100 reused Spatial episodes complete the original 500-episode matrix. The existing four-expert Table1 endpoint remains read-only reuse and is not evaluated here.

## Engineering correction

`finish_completed()` first requires successful child exit, audits complete stage artifacts, and durably records acceptance. It then takes one advisory GPU snapshot. A busy card, unknown host-namespace PID, changed UUID or unavailable query does not invalidate that result or terminate another worker. Future dispatch still requires a fresh strictly idle-card gate and the unchanged dual, compatibility and build-slot leases. The queue never signals a PID learned from `nvidia-smi`.

Actual stage failure or an owner stop retains the existing owned-child cleanup policy. The new terminal failure record atomically refreshes `state.json` with `active: {}` and records interrupted child identities; it no longer presents dead children as active. Resource-query failures during dispatch defer new admission.

All construction stages retain `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`; formal evaluation clears that variable. The original fixed reset bank, repeat-01, seed 274001, native horizons and frozen evaluator are retained. No parameter tuning or score-dependent gate is added.

## CPU evidence and launch boundary

Fifteen CPU tests passed, including acceptance-before-busy-card ordering while another worker remains untouched, unavailable queries, UUID changes, malformed stage binding, incomplete episodes, wrong resets, exact resource thresholds, inherited locks, backend separation, precise single-use permits, and terminal state cleanup. Preparation and subsequent preflight passed with 223 frozen assets. No GPU stage or execution permit was consumed by this delivery.

Remote source: `/mnt/workspace/Wilson/parameter-fusion/vla-merge/experiments/fig4-baselines-codex-20260925/featcal_queue_v3/queue_featcal.py`.

Remote runtime: `/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/fig4-baselines-codex-20260925/featcal-queue-v3`.

Frozen plan SHA: `7f4bc975eaca6c8a685bd471eeb90ec59e0124ae52a3363cbfdf6b97b3c5adc0`.
Source SHA: `ede2a14067b98422b13bfff0427c8fe7ca89cd43aa8cec6b9eb6da1d4eb098e3`.

The resource owner must review and supply a new v3 permit before launch. The template remains `allowed: false`; it binds nine job IDs, `formal_episodes: 400`, `reused_formal_episodes: 100`, the plan hash and exact host/GPU UUIDs. V2 permits are rejected. The four eligible physical-card candidates remain 1,4,5,6 on host `dsw-824375-57c745db88-n6tv9`, with no promise that they are currently idle.

```bash
TASK_WORK=/mnt/workspace/Wilson/parameter-fusion
TASK_PYTHON=$TASK_WORK/pi05_lora_finetune_v2_20260826/.venv/bin/python
TASK_SOURCE=$TASK_WORK/vla-merge/experiments/fig4-baselines-codex-20260925/featcal_queue_v3
TASK_RUN=$TASK_WORK/vla-merge-runtime/experiments/fig4-baselines-codex-20260925/featcal-queue-v3
CUDA_VISIBLE_DEVICES= "$TASK_PYTHON" "$TASK_SOURCE/queue_featcal.py" preflight
# Only after resource-owner review and a matching new explicit permit:
"$TASK_PYTHON" -u "$TASK_SOURCE/queue_featcal.py" run --permit "$TASK_RUN/EXECUTION-PERMIT.json"
```
