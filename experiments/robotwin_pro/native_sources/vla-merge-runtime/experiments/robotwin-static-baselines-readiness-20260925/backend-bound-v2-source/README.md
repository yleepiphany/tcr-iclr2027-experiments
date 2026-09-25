# RoboTwin FeatCal M3: artifact audit and TF32-bound v2

The original M3 model passed an independent artifact-integrity audit: full checkpoint SHA `7b1598f577e6f5d6fc1d2ee75808957cdd8502925fe70dcc7cf33fb7ae5c6252`, 47 atomic prefix barriers, 418 solved weights, 3,330,900 actual regression rows, 813 finite export tensors, and 395 unchanged Soup tensors. All solved snapshots were fully hashed and their tensors compared to export. The 2,820 teacher/student shard manifests and tensor headers have complete identities, row quotas and paired row-key hashes. The large factor data files were not all independently rehashed by this audit. Three teacher markers, the four-model native smoke and bitwise reload receipt are present. The worker returned 0; both controller and child exited; no failure marker or formal episode exists.

The old checkpoint is nevertheless **backend-unbound and ineligible for paper/formal evaluation**. Its frozen plan and inherited-environment supervisor do not bind or record TF32. They are preserved unchanged. The original Table1 plan explicitly requires `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, and its driver rejects a different value. Fresh CPU processes using the installed PyTorch 2.7.1+cu128 reported CUDA matmul `allow_tf32=False` when unset/0 and `True` for 1, without initializing CUDA. Actual action/time interface weights are FP32, so this setting controls a relevant forward precision path. This establishes a possible numerical difference; no old-versus-new output or performance difference has been measured. [PyTorch 2.7 documents the override semantics](https://docs.pytorch.org/docs/2.7/cuda_environment_variables.html).

V2 creates a fresh `featcal-m3-tf32-v2` core plan through the **unchanged original API** in `source-v1/featcal_m3.py`. A separate immutable `backend-contract.json` binds that core plan to TF32=1 for native smoke, teacher capture, student capture, solving and reload. No mathematical API function or kernel is copied or edited. All original models, Soup, raw observation/noise bank, three cameras, 150 observations/450 t=1 calls, alpha .3/rho 2/lambda .05/eps 1e-8, 47 stages and 418-weight scope remain identical.

The child environment explicitly sets the PyTorch override to 1 and removes the NVIDIA global override, then verifies effective `torch.backends.cuda.matmul.allow_tf32=True`. The wrapper repeats that guard at forward and kernel calls. Each completed stage gets a new backend receipt binding environment, effective flag, core plan, backend contract, runner SHA and exact native artifact SHA. Resuming any stage without that receipt is rejected. **No v1 smoke, teacher factor, student factor, solved prefix or model is reused**; only immutable raw input observations/noises and original source models are shared.

The successor retains empty-card admission, 48 GiB free memory, two inherited GPU leases, duplicate-stage and successor locks, 0.5 allocator fraction, 12 GiB runtime reserve and 120 GiB free disk. It has no process-signaling code. No wait supervisor or GPU worker has been started for v2.

Frozen identities:

- Core plan: `987e95f0db4708079fbfab59e96e5147181619f6ebf9e337b9abb4c0e691495f`.
- Backend contract: `c5d726340ee390118910e661cceb4f037316fe3ab0dca24334e886d8a3a64ada`.
- Runner: `fbe7672600866663f4f2a3caa828efb6f62658b9403f0d94fb0c553d3b771c33`.
- Execution-readiness receipt: `62e4280989f7733dd72ebec06eb092af5dbf2b206b2150f008a6841374658903`.

Twelve CPU tests passed, including missing/zero override, global NVIDIA disable, disabled effective flag, altered/missing stage evidence, exact unchanged bank/recipe/API, no existing v1 stage reuse, occupied-card/race/lease/disk gates and real FD inheritance. CPU preparation and preflight passed. These checks do not replace the required new native GPU smoke.

## Host paths and next launch boundary

Root: `/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/robotwin-static-baselines-readiness-20260925`.

- New source: `backend-bound-v2-source/safe_successor_v2.py`.
- New core/output: `featcal-m3-tf32-v2/`.
- Immutable backend contract: `featcal-m3-tf32-v2/backend-contract.json`.
- CPU proof: `featcal-m3-tf32-v2/EXECUTION-READINESS.json` and `backend-bound-v2-source/cpu-tests.log`.

After owner review, use the usual native `.venv/bin/python` to invoke the new successor with `--workspace-root`, a new `--queue-root` under `gpu-successor-v2`, `--mode wait`, an owner-approved `--gpus` list, `--target build`, and `--authorize-gpu-run`. It first performs a new bound native smoke and then all construction stages. This delivery did not invoke that command.

Only the new backend-bound, reload-verified checkpoint may enter formal evaluation. The shortest valid formal follow-up is the existing native RoboTwin dense evaluator with a new isolated FeatCal method binding: 3 groups × 3 repeats × 10 tasks × 6 episodes = **9 jobs / 540 episodes**, protocol SHA `ed47fc5340cfb53a4a1c497a08ef3aca984c0ad50182748ac3df3ca7e060d0de`. Preserve its exact reset banks and the receptacle repeat-2 task-27 amendment to seed 877886121, demo_clean/joint control, full native horizons and checkpoint processors. Clear the construction TF32 override for the explicitly recorded formal environment. Bind every job to the new model SHA, complete/backend-stage receipts, 14-dimensional quantile normalizer and original 60 task/seed pairs; do not evaluate old `7b1598…` as a substitute.
