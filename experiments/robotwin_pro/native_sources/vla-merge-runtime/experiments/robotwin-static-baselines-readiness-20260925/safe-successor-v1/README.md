# RoboTwin FeatCal M3: safe single-card successor

This is an independent CPU supervisor and GPU worker for the previously frozen static-observation FeatCal M3 adapter. It leaves `source-v1`, the input bank, plan, algorithm, and existing RoboTwin/PRO queues unchanged. It has no stop, pause, kill, or preemption code. It launches zero formal evaluation episodes.

The bound plan is `../featcal-m3-cpu-plan-v1/plan.json`, SHA `f9fbc255cf37f5bec14fc2ba36d1b7c147b2d7205e6999f3a6a007df331c7eb5`. It uses 150 observations / 450 independent t=1 input states from the RoboTwin calibration split, three valid cameras, the selected three dense experts and their Soup, all 47 stages / 418 linear weights, and unchanged Table 1 FeatCal constants and exact solver.

`--mode preflight` is the default and uses only the Python standard library. It validates the frozen source/input hashes, model stats from full-hash CPU preparation, and deployment files. `--mode wait` requires both an explicit physical GPU list and `--authorize-gpu-run`. It takes an experiment-wide successor lock, then waits for an eligible card without allocating CUDA or holding a card lease while waiting.

Admission requires no compute process on the card, at least 48 GiB free memory, both existing legacy and UUID lease paths, a fresh GPU-identity/memory/compute check after locking, and 120 GiB free disk. The child inherits the lease file descriptors and checks the parent's admission receipt, PID, source SHA, FD file identities, and another empty-card snapshot before importing/initializing CUDA. Standard per-stage locks also exclude duplicate direct CLI runs. The worker preserves the original 0.5 allocator fraction and 12 GiB runtime free-memory floor.

`--target smoke` (default) runs only the native full-joint bitwise oracle against the explicit graph on Soup and the three experts, each with a five-observation batch. It stops after accepted smoke. `--target build` adds fresh teacher captures, all 47 sequential merged-prefix student solves, checkpoint export, and bitwise native reload. It fails closed on engineering or numerical failure and does not retry a failed GPU build on another card. Existing valid smoke and teacher markers are reused; completed prefix barriers permit solver resume. An existing final completion marker is checked against actual checkpoint SHA.

Seven CPU gate tests use fake GPU snapshots plus **real file locks and an actual child FD-inheritance test**. They cover occupied cards, insufficient memory, process/UUID changes after acquiring both leases, held leases, release behavior, duplicate stage/successor owners, disk shortage, and absence of process signaling. These do not constitute native GPU validation.

## Commands

```bash
TASK_WORK=/mnt/workspace/Wilson/parameter-fusion
TASK_ROOT=$TASK_WORK/vla-merge-runtime/experiments/robotwin-static-baselines-readiness-20260925
TASK_PYTHON=$TASK_WORK/pi05_lora_finetune_v2_20260826/.venv/bin/python
cd "$TASK_ROOT/safe-successor-v1"
"$TASK_PYTHON" -m unittest -v test_safe_successor_cpu
CUDA_VISIBLE_DEVICES= "$TASK_PYTHON" safe_successor.py --workspace-root "$TASK_WORK" \
  --queue-root "$TASK_ROOT/gpu-successor-v1" --mode preflight
```

Future resource-owner launch, with the physical card list chosen by that owner:

```bash
"$TASK_PYTHON" "$TASK_ROOT/safe-successor-v1/safe_successor.py" \
  --workspace-root "$TASK_WORK" --queue-root "$TASK_ROOT/gpu-successor-v1" \
  --mode wait --gpus 1,4,5,6 --target build --authorize-gpu-run
```

The list above is an example, not a reservation or a claim that those cards are idle. Default polling interval is 60 seconds; maximum waiting time is 24 hours. No card is admitted while it contains any compute process, including a training or evaluator process that currently uses little VRAM. A `STOP` file under the queue directory cancels waiting before admission; it never signals an already running worker.

Supervisor progress and latest state are `gpu-successor-v1/progress.jsonl` and `state.json`; worker log is `worker.log`. Native smoke, teacher markers, 47-stage receipts, and final checkpoint remain under `featcal-m3-cpu-plan-v1`. Final model completion is `featcal-m3-cpu-plan-v1/complete.json`; this means a reload-verified built model, not a formal RoboTwin result. Formal evaluation remains a separate owner-managed step.
