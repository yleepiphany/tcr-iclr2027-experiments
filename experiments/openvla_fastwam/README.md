# Table 4: OpenVLA-OFT and Fast-WAM

This package contains a SHA-256-pinned snapshot of the actual experiment code and
current execution plans captured on 2026-09-24. It excludes real-robot work,
training queues, credentials, model binaries, datasets, and private machine access.

The supported interface is a **host-runtime adapter**. Mount the original asset
tree (or its relocated copy) and pass `--workspace-root`. The smoke/preflight code
resolves data relative to that root. Native full-experiment controllers still use
their original host, lease and provenance contracts; `native.py` checks that the
installed controller matches this Git snapshot before dispatch. We do not claim
that copying a Python virtualenv makes it portable to another OS or CUDA stack.
Some frozen reset manifests and simulator configurations themselves contain
absolute paths. A relocated runtime must provide compatible read-only mounts or
re-freeze and validate those configurations; `--workspace-root` does not silently
rewrite SHA-bound artifact contents.

## Quick checks

From this directory, with ordinary Python 3:

```bash
python3 cli.py package-check
python3 -m unittest -v test_adapter.py
python3 smoke.py --workspace-root /mnt/workspace/Wilson/parameter-fusion --mode cpu
```

The first two commands need no model files or GPU. The runtime CPU check runs the
mounted `vla-merge-runtime/envs/mergevla/bin/python` and validates:

- OpenVLA checkpoint sidecars against frozen hashes, native imports, ten task
  identities and 100 procedural reset inputs.
- Fast-WAM normalizer identity, strict CPU-mmap structure (1,649 `mot` tensors and
  two proprio tensors), 100 procedural reset inputs, and the frozen native request
  used by the optional inference smoke.
- No CUDA device is visible to the CPU subprocesses. Metadata-only checks do not
  claim to hash all model weights; `cli.py preflight --verify-weights` does that.

The JSON result is `pass`, `blocked` (missing assets/resources), or `failed`.
Outputs are written to a new temporary directory unless `--output` is provided.

## One-request inference smoke

```bash
python3 smoke.py --workspace-root /mnt/workspace/Wilson/parameter-fusion \
  --mode gpu --backend openvla --gpu 5 --output /tmp/oft-release-smoke-new

python3 smoke.py --workspace-root /mnt/workspace/Wilson/parameter-fusion \
  --mode gpu --backend fastwam --gpu 5 --output /tmp/fastwam-release-smoke-new
```

Each invocation performs **at most one policy request**, never a formal episode,
training update or full experiment. Omitting `--backend` in GPU mode selects only
OpenVLA. The OpenVLA check loads an official expert, restores one frozen reset,
obtains an eight-action request and verifies exact request replay. Fast-WAM loads
the selected expert and executes one SHA-verified native calibration request,
requiring finite output with shape `[32,7]`; it does not report task success.

Admission requires **no compute PIDs, at most 512 MiB used, and at least 40,000 MiB
free**. Both existing legacy and UUID GPU leases are held, followed by a second
resource check. A busy device returns `blocked`; no process is signaled. All model
hashes for the selected expert are checked before the GPU worker is launched.

## Assets and environments

`assets.json` lists external checkpoint paths, sizes and known SHA-256 values;
`provenance.json` lists every copied source/config hash. Paths are relative to the
workspace root unless an original frozen manifest records the historical path.

Required directories include:

| Asset | Relative path |
|---|---|
| Runtime Python | `vla-merge-runtime/envs/mergevla` |
| OpenVLA implementation/overrides | `vla-merge_table4/OpenVLA-OFT/{source,dependencies,python_overlay}` |
| Official OpenVLA experts | `vla-merge_table4/OpenVLA-OFT/weights/{spatial,object,goal,long}` |
| Fast-WAM implementation | `vla-merge_table4/Fast-WAM/{source,.python-packages}` |
| Fast-WAM expert weights/statistics | `vla-merge_table4/Fast-WAM/weights/` |
| Fast-WAM pretrained components | `.datasets/FastWAM/models` and `text_embeds_cache` |
| LIBERO implementation | `vla-merge-runtime/references/LIBERO-MergeVLA` |
| OpenVLA data runtime | `vla-merge-runtime/references/dlimp-openvla` |
| Native simulator configurations | `.datasets/LIBERO/20260919/config-standard`; `vla-merge_table4/DreamZero/run/libero_config` |
| Formal reset bank | `vla-merge-runtime/experiments/iclr2027-table1-20260910/reset-banks/libero-procedural-clean-v1` |
| Fast-WAM native request cache | `vla-merge-runtime/experiments/claude-fastwam-tcr-20260923/calibration-v1` |

The bank's referenced source assets must also exist. Checkpoints require their
tokenizer, normalization and processor sidecars, not only backbone weights.
OpenVLA official experts are Spatial/Object/Long 150k and Goal 50k; Fast-WAM uses
Spatial 12k, Object 10k, Goal 10k and Long 9k. TensorFlow is CPU preprocessing only;
PyTorch/CUDA, EGL/MuJoCo, Hydra and the pinned model-specific Transformers runtime
are required. No model download is attempted by smoke.

## Current full-experiment entrypoints

Use `native.py` to inspect exact commands without executing them:

```bash
python3 native.py --workspace-root /mnt/workspace/Wilson/parameter-fusion \
  --entry openvla-r2-formal -- --run /path/to/registered-run
python3 native.py --workspace-root /mnt/workspace/Wilson/parameter-fusion \
  --entry fastwam-experts-v2 -- audit --job-id experts-spatial-repeat-01
```

Add `--execute` only to run that explicitly selected native command. Use the
controller's `--help` to inspect its complete native interface. Existing jobs must
be resumed by their original owner/receipt protocol; do not create duplicate
formal evaluations or rewrite old plans to bypass identity checks.

| Entry | Snapshot controller and frozen context |
|---|---|
| `openvla-r2-formal` | `claude-openvla-tcr-repair-20260922/run_repair_formal1200.py`; `R2-formal1200-attempt-01/plan.json` |
| `openvla-experts-r23` | `claude-openvla-tcr-repair-20260922/run_expert_formal_repeats23.py`; `expert-formal-repeats23-attempt-03/plan.json` |
| `fastwam-experts-v2` | `fastwam-formal-20260924/fastwam_experts_formal_v2.py`; **v2** contract, not the obsolete v1 bank runner |
| `fastwam-fixed-interface-build` | `fastwam-interface-recalibration-codex-20260924/run_fixed_build.py`; fixed Spatial-interface offline candidate |
| `fastwam-fixed-interface-eval` | same directory `run_fixed_eval.py`; same-request diagnostic, not formal success |

The `snapshot/` includes versioned dependencies needed to understand the current
controllers. Older helper names in that snapshot are **not additional recommended
entrypoints**. Source SHA changes on the host are rejected by `native.py`.

## Honest completion boundaries

- OpenVLA R2 and matched experts have current runnable native evaluation paths.
  Soup, TIES, RegMean++ and FeatCal rows require their own verified checkpoint and
  matching evaluation contract; this package does not invent missing models.
- Fast-WAM Experts have a current resumable v2 evaluation implementation. Prior
  Soup/TCR development gates failed. The fixed-interface candidate is explicitly
  an offline adaptation and must not be advertised as completed formal TCR.
- `smoke.py` validates an **expert/native interface**, not the performance or
  correctness of every uncompleted merged-model candidate.
- Full controllers remain tied to the mounted runtime's independent package
  dependencies and historical artifacts. An asset-only preflight is not a claim
  of an end-to-end formal reproduction on a clean machine.

Measured release checks and exact limitations are in `VALIDATION.md` and
`evidence/`. No giant weights are committed to normal Git.
