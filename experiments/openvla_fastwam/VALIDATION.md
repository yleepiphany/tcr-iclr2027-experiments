# Release validation — 2026-09-24

**PASS: CPU preflight and one native GPU inference request for both backends.**
These checks used the mounted Linux experiment runtime and an A100 80GB, not a
fresh-machine dependency installation. Missing external assets remain explicit
preflight blockers. This does not certify incomplete formal experiment rows.

| Check | Result | Evidence |
|---|---|---|
| Source/config identity and syntax | PASS; 150 redistributed snapshot files SHA-verified and Python parsed | `provenance.json`; `test_adapter.py` |
| Adapter tests | PASS, 5/5 | Missing assets fail closed; relocation; no-CUDA CPU env; busy GPU cannot launch; snapshot identity |
| OpenVLA CPU native preflight | PASS; ten tasks and 100 selected resets verified | `evidence/openvla-cpu-contract.json`, `cpu-smoke-report.json` |
| Fast-WAM CPU preflight | PASS; 1,651 mmap tensors, 100 selected resets, frozen request SHA | `evidence/fastwam-cpu-preflight.json` |
| OpenVLA native GPU request | PASS; action shape `[8,7]`, exact replay `max_abs=0.0`, one request | `evidence/openvla-native-smoke.json`, `openvla-gpu-report.json` |
| Fast-WAM native GPU request | PASS; action shape `[32,7]`, finite, one request, 12.87 seconds inference | `evidence/fastwam-native-smoke.json`, `fastwam-gpu-report.json` |
| Formal evaluation / training | **Not run**; zero formal episodes and zero training updates | Both GPU launch/result reports |

Fast-WAM GPU peak allocated memory was **23,949.97 MiB**. Its verified Spatial12k
checkpoint SHA-256 is
`540249aee838c87fc57ae6f93fb320092a8d7aeaed1ebe4004455eee479cbaf8`;
native request SHA-256 is
`aca1c947ba955da3cc6cdc40cb9620e93fe6e2cea5e48c8eec7354fc422fca00`.
The OpenVLA GPU run hashed all selected official expert checkpoint files,
including its four backbone shards and separate action/proprio heads, against
the frozen expert identity manifest before loading.

Both successful GPU launches recorded zero existing compute PIDs, 1 MiB used,
81,153 MiB free, and held both existing legacy and UUID leases. Earlier attempts
correctly returned BLOCKED when another paper job still owned a lease. The
coordinator released only identified paper jobs before the successful retries;
this package never sends signals to other processes. Both inference workers and
their wrappers exited; GPU5 was verified free again after completion.

Commands used on the runtime host (the package was copied to a separate `/tmp`
directory, leaving the active experiment source untouched):

```bash
python smoke.py --workspace-root /mnt/workspace/Wilson/parameter-fusion \
  --mode cpu --output /tmp/tcr-table4-release-evidence-cpu-v2-20260924

python smoke.py --workspace-root /mnt/workspace/Wilson/parameter-fusion \
  --mode gpu --backend openvla --gpu 5 --wait-idle-seconds 300 \
  --output /tmp/tcr-table4-release-evidence-openvla-gpu-v3-20260924

python smoke.py --workspace-root /mnt/workspace/Wilson/parameter-fusion \
  --mode gpu --backend fastwam --gpu 5 --wait-idle-seconds 600 \
  --output /tmp/tcr-table4-release-evidence-fastwam-gpu-20260924
```

The first CPU attempt exposed a real Torch 2.2 compatibility problem: mmap
loading required `str(path)`, not `Path`. Only this new adapter was corrected;
the second CPU attempt and both actual GPU checks passed. Runtime versions are
recorded in `evidence/runtime-versions.json`. OpenVLA selects its own pinned
Transformers overlays, so the installed package version alone is insufficient.

`evidence/receipt-index.json` binds the copied GPU artifacts by SHA-256. The
executed provenance recorded 164 files. For publication, 14 unlicensed upstream
Fast-WAM YAML copies were removed; their **hashes only** remain as external
dependency metadata. No model/inference code changed for that removal. The
published CLI additionally summarizes the large OpenVLA call trace in stdout;
the exact original trace is retained in its native smoke JSON.

No weights, native request tensors, checkpoint binaries or upstream Fast-WAM
source/config texts are included in normal Git. External asset metadata and
our experiment wrappers are included. Full native controllers are supplied with
their precise current entrypoints, but their unfinished outputs are not claimed
as reproduced by these expert-interface smoke checks.
