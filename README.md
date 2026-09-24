# TCR ICLR 2027 experiment package

Reproducible code and artifact identities for the unfinished non-real-robot experiments in *Trajectory-Calibrated Merging of Embodied Policies*. The package keeps an experiment's **code**, **frozen plan**, **checkpoint identity**, and **validation status** together. It does not silently substitute another model or reset bank when an artifact is unavailable.

This repository is being assembled from the live research workspace. The status of each family is recorded in its own `experiments/*/README.md`; `PASS` means a stated check actually ran, while `BLOCKED` names an unavailable checkpoint, data set, environment, or frozen protocol. A CPU syntax check alone does not establish closed-loop success.

| Package | Scope |
|---|---|
| `experiments/openvla_fastwam/` | OpenVLA-OFT and Fast-WAM cross-backbone experiments |
| `experiments/robotwin_pro/` | RoboTwin 2.0 and LIBERO-PRO cross-benchmark experiments |
| `experiments/appendix_cost_training/` | Calibration controls, action/cost accounting, and the planned continued-training study |

Real-robot experiments are excluded.

## Quick checks

```bash
python3 tools/doctor.py --code-only
python3 tools/doctor.py --workspace-root /mnt/workspace/Wilson/parameter-fusion --mode cpu
```

The second command checks the available source checkout, contracts, and required assets on the research server. A GPU smoke check uses each package's documented `smoke.py --mode gpu` and must run on a card with enough free memory; it does not start the full formal matrix. Formal runs use the package-specific frozen manifest and a new output directory.

Weights, calibration caches, reset banks, and simulator data are declared in `assets/manifest.json`. The original server paths are **sources**, not portability guarantees. Use `python3 tools/check_assets.py --workspace-root ...` to see what is available. Checkpoint sidecars, normalizers, tokenizers, task text, and reset selections matter as much as the main tensor file.

GitHub limits ordinary Git blobs to 100 MB and Git LFS on a Free/Pro account to 2 GB per file, with 10 GB of included LFS storage. Individual project checkpoints exceed those limits. Large assets must be published as SHA-verified, sub-2-GB release chunks or supplied from the declared artifact source. A Git pointer or path-only manifest does **not** mean the weight bytes have been published. See [GitHub's LFS file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage), [LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs), and [release asset limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Provenance

The source checkout is `/mnt/workspace/Wilson/parameter-fusion/vla-merge`; related model runtimes are sibling workspaces. Code snapshots in each experiment folder preserve their original source paths and SHA-256 digests. Older `state.json` files may say `running` after an immutable completion receipt exists. A result is complete only when its checkpoint, reset selection, episode count, and per-episode outputs match its frozen plan.
