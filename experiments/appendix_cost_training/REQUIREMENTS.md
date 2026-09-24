# Runtime boundaries

The release CPU smoke uses Python **3.11+**, standard library modules, and the copied pure-Python appendix contracts. It does not need Torch, CUDA, LIBERO, a model checkpoint, or calibration tensors.

The recorded native runtime used Python3.12 and the following observed installed package versions for the original action-metric CPU check:

| Package | Observed version |
|---|---|
| torch | 2.7.1+cu128 |
| safetensors | 0.5.3 |
| numpy | 2.2.6 |
| transformers | 5.5.4 |

These are observed environment metadata, not a claim that `pip install` of these four packages alone reproduces the patched native PI0.5 stack. Restore the source/runtime, runtime overlay, tokenizer, processor files and benchmark assets listed in `ASSETS.json`. The original runners enforce their own model/config/hash contracts. The public CPU smoke does not weaken those checks.

No reviewed continued-training environment/optimizer configuration has been frozen for the six arms in Tables6/15. Do not substitute an unrelated RobotTwin or expert-training environment and claim a matched continuation experiment.
