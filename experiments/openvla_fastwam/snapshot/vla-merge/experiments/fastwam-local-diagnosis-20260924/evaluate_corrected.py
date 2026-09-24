"""Use the unchanged v6 native evaluator on the isolated repaired build.

The legacy tcr-build-v5/pipeline-v5 subdirectory names are an adapter layout in
the NEW output root, never the original model or original stopped v6 queue.
"""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'vla-merge/experiments/claude-fastwam-tcr-20260923/run_eval_job_v6.py'
spec = importlib.util.spec_from_file_location('corrected_native_evaluator', SOURCE)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
native.BASE = ROOT / 'vla-merge-runtime/experiments/fastwam-local-diagnosis-20260924/corrected-attempt-01'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--job-id', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not args.job_id.startswith('development-'):
        raise ValueError('Only the fixed 40 development episodes are authorized')
    import torch
    torch.cuda.set_per_process_memory_fraction(28 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    native.run(args)
