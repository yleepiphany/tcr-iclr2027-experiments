"""Memory-capped entry for opt-in Table-3 solve or procedural evaluation."""
import argparse
import runpy
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--table3-stage',choices=('solve','eval'),required=True)
    args,remaining=parser.parse_known_args()
    import torch
    torch.cuda.set_per_process_memory_fraction(.45 if args.table3_stage=='solve' else .30,0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    entry='materialize_pi05_tcr_e_peft_safe.py' if args.table3_stage=='solve' else 'eval_pi05_policy_with_procedural_bank.py'
    path=Path(__file__).resolve().parent/entry
    sys.argv=[str(path),*remaining]
    runpy.run_path(str(path),run_name='__main__')


if __name__=='__main__':
    main()
