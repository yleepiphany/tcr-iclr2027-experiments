#!/usr/bin/env python3
"""Alias for the no-subprocess, no-GPU release preflight."""
import sys
from smoke import main
if __name__ == '__main__':
    raise SystemExit(main(['--preflight-only', *sys.argv[1:]]))
