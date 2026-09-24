"""Check packaged code and, on the research host, run family CPU preflights."""
from __future__ import annotations

import argparse
import compileall
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("openvla_fastwam", "robotwin_pro", "appendix_cost_training")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--workspace-root", type=Path, default=Path(os.environ.get("TCR_WORKSPACE_ROOT", "/mnt/workspace/Wilson/parameter-fusion")))
    parser.add_argument("--mode", choices=("cpu",), default="cpu")
    args = parser.parse_args()

    if not compileall.compile_dir(ROOT / "experiments", quiet=1):
        print(json.dumps({"status": "FAIL", "reason": "Python syntax check failed"}))
        return 1
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("assets"), list):
        print(json.dumps({"status": "FAIL", "reason": "Asset manifest schema differs"}))
        return 1
    if args.code_only:
        print(json.dumps({"status": "PASS", "scope": "Python syntax and root manifest schema", "packages": PACKAGES}))
        return 0

    results = {}
    for family in PACKAGES:
        entry = ROOT / "experiments" / family / "smoke.py"
        if not entry.is_file():
            results[family] = {"status": "BLOCKED", "reason": "Package smoke entry is not published yet"}
            continue
        completed = subprocess.run(
            [sys.executable, str(entry), "--workspace-root", str(args.workspace_root), "--mode", args.mode],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        output = completed.stdout.strip()
        payload = None
        decoder = json.JSONDecoder()
        for offset, character in enumerate(output):
            if character != "{":
                continue
            try:
                candidate, end = decoder.raw_decode(output[offset:])
            except json.JSONDecodeError:
                continue
            if not output[offset + end:].strip() and isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            payload = {"status": "FAIL", "reason": "Smoke did not emit JSON", "stderr": completed.stderr[-1000:]}
        payload["status"] = str(payload.get("status", "FAIL")).upper()
        if completed.returncode and payload.get("status") == "PASS":
            payload = {"status": "FAIL", "reason": "Smoke exited nonzero despite PASS", "exit_code": completed.returncode}
        results[family] = payload
    status = "PASS" if all(row.get("status") == "PASS" for row in results.values()) else "BLOCKED"
    print(json.dumps({"status": status, "workspace_root": str(args.workspace_root), "packages": results}, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
