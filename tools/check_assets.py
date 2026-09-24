"""Validate paths and optional SHA-256 values from the root asset manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path(os.environ.get("TCR_WORKSPACE_ROOT", "/mnt/workspace/Wilson/parameter-fusion")))
    parser.add_argument("--hash", action="store_true", help="Read each file and verify its declared SHA-256; can be slow for large checkpoints")
    args = parser.parse_args()
    data = json.loads((ROOT / "assets/manifest.json").read_text())
    rows = []
    seen = set()
    for entry in data["assets"]:
        ident = entry["id"]
        if ident in seen:
            raise ValueError(f"Duplicate asset id: {ident}")
        seen.add(ident)
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Asset path must stay below workspace root: {relative}")
        source = args.workspace_root / relative
        exists = source.is_file() if entry.get("kind", "file") == "file" else source.is_dir()
        row = {"id": ident, "path": str(relative), "exists": exists, "published": bool(entry.get("release_url"))}
        if exists and source.is_file():
            row["bytes"] = source.stat().st_size
            if "bytes" in entry and row["bytes"] != entry["bytes"]:
                row["error"] = "byte length differs"
            if args.hash and entry.get("sha256"):
                row["sha256_ok"] = file_hash(source) == entry["sha256"]
        rows.append(row)
    blocked = [row["id"] for row in rows if not row["exists"] or row.get("error") or row.get("sha256_ok") is False]
    if not rows:
        blocked.append("manifest_has_no_assets")
    print(json.dumps({"status": "PASS" if not blocked else "BLOCKED", "checked": len(rows), "blocked": blocked, "assets": rows}, ensure_ascii=False))
    return 0 if not blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
