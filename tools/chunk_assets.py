"""Split large external weights into release-sized chunks and verify restoration.

This is an offline preparation tool. It never uploads assets or changes an
existing checkpoint. GitHub release URLs are recorded only after upload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNK_BYTES = 1_500_000_000  # safely below GitHub's 2 GiB asset limit
MAX_CHUNK_BYTES = 2_000_000_000
BLOCK = 8 * 1024 * 1024


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_by_id(ident: str) -> dict:
    data = json.loads((ROOT / "assets/manifest.json").read_text())
    matches = [entry for entry in data["assets"] if entry["id"] == ident]
    if len(matches) != 1:
        raise ValueError(f"Expected one file asset with id {ident!r}")
    entry = matches[0]
    if entry["kind"] != "file":
        raise ValueError("Chunking requires an individual file, not a directory")
    return entry


def create(args: argparse.Namespace) -> dict:
    if not 1 <= args.chunk_bytes <= MAX_CHUNK_BYTES:
        raise ValueError("Chunk size is outside the accepted release-asset range")
    entry = asset_by_id(args.asset_id)
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Manifest path escapes workspace root")
    source = args.workspace_root / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    if entry.get("bytes") is not None and source.stat().st_size != entry["bytes"]:
        raise ValueError("Source byte length differs from frozen manifest")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    full = hashlib.sha256()
    parts = []
    with source.open("rb") as stream:
        index = 0
        while True:
            first = stream.read(min(BLOCK, args.chunk_bytes))
            if not first:
                break
            index += 1
            name = f"{args.asset_id.replace(':', '-')}.part{index:04d}"
            target = out / name
            part_hash = hashlib.sha256()
            size = 0
            with target.open("xb") as writer:
                chunk = first
                while chunk:
                    writer.write(chunk)
                    part_hash.update(chunk)
                    full.update(chunk)
                    size += len(chunk)
                    if size == args.chunk_bytes:
                        break
                    chunk = stream.read(min(BLOCK, args.chunk_bytes - size))
                writer.flush()
                os.fsync(writer.fileno())
            parts.append({"name": name, "bytes": size, "sha256": part_hash.hexdigest()})
    if not parts:
        raise ValueError("Empty model file is not a valid release asset")
    if entry.get("sha256") and full.hexdigest() != entry["sha256"]:
        raise ValueError("Source SHA-256 differs from frozen manifest; do not upload chunks")
    record = {"schema_version": 1, "asset_id": args.asset_id, "workspace_path": entry["path"],
              "source_bytes": source.stat().st_size, "source_sha256": full.hexdigest(),
              "chunk_bytes": args.chunk_bytes, "parts": parts}
    (out / "chunks.json").write_text(json.dumps(record, indent=2) + "\n")
    return {"status": "PASS", "manifest": str(out / "chunks.json"), "parts": len(parts), "bytes": record["source_bytes"]}


def restore(args: argparse.Namespace) -> dict:
    manifest = args.manifest.resolve()
    record = json.loads(manifest.read_text())
    if record.get("schema_version") != 1:
        raise ValueError("Chunk manifest schema differs")
    relative = Path(record["workspace_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Chunk manifest path escapes output root")
    target = args.output_root / relative
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".reassembling")
    if temporary.exists():
        raise FileExistsError(temporary)
    full = hashlib.sha256()
    total = 0
    try:
        with temporary.open("xb") as writer:
            for part in record["parts"]:
                name = part["name"]
                if Path(name).name != name:
                    raise ValueError("Chunk name escapes its manifest directory")
                path = manifest.parent / name
                if path.stat().st_size != part["bytes"] or digest_file(path) != part["sha256"]:
                    raise ValueError(f"Chunk identity differs: {name}")
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(BLOCK), b""):
                        writer.write(block)
                        full.update(block)
                        total += len(block)
            writer.flush()
            os.fsync(writer.fileno())
        if total != record["source_bytes"] or full.hexdigest() != record["source_sha256"]:
            raise ValueError("Reassembled artifact identity differs")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"status": "PASS", "restored": str(target), "bytes": total, "sha256": full.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    pack = subs.add_parser("create")
    pack.add_argument("--asset-id", required=True)
    pack.add_argument("--workspace-root", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    unpack = subs.add_parser("restore")
    unpack.add_argument("--manifest", type=Path, required=True)
    unpack.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args) if args.command == "create" else restore(args)))


if __name__ == "__main__":
    main()
