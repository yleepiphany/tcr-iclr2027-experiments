"""Combine family ASSETS.json files without scanning large model directories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/manifest.json")
    args = parser.parse_args()
    by_path: dict[str, dict] = {}
    for path in sorted((ROOT / "experiments").glob("*/ASSETS.json")):
        payload = json.loads(path.read_text())
        rows = payload.get("assets", payload.get("external_assets")) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"Expected asset list in {path}")
        for outer in rows:
            expanded = [outer, *outer.get("files", [])]
            for index, row in enumerate(expanded):
                value = row.get("path", row.get("workspace_relative_path"))
                if not value:
                    raise ValueError(f"Missing workspace-relative asset path: {path}: {row}")
                relative = Path(value)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Path must be under workspace root: {relative}")
                key = relative.as_posix()
                declared = row.get("kind")
                if declared in ("file", "directory"):
                    kind = declared
                elif row.get("bytes") is not None or relative.suffix in (".json", ".yaml", ".yml", ".pt", ".pth", ".safetensors", ".parquet", ".npz"):
                    kind = "file"
                else:
                    kind = "directory"
                item = by_path.setdefault(key, {"id": f"asset-{len(by_path)+1:04d}", "path": key, "kind": kind,
                                                "required": bool(outer.get("required", True)), "experiments": [], "roles": []})
                item["required"] = item["required"] or bool(outer.get("required", True))
                if item["kind"] != kind:
                    raise ValueError(f"Asset kind differs for {key}")
                if path.parent.name not in item["experiments"]:
                    item["experiments"].append(path.parent.name)
                role = row.get("role", outer.get("role", "external_artifact"))
                if role not in item["roles"]:
                    item["roles"].append(role)
                for field in ("bytes", "sha256"):
                    if row.get(field) is not None:
                        if field in item and item[field] != row[field]:
                            raise ValueError(f"Conflicting {field} for {key}")
                        item[field] = row[field]
                if outer.get("public_source"):
                    item["public_source"] = outer["public_source"]
    if not by_path:
        raise ValueError("No family asset declarations have been published")
    assets = []
    for ordinal, key in enumerate(sorted(by_path), 1):
        item = by_path[key]
        item["id"] = f"asset-{ordinal:04d}"
        assets.append(item)
    result = {"schema_version": 1, "description": "External checkpoints, calibration caches, reset banks and data sets required by unfinished non-real-robot experiments.", "assets": assets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "assets": len(assets)}))


if __name__ == "__main__":
    main()
