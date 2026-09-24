"""Round-trip and corruption checks for release asset chunks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import chunk_assets


class ChunkTests(unittest.TestCase):
    def test_round_trip_and_reject_corrupt_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source" / "weights" / "fixture.safetensors"
            source.parent.mkdir(parents=True)
            data = bytes(range(256)) * 1027
            source.write_bytes(data)
            entry = {"id": "test-weight", "path": "weights/fixture.safetensors", "kind": "file",
                     "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            args = type("Args", (), {"asset_id": "test-weight", "workspace_root": root / "source",
                                     "output": root / "chunks", "chunk_bytes": 50000})()
            with patch.object(chunk_assets, "asset_by_id", return_value=entry):
                result = chunk_assets.create(args)
            self.assertEqual(result["status"], "PASS")
            record = json.loads((root / "chunks/chunks.json").read_text())
            self.assertGreater(len(record["parts"]), 1)
            restore = type("Restore", (), {"manifest": root / "chunks/chunks.json",
                                          "output_root": root / "restored"})()
            self.assertEqual(chunk_assets.restore(restore)["status"], "PASS")
            self.assertEqual((root / "restored/weights/fixture.safetensors").read_bytes(), data)
            (root / "chunks" / record["parts"][0]["name"]).write_bytes(b"bad")
            restore.output_root = root / "corrupt"
            with self.assertRaises(ValueError):
                chunk_assets.restore(restore)
            self.assertFalse((root / "corrupt/weights/fixture.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
