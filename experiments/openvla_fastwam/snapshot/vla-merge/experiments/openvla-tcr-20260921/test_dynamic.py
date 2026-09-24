import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_dynamic as r


class DynamicCards(unittest.TestCase):
    def test_only2_until_cap24_completes(self):
        with tempfile.TemporaryDirectory() as td, patch.object(r.base, "DEPENDENCY", Path(td)):
            self.assertEqual(list(r.AvailableCards()), [2])
            (Path(td) / "queue-ended.json").write_text(json.dumps({"status": "complete", "completed_jobs": 4}))
            self.assertEqual(list(r.AvailableCards()), [2, 3])

    def test_failed_cap24_blocks_dispatch(self):
        with tempfile.TemporaryDirectory() as td, patch.object(r.base, "DEPENDENCY", Path(td)):
            (Path(td) / "batch-stop-report.json").write_text("{}")
            with self.assertRaises(RuntimeError): list(r.AvailableCards())

    def test_gpu_environment(self):
        for gpu in (2, 3):
            self.assertEqual(r.environment(gpu)["CUDA_VISIBLE_DEVICES"], str(gpu))
        for gpu in (1, 4, 5, 6, 7):
            with self.assertRaises(ValueError): r.environment(gpu)


if __name__ == "__main__":
    unittest.main()
