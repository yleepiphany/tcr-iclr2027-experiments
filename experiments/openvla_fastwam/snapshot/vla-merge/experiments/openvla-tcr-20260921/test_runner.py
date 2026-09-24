import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_local as r


class Runner(unittest.TestCase):
    def test_terminal_episode_count(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "queue-ended.json"
            r.save_with_actual_counts(path, {"episodes": 300, "finished": [
                {"outcome": "success", "artifacts": {"episodes": 0}},
                {"outcome": "success", "artifacts": {"episodes": 100}},
                {"outcome": "failed"}]})
            result = json.loads(path.read_text())
            self.assertEqual(result["episodes"], 100)
            self.assertEqual(result["accepted_jobs"], 2)

    def test_environment_isolated(self):
        with patch.dict(r.os.environ, {"PYTHONPATH": "/wrong", "PI05_LIBERO_INIT_STATE_OFFSET": "30",
                                      "LIBERO_PRO_REPO": "/wrong", "MUJOCO_EGL_DEVICE_ID": "2"}):
            env = r.environment(3)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
        self.assertNotIn("/wrong", env["PYTHONPATH"])
        for name in ("PI05_LIBERO_INIT_STATE_OFFSET", "LIBERO_PRO_REPO", "MUJOCO_EGL_DEVICE_ID"):
            self.assertNotIn(name, env)
        with self.assertRaises(ValueError): r.environment(2)

    def test_failed_dependency_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            dep = Path(td) / "dependency"
            dep.mkdir()
            (dep / "queue-ended.json").write_text(json.dumps({"status": "incomplete", "completed_jobs": 1}))
            with patch.object(r, "DEPENDENCY", dep), patch.object(r.formal.BatchStop, "install_signal_handlers"):
                with self.assertRaisesRegex(RuntimeError, "did not finish healthily"):
                    r.wait_dependency(Path(td) / "new-run")


if __name__ == "__main__":
    unittest.main()
