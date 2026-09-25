"""Portable CPU evidence for the byte-preserved, already-launched shared runner.

No CUDA, GPU queries, native simulator, /proc access or remote mutations.
This is new package verification, not an invented historical remote test receipt.
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "evidence/vla-merge-runtime/experiments"
RUN = RUNTIME / "robotwin-featcal-local-20260925/formal-v1"
BUILD = RUNTIME / "robotwin-static-baselines-readiness-20260925/featcal-m3-tf32-v2"
SOURCE = ROOT / "native_sources/experiments/robotwin-tcr-local-codex-20260924/featcal_shared_formal.py"
read = lambda p: json.loads(p.read_text())


class FeatCalSharedFormalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper = ROOT / "native_sources/experiments/claude-firstpass-cause-20260920"
        old_path = sys.path[:]
        try:
            sys.path.insert(0, str(helper))
            spec = importlib.util.spec_from_file_location("packaged_featcal_shared_cpu", SOURCE)
            cls.runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.runner)
        finally:
            sys.path[:] = old_path
        cls.plan = read(RUN / "plan.json")
        cls.records = {r["source_path"]: ROOT / r["path"]
                       for r in read(ROOT / "PROVENANCE.json")["files"]}

    def test_exact_plan_sources_and_540_unchanged_native_keys(self):
        r = self.runner
        self.assertEqual(r.sha(RUN / "plan.json"), read(RUN / "PLAN-SHA256.json")["sha256"])
        parent = read(self.records[self.plan["source_plan"]["path"]])
        self.assertEqual(r.sha(self.records[self.plan["source_plan"]["path"]]),
                         self.plan["source_plan"]["sha256"])
        for binding in self.plan["sources"]:
            self.assertEqual(r.sha(self.records[binding["path"]]), binding["sha256"])
        before = {j["id"]: j for j in parent["jobs"]}
        self.assertEqual(len(self.plan["jobs"]), 9)
        total = 0
        for job in self.plan["jobs"]:
            restored = copy.deepcopy(job)
            original = before[job["id"]]
            restored["output"] = original["output"]
            for key in ("checkpoint", "purpose"):
                restored["runtime"][key] = original["runtime"][key]
            self.assertEqual(restored, original)
            self.assertEqual(job, read(RUN / "manifests" / (job["id"] + ".json")))
            total += len(r.expected_keys(job["tasks"]))
            self.assertEqual(len(job["reference_reset_sha256"]), 60)
        self.assertEqual(total, self.plan["episodes"])
        self.assertEqual(total, 540)

    def test_backend_bound_completion_chain_and_reload(self):
        r = self.runner
        complete = read(BUILD / "complete.json")
        self.assertEqual(r.sha(BUILD / "complete.json"), self.plan["build_complete"]["sha256"])
        self.assertTrue(complete["reload_bitwise_equal"])
        self.assertEqual(complete["checkpoint"]["model_sha256"], self.plan["model_sha256"])
        self.assertEqual(complete["checkpoint"]["tensor_count"], 813)
        self.assertEqual(complete["checkpoint"]["linear_weights"], 418)
        self.assertEqual(complete["formal_episodes"], 0)
        receipts = list((BUILD / "backend-stage-receipts").glob("*.json"))
        self.assertEqual(len(receipts), 5)
        for path in receipts:
            d = read(path)
            self.assertEqual(d["artifact_sha256"], r.sha(self.records[d["artifact"]]))
            self.assertEqual(d["backend_contract_sha256"], r.sha(BUILD / "backend-contract.json"))
            self.assertEqual(d["core_plan_sha256"], r.sha(BUILD / "plan.json"))
            self.assertEqual(d["backend"]["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"], "1")
            self.assertIsNone(d["backend"]["NVIDIA_TF32_OVERRIDE"])
            self.assertTrue(d["backend"]["torch_cuda_matmul_allow_tf32"])
            self.assertEqual(d["formal_episodes"], 0)

    def test_launch_state_and_inherited_leases_match_model(self):
        state = read(RUN / "state.json")
        self.assertFalse(state["failed"])
        self.assertEqual(state["accepted"], {})
        self.assertEqual(len(state["active"]), 2)
        self.assertEqual(len(state["pending"]), 7)
        self.assertEqual(set(state["active"]) | set(state["pending"]),
                         {j["id"] for j in self.plan["jobs"]})
        for jid, active in state["active"].items():
            launch = read(RUN / "launches" / (jid + ".json"))
            started = read(RUN / "jobs" / jid / "STARTED.json")
            lease = read(RUN / "jobs" / jid / "LEASE.json")
            self.assertEqual(launch["pid"], active["pid"])
            self.assertEqual(started["pid"], active["pid"])
            self.assertEqual(started["model_sha256"], self.plan["model_sha256"])
            self.assertEqual(lease["gpu"], active["gpu"])
            self.assertEqual(lease["uuid"], self.plan["gpu_uuids"][str(active["gpu"])])
            self.assertEqual(len(lease["leases"]), 3)
            self.assertTrue(lease["shared_known_pro"])
            self.assertTrue(launch["shared_known_pro"])

    def test_duplicate_seed_is_rejected_by_original_function(self):
        tasks = copy.deepcopy(self.plan["jobs"][0]["tasks"])
        tasks[0]["seeds"][1] = tasks[0]["seeds"][0]
        with self.assertRaises(AssertionError):
            self.runner.expected_keys(tasks)

    def test_owner_identity_mismatch_dead_or_wrong_command_rejected(self):
        r = self.runner
        for gpu in (5, 7):
            owner = self.plan["owners"][str(gpu)]
            command = f"python resume_matched_pro.py run --method {owner['method']} --gpu {gpu}"
            with patch.object(r, "process_identity", return_value=owner), \
                 patch.object(Path, "read_bytes", return_value=command.encode()):
                self.assertTrue(r.owner_alive(gpu, self.plan))
            changed = {**owner, "startticks": owner["startticks"] + 1}
            with patch.object(r, "process_identity", return_value=changed), \
                 patch.object(Path, "read_bytes", return_value=command.encode()):
                self.assertFalse(r.owner_alive(gpu, self.plan))
            with patch.object(r, "process_identity", return_value=owner), \
                 patch.object(Path, "read_bytes", return_value=b"unrelated project"):
                self.assertFalse(r.owner_alive(gpu, self.plan))
            with patch.object(Path, "read_bytes", side_effect=FileNotFoundError):
                self.assertFalse(r.owner_alive(gpu, self.plan))

    def test_frozen_resource_and_formal_backend_contract(self):
        p = self.plan
        self.assertEqual((p["memory_fraction"], p["minimum_free_mib"],
                          p["runtime_free_floor_mib"], p["max_active"]),
                         (.35, 49152, 12288, 2))
        self.assertTrue(p["no_retry"])
        self.assertEqual(set(p["gpu_uuids"]), {"5", "7"})
        tree = ast.parse(SOURCE.read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        removals = [n for n in ast.walk(run) if isinstance(n, ast.For)
                    and isinstance(n.iter, ast.List)
                    and [getattr(x, "value", None) for x in n.iter.elts] ==
                    ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG"]]
        self.assertEqual(len(removals), 1)
        self.assertIn("env.pop(k, None)", ast.unparse(removals[0]))


if __name__ == "__main__":
    unittest.main()
