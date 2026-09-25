"""Portable read-only checks of the new frozen successor evidence."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
W = "/mnt/workspace/Wilson/parameter-fusion/"
R = ROOT / "evidence/vla-merge-runtime/experiments"
REG = R / "robotwin-regmeanpp-m3-codex-20260925"
FEAT = R / "robotwin-static-baselines-readiness-20260925"
SRC = ROOT / "native_sources"
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


class SuccessorBindings(unittest.TestCase):
    def test_accepted_model_is_not_a_score(self):
        accepted_path = REG / "materialize-supervisor-v1/MODEL-ACCEPTED.json"
        accepted = read(accepted_path)
        plan = read(REG / "formal-540-v1/plan.json")
        self.assertEqual(sha(accepted_path), plan["model_accepted"]["sha256"])
        self.assertEqual(accepted["model_sha256"], plan["model_sha256"])
        self.assertEqual(accepted["status"], "PASS")
        self.assertEqual(accepted["diagnostics"]["modules"], 418)
        self.assertEqual(accepted["saved_and_loaded_tensors_bitwise"], 813)
        self.assertEqual(accepted["formal_episodes"], 0)
        self.assertFalse(accepted["formal_performance_claim"])
        self.assertTrue(all(x["finite"] for x in accepted["native_reload_reports"]))
        manifest = REG / "cpu-plan-v1/checkpoint/block_regmeanpp_manifest.json"
        self.assertEqual(sha(manifest), accepted["model_manifest_sha256"])

    def test_formal_panel_permit_launch_and_waiting_state(self):
        base = REG / "formal-540-v1"
        plan = read(base / "plan.json")
        permit = read(base / "FORMAL-EXECUTION-PERMIT.json")
        launch = read(base / "launch-receipt.json")
        state = read(base / "state.json")
        self.assertEqual((len(plan["jobs"]), plan["episodes"]), (9, 540))
        self.assertFalse(plan["automatic_retry"])
        self.assertEqual(plan["maximum_attempts_per_job"], 1)
        self.assertEqual(launch["plan_sha256"], sha(base / "plan.json"))
        self.assertEqual(permit["plan_sha256"], sha(base / "plan.json"))
        self.assertEqual(launch["permit_sha256"], sha(base / "FORMAL-EXECUTION-PERMIT.json"))
        self.assertEqual(launch["model_sha256"], plan["model_sha256"])
        self.assertEqual(set(state["pending"]), {j["id"] for j in plan["jobs"]})
        self.assertEqual((state["active"], state["accepted"], state["failed"]), ([], [], {}))
        self.assertEqual(sum(j["episodes"] for j in plan["jobs"]), 540)
        seen = set()
        for job in plan["jobs"]:
            keys = [(job["group"], job["repeat"], t["task_index"], seed)
                    for t in job["tasks"] for seed in t["seeds"]]
            self.assertEqual(len(keys), 60)
            self.assertEqual(len(set(keys)), 60)
            self.assertFalse(seen.intersection(keys))
            seen.update(keys)
            self.assertEqual(len(job["reference_reset_sha256"]), 60)
        self.assertEqual(len(seen), 540)

    def test_formal_sources_and_cpu_receipt_are_exact(self):
        records = {x["source_path"]: ROOT / x["path"]
                   for x in read(ROOT / "PROVENANCE.json")["files"]}
        plan = read(REG / "formal-540-v1/plan.json")
        for source, digest in plan["sources_sha256"].items():
            self.assertEqual(sha(records[source]), digest, source)
        source = SRC / "experiments/robotwin-regmeanpp-m3-codex-20260925/formal_540_v1"
        cpu = read(source / "CPU-TEST-RECEIPT.json")
        self.assertEqual((cpu["status"], cpu["tests"], cpu["errors"], cpu["failures"]), ("PASS", 15, 0, 0))
        self.assertEqual(cpu["plan_sha256"], sha(REG / "formal-540-v1/plan.json"))
        self.assertFalse(cpu["gpu_used"])
        self.assertEqual(cpu["native_jobs_started"], 0)

    def test_featcal_backend_gap_and_fresh_contract(self):
        contract_path = FEAT / "featcal-m3-tf32-v2/backend-contract.json"
        backend = read(contract_path)
        plan_path = FEAT / "featcal-m3-tf32-v2/plan.json"
        old = read(ROOT / "evidence/ROBOTWIN-FEATCAL-M3-CPU-PLAN.json")
        new = read(plan_path)
        gap = read(ROOT / "evidence/coordination/2026-09-25/robotwin-featcal-m3-tf32-backend-gap.json")
        self.assertFalse(gap["old_model_formal_eligible"])
        self.assertEqual(gap["formal_episodes"], 0)
        self.assertEqual(backend["core_plan_sha256"], sha(plan_path))
        self.assertEqual(backend["required_environment"], {
            "NVIDIA_TF32_OVERRIDE": None, "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"})
        self.assertTrue(backend["required_torch_cuda_matmul_allow_tf32"])
        self.assertTrue(backend["original_api_unchanged"])
        self.assertEqual(backend["predecessor"]["complete_sha256"],
                         sha(FEAT / "featcal-m3-cpu-plan-v1/complete.json"))
        for key in ["bank", "bank_sha256", "groups", "models", "soup", "base", "implementations",
                    "observations", "noise_states", "physical_timestep", "regression_rows",
                    "stages", "linear_weights", "alpha", "rho", "lambda", "eps", "solve_dtype",
                    "workspace_cap_bytes"]:
            self.assertEqual(old[key], new[key], key)

    def test_featcal_new_waiter_cpu_proof_and_no_formal_authorization(self):
        path = FEAT / "gpu-successor-v2/build-wait-attempt-01"
        launch = read(path / "launch-receipt.json")
        state = read(path / "state.json")
        backend = FEAT / "featcal-m3-tf32-v2/backend-contract.json"
        plan = FEAT / "featcal-m3-tf32-v2/plan.json"
        source = SRC / "vla-merge-runtime/experiments/robotwin-static-baselines-readiness-20260925/backend-bound-v2-source"
        self.assertEqual(launch["plan_sha256"], sha(plan))
        self.assertEqual(launch["backend_contract_sha256"], sha(backend))
        self.assertEqual(launch["runner_sha256"], sha(source / "safe_successor_v2.py"))
        self.assertEqual(state["controller_pid"], launch["pid"])
        self.assertEqual(state["event"], "waiting_for_empty_card")
        self.assertEqual(state["target"], "build")
        self.assertFalse(launch["old_backend_unbound_model_reused"])
        self.assertFalse(read(plan)["formal_evaluation_authorized_by_plan"])
        log = (source / "cpu-tests.log").read_text()
        self.assertIn("Ran 12 tests", log)
        self.assertTrue(log.rstrip().endswith("OK"))
        self.assertEqual(read(source / "cpu-preflight.json")["backend_contract_sha256"], sha(backend))


if __name__ == "__main__":
    unittest.main()
