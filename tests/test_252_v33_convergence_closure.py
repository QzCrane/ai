#!/usr/bin/env python3
"""Independent v33.0 Convergence Closure, DLF BNF, 5-Gate Adjudication & Core Closure Test Suite."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Test252V33ConvergenceClosure(unittest.TestCase):
    """Rule 31-34 & v33.0 convergence closure verification (zero mock lambdas)."""

    def test_252a_whitepaper_v33_dlf_bnf_5gate_and_core_closure_anchors(self) -> None:
        wp = (REPO_ROOT / "docs" / "whitepapers" / "world_cognition_unified_model.md").read_text(encoding="utf-8")
        required_tokens = (
            "### 3.1F",
            "DLF 形式化 BNF 与充要判定边界",
            "DLF_Program ::=",
            "TransitionDecl ::=",
            "EffectAuthClause ::=",
            "ExecutableDeterministic",
            "PureDeclarativeContract",
            "HaltedOutsideDlf",
            "5-Gate 双时态与 Chisholm 归属裁决",
            "FIVE_GATE_ADJUDICATION",
            "M1_STANDARD_CALCULUS_LIB",
            "BitemporalFactLedger",
            "ChisholmDeonticEngine",
            "类 C 5-Primitive 内核封闭定理",
            "收敛于控制结构而非开放业务领域",
            "反例压力测试集",
            "core_closure_locked",
            "v32.0→v33.0",
        )
        for tok in required_tokens:
            self.assertIn(tok, wp, f"Missing v33.0 whitepaper anchor: {tok}")

    def test_252b_manifest_v33_82_active_five_layer_and_zero_evaporation(self) -> None:
        from scripts.repo_doctor import audit_theory_preservation_manifest

        mpath = REPO_ROOT / "docs" / "architecture" / "THEORY_PRESERVATION_MANIFEST.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        self.assertGreaterEqual(int(str(manifest["manifest_version"]).lstrip("v").split(".")[0]), 33)

        entries_by_id = {e["id"]: e for e in manifest["entries"]}
        expected_v33 = {
            "EXEC-DLF-FORMAL-BNF-BOUNDARY": "LAYER_D_L2_L3_PHYSICAL_EXECUTION",
            "CALC-5GATE-BITEMPORAL-CHISHOLM-ADJUDICATION": "LAYER_C_L1_DETERMINISTIC_KERNEL",
            "COMP-C-LIKE-5PRIM-CORE-CLOSURE": "LAYER_C_L1_DETERMINISTIC_KERNEL",
        }
        for eid, expected_layer in expected_v33.items():
            self.assertIn(eid, entries_by_id, f"Missing v33.0 manifest entry: {eid}")
            self.assertEqual(entries_by_id[eid]["status"], "ACTIVE")
            self.assertEqual(entries_by_id[eid]["arch_layer"], expected_layer)

        res = audit_theory_preservation_manifest()
        self.assertTrue(res["ok"], res)
        self.assertGreaterEqual(int(str(res["manifest_version"]).lstrip("v").split(".")[0]), 33)
        self.assertGreaterEqual(res["active_theories_verified"], 82)
        self.assertEqual(res["evaporated_urls"], [])
        self.assertEqual(res["evaporated_arxiv"], [])
        v33_floor = {
            "LAYER_A_META_WORLD": 27,
            "LAYER_B_L0_INTENT_SURFACE": 14,
            "LAYER_C_L1_DETERMINISTIC_KERNEL": 20,
            "LAYER_D_L2_L3_PHYSICAL_EXECUTION": 6,
            "LAYER_E_EXTERNAL_USERS_AI_BRAIN": 15,
        }
        for layer_name, min_cnt in v33_floor.items():
            self.assertGreaterEqual(res["layer_distribution"].get(layer_name, 0), min_cnt)
        self.assertEqual(sum(res["layer_distribution"].values()), res["active_theories_verified"])

    def test_252c_five_primitive_closure_adversarial_business_stress_projection(self) -> None:
        """Verify heterogeneous business domains project onto the 5-primitive spine + M1 calculus
        as counterexample stress tests without requiring a 6th core primitive."""
        from src.brain.world_kernel_engine import (
            REFUTE,
            SUPPORT,
            BitemporalFactLedger,
            ChisholmDeonticEngine,
            ChisholmNormSpec,
            DependencyGraph,
            FiniteDomainCSPSolver,
            FourDTimeCut,
            NamedConstraint,
        )

        # Stress Case 1 (Financial Settlement): retroactive compliance drift + transitive invalidation
        def margin_v1(val: int) -> bool:
            return val >= 1000

        def margin_v2(val: int) -> bool:
            return val >= 1500

        ledger = BitemporalFactLedger()
        ledger.register_norm("margin_floor", 1, effective_from=1, effective_to=20, recorded_at=1, predicate=margin_v1)
        ledger.register_norm("margin_floor", 2, effective_from=20, effective_to=None, recorded_at=20, predicate=margin_v2)
        ledger.append_fact("tx_margin", 1200, valid_from=10, recorded_at=10, generation=1)
        q = ledger.query_at(FourDTimeCut(10, 25, 20, 25), "tx_margin", norm_id="margin_floor")
        self.assertTrue(q.retroactive_norm_detected)
        self.assertEqual(q.primary_verdict, SUPPORT)
        self.assertEqual(q.compliant_at_norm_time, REFUTE)

        # Stress Case 2 (Supply-Chain Escrow): Chisholm CTD + R3 hard redline + CSP MUS
        def no_sanction_bypass(assign: dict) -> bool:
            return not assign["bypass_sanction"]

        def prefer_fast_route(assign: dict) -> bool:
            return assign["route_days"] <= 2

        spec = ChisholmNormSpec(
            norm_id="customs_sla",
            primary_obligation="clear_customs_48h",
            compliant_followup="release_escrow",
            contrary_to_duty_compensation="credit_delay_rebate",
            hard_redline_guards=(NamedConstraint("no_sanction_bypass", ("bypass_sanction",), no_sanction_bypass, hard=True),),
            soft_preferences=(NamedConstraint("prefer_fast_route", ("route_days",), prefer_fast_route, hard=False, priority=999_999),),
        )
        eng = ChisholmDeonticEngine()
        ctd_res = eng.evaluate(spec, {"bypass_sanction": False, "route_days": 5}, primary_observed=REFUTE, compensation_observed=SUPPORT)
        self.assertEqual(ctd_res.state, "CTD_COMPENSATION_ACTIVE")
        self.assertTrue(ctd_res.explosion_prevented)
        r3_res = eng.evaluate(spec, {"bypass_sanction": True, "route_days": 1}, primary_observed=SUPPORT, followup_observed=SUPPORT)
        self.assertEqual(r3_res.state, "HARD_REDLINE_HALT")

        # Stress Case 3 (Inventory Allocation): real CSP MUS + positive/negative dependency invalidation
        def slot_eq_1(a: dict) -> bool:
            return a["w1"] == 1

        def slot_eq_2(a: dict) -> bool:
            return a["w2"] == 1

        def slots_disjoint(a: dict) -> bool:
            return a["w1"] != a["w2"]

        solver = FiniteDomainCSPSolver()
        mus = solver.extract_mus_and_safe_subset(
            {"w1": [1, 2], "w2": [1, 2]},
            [
                NamedConstraint("w1_is_1", ("w1",), slot_eq_1),
                NamedConstraint("w2_is_1", ("w2",), slot_eq_2),
                NamedConstraint("disjoint", ("w1", "w2"), slots_disjoint),
            ],
        )
        self.assertEqual(mus["status"], "UNSAT")
        self.assertEqual(set(mus["unsat_core"]), {"w1_is_1", "w2_is_1", "disjoint"})

        dg = DependencyGraph()
        dg.facts["stock"] = 10
        dg.record_derivation("can_ship", True, reads=["stock"], neg_reads=["warehouse_lock"])
        inv = dg.mutate_fact("warehouse_lock", True)
        self.assertEqual(inv["invalidated"], ["can_ship"])

    def test_252d_repo_doctor_v33_gates_zero_dead_code_9md_and_zero_ai_firewall(self) -> None:
        from scripts.repo_doctor import (
            audit_ssot_governance,
            audit_v33_convergence_closure,
            audit_zero_dead_code,
        )
        from src.brain.five_axis_microworld import validate_external_user_protocol

        v33 = audit_v33_convergence_closure()
        self.assertTrue(v33["ok"], v33)
        self.assertTrue(v33["core_closure_locked"])
        self.assertTrue(v33["dlf_formal_spec_present"])
        self.assertTrue(v33["five_gate_adjudication_present"])
        self.assertTrue(v33["manifest_v33_upgraded"])

        self.assertTrue(audit_zero_dead_code()["ok"])
        ssot = audit_ssot_governance()
        self.assertTrue(ssot["ok"], ssot)
        self.assertTrue(ssot["md_file_count_locked_at_9"])

        md_files = [p for p in REPO_ROOT.rglob("*.md") if ".git" not in p.parts]
        self.assertEqual(len(md_files), 9, f"Expected exactly 9 .md files, found: {md_files}")

        ext_ok = validate_external_user_protocol(
            {"zero_ai_inside_compiler": True, "ai_role": "EXTERNAL_USER_ONLY"},
            {"message": "业务硬红线校验通过"},
        )
        self.assertTrue(ext_ok["ok"])
        ext_bad = validate_external_user_protocol(
            {"zero_ai_inside_compiler": False, "ai_role": "INTERNAL_SOLVER"},
            {"message": "llm_oracle_fallback"},
        )
        self.assertFalse(ext_bad["ok"])

        # End-to-end CLI execution of scripts/repo_doctor.py
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "repo_doctor.py")],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        report = json.loads(proc.stdout)
        self.assertTrue(report["overall_healthy"], report)
        self.assertTrue(report["v33_convergence_closure"]["ok"])
        self.assertTrue(report["v33_convergence_closure"]["core_closure_locked"])
        self.assertTrue(report["v33_convergence_closure"]["dlf_formal_spec_present"])
        self.assertTrue(report["v33_convergence_closure"]["five_gate_adjudication_present"])
        self.assertTrue(report["ssot_governance"]["md_file_count_locked_at_9"])
        self.assertTrue(report["ast_zero_dead_code"]["ok"])


if __name__ == "__main__":
    unittest.main()
