#!/usr/bin/env python3
"""Independent v35.0 Wheel-First Anti-NIH, Lean 4 Tiered Boundary & 10-Benchmark PL Landscape Test Suite."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Test254V35WheelFirstLandscape(unittest.TestCase):
    """Rule 31-34 & v35.0 Wheel-First loop, Lean 4 tiered correction & 10 PL/formal benchmark audit."""

    def test_254a_whitepaper_v35_lean4_tiered_correction_and_ten_benchmarks(self) -> None:
        wp = (REPO_ROOT / "docs" / "whitepapers" / "world_cognition_unified_model.md").read_text(encoding="utf-8")

        # 1. Lean 4 tiered boundary correction in §3.1G
        lean_tiered_tokens = (
            "### 3.1G",
            "日常业务编译热路径",
            "离线元证明",
            "交互式 Tactic",
            "Refinement Types",
            "Liquid",
            "Verus",
            "Dafny",
            "https://arxiv.org/html/2403.14064v3",
            "https://arxiv.org/html/2611.04447v1",
            "https://arxiv.org/html/2607.28459v1",
        )
        for tok in lean_tiered_tokens:
            self.assertIn(tok, wp, f"Missing Lean 4 tiered boundary token in whitepaper: {tok}")

        # Verify overly dogmatic "坚决拒绝（3 项）" is removed from §3.1G
        sec_31g = wp.split("### 3.1G", 1)[1].split("### 3.1H", 1)[0]
        self.assertNotIn("坚决拒绝（3 项）", sec_31g, "Overly dogmatic Lean 4 blanket rejection must be corrected in §3.1G")

        # 2. §3.1H Wheel-First + Rust pure-subset bridge + 10 PL/formal benchmarks + §4.1 12th transition
        v35_tokens = (
            "### 3.1H",
            "轮子优先（Wheel-First）五步工作循环与反 NIH 铁律",
            "绝不重造通用编程语言",
            "Rust 纯子集桥接（`syn`）与薄意图层边界",
            "PL/形式化全谱系十大标杆横向审计",
            "Verus",
            "Aeneas",
            "Dafny",
            "Quint",
            "TLA+",
            "Alloy",
            "Koka",
            "Effekt",
            "Flix",
            "Salsa",
            "egglog",
            "Z3",
            "OR-Tools",
            "直接复用轮子",
            "吸收思想",
            "归属离线层",
            "排除边界",
            "LSP 3.17",
            "SMT-LIB2",
            "v34.0→v35.0",
            "第十二次跃迁",
        )
        for tok in v35_tokens:
            self.assertIn(tok, wp, f"Missing v35.0 whitepaper token: {tok}")

    def test_254b_constitution_v35_wheel_first_anti_nih_and_9md_lock(self) -> None:
        const_txt = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
        for tok in (
            "v35.0",
            "Wheel-First",
            "五步工作循环",
            "反 NIH",
            "绝不重造通用编程语言",
            "syn",
            "SMT-LIB2",
            "Z3",
            "OR-Tools",
            "LSP 3.17",
            "Salsa",
            "egglog",
            "日常业务编译热路径",
            "离线元证明",
            "交互式 Tactic",
            "Refinement Types",
            "zero_ai_inside_compiler: true",
            "EXTERNAL_USER_ONLY",
            "v35_wheel_first_and_pl_landscape",
        ):
            self.assertIn(tok, const_txt, f"Missing v35.0 SYSTEM_CONSTITUTION.md token: {tok}")

        rule34_sec = const_txt.split("## Rule 34:", 1)[1]
        self.assertNotIn("坚决拒绝依赖类型", rule34_sec, "Rule 34 must use tiered Lean 4 boundary instead of blanket rejection")

        md_files = [p for p in REPO_ROOT.rglob("*.md") if ".git" not in p.parts]
        self.assertEqual(len(md_files), 9, f"Expected exactly 9 .md files, found: {md_files}")

    def test_254c_manifest_v35_87_active_five_layer_and_zero_evaporation(self) -> None:
        from scripts.repo_doctor import audit_theory_preservation_manifest

        mpath = REPO_ROOT / "docs" / "architecture" / "THEORY_PRESERVATION_MANIFEST.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        self.assertGreaterEqual(int(str(manifest["manifest_version"]).lstrip("v").split(".")[0]), 35)

        entries_by_id = {e["id"]: e for e in manifest["entries"]}
        expected_v35 = {
            "WORKFLOW-WHEEL-FIRST-ANTI-NIH-LAW": "LAYER_A_META_WORLD",
            "SURF-RUST-PURE-SUBSET-BRIDGE-SYN": "LAYER_B_L0_INTENT_SURFACE",
            "COMP-PL-FORMAL-TEN-BENCHMARK-LANDSCAPE": "LAYER_C_L1_DETERMINISTIC_KERNEL",
        }
        for eid, expected_layer in expected_v35.items():
            self.assertIn(eid, entries_by_id, f"Missing v35.0 manifest entry: {eid}")
            self.assertEqual(entries_by_id[eid]["status"], "ACTIVE")
            self.assertEqual(entries_by_id[eid]["arch_layer"], expected_layer)

        res = audit_theory_preservation_manifest()
        self.assertTrue(res["ok"], res)
        self.assertGreaterEqual(int(str(res["manifest_version"]).lstrip("v").split(".")[0]), 35)
        self.assertGreaterEqual(res["active_theories_verified"], 87)
        self.assertEqual(res["evaporated_urls"], [])
        self.assertEqual(res["evaporated_arxiv"], [])

        v35_floor = {
            "LAYER_A_META_WORLD": 29,
            "LAYER_B_L0_INTENT_SURFACE": 15,
            "LAYER_C_L1_DETERMINISTIC_KERNEL": 21,
            "LAYER_D_L2_L3_PHYSICAL_EXECUTION": 7,
            "LAYER_E_EXTERNAL_USERS_AI_BRAIN": 15,
        }
        for layer_name, min_cnt in v35_floor.items():
            self.assertGreaterEqual(res["layer_distribution"].get(layer_name, 0), min_cnt)
        self.assertEqual(sum(res["layer_distribution"].values()), res["active_theories_verified"])

    def test_254d_repo_doctor_v35_gate_zero_dead_code_and_cli_overall_healthy(self) -> None:
        from scripts.repo_doctor import (
            audit_ssot_governance,
            audit_v33_convergence_closure,
            audit_v34_unification_and_lean_synthesis,
            audit_v35_wheel_first_and_pl_landscape,
            audit_zero_dead_code,
        )

        v35 = audit_v35_wheel_first_and_pl_landscape()
        self.assertTrue(v35["ok"], v35)
        self.assertTrue(v35["lean4_tiered_correction_verified"])
        self.assertTrue(v35["wheel_first_constitution_synced"])
        self.assertTrue(v35["pl_landscape_ten_benchmarks_verified"])
        self.assertTrue(v35["manifest_v35_upgraded"])
        self.assertTrue(v35["md_file_count_locked_at_9"])

        self.assertTrue(audit_v34_unification_and_lean_synthesis()["ok"])
        self.assertTrue(audit_v33_convergence_closure()["ok"])
        self.assertTrue(audit_zero_dead_code()["ok"])
        ssot = audit_ssot_governance()
        self.assertTrue(ssot["ok"], ssot)
        self.assertTrue(ssot["md_file_count_locked_at_9"])

        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "repo_doctor.py")],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        report = json.loads(proc.stdout)
        self.assertTrue(report["overall_healthy"], report)
        self.assertTrue(report["v35_wheel_first_and_pl_landscape"]["ok"])
        self.assertTrue(report["v34_unification_and_lean_synthesis"]["ok"])
        self.assertTrue(report["v33_convergence_closure"]["ok"])
        self.assertTrue(report["ssot_governance"]["md_file_count_locked_at_9"])
        self.assertTrue(report["ast_zero_dead_code"]["ok"])


if __name__ == "__main__":
    unittest.main()
