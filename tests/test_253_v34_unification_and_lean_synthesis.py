#!/usr/bin/env python3
"""Independent v34.0 Conceptual Unification, Orthogonal Map & Lean 4 Synthesis Test Suite."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.repo_doctor import ssot_markdown_files


class Test253V34UnificationAndLeanSynthesis(unittest.TestCase):
    """Rule 31-34 & v34.0 conceptual unification + Lean 4 de Bruijn synthesis verification."""

    def test_253a_whitepaper_v34_triad_orthogonal_and_lean4_anchors(self) -> None:
        wp = (REPO_ROOT / "docs" / "whitepapers" / "world_cognition_unified_model.md").read_text(encoding="utf-8")
        required_tokens = (
            "### 3.1G",
            "正交坐标清晰分离",
            "L0–L3",
            "空间架构栈",
            "M0 / M1",
            "演化代际",
            "极简三联统一压缩视图",
            "内核本体",
            "物理边界",
            "离散有限 + 离线立法 + 诚实停机",
            "运行与演化护栏",
            "日常写 `.gift` 完全无需背诵",
            "Lean 4 第一性原理悟性融合",
            "de Bruijn",
            "kernel_checker",
            "PTXLean4",
            "非空真性",
            "证书可复验",
            "依赖类型",
            "宇宙多态",
            "Mathlib",
            "https://arxiv.org/html/2403.14064v3",
            "https://arxiv.org/html/2611.04447v1",
            "https://arxiv.org/html/2607.28459v1",
            "v33.0→v34.0",
        )
        for tok in required_tokens:
            self.assertIn(tok, wp, f"Missing v34.0 whitepaper anchor: {tok}")

    def test_253b_manifest_v34_84_active_five_layer_and_zero_evaporation(self) -> None:
        from scripts.repo_doctor import audit_theory_preservation_manifest

        mpath = REPO_ROOT / "docs" / "architecture" / "THEORY_PRESERVATION_MANIFEST.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        self.assertGreaterEqual(int(str(manifest["manifest_version"]).lstrip("v").split(".")[0]), 34)

        entries_by_id = {e["id"]: e for e in manifest["entries"]}
        expected_v34 = {
            "ARCH-CONCEPTUAL-UNIFICATION-ORTHOGONAL-MAP": "LAYER_A_META_WORLD",
            "EXEC-LEAN4-DEBRUIJN-CHECKER-SYNTHESIS": "LAYER_D_L2_L3_PHYSICAL_EXECUTION",
        }
        for eid, expected_layer in expected_v34.items():
            self.assertIn(eid, entries_by_id, f"Missing v34.0 manifest entry: {eid}")
            self.assertEqual(entries_by_id[eid]["status"], "ACTIVE")
            self.assertEqual(entries_by_id[eid]["arch_layer"], expected_layer)

        res = audit_theory_preservation_manifest()
        self.assertTrue(res["ok"], res)
        self.assertGreaterEqual(int(str(res["manifest_version"]).lstrip("v").split(".")[0]), 34)
        self.assertGreaterEqual(res["active_theories_verified"], 84)
        self.assertEqual(res["evaporated_urls"], [])
        self.assertEqual(res["evaporated_arxiv"], [])
        v34_floor = {
            "LAYER_A_META_WORLD": 28,
            "LAYER_B_L0_INTENT_SURFACE": 14,
            "LAYER_C_L1_DETERMINISTIC_KERNEL": 20,
            "LAYER_D_L2_L3_PHYSICAL_EXECUTION": 7,
            "LAYER_E_EXTERNAL_USERS_AI_BRAIN": 15,
        }
        for layer_name, min_cnt in v34_floor.items():
            self.assertGreaterEqual(res["layer_distribution"].get(layer_name, 0), min_cnt)
        self.assertEqual(sum(res["layer_distribution"].values()), res["active_theories_verified"])

    def test_253c_constitution_rule34_v34_sync_and_9md_lock(self) -> None:
        const_txt = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
        for tok in (
            "v34.0",
            "极简三联",
            "正交坐标",
            "de Bruijn",
            "kernel_checker",
            "PTXLean4",
            "非空真性",
            "依赖类型",
            "宇宙多态",
            "Mathlib",
            "zero_ai_inside_compiler: true",
            "EXTERNAL_USER_ONLY",
            "v34_unification_and_lean_synthesis",
        ):
            self.assertIn(tok, const_txt, f"Missing v34.0 SYSTEM_CONSTITUTION.md token: {tok}")

        md_files = ssot_markdown_files()
        self.assertEqual(len(md_files), 9, f"Expected exactly 9 .md files, found: {md_files}")

    def test_253d_repo_doctor_v34_gate_zero_dead_code_and_cli_overall_healthy(self) -> None:
        from scripts.repo_doctor import (
            audit_ssot_governance,
            audit_v33_convergence_closure,
            audit_v34_unification_and_lean_synthesis,
            audit_zero_dead_code,
        )

        v34 = audit_v34_unification_and_lean_synthesis()
        self.assertTrue(v34["ok"], v34)
        self.assertTrue(v34["orthogonal_map_verified"])
        self.assertTrue(v34["triad_unification_verified"])
        self.assertTrue(v34["lean4_debruijn_synthesis_verified"])
        self.assertTrue(v34["constitution_v34_synced"])
        self.assertTrue(v34["manifest_v34_upgraded"])
        self.assertTrue(v34["md_file_count_locked_at_9"])

        v33 = audit_v33_convergence_closure()
        self.assertTrue(v33["ok"], v33)
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
        self.assertTrue(report["v34_unification_and_lean_synthesis"]["ok"])
        self.assertTrue(report["v33_convergence_closure"]["ok"])
        self.assertTrue(report["ssot_governance"]["md_file_count_locked_at_9"])
        self.assertTrue(report["ast_zero_dead_code"]["ok"])


if __name__ == "__main__":
    unittest.main()
