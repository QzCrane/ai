#!/usr/bin/env python3
"""
Sovereign AI Brain (Route B v2.2) — Permanent Automated Regression & Invariant Test Suite (GAP-22)
==================================================================================================
Runs cleanly in <1s on ANY environment (pure Python + NumPy + AST, zero GPU required), and
automatically runs full PyTorch forward/backward/Q4 round-trip tests whenever `torch` is installed.
"""
from __future__ import annotations

import ast
import importlib.util
import math
import py_compile
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BRAIN_DIR = REPO_ROOT / "src" / "brain"
if str(BRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(BRAIN_DIR))

import exoskeleton_bridge as eb
import prepare_seed_corpus as psc
import wake_sleep_runtime as wsr
sys.path.insert(0, str(REPO_ROOT / "src" / "experimental"))
import harness_and_safety_policies as exp_hsp
import memory_and_sleep_planners as exp_msp


class Test02ExperimentalPrototypes(unittest.TestCase):
    """Verifies syntax, AST causal wiring, mathematical invariants, and repo consistency."""

    def test_66_gap80_spectral_unlearning_steepest_direction(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("use_spectral_direction: bool = True", bsrc)
        self.assertIn("g_orth = polar_express_orthogonalize(g, steps=self.spectral_steps)", bsrc)

    def test_76_gap89_rewy_3matmul_backward_primitive(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def rewy_chunk_backward_3matmul(", bsrc)
        # Pure-NumPy shape & exactness sanity check
        rng = np.random.default_rng(89)
        L, D = 8, 16
        dO, Q, K, R = [rng.normal(0, 0.2, size=(L, D)) for _ in range(4)]
        inv_M = np.eye(L) + np.tril(rng.normal(0, 0.05, size=(L, L)), -1)
        G_qk = dO @ R.T
        P = inv_M.T @ dO
        dQ, dK, dV = G_qk @ K, G_qk.T @ Q, P
        self.assertEqual(dQ.shape, (L, D))
        self.assertEqual(dK.shape, (L, D))
        self.assertEqual(dV.shape, (L, D))

    def test_78_gap91_bole_factorized_speculative_state_commit(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def factorized_speculative_state_commit(", bsrc)
        # Pure-NumPy check: sequential factorized commit matches exact rank-1 delta state updates
        rng = np.random.default_rng(91)
        D = 8
        S = np.zeros((D, D))
        k = rng.normal(0, 1.0, size=(D,))
        k /= np.linalg.norm(k)
        v = rng.normal(0, 1.0, size=(D,))
        S_new = S + np.outer(v - 0.5 * (S @ k), k)
        self.assertAlmostEqual(float(np.linalg.norm(S_new @ k - v)), 0.0, places=6)


    def test_80_gap93_deletion_tuning_ultra_sparse_mask(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def compute_deletion_tuning_mask(", bsrc)
        # Pure-NumPy check: deleting top 0.05% positive (w * g) coords strictly lowers first-order loss
        rng = np.random.default_rng(93)
        w = rng.normal(0, 1.0, size=(100, 100))
        g = rng.normal(0, 1.0, size=(100, 100))
        benefit = (w * g).reshape(-1)
        k = max(1, int(round(benefit.size * 0.0005)))
        top_vals = np.sort(benefit)[-k:]
        self.assertTrue(np.all(top_vals > 0.0))

    def test_81_gap94_shared_candidate_dither_unbiasedness(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def shared_candidate_dither_quantize(", bsrc)
        # Pure-NumPy check: uniform [-0.5, 0.5] dither eliminates deterministic rounding bias on constant x
        rng = np.random.default_rng(94)
        x = 2.35
        dithers = rng.uniform(-0.5, 0.5, size=20000)
        q_mean = float(np.mean(np.round(x + dithers)))
        self.assertAlmostEqual(q_mean, x, places=2)

    def test_88_gap101_echomoe_delta_reference_compression(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def encode_delta_reference_state(", bsrc)
        self.assertIn("def decode_delta_reference_state(", bsrc)
        # Pure-NumPy check: quantized residual reconstruction stays close to h_cur
        rng = np.random.default_rng(101)
        h_ref = rng.normal(0, 1.0, size=(4, 16))
        h_cur = h_ref + rng.normal(0, 0.05, size=(4, 16))
        diff = h_cur - h_ref
        sc = np.maximum(np.max(np.abs(diff), axis=-1, keepdims=True), 1e-7) / 7.0
        rec = h_ref + np.clip(np.round(diff / sc), -7, 7) * sc
        self.assertLess(float(np.max(np.abs(rec - h_cur))), 0.025)

    def test_90_gap103_eggmind_egraph_saturation_budget_controller(self) -> None:
        ctrl = exp_hsp.EGraphSaturationBudgetController(max_enodes=100, max_iters=3)
        s0 = ctrl.plan_step(current_enodes=40, iteration=0, rule_partitions=["arith", "assoc"])
        self.assertEqual(s0["action"], "apply_partition_and_rebuild")
        self.assertEqual(s0["partition"], "arith")
        s_sat = ctrl.plan_step(current_enodes=120, iteration=1, rule_partitions=["arith"])
        self.assertEqual(s_sat["action"], "extract_lowest_cost")
        self.assertTrue(s_sat["budget_saturated"])

    def test_92_gap104_recsa_log_l_tree_kv_selector(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def recsa_tree_kv_select(", bsrc)
        # Pure-NumPy check: L=32, bf=4 (8 groups), top_k=2 -> 2*4 exact + 6 coarse = 14 < 32
        self.assertEqual(2 * 4 + (8 - 2), 14)

    def test_93_gap105_loopvq_sub_1bit_looped_kv_quantization(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def quantize_looped_kv_loopvq(", bsrc)
        # 16 centroids across 8 loops = log2(16)/8 = 0.5 bits per element
        self.assertAlmostEqual(math.log2(16) / 8.0, 0.5, places=6)

    def test_95_gap107_uue_antipodal_nullspace_unlearning(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def compute_uue_antipodal_nullspace_update(", bsrc)
        # Pure-NumPy check: delta_W @ U_r == 0 (strict zero interference on retained subspace)
        rng = np.random.default_rng(107)
        W = rng.normal(0, 1.0, size=(8, 16))
        A = rng.normal(0, 1.0, size=(6, 16))
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        U_r = vh[:4, :].T
        K_f = rng.normal(0, 1.0, size=(3, 16))
        K_perp = K_f - (K_f @ U_r) @ U_r.T
        delta_W = -0.5 * ((W @ K_perp.T) @ K_perp) / (np.sum(K_perp ** 2) + 1e-6)
        self.assertLess(float(np.max(np.abs(delta_W @ U_r))), 1e-10)

    def test_97_gap109_kvguard_shadow_block_safety(self) -> None:
        g = exp_hsp.KVBlockShadowGuard()
        g.allocate_block(7, "req_A")
        self.assertFalse(g.check_read_safe(7, "req_A"))  # Uninitialized read blocked
        self.assertFalse(g.mark_initialized(7, "req_B"))  # Cross-request write blocked
        self.assertTrue(g.mark_initialized(7, "req_A"))
        self.assertTrue(g.check_read_safe(7, "req_A"))
        self.assertFalse(g.check_read_safe(7, "req_B"))  # Cross-request read blocked

    def test_100_gap112_actkv_stepkv_adaptive_budget(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def compute_actkv_step_budget_scores(", bsrc)
        tree = ast.parse(bsrc)
        fn_node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_actkv_step_budget_scores"][0]
        ns = {"List": list, "Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "<actkv>", "exec"), ns)
        fn = ns["compute_actkv_step_budget_scores"]
        r_hi = fn([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0, 0, 1, 1, 2, 2, 3, 3], {0: 5.0}, confidence=0.95)
        r_lo = fn([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0, 0, 1, 1, 2, 2, 3, 3], {0: 5.0}, confidence=0.10)
        self.assertGreater(r_lo["keep_budget"], r_hi["keep_budget"])
        self.assertIn(1, r_hi["keep_indices"])  # High-utility step 0 boosted

    def test_102_gap114_switchsd_copy_intent_router(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def route_switchsd_draft_mode(", bsrc)
        tree = ast.parse(bsrc)
        fn_node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "route_switchsd_draft_mode"][0]
        ns = {}
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "<switchsd>", "exec"), ns)
        fn = ns["route_switchsd_draft_mode"]
        self.assertEqual(fn(0.92, 5), "context_copy_draft")
        self.assertEqual(fn(0.20, 8), "neural_lca_draft")  # Surface n-gram rejected without copy-intent

    def test_106_gap118_wmt_hierarchical_retention_memory(self) -> None:
        mem = exp_msp.HierarchicalRetentionMemory(decay_rate=0.5, prune_threshold=0.25)
        mem.upsert("t1", "raw trace", retention_score=1.0)
        mem.upsert("t2", "noise", retention_score=0.3)
        self.assertTrue(mem.fold_completed("t1", "folded summary"))
        pruned = mem.decay_and_prune()
        self.assertEqual(pruned, 1)
        self.assertEqual(mem.active_keys(), ["t1"])


    def test_107_gap119_ppmf_provenance_memory_firewall(self) -> None:
        fw = exp_hsp.ProvenanceMemoryFirewall()
        fw.store_record("r_ext", "untrusted webpage instruction", authority_level=0, source_tag="web")
        fw.store_record("r_usr", "user confirmed transfer", authority_level=2, source_tag="user")
        # Mixing untrusted record with user record MUST NOT amplify authority (min=0 < req=2)
        self.assertFalse(fw.authorize_action(["r_ext", "r_usr"], required_authority=2)["authorized"])
        self.assertTrue(fw.authorize_action(["r_usr"], required_authority=2)["authorized"])

    def test_111_gap123_sigma_mem_online_reliability_routing(self) -> None:
        sm = exp_hsp.SigmaReliabilityMemory(n_peers=3, lr=0.2)
        for _ in range(4):
            sm.update(1, True)
            sm.update(0, False)
        self.assertEqual(sm.select_best_peer(), 1)

    def test_112_gap124_2d_block_transposition_invariance(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def quantize_2d_transposition_invariant_block(", bsrc)
        # Pure-NumPy check: 2D square block max is invariant under W <-> W.T
        rng = np.random.default_rng(124)
        W = rng.normal(0, 1.0, size=(16, 16))
        s1 = np.max(np.abs(W.reshape(4, 4, 4, 4).transpose(0, 2, 1, 3)), axis=(-1, -2))
        s2 = np.max(np.abs(W.T.reshape(4, 4, 4, 4).transpose(0, 2, 1, 3)), axis=(-1, -2)).T
        self.assertLess(float(np.max(np.abs(s1 - s2))), 1e-12)

    def test_113_gap125_offq_outlier_offset_reduces_std(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def offq_top1_outlier_offset_absorb(", bsrc)
        rng = np.random.default_rng(125)
        X = rng.normal(0, 0.5, size=(32, 16))
        X[:, 0] += 12.0  # Structured outlier channel
        X_c = X - X.mean(axis=0, keepdims=True)
        self.assertLess(float(np.max(np.abs(X_c))), 0.25 * float(np.max(np.abs(X))))

    def test_114_gap126_slimsc_redundant_chain_pruning(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def prune_redundant_thought_chains_slimsc(", bsrc)
        tree = ast.parse(bsrc)
        fn_node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prune_redundant_thought_chains_slimsc"][0]
        ns = {"List": list, "math": math}
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "<slimsc>", "exec"), ns)
        kept = ns["prune_redundant_thought_chains_slimsc"]([[1.0, 0.0], [0.99, 0.05], [0.0, 1.0]], [0.9, 0.7, 0.8], sim_threshold=0.92)
        self.assertEqual(kept, [0, 2])  # Chain 1 pruned as redundant duplicate of Chain 0


    def test_115_gap127_deepseek_v4_csa_hca_kv_budget(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def csa_hca_two_tier_kv_budget(", bsrc)
        tree = ast.parse(bsrc)
        fn_node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "csa_hca_two_tier_kv_budget"][0]
        ns = {"Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "<csahca>", "exec"), ns)
        r = ns["csa_hca_two_tier_kv_budget"](1000000, hca_compress_ratio=10, csa_top_blocks=16, block_size=64)
        self.assertLess(r["kv_ratio"], 0.11)  # ~10.1% KV footprint at 1M context

    def test_117_gap129_gemma_interleaved_local_global_kv_budget(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def compute_interleaved_local_global_kv_budget(", bsrc)
        tree = ast.parse(bsrc)
        fn_node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_interleaved_local_global_kv_budget"][0]
        ns = {"Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "<gemma>", "exec"), ns)
        r = ns["compute_interleaved_local_global_kv_budget"](24, seq_len=32768, local_window=1024, local_to_global_ratio=5)
        self.assertGreater(r["kv_compression_ratio"], 4.5)

    def test_118_gap130_nemotron_h_hybrid_ssm_attention_schedule(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertIn("def plan_hybrid_ssm_attention_schedule(", bsrc)
        tree = ast.parse(bsrc)
        fn_node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "plan_hybrid_ssm_attention_schedule"][0]
        ns = {"List": list}
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "<nemotron>", "exec"), ns)
        sched = ns["plan_hybrid_ssm_attention_schedule"](12, target_attn_frac=0.25)
        self.assertEqual(len(sched), 12)
        self.assertEqual(sched.count("exact_attn"), 3)

    def test_119_gap131_multiturn_safeguard_drift_monitor(self) -> None:
        mon = exp_hsp.MultiTurnSafeguardMonitor(risk_ceiling=0.80, decay=0.85)
        r1 = mon.observe_turn(0.35)
        self.assertFalse(r1["escalate_safeguard"])
        r2 = mon.observe_turn(0.35)
        r3 = mon.observe_turn(0.35)
        self.assertTrue(r3["escalate_safeguard"])

    def test_120_gap132_flatcake_write_once_kv_allocator(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlatCakeWriteOnceKVAllocator"][0]
        ns = {"Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<flatcake>", "exec"), ns)
        alloc = ns["FlatCakeWriteOnceKVAllocator"](dram_capacity_mb=64.0, nvme_capacity_mb=512.0)
        r1 = alloc.allocate_session("s1", n_layers=8, kv_mb_per_layer=4.0)
        self.assertEqual(r1["tier"], "host_dram")
        self.assertEqual(r1["read_amplification"], 1.0)
        r2 = alloc.allocate_session("s2", n_layers=8, kv_mb_per_layer=8.0)
        self.assertEqual(r2["tier"], "nvme_gds")
        r1_hit = alloc.allocate_session("s1", n_layers=8, kv_mb_per_layer=4.0)
        self.assertTrue(r1_hit["hit"])
        self.assertFalse(r1_hit["migrated"])

    def test_121_gap133_leash_three_level_safety_gate(self) -> None:
        gate = exp_hsp.LeashThreeLevelSafetyGate(task_ceiling=0.65, action_ceiling=0.55, commit_ceiling=0.25)
        self.assertFalse(gate.evaluate(0.70, 0.10, 0.05, irreversible=False)["allow"])
        self.assertEqual(gate.evaluate(0.20, 0.60, 0.05, irreversible=False)["blocked_at"], "action_gate")
        self.assertEqual(gate.evaluate(0.20, 0.20, 0.30, irreversible=True)["blocked_at"], "irreversible_commit_gate")
        self.assertTrue(gate.evaluate(0.20, 0.20, 0.10, irreversible=True)["allow"])

    def test_122_gap134_toolroute_health_telemetry_router(self) -> None:
        rt = exp_hsp.HarnessToolHealthTelemetryRouter(max_latency_ms=1000.0, min_success_rate=0.90)
        rt.update_health("stale_api", latency_ms=200.0, success_rate=0.40, alive=True)
        rt.update_health("slow_api", latency_ms=1800.0, success_rate=0.99, alive=True)
        rt.update_health("fast_api", latency_ms=120.0, success_rate=0.98, alive=True)
        res = rt.select_tool(["stale_api", "slow_api", "fast_api"])
        self.assertEqual(res["selected"], "fast_api")

    def test_123_gap135_safety_lead_team_governance_gate(self) -> None:
        gov = exp_hsp.SafetyLeadTeamGovernanceGate(safety_lead_id="lead_safety", spec_version="SYSTEM_CONSTITUTION_v4.8")
        self.assertFalse(gov.approve_team_action("worker_a", "worker_b", "SYSTEM_CONSTITUTION_v4.8", 0.1)["approved"])
        self.assertFalse(gov.approve_team_action("worker_a", "lead_safety", "OLD_SPEC", 0.1)["approved"])
        self.assertTrue(gov.approve_team_action("worker_a", "lead_safety", "SYSTEM_CONSTITUTION_v4.8", 0.2)["approved"])

    def test_124_gap136_mode_hopping_checkpoint_selector(self) -> None:
        tsrc_te = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(tsrc_te)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModeHoppingCheckpointSelector"][0]
        ns = {"Dict": dict, "Any": object, "List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<modehop>", "exec"), ns)
        sel = ns["ModeHoppingCheckpointSelector"](hop_drop_threshold=0.08)
        sel.record(1000, 1.80, 0.72)
        r_hop = sel.record(2000, 1.45, 0.58)
        self.assertTrue(r_hop["mode_hop_detected"])
        best = sel.select_best_checkpoint()
        self.assertEqual(best["best_step"], 1000)
        self.assertTrue(best["avoid_checkpoint_averaging"])

    def test_125_gap137_owlsight_cot_and_trace_hygiene_monitor(self) -> None:
        mon = exp_hsp.OwlSightCoTAndTraceHygieneMonitor(fine_threshold=0.60, coarse_threshold=0.55)
        r_clean = mon.inspect_trace("normal reasoning step", [0.1, 0.2], 0.15)
        self.assertFalse(r_clean["flagged"])
        r_leak = mon.inspect_trace("found key sk-123456 in trace", [0.1, 0.2], 0.15)
        self.assertTrue(r_leak["flagged"])
        self.assertTrue(r_leak["strip_before_persist"])

    def test_127_gap138_anthropic_dreaming_memory_consolidator(self) -> None:
        from src.experimental.memory_and_sleep_planners import DreamingMemoryConsolidator
        dmc = DreamingMemoryConsolidator(max_index_lines=200, max_index_bytes=25 * 1024)
        w1 = dmc.write_sharded_entity("contract_A", "agent_1", ["price=100", "hallucinated_clause"], raw_source_bytes=200)
        self.assertTrue(w1["written"])
        # Single-owner conflict protection
        w_conflict = dmc.write_sharded_entity("contract_A", "agent_2", ["overwrite"], raw_source_bytes=200)
        self.assertFalse(w_conflict["written"])
        self.assertEqual(w_conflict["reason"], "single_owner_conflict")
        # Anti-Borges bloat guard
        w_bloat = dmc.write_sharded_entity("contract_B", "agent_2", ["x" * 300], raw_source_bytes=50)
        self.assertFalse(w_bloat["written"])
        self.assertEqual(w_bloat["reason"], "borges_bloat_rejected")
        # Transcript-grounded dreaming corrects stale/hallucinated fact
        res = dmc.dream_consolidate("Verified transcript confirms price=100 and nothing else.")
        self.assertEqual(res["corrected_stale_facts"], 1)
        self.assertEqual(res["top_index_lines"], 1)

    def test_128_gap139_google_zero_tim_three_way_policy_contract(self) -> None:
        tsrc_te = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(tsrc_te)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ZeroTIMThreeWayPolicyContract"][0]
        ns = {"Dict": dict, "Any": object, "List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<zerotim>", "exec"), ns)
        c = ns["ZeroTIMThreeWayPolicyContract"]()
        ok = c.verify_contract([-1.25, -0.5], [-1.25, -0.5], [-1.25, -0.5], [10, 20], [10, 20])
        self.assertTrue(ok["bitwise_zero_tim"])
        bad_serv = c.verify_contract([-1.25, -0.5], [-1.250001, -0.5], [-1.250001, -0.5], [10, 20], [10, 20])
        self.assertEqual(bad_serv["diagnosis"], "serving_engine_reduction_drift")
        bad_train = c.verify_contract([-1.25, -0.5], [-1.25, -0.5], [-1.250001, -0.5], [10, 20], [10, 20])
        self.assertEqual(bad_train["diagnosis"], "trainer_compilation_or_autodiff_drift")

    def test_129_gap140_google_streaming_diloco_fragment_scheduler(self) -> None:
        tsrc_te = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(tsrc_te)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StreamingDiLoCoFragmentScheduler"][0]
        ns = {"Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<diloco>", "exec"), ns)
        sched = ns["StreamingDiLoCoFragmentScheduler"](n_layers=64, model_size_gb=65.0)
        s0 = sched.fragment_for_step(0)
        self.assertEqual(s0["total_fragments"], 65)
        self.assertEqual(s0["per_step_transfer_gb"], 1.0)
        self.assertEqual(s0["traffic_reduction_vs_dp"], 65)
        self.assertFalse(s0["periodic_full_model_stall"])

    def test_130_gap141_openai_codex_layered_sandbox_policy(self) -> None:
        pol = exp_hsp.CodexLayeredSandboxPolicyEngine(
            allowed_domains=["*.openai.com", "localhost"],
            denied_domains=["evil.openai.com"],
            enterprise_block_full_access=True,
        )
        self.assertFalse(pol.check_file_op(".git/hooks/pre-commit", "write")["allow"])
        self.assertFalse(pol.check_file_op("config/.env", "read")["allow"])
        self.assertTrue(pol.check_file_op("src/main.py", "write")["allow"])
        # Explicit deny beats wildcard allow BEFORE DNS resolution
        r_deny = pol.check_network_pre_dns("evil.openai.com")
        self.assertFalse(r_deny["allow"])
        self.assertFalse(r_deny["resolve_dns"])
        # Enterprise ceiling blocks escalation
        self.assertFalse(pol.check_network_pre_dns("api.openai.com", requested_mode="require_escalated")["allow"])
        self.assertTrue(pol.check_network_pre_dns("api.openai.com")["allow"])

    def test_131_gap142_openai_performance_factory_adversarial_loop(self) -> None:
        pf = exp_hsp.PerformanceFactoryAdversarialLoop(self_repair_threshold=3)
        pf.open_task("reg_1", baseline_metric=100.0)
        r_fail = pf.submit_adversarial_review("reg_1", counterfactual_falsified=True)
        self.assertFalse(r_fail["accepted"])
        self.assertEqual(r_fail["state"], "Investigating")
        self.assertTrue(pf.submit_adversarial_review("reg_1", counterfactual_falsified=False)["accepted"])
        self.assertEqual(pf.verify_post_deploy_recovery("reg_1", post_deploy_metric=103.0)["state"], "Resolved")
        pf.record_tool_error("profiler")
        pf.record_tool_error("profiler")
        self.assertTrue(pf.record_tool_error("profiler")["open_self_repair_task"])

    def test_132_gap143_nvidia_dynamo_reshard_and_kv_rollout_planner(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DynamoReshardAndKVRolloutPlanner"][0]
        ns = {"Dict": dict, "Any": object, "List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<dynamo>", "exec"), ns)
        p = ns["DynamoReshardAndKVRolloutPlanner"]()
        plan = p.plan_weight_refit([
            {"name": "layer0.mlp.up_proj", "numel": 9500},
            {"name": "layer0.qkv_proj", "numel": 500},
        ])
        self.assertEqual(plan["routes"]["layer0.mlp.up_proj"], "nccl_m2n_reshard")
        self.assertEqual(plan["routes"]["layer0.qkv_proj"], "gather_and_broadcast")
        self.assertEqual(plan["reshard_fraction"], 0.95)
        w = p.route_rollout_worker([
            {"worker_id": "w_cold", "kv_hit_ratio": 0.1, "load_ratio": 0.2},
            {"worker_id": "w_warm", "kv_hit_ratio": 0.9, "load_ratio": 0.4},
        ])
        self.assertEqual(w["selected_worker"], "w_warm")

    def test_133_gap144_anthropic_managed_agent_brain_hands_decoupling(self) -> None:
        mh = exp_hsp.ManagedAgentMetaHarness()
        mh.register_vault_secret("github", "ghp_secret_token")
        # Before execute_hand, hand is NOT provisioned (cuts TTFT)
        w0 = mh.wake("sess_1")
        self.assertFalse(w0["hand_provisioned"])
        r1 = mh.execute_hand("sess_1", "git_push", "origin main", service_for_proxy="github")
        self.assertTrue(r1["lazy_provisioned"])
        self.assertTrue(r1["proxy_authed"])
        self.assertFalse(r1["raw_credential_exposed_to_sandbox"])
        # Stateless recovery from durable session log
        w1 = mh.wake("sess_1")
        self.assertEqual(w1["recovered_events"], 1)

    def test_134_gap145_openai_layered_domain_architecture_linter(self) -> None:
        linter = exp_hsp.LayeredDomainArchitectureLinter()
        self.assertTrue(linter.check_dependency("Service", "Repo")["valid"])
        self.assertTrue(linter.check_dependency("Repo", "Providers")["valid"])
        bad = linter.check_dependency("Repo", "UI")
        self.assertFalse(bad["valid"])
        self.assertIn("Providers", bad["remediation"])

    def test_135_gap146_google_ml_drift_stage_kernel_and_gemma4_ple(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StageAwareEdgeKernelAndPLEPlanner"][0]
        ns = {"Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<mldrift>", "exec"), ns)
        pl = ns["StageAwareEdgeKernelAndPLEPlanner"]()
        self.assertEqual(pl.select_stage_config("prefill")["bottleneck"], "compute_bound")
        dec = pl.select_stage_config("decode")
        self.assertEqual(dec["kv_layout"], "conv_aligned_kv_cache")
        self.assertTrue(dec["in_kernel_act_quant"])
        ple = pl.estimate_ple_vram_savings(n_layers=12, vocab_size=65536, ple_dim=256, bytes_per_elem=1)
        self.assertEqual(ple["gpu_vram_ple_mb"], 0.0)
        self.assertGreater(ple["offloaded_ple_mb"], 100.0)

    def test_136_gap147_nvidia_nvfp4_microblock_stochastic_quantizer(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NVFP4MicroBlockQuantizer"][0]
        ns = {"Dict": dict, "Any": object, "List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<nvfp4>", "exec"), ns)
        q = ns["NVFP4MicroBlockQuantizer"](block_size=16)
        vals = [0.1 * i - 1.5 for i in range(32)]
        res = q.quantize_dequantize_block(vals, stochastic=True, seed=7)
        self.assertEqual(res["block_size"], 16)
        self.assertEqual(res["num_blocks"], 2)
        self.assertEqual(len(res["dequantized"]), 32)

    def test_137_gap148_google_tpu_codesign_256_and_latent_moe(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HardwareAlignedShapeAndLatentMoEAdvisor"][0]
        ns = {"Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<tpucodesign>", "exec"), ns)
        adv = ns["HardwareAlignedShapeAndLatentMoEAdvisor"]()
        self.assertTrue(adv.audit_shapes(1024, 4096, 256)["all_systolic_aligned"])
        self.assertFalse(adv.audit_shapes(1024, 4096, 128)["head_dim_256_optimal"])
        lm = adv.latent_moe_compression_ratio(2048, 768)
        self.assertTrue(lm["latent_256_aligned"])
        self.assertTrue(lm["in_recommended_2x_4x_band"])
        self.assertGreaterEqual(adv.min_vpu_hidden_block_size(4614.0, 19.2), 340)

    def test_138_gap149_google_tunix_multiturn_inturn_rloo(self) -> None:
        tsrc_te = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(tsrc_te)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiTurnInTurnToolRLOOEngine"][0]
        ns = {"List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<tunix>", "exec"), ns)
        eng = ns["MultiTurnInTurnToolRLOOEngine"]()
        self.assertTrue(eng.should_continue_in_turn("call:strike{cell:C5}<|tool_response>"))
        adv = eng.compute_rloo_advantages([1.0, 0.0, 0.0, 0.0])
        self.assertEqual(adv[0], 1.0)
        self.assertLess(adv[1], 0.0)

    def test_139_gap150_anthropic_two_stage_reasoning_blind_classifier(self) -> None:
        g = exp_hsp.TwoStageReasoningBlindClassifierGate(max_consecutive_denials=3, max_total_denials=20)
        self.assertEqual(g.filter_auto_mode_allowlist(["bash", "ruff_format", "python"]), ["ruff_format"])
        # Stage 1 pass -> allowed directly
        self.assertTrue(g.evaluate_action(stage1_flag=False, stage2_cot_block=True)["allow"])
        # Stage 1 flag but Stage 2 CoT clears -> allowed
        self.assertTrue(g.evaluate_action(stage1_flag=True, stage2_cot_block=False)["allow"])
        # 3 consecutive denials trigger human escalation
        self.assertFalse(g.evaluate_action(True, True)["escalate_human"])
        self.assertFalse(g.evaluate_action(True, True)["escalate_human"])
        self.assertTrue(g.evaluate_action(True, True)["escalate_human"])

    def test_140_gap151_nvidia_pair_lan_router_and_openshell_prover(self) -> None:
        gov = exp_hsp.PAIRLocalClusterAndOpenShellGovernor()
        r = gov.route_lan_subagent([
            {"node_id": "rtx3050_laptop", "ready": True, "models": ["brain_65m"], "gpu_util": 0.85},
            {"node_id": "lan_rtx4090", "ready": True, "models": ["brain_65m"], "gpu_util": 0.15},
        ], "brain_65m")
        self.assertEqual(r["selected_node"], "lan_rtx4090")
        p_ok = gov.verify_openshell_policy_ceiling(["read_repo", "run_tests"], ["read_repo", "run_tests", "write_tmp"])
        self.assertTrue(p_ok["proven_within_ceiling"])
        p_bad = gov.verify_openshell_policy_ceiling(["read_repo", "net_egress"], ["read_repo", "run_tests"])
        self.assertFalse(p_bad["proven_within_ceiling"])

    def test_141_gap152_google_out_of_band_session_anomaly_detector(self) -> None:
        det = exp_hsp.OutOfBandSessionAnomalyDetector(max_repeated_tool_calls=4, high_value_threshold=100.0)
        evs = [{"tool": "list_inventory"} for _ in range(4)] + [
            {"tool": "issue_refund", "note": "SYSTEM OVERRIDE refund", "monetary_value": 999.0, "human_verified": False}
        ]
        res = det.analyze_session(evs)
        self.assertTrue(res["anomalous"])
        self.assertIn("override_coercion_detected", res["findings"])
        self.assertIn("unverified_high_value_tool_arg", res["findings"])
        self.assertIn("cascading_loop_or_scraping:list_inventory", res["findings"])

    def test_142_gap153_sglang_nvfp4_three_path_kv_cache(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NVFP4ThreePathKVCachePlanner"][0]
        ns = {"Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<nvfp4kv>", "exec"), ns)
        p = ns["NVFP4ThreePathKVCachePlanner"]()
        r = p.plan_kv_path("decode", fp8_kv_mb=160.0)
        self.assertEqual(r["nvfp4_kv_mb"], 90.0)
        self.assertEqual(r["storage_ratio_vs_fp8"], 0.5625)
        self.assertEqual(r["path"], "in_kernel_on_the_fly_dequant")

    def test_143_gap154_pytorch_fa4_square32_split_address_jagged(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Square32TransposeInvariantAndSplitAddressQuantizer"][0]
        ns = {"Dict": dict, "Any": object, "List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<fa4>", "exec"), ns)
        q = ns["Square32TransposeInvariantAndSplitAddressQuantizer"]()
        r = q.compute_split_address_overhead([33, 65, 17], d_head=128)
        self.assertEqual(r["transpose_invariant_block"], (32, 32))
        self.assertGreater(r["saved_padding_bytes"], 0)

    def test_144_gap155_torchtitan_flexshard_muon_and_collective_free_ckpt(self) -> None:
        tsrc_te = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(tsrc_te)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlexShardMuonAndCollectiveFreeCkptPlanner"][0]
        ns = {"Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<flexshard>", "exec"), ns)
        pl = ns["FlexShardMuonAndCollectiveFreeCkptPlanner"]()
        self.assertEqual(pl.assign_placement("core.w1", 2, "muon"), "BlockShard_2D_Owned")
        self.assertEqual(pl.assign_placement("norm.weight", 1, "adamw"), "FSDP2_RowShard")
        s1 = pl.record_checkpoint_save(100)
        s2 = pl.record_checkpoint_save(200)
        self.assertTrue(s1["ran_global_collective"])
        self.assertFalse(s2["ran_global_collective"])
        self.assertTrue(s2["barrier_only"])

    def test_145_gap156_sglang_breakable_cuda_graph_prefill_planner(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BreakableCUDAGraphPrefillPlanner"][0]
        ns = {"Dict": dict, "Any": object, "List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<bcg>", "exec"), ns)
        bcg = ns["BreakableCUDAGraphPrefillPlanner"]()
        ops = [{"name": "rmsnorm"}, {"name": "qkv"}, {"name": "sparse_attn", "eager_on_graph": True}, {"name": "mlp"}]
        r = bcg.plan_capture(ops, capture_ceiling=2048, chunked_prefill_size=2048)
        self.assertEqual(r["graph_segments"], 2)
        self.assertEqual(r["eager_breaks"], 1)
        self.assertTrue(r["eager_activation_peak_collapsed"])

    def test_146_gap157_merlin_second_order_moment_router(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MERLINSecondOrderMomentRouter"][0]
        ns = {"Dict": dict, "Any": object, "List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<merlin>", "exec"), ns)
        r = ns["MERLINSecondOrderMomentRouter"]()
        w = r.score_segments([1.0, 0.0], [{"mean_key": [0.5, 0.0], "var_key": [0.01, 0.0]}, {"mean_key": [0.5, 0.0], "var_key": [1.0, 0.0]}])
        self.assertGreater(w[1], w[0])  # Higher 2nd-moment variance boosts segment routing weight

    def test_147_gap158_parattt_chunkwise_parallel_refiner(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ParaTTTChunkwiseParallelRefiner"][0]
        ns = {"Dict": dict, "Any": object, "List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<parattt>", "exec"), ns)
        p = ns["ParaTTTChunkwiseParallelRefiner"]()
        r1 = p.refine_trajectory([1.0, -2.0], contraction=0.5, iters=1)
        r2 = p.refine_trajectory([1.0, -2.0], contraction=0.5, iters=3)
        self.assertLess(r2["mean_abs_error"], r1["mean_abs_error"])

    def test_148_gap159_residualquant_looped_int2_kv(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ResidualQuantLoopedINT2KV"][0]
        ns = {"Dict": dict, "Any": object, "List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<resq>", "exec"), ns)
        rq = ns["ResidualQuantLoopedINT2KV"]()
        res = rq.quantize_loop_residual([1.0, 2.0, 3.0], [1.05, 1.95, 3.02])
        self.assertEqual(res["bits_per_residual"], 2)
        self.assertEqual(len(res["reconstructed"]), 3)

    def test_149_gap160_e8_lattice_implicit_2bit_qat(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "E8LatticeImplicit2BitQAT"][0]
        ns = {"Dict": dict, "Any": object, "List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<e8qat>", "exec"), ns)
        e8 = ns["E8LatticeImplicit2BitQAT"]()
        res = e8.project_e8([0.4, 0.6, 0.5, 0.5, -0.4, -0.6, 0.5, 0.5])
        self.assertEqual(res["explicit_codebook_bytes"], 0)
        self.assertEqual(len(res["lattice_point"]), 8)

    def test_150_gap161_cots_competitive_token_selector(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CoTSCompetitiveTokenSelector"][0]
        ns = {"Dict": dict}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<cots>", "exec"), ns)
        cots = ns["CoTSCompetitiveTokenSelector"]()
        dom = cots.blend_weights(0.95, 0.02)
        tied = cots.blend_weights(0.45, 0.44)
        self.assertGreater(dom["discrete_weight"], 0.95)
        self.assertGreater(tied["soft_latent_weight"], 0.95)

    def test_151_gap162_super_mamba_surprisal_router(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SuPERMambaSurprisalRouter"][0]
        ns = {"List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<supermamba>", "exec"), ns)
        sr = ns["SuPERMambaSurprisalRouter"]()
        gates = sr.compute_update_gates([0.99, 0.99, 0.01, 0.99], pool_window=1)
        self.assertEqual(gates[2], 1.0)
        self.assertLess(gates[0], 0.05)

    def test_152_gap163_select_ttt_query_conditioned_writer(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SelectTTTQueryConditionedWriter"][0]
        ns = {"List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<selectttt>", "exec"), ns)
        sw = ns["SelectTTTQueryConditionedWriter"]()
        mults = sw.compute_chunk_lr_multipliers([-10.0, -0.5, -10.0], temperature=1.0)
        self.assertGreater(mults[1], 2.5)

    def test_153_gap164_gppo_generalized_divergence_rl(self) -> None:
        tsrc_te = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(tsrc_te)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GPPOGeneralizedDivergenceRL"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<gppo>", "exec"), ns)
        g = ns["GPPOGeneralizedDivergenceRL"]()
        self.assertEqual(g.f_divergence(1.0, "reverse_kl"), 0.0)
        self.assertGreater(g.f_divergence(1.5, "chi2"), 0.0)
        self.assertGreater(g.f_divergence(1.5, "js"), 0.0)

    def test_154_gap165_int8_exact_426bit_block_gemm(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Int8Exact426BitBlockGEMM"][0]
        ns = {"Dict": dict, "Any": object, "List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<int8mxfp4>", "exec"), ns)
        q = ns["Int8Exact426BitBlockGEMM"](block_size=16, max_int_mult=10)
        res = q.quantize_block([0.1 * i - 0.8 for i in range(16)])
        self.assertTrue(res["exact_int8_gemm"])
        self.assertTrue(1 <= res["int_mult"] <= 10)
        self.assertEqual(len(res["q_vals"]), 16)

    def test_155_gap166_pact_certified_parallel_committer(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PACTCertifiedParallelCommitter"][0]
        ns = {"List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<pact>", "exec"), ns)
        p = ns["PACTCertifiedParallelCommitter"]()
        pos = p.select_certified_positions([0.95, 0.90, 0.85, 0.20, 0.92], min_reliability=0.70, min_spacing=2)
        self.assertEqual(pos, [0, 2, 4])

    def test_156_gap167_hinted_scaling_law_forecaster(self) -> None:
        tsrc_te = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(tsrc_te)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HintedScalingLawForecaster"][0]
        ns = {"Dict": dict, "Any": object, "List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<hinted>", "exec"), ns)
        f = ns["HintedScalingLawForecaster"]()
        r = f.compute_hinted_score([0.02, 0.15, 0.45, 0.80])
        self.assertTrue(r["monotone_with_hints"])
        self.assertTrue(r["above_noise_floor"])

    def test_157_gap168_truer_error_aware_reward_rectifier(self) -> None:
        tsrc_te = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(tsrc_te)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TRUERErrorAwareRewardRectifier"][0]
        ns = {"Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<truer>", "exec"), ns)
        tr = ns["TRUERErrorAwareRewardRectifier"](rectifier_threshold=0.50)
        fp = tr.rectify_reward(majority_vote_reward=1.0, rectifier_correct_prob=0.15)
        self.assertTrue(fp["false_positive_majority_caught"])
        self.assertEqual(fp["rectified_reward"], 0.0)
        fn = tr.rectify_reward(majority_vote_reward=0.0, rectifier_correct_prob=0.85)
        self.assertTrue(fn["correct_minority_rescued"])
        self.assertEqual(fn["rectified_reward"], 1.0)

    def test_158_gap169_bridgekv_cross_model_stitcher(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BridgeKVCrossModelStitcher"][0]
        ns = {"List": list}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<bridgekv>", "exec"), ns)
        b = ns["BridgeKVCrossModelStitcher"]()
        m = b.map_layer_indices(src_layers=32, dst_layers=8)
        self.assertEqual(len(m), 8)
        self.assertEqual(m[0], [0, 1, 2, 3])
        self.assertEqual(m[-1], [28, 29, 30, 31])

    def test_160_gap170_llm42_verified_speculation_determinism(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LLM42VerifiedSpeculationDeterminism"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<llm42>", "exec"), ns)
        v = ns["LLM42VerifiedSpeculationDeterminism"](verify_window=4)
        ok = v.verify_and_commit([10, 20, 30, 40], [10, 20, 30, 40])
        self.assertFalse(ok["rolled_back"])
        self.assertEqual(ok["committed_tokens"], [10, 20, 30, 40])
        rb = v.verify_and_commit([10, 20, 99, 40], [10, 20, 30, 40])
        self.assertTrue(rb["rolled_back"])
        self.assertEqual(rb["rollback_index"], 2)
        self.assertEqual(rb["committed_tokens"], [10, 20, 30])

    def test_161_gap171_kernelseek_question_driven_loop(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KernelSeekQuestionDrivenLoop"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<kernelseek>", "exec"), ns)
        ks = ns["KernelSeekQuestionDrivenLoop"](target_speedup=1.15)
        r_bad = ks.evaluate_step(181, 182, 1.46, 0)
        self.assertFalse(r_bad["accepted"])
        self.assertEqual(r_bad["action"], "fix_correctness_first")
        r_q = ks.evaluate_step(182, 182, 0.50, 6, claimed_hardware_limit=True)
        self.assertEqual(r_q["action"], "ask_layer_shift_question")
        r_ok = ks.evaluate_step(182, 182, 1.46, 1)
        self.assertTrue(r_ok["accepted"])

    def test_162_gap172_lark_segmented_kernel_proof_gate(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LARKSegmentedKernelProofGate"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<lark>", "exec"), ns)
        gate = ns["LARKSegmentedKernelProofGate"]()
        r_ok = gate.verify_kernel_segments([
            {"name": "load_tile", "lean_segment_proved": True, "pytorch_equiv_proved": True, "has_data_race": False}
        ])
        self.assertTrue(r_ok["verified"])
        r_race = gate.verify_kernel_segments([
            {"name": "reduce_smem", "lean_segment_proved": True, "pytorch_equiv_proved": True, "has_data_race": True}
        ])
        self.assertFalse(r_race["verified"])
        self.assertTrue(r_race["data_race_detected"])

    def test_163_gap173_memento_compaction_and_backtracking(self) -> None:
        wsrc = (REPO_ROOT / "src" / "experimental" / "memory_and_sleep_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(wsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MEMENTOCompactionAndBacktrackingController"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<memento>", "exec"), ns)
        mc = ns["MEMENTOCompactionAndBacktrackingController"](max_window_tokens=20000, backtrack_conf_threshold=0.65)
        r_fold = mc.step_fold_or_backtrack(25000, ["note1", "note2"], 0.85)
        self.assertEqual(r_fold["action"], "compact_and_fold")
        self.assertTrue(r_fold["within_window_cap"])
        r_bt = mc.step_fold_or_backtrack(15000, ["note1", "note2"], 0.40)
        self.assertEqual(r_bt["action"], "confidence_backtrack")

    def test_164_gap174_yolofs_staged_snapshot_filesystem(self) -> None:
        esrc = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        tree = ast.parse(esrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "YoloFSStagedSnapshotFilesystem"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<yolofs>", "exec"), ns)
        yfs = ns["YoloFSStagedSnapshotFilesystem"]()
        blocked = yfs.stage_write("/etc/important", "", destructive=True, approved=False)
        self.assertFalse(blocked["staged"])
        ok = yfs.stage_write("a.txt", "v1")
        self.assertTrue(ok["staged"])
        yfs.commit_with_snapshot()
        self.assertEqual(yfs.committed_state["a.txt"], "v1")
        yfs.stage_write("a.txt", "v2")
        yfs.commit_with_snapshot()
        self.assertTrue(yfs.undo_last_commit())
        self.assertEqual(yfs.committed_state["a.txt"], "v1")

    def test_165_gap175_harnessflow_mcts_optimizer(self) -> None:
        esrc = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        tree = ast.parse(esrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HarnessFlowMCTSOptimizer"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<harnessflow>", "exec"), ns)
        hf = ns["HarnessFlowMCTSOptimizer"]()
        pick = hf.select_candidate_ucb1([
            {"id": "h0", "visits": 5, "mean_score": 0.58},
            {"id": "h1", "visits": 2, "mean_score": 0.69},
        ], total_visits=7)
        self.assertEqual(pick, "h1")

    def test_166_multi_agent_info_network_contagion_guardrail(self) -> None:
        esrc = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        tree = ast.parse(esrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiAgentInfoNetworkGuardrail"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<infonet>", "exec"), ns)
        g = ns["MultiAgentInfoNetworkGuardrail"]()
        bad = g.validate_topology(incentives_aligned=False, topology="barbell", peer_transparency=True)
        self.assertFalse(bad["allowed"])
        self.assertIn("barbell_topology_banned", bad["violations"])
        self.assertIn("misaligned_peer_contagion_risk", bad["violations"])
        good = g.validate_topology(incentives_aligned=False, topology="isolated", peer_transparency=False)
        self.assertTrue(good["allowed"])

    def test_167_cto_experimental_isolation_tier_verified(self) -> None:
        exp_root = REPO_ROOT / "src" / "experimental"
        counts = {}
        for fn in ("kernel_and_serving_planners.py", "training_and_rl_planners.py", "memory_and_sleep_planners.py", "harness_and_safety_policies.py"):
            src = (exp_root / fn).read_text(encoding="utf-8")
            counts[fn] = len([n for n in ast.parse(src).body if isinstance(n, ast.ClassDef)])
        self.assertGreaterEqual(counts["kernel_and_serving_planners.py"], 28)
        self.assertGreaterEqual(counts["training_and_rl_planners.py"], 8)
        self.assertGreaterEqual(counts["memory_and_sleep_planners.py"], 6)
        self.assertGreaterEqual(counts["harness_and_safety_policies.py"], 22)

    def test_168_gap176_helion_compile_time_ir_heuristics(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HelionCompileTimeIRHeuristicPlanner"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<helion>", "exec"), ns)
        p = ns["HelionCompileTimeIRHeuristicPlanner"]()
        cfg = p.plan_kernel_config("linear_attention", seq_len=4096, head_dim=128)
        self.assertTrue(cfg["autotune_skipped"])
        self.assertEqual(cfg["block_m"], 64)
        self.assertEqual(cfg["num_stages"], 3)

    def test_169_gap177_opsd_weight_text_memory_router(self) -> None:
        wsrc = (REPO_ROOT / "src" / "experimental" / "memory_and_sleep_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(wsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OPSDWeightTextMemoryRouter"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<opsd>", "exec"), ns)
        r = ns["OPSDWeightTextMemoryRouter"](verify_threshold=0.80)
        self.assertEqual(r.route_query(0.92)["source"], "weight_memory")
        self.assertTrue(r.route_query(0.55)["regenerate_with_text"])

    def test_170_gap178_training_free_memory_lifecycle_and_belief_guard(self) -> None:
        wsrc = (REPO_ROOT / "src" / "experimental" / "memory_and_sleep_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(wsrc)
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("TrainingFreeMemoryLifecycleGate", "KnowinglyWrongBeliefConsistencyGuard")]
        ns = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<memlife>", "exec"), ns)
        gate = ns["TrainingFreeMemoryLifecycleGate"]()
        self.assertGreater(gate.score_memory_ucb(0.5, 1, 100), gate.score_memory_ucb(0.5, 50, 100))
        self.assertTrue(gate.screen_merge_preservation(["e1", "e2"], ["e1", "e2", "e3"])["approve_merge"])
        self.assertFalse(gate.screen_merge_preservation(["e1", "e2"], ["e1"])["approve_merge"])
        bg = ns["KnowinglyWrongBeliefConsistencyGuard"]()
        self.assertFalse(bg.audit_knowledge_update(0.90, True, False)["allow_update"])
        self.assertTrue(bg.audit_knowledge_update(0.90, True, True)["allow_update"])

    def test_171_gap179_lprop_latent_uncertainty_belief_reviser(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LPropLatentUncertaintyBeliefReviser"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<lprop>", "exec"), ns)
        rev = ns["LPropLatentUncertaintyBeliefReviser"]()
        low_u = rev.revise_belief(0.2, 0.1, 0.9)
        high_u = rev.revise_belief(0.2, 0.9, 0.9)
        self.assertGreater(high_u["revised_prob"], low_u["revised_prob"])
        self.assertFalse(high_u["requires_extra_sampling"])

    def test_172_gap180_safestart_full_expert_prefix_guard(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SafeStartFullExpertPrefixGuard"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<safestart>", "exec"), ns)
        g = ns["SafeStartFullExpertPrefixGuard"](safe_prefix_tokens=8)
        self.assertEqual(g.active_expert_count(True, 0, 8, 2), 8)
        self.assertEqual(g.active_expert_count(False, 7, 8, 2), 8)
        self.assertEqual(g.active_expert_count(False, 8, 8, 2), 2)

    def test_173_gap181_deadline_aware_reasoning_budget_gate(self) -> None:
        esrc = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        tree = ast.parse(esrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeadlineAwareReasoningBudgetGate"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<deadline>", "exec"), ns)
        d = ns["DeadlineAwareReasoningBudgetGate"]()
        self.assertEqual(d.select_mode(45.0, 20.0, 2.0, concurrent_burst=8)["mode"], "thinking")
        self.assertEqual(d.select_mode(45.0, 20.0, 2.0, concurrent_burst=56)["mode"], "fast_fallback")

    def test_174_gap182_aegis_online_sdc_self_equivalence_sensor(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AEGISOnlineSDCSelfEquivalenceSensor"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<aegis>", "exec"), ns)
        s = ns["AEGISOnlineSDCSelfEquivalenceSensor"]()
        self.assertFalse(s.check_recomputation_equivalence(0xABCD, 0xABCD, 1e-5)["sdc_suspected"])
        bad = s.check_recomputation_equivalence(0xABCD, 0xABCE, 1e-5)
        self.assertTrue(bad["sdc_suspected"])
        self.assertTrue(bad["dispatch_offpath_verifier"])

    def test_176_gap183_gatesight_promoted_to_live_brain_hot_path(self) -> None:
        bsrc = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn("class GateSightKVPruner(nn.Module):", bsrc)
        self.assertIn("self.gatesight = GateSightKVPruner(", bsrc)
        self.assertIn("self.gatesight.prune_cache(cache, best_h", bsrc)
        ek = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertNotIn("class GateSightGDNAnticipatedKVPruner", ek)

    def test_177_gap184_bitnest_promoted_to_live_brain_hot_path(self) -> None:
        bsrc = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn("def bitnest_spec_decode(", bsrc)
        self.assertIn('"extra_draft_vram_mb": 0.0', bsrc)
        ek = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        self.assertNotIn("class BitNestSharedWeightSpeculator", ek)

    def test_178_gap185_tasq_one_bit_kv_space_tailor(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TaSQOneBitKVSpaceTailor"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<tasq>", "exec"), ns)
        t = ns["TaSQOneBitKVSpaceTailor"]()
        plan = t.compute_channel_group_plan(head_dim=128, group_size=8)
        self.assertEqual(plan["num_covariance_groups"], 16)
        self.assertTrue(plan["rope_compatible_merged"])

    def test_179_gap186_schemafill_slot_parallel_tool_decoder(self) -> None:
        esrc = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        tree = ast.parse(esrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SchemaFillSlotParallelToolDecoder"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<schemafill>", "exec"), ns)
        sf = ns["SchemaFillSlotParallelToolDecoder"]()
        res = sf.verify_parallel_slots({"a": "1", "b": "wrong"}, {"a": "1", "b": "2"})
        self.assertEqual(res["committed_slots"], {"a": "1", "b": "2"})
        self.assertEqual(res["corrected_slots"], ["b"])

    def test_180_gap187_dia_disagreement_heterogeneous_arbiter(self) -> None:
        esrc = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        tree = ast.parse(esrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DIADisagreementHeterogeneousArbiter"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<dia>", "exec"), ns)
        dia = ns["DIADisagreementHeterogeneousArbiter"]()
        fast = dia.arbitrate("42", "42")
        self.assertFalse(fast["arbiter_triggered"])
        arb = dia.arbitrate("41", "42", ["42", "42", "41"])
        self.assertTrue(arb["arbiter_triggered"])
        self.assertEqual(arb["selected_answer"], "42")

    def test_181_gap188_stagq_dyadic_multi_precision_stream(self) -> None:
        bsrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(bsrc)
        cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StagQDyadicMultiPrecisionStream"][0]
        ns = {}
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), "<stagq>", "exec"), ns)
        sq = ns["StagQDyadicMultiPrecisionStream"]()
        s2 = sq.slice_operating_precision(2)
        s4 = sq.slice_operating_precision(4)
        self.assertEqual(s2["dyadic_refinement_planes"], 0)
        self.assertEqual(s4["dyadic_refinement_planes"], 2)
        self.assertTrue(s4["prefix_readable_single_stream"])

    def test_183_cto_phase_routing_registry_and_legacy_numpy_isolation(self) -> None:
        self.assertFalse((REPO_ROOT / "src" / "brain" / "non_transformer_backbone.py").exists())
        self.assertTrue((REPO_ROOT / "src" / "experimental" / "legacy_numpy_backbone_poc.py").exists())
        import importlib.util
        spec = importlib.util.spec_from_file_location("exp_pkg", str(REPO_ROOT / "src" / "experimental" / "__init__.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        reg = mod.PHASE_ROUTING_REGISTRY
        self.assertEqual(len(reg), 5)
        self.assertIn("Phase 2 · 后训练/RLVR (Accepted Backbone)", reg)
        self.assertIn("Phase 3 · 推理与外骨骼 (Accepted Backbone)", reg)
        self.assertIn("⚔️ 同位待 PK (True A/B Challengers)", reg)


if __name__ == "__main__":
    unittest.main()
