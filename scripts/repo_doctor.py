#!/usr/bin/env python3
"""
Sovereign AI Brain — Automated CTO Repo Doctor, ADR Compiler & 11-Dim Graph SSOT CLI (v10.32).
Single-command health check verifying:
  1. Zero dead classes & zero dead functions across src/brain/ hot path (AST call graph).
  2. Zero-duplicate 900 refs & unified 3-Verdict x 6-Substate closure.
  3. ARCHITECTURE_GRAPH.json -> ARCHITECTURE_DECISIONS.md one-way compiled byte-for-byte sync (`--compile-adr`).
  4. ARCHITECTURE_DIAGRAMS.md version & live Mermaid node sync gate.
  5. ARCHITECTURE_GRAPH.json 11-Dim AST bidirectional verification (code_anchors, gate_tests, mermaid_nodes, ref_ids, cross_organ_edges).
  6. `--trace <query>` instant reverse-lookup across the 11-dim architecture graph.
  7. Optional full 3-tier unittest regression (`--run-tests`).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_VERSION = "v10.32"
EXPECTED_REFS = 2236


def compile_adr_markdown(data: dict) -> str:
    ver = data["repo_version"]
    lines = [
        f"# Sovereign AI Brain — CTO 唯一架构决策与全链路追踪中枢（ADR-001 ~ ADR-013，{ver}）",
        "",
        "> **单向编译产物声明（SSOT 终态）**：本文件由 [`ARCHITECTURE_GRAPH.json`](ARCHITECTURE_GRAPH.json) 经 `python3 scripts/repo_doctor.py --compile-adr` 自动编译生成，并由 `adr_compiled_in_sync` 字节级门禁强制守卫（支持 `Phase 1/2/3` 与 `Scale-Up` 全阶段追踪）。严禁手工改坏同步。",
        "",
        "| ADR | 器官 / 问题 | `✅ 既定主干` × 生效阶段 | `⚔️ 同位待 PK` / `⚡扩展储备` | `❌ 淘汰` / 禁令 | 代码落点 & 守门单测 | 成熟度 & 硬件包络 & 推翻条件 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for a in data["adrs"]:
        code_parts = []
        for ca in a["code_anchors"]:
            syms = ", ".join(f"`{s}`" for s in ca["symbols"])
            code_parts.append(f"`{ca['file']}` ({syms})")
        code_str = " + ".join(code_parts)
        tests_str = ", ".join(f"`{t.split('_')[0]}_{t.split('_')[1]}`" for t in a["gate_tests"])
        col_main = f"{a['verdict']} `[{a['phase']}]` {a['core_mechanism']}"
        col_chal = a.get("challenger_or_reserve", "-")
        col_rej = a.get("rejected_or_redlines", "-")
        col_ev = f"`[{a['evidence_maturity']}]` {a['hardware_envelope']}；**推翻**：{a['rollback_trigger']}"
        lines.append(
            f"| **{a['adr_id']}** | **{a['organ']}** | {col_main} | {col_chal} | {col_rej} | {code_str} \\| {tests_str} | {col_ev} |"
        )
    return "\n".join(lines) + "\n"


def audit_zero_dead_code() -> dict:
    core_specs = [
        ("sovereign_brain_pytorch.py", {"SovereignRouteBBrain", "RouteBConfig", "RouteBCache", "PolarExpressMuonOptimizer"}),
        ("training_engine.py", {"TrainConfig", "AdaptiveDepthPriorController", "StreamingByteCorpus"}),
        ("wake_sleep_runtime.py", {"WakeTurnResult", "WakeSleepRuntimeState", "WakeSleepBrainRuntime"}),
        ("exoskeleton_bridge.py", {
            "ByteStreamControlProtocol", "UncertaintyReadout", "AsymmetricWritePort",
            "UniversalByteCodec", "DecomposedUncertainty", "SplitConformalAbstentionCalibrator",
            "VersionedWeightSnapshot", "ConditionalFailurePlaybook",
        }),
        ("five_axis_microworld.py", {"FiveAxisMicroworldGenerator"}),
        ("world_kernel_engine.py", {"EvidenceBit", "NamedConstraint", "FiniteDomainCSPSolver", "DependencyGraph", "FourDTimeCut", "BitemporalFactLedger", "ChisholmDeonticEngine"}),
    ]
    dead_classes = {}
    for fname, roots in core_specs:
        src = (REPO_ROOT / "src" / "brain" / fname).read_text(encoding="utf-8")
        tree = ast.parse(src)
        cnodes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        refs = {}
        for n in tree.body:
            names = {s.id for s in ast.walk(n) if isinstance(s, ast.Name)} | {
                s.attr for s in ast.walk(n) if isinstance(s, ast.Attribute)
            }
            refs[id(n)] = names
        live = set(roots)
        for n in tree.body:
            if not isinstance(n, ast.ClassDef):
                for cn in cnodes:
                    if cn in refs[id(n)]:
                        live.add(cn)
        changed = True
        while changed:
            changed = False
            for cn in list(live):
                if cn in cnodes:
                    for other in cnodes:
                        if other != cn and other not in live and other in refs[id(cnodes[cn])]:
                            live.add(other)
                            changed = True
        dead = sorted(set(cnodes) - live)
        if dead:
            dead_classes[fname] = dead

    bsrc = (REPO_ROOT / "src" / "brain" / "sovereign_brain_pytorch.py").read_text(encoding="utf-8")
    btree = ast.parse(bsrc)
    bfuncs = {n.name: n for n in btree.body if isinstance(n, ast.FunctionDef)}
    brefs = {
        id(n): (
            {s.id for s in ast.walk(n) if isinstance(s, ast.Name)}
            | {s.attr for s in ast.walk(n) if isinstance(s, ast.Attribute)}
        )
        for n in btree.body
    }
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
    dead_funcs = sorted(set(bfuncs) - lf)
    return {"ok": (not dead_classes and not dead_funcs), "dead_classes": dead_classes, "dead_funcs": dead_funcs}


def audit_architecture_graph() -> dict:
    gpath = REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_GRAPH.json"
    if not gpath.exists():
        return {"ok": False, "errors": ["Missing docs/architecture/ARCHITECTURE_GRAPH.json"]}
    data = json.loads(gpath.read_text(encoding="utf-8"))
    errors: list[str] = []

    if data.get("repo_version") != REPO_VERSION:
        errors.append(f"Graph repo_version={data.get('repo_version')} != {REPO_VERSION}")
    if data.get("total_refs") != EXPECTED_REFS:
        errors.append(f"Graph total_refs={data.get('total_refs')} != {EXPECTED_REFS}")

    adrs = data.get("adrs", [])
    adr_ids = {a["adr_id"] for a in adrs}
    expected_adrs = {f"ADR-{i:03d}" for i in range(1, 14)}
    if adr_ids != expected_adrs:
        errors.append(f"ADR mismatch: missing={sorted(expected_adrs - adr_ids)}")

    real_tests: set[str] = set()
    for tf in (REPO_ROOT / "tests").glob("*.py"):
        ttree = ast.parse(tf.read_text(encoding="utf-8"))
        for node in ast.walk(ttree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                real_tests.add(node.name)

    reg_txt = (REPO_ROOT / "docs" / "whitepapers" / "brain_paper_organ_registry.md").read_text(encoding="utf-8")
    real_refs = {int(m) for m in re.findall(r"(?m)^\|\s*`\[(\d+)\]`\s*\|", reg_txt)}
    diag_txt = (REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_DIAGRAMS.md").read_text(encoding="utf-8")

    file_syms: dict[str, set[str]] = {}
    def get_symbols(rel_path: str) -> set[str]:
        if rel_path not in file_syms:
            fp = REPO_ROOT / rel_path
            if not fp.exists():
                file_syms[rel_path] = set()
            else:
                tr = ast.parse(fp.read_text(encoding="utf-8"))
                file_syms[rel_path] = {
                    n.name for n in ast.walk(tr) if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                }
        return file_syms[rel_path]

    required_fields = (
        "adr_id", "organ", "verdict", "phase", "sources", "core_mechanism",
        "code_anchors", "gate_tests", "hardware_envelope", "evidence_maturity",
        "cross_organ_edges", "rollback_trigger", "mermaid_nodes", "benchmark_reports",
        "challenger_or_reserve", "rejected_or_redlines",
    )
    checked_symbols = 0
    checked_tests = 0
    for a in adrs:
        aid = a.get("adr_id", "UNKNOWN")
        for f in required_fields:
            if f not in a or a[f] in (None, "", []):
                errors.append(f"{aid}: missing or empty field '{f}'")
        for ca in a.get("code_anchors", []):
            rel = ca["file"]
            syms = get_symbols(rel)
            if not syms:
                errors.append(f"{aid}: code_anchor file missing '{rel}'")
            for sym in ca.get("symbols", []):
                checked_symbols += 1
                if sym not in syms:
                    errors.append(f"{aid}: symbol '{sym}' not found in '{rel}'")
        for gt in a.get("gate_tests", []):
            checked_tests += 1
            if gt not in real_tests:
                errors.append(f"{aid}: gate_test '{gt}' not found in tests/")
        for rid in a.get("sources", {}).get("ref_ids", []):
            if int(rid) not in real_refs:
                errors.append(f"{aid}: ref_id [{rid}] not found in registry")
        for mn in a.get("mermaid_nodes", []):
            if mn not in diag_txt:
                errors.append(f"{aid}: mermaid_node '{mn}' not found in ARCHITECTURE_DIAGRAMS.md")
        edges = a.get("cross_organ_edges", {})
        for etype in ("depends_on", "conflicts_with", "feeds_into"):
            for target in edges.get(etype, []):
                if target not in adr_ids or target == aid:
                    errors.append(f"{aid}: invalid {etype} edge -> '{target}'")

    adr_disk = (REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_DECISIONS.md").read_text(encoding="utf-8")
    adr_compiled_in_sync = (adr_disk == compile_adr_markdown(data))
    if not adr_compiled_in_sync:
        errors.append("ARCHITECTURE_DECISIONS.md out of sync with compiled ARCHITECTURE_GRAPH.json (run --compile-adr)")

    return {
        "ok": len(errors) == 0,
        "adrs_verified": len(adrs),
        "ast_symbols_verified": checked_symbols,
        "gate_tests_verified": checked_tests,
        "adr_compiled_in_sync": adr_compiled_in_sync,
        "errors": errors,
    }


def audit_ssot_governance() -> dict:
    reg = (REPO_ROOT / "docs" / "whitepapers" / "brain_paper_organ_registry.md").read_text(encoding="utf-8")
    non_fixed_tags = {"⚔️同位待PK", "⚡扩展储备", "⛔负向禁令", "📜历史淘汰"}
    rows = re.findall(r"(?m)^\|\s*`\[(\d+)\]`\s*\|\s*`\[([^]]+)\]`\s*\|", reg)
    bad_tags = sorted({
        t for _, t in rows
        if t not in non_fixed_tags and not re.match(r"^✅既定·(?:活体已接电|待阶段接电)·[^·]+$", t)
    })
    adr = (REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_DECISIONS.md").read_text(encoding="utf-8")
    adrs_ok = all(f"ADR-{i:03d}" in adr for i in range(1, 14))
    bench_sum = json.loads((REPO_ROOT / "benchmarks" / "reports" / "BENCHMARK_SUMMARY.json").read_text(encoding="utf-8"))
    raw_cnt = len([p for p in (REPO_ROOT / "benchmarks" / "reports").glob("*.json") if p.name != "BENCHMARK_SUMMARY.json"])
    bench_ok = bench_sum.get("total_raw_reports") == raw_cnt

    diag = (REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_DIAGRAMS.md").read_text(encoding="utf-8")
    required_diag_tokens = [
        REPO_VERSION,
        "GateSightKVPruner", "BOS Summary", "SGA-Sparse", "BitNest",
        "Mixture-then-Select", "Soteria", "TenonAgent", "EpiLoRA", "Fairway",
        "PrefixCacheMetastabilityBudgeter", "QHijackCrossPrecisionAuditGate",
        "ShapleyCompactMemoryPruner", "GateFloor", "GroupClip", "Airlock",
        "PRISM", "ContinualHarnessRegressionGuard", "CISA", "TAC-CoT",
        "Host-Aware", "PT/MT×LR协同退火", "GovernedExperimentAppendOnlyLedger", "MemShift", "Plan-and-Patch", "ReplaySSMCheckpointBuffer", "TransactionalKVStateGuard", "LTTuningConfidenceSwitch", "KernelRidgeAttentionChallenger", "EarlyLayerW8QuantGuard", "StopAndRedrawLoopBreaker", "GuidedRLVRLogicController", "VerifierYoudenAuditGate", "GPUAdaptiveSpecWidthController", "ROSESubspaceRiskMask", "AbstentionDirectionTriRouter", "FOSCARAffineFoldKVKernel", "WriteGatedKVAdmission", "TurnWiseCreditAssigner", "ToolUpdateIntegrityGate", "LoopSkimMultiDepthVerifier", "ShallowWriteAnchorExtractor", "CVSAZeroDenseCoverageGate", "SpectralUnlearningMuonUpdater", "DISCERNReadTimeResolver", "ExecutableFSMProcedureGuard", "DecodeFeedbackKVEvictionTimer", "SynLatSyntaxAlignedCompressor", "TACOOpeNormOneSparseUpdater", "InfoGRPODoubleRolloutEstimator", "UMASMultiAgentUncertaintyGate", "HostSecondaryToolAuthGuard", "SPARECompactRecurrentStateProjector", "VerKVFathomCertificateFetcher", "MarginProximalEntropyRLVRController", "TVPDRealSpeculativeLossScheduler", "LMMRDeltaProceduralMemoryBridge", "MCPProviderBindingAndOverServeGuard", "QEDTailReplayHybridController", "LoopMTPFixedLNSScheduler", "QUASAROnlineSaliencyQATUpdater", "CueGRPOVerifierAnchoredCredit", "TFRNetDriftGuardMemoryRuntime", "HardwareKeystoreSkillGuardEnforcer", "QDeltaCCQEraseReadController", "LoopedFactorScalingVaSEGuard", "APEX4MixQuantMuonHyperballGuard", "VeriGateAsymGRPOEngine", "RepSelectISMSchemaUnlearnRuntime", "StoredPISafeMCPKernelFuzzGuard", "ByteFlowAMOREntropyGateController", "SpectralGuardCHONHotChannelShield", "MuonPlusMSignRankOptimizerGuard", "FRPOFutureKLOARCreditEngine", "SBUSAGOSynchronizedUnlearnRuntime", "MCPSHIELDOwnerHarmAnumatiGuard", "TrainingEngineGovernanceSuite",
    ]
    missing_diag = [t for t in required_diag_tokens if t not in diag]
    diagrams_synced = len(missing_diag) == 0
    md_locked = len([p for p in REPO_ROOT.rglob("*.md") if ".git" not in p.parts]) == 9
    graph_res = audit_architecture_graph()
    return {
        "ok": (
            len(rows) == EXPECTED_REFS
            and not bad_tags
            and adrs_ok
            and bench_ok
            and diagrams_synced
            and md_locked
            and graph_res["ok"]
        ),
        "registry_refs": len(rows),
        "bad_tags": bad_tags,
        "adrs_complete": adrs_ok,
        "diagrams_synced": diagrams_synced,
        "missing_diagram_tokens": missing_diag,
        "benchmark_summary_synced": bench_ok,
        "md_file_count_locked_at_9": md_locked,
        "architecture_graph_ast_verified": graph_res,
    }



def print_audit_table() -> int:
    data = json.loads((REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_GRAPH.json").read_text(encoding="utf-8"))
    print("| ADR | 器官 | 裁决 × 阶段 | 成熟度 | 代码符号数 | 守门单测数 | Mermaid 节点 | 回滚/推翻触发条件 |")
    print("| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |")
    for a in data["adrs"]:
        sym_cnt = sum(len(ca["symbols"]) for ca in a["code_anchors"])
        t_cnt = len(a["gate_tests"])
        mn = ", ".join(a["mermaid_nodes"][:3])
        print(f"| **{a['adr_id']}** | {a['organ']} | {a['verdict']} `[{a['phase']}]` | `{a['evidence_maturity']}` | {sym_cnt} | {t_cnt} | `{mn}` | {a['rollback_trigger']} |")
    return 0


def trace_query(query: str) -> int:
    data = json.loads((REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_GRAPH.json").read_text(encoding="utf-8"))
    q = query.lower()
    hits = [a for a in data["adrs"] if q in json.dumps(a, ensure_ascii=False).lower()]
    print(json.dumps({"query": query, "match_count": len(hits), "matches": hits}, indent=2, ensure_ascii=False))
    return 0 if hits else 1



def check_arxiv_duplicates(candidate_ids: list[str]) -> int:
    wp = (REPO_ROOT / "docs" / "whitepapers" / "sovereign_ai_brain_whitepaper.md").read_text(encoding="utf-8")
    existing = set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", wp))
    brand_new = []
    duplicates = []
    for raw in candidate_ids:
        cid = re.sub(r"v\d+$", "", raw.strip().replace("arXiv:", ""))
        if cid in existing:
            duplicates.append(cid)
        else:
            brand_new.append(cid)
    print(json.dumps({"total_existing_ids": len(existing), "brand_new": brand_new, "duplicates": duplicates}, indent=2, ensure_ascii=False))
    return 0



def audit_three_pillar_freshness(online: bool = False) -> dict:
    """Rule 27: Unified 3-pillar freshness & anti-rot audit (Wheels x Datasets x Papers)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.brain.ecosystem_backends import audit_ecosystem_wheel_freshness
    from src.brain.prepare_seed_corpus import audit_corpus_sources_liveness

    wheels = audit_ecosystem_wheel_freshness(online=online)
    datasets = audit_corpus_sources_liveness("rtx3050_4gb_prod_30gb", online=online)

    reg = (REPO_ROOT / "docs" / "whitepapers" / "brain_paper_organ_registry.md").read_text(encoding="utf-8")
    arxiv_ids = re.findall(r"arXiv:(\d{4}\.\d{4,5})", reg)
    recent_25_26 = sum(1 for a in arxiv_ids if a.startswith(("25", "26")))
    ratio_25_26 = round(recent_25_26 / max(1, len(arxiv_ids)), 4)
    papers = {
        "ok": ratio_25_26 >= 0.95,
        "total_arxiv_mentions": len(arxiv_ids),
        "unique_arxiv_ids": len(set(arxiv_ids)),
        "recent_2025_2026_ratio": ratio_25_26,
    }
    return {
        "ok": bool(wheels["ok"] and datasets["ok"] and papers["ok"]),
        "mode": "online_live_api" if online else "offline_metadata",
        "wheels": wheels,
        "datasets": datasets,
        "papers": papers,
    }


def audit_anti_drift_and_slimming() -> dict:
    """Rule 30 (v23.0): Zero-magic-number structural invariants gate (zero-loss, orthogonal fields, natural cleavage)."""
    l1_txt = (REPO_ROOT / "docs" / "whitepapers" / "world_cognition_unified_model.md").read_text(encoding="utf-8")
    required_zero_loss_anchors = (
        "Gray–Scott", "0.696 AUROC", "95.5%", "6/6 vs 基线 0/12",
        "ADEPTS 30.6%", "复式记账", "V(D)J", "诺特定理", "重整化群",
        "evidence_both", "precedent_open_texture", "blast_radius_cap", "25 万页", "克雷所",
        "跨域三角定位律", "领域特征坍缩", "内生自洽律", "宇宙五层涌现阶梯", "四维双时态",
        "背压四难不可能定理", "控制-数据割序", "河床与水流三尺度互塑", "核心—表面双层可用性权衡律", "意图行为本质层", "Telos", "Grounding",
    )
    missing_anchors = [a for a in required_zero_loss_anchors if a not in l1_txt]
    zero_loss_ok = len(missing_anchors) == 0

    const_txt = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
    const_tokens = ("第 30 条", "第 31 条", "数字三源合法律", "核心—表面双层可用性权衡律", "全原则·全约束·全目标统一总纲", "Step 4.5")
    const_ok = all(t in const_txt for t in const_tokens)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.brain.five_axis_microworld import (
        FiveAxisMicroworldGenerator,
        desugar_human_surface_to_msk,
        audit_core_surface_four_rulers,
        lift_conflict_across_layers,
        validate_layer_descent_chain,
        validate_layered_kernel_packet,
    )

    sample = FiveAxisMicroworldGenerator().generate_axis1_non_additive_triad(seed=0)["msk_v1"]
    clean_l0 = validate_layered_kernel_packet(sample, "L0_INTENT", step_budget_remaining=1)
    leaked_pkt = dict(sample)
    leaked_pkt["Constraint"] = {**sample["Constraint"], "payload": {"detail": "acquire mutex on vram"}}
    bad_l0 = validate_layered_kernel_packet(leaked_pkt, "L0_INTENT", step_budget_remaining=1)
    undeclared_pkt = dict(sample)
    undeclared_pkt["Anchor"] = {**sample["Anchor"], "junk_field": True}
    bad_schema = validate_layered_kernel_packet(undeclared_pkt, "L0_INTENT", step_budget_remaining=1)
    budget_fail = validate_layered_kernel_packet(sample, "L3_EXECUTION", step_budget_remaining=0)
    chain_ok = validate_layer_descent_chain(["L0_INTENT", "L1_COORDINATION", "L2_DECIDABLE_IR", "L3_EXECUTION"])
    chain_cycle = validate_layer_descent_chain(["L0_INTENT", "L1_COORDINATION", "L0_INTENT"])
    lifted = lift_conflict_across_layers("vram_oom", "L3_EXECUTION", "L0_INTENT")
    desugared = desugar_human_surface_to_msk({"entity": "order", "hard_rule": "no_oversell", "action": "buy", "fallback": "owner_review"})
    desugar_ok = validate_layered_kernel_packet(desugared, "L0_INTENT", step_budget_remaining=1)["ok"]

    layered_ok = bool(
        clean_l0["ok"]
        and (not bad_l0["ok"])
        and any("layer_hiding_violation" in e for e in bad_l0["errors"])
        and (not bad_schema["ok"])
        and any("undeclared_non_orthogonal_fields" in e for e in bad_schema["errors"])
        and (not budget_fail["ok"])
        and budget_fail["halt_tag"] == "<|halt:rlimit|>"
        and chain_ok["ok"]
        and (not chain_cycle["ok"])
        and lifted["ok"]
        and len(lifted["leaked_tokens"]) == 0
        and desugar_ok
    )
    four_rulers = audit_core_surface_four_rulers()
    return {
        "ok": bool(zero_loss_ok and const_ok and layered_ok and four_rulers["ok"]),
        "zero_loss_anchors_ok": zero_loss_ok,
        "missing_anchors": missing_anchors,
        "constitution_v23_locked": const_ok,
        "four_rulers_gate": four_rulers,
        "structural_invariants_ok": layered_ok,
    }


def audit_theory_preservation_manifest() -> dict:
    """Rule 32 (v26.0): Machine-checkable Theory Manifest + Full Git History Union Zero-Evaporation Gate."""
    mpath = REPO_ROOT / "docs" / "architecture" / "THEORY_PRESERVATION_MANIFEST.json"
    if not mpath.exists():
        return {"ok": False, "errors": ["Missing THEORY_PRESERVATION_MANIFEST.json"]}
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    target_rel = manifest["target_document"]
    target_txt = (REPO_ROOT / target_rel).read_text(encoding="utf-8")
    errors: list[str] = []
    valid_layers = set(manifest.get("layer_schema", {}).keys())
    if len(valid_layers) != 5:
        errors.append(f"invalid_layer_schema_count:{len(valid_layers)}")
    layer_distribution: dict[str, int] = {k: 0 for k in sorted(valid_layers)}
    active_checked = 0
    for entry in manifest.get("entries", []):
        eid = entry.get("id", "UNKNOWN")
        st = entry.get("status")
        if st == "ACTIVE":
            active_checked += 1
            anc = entry.get("required_anchor", "")
            if not anc or anc not in target_txt:
                errors.append(f"missing_active_theory_anchor:{eid}:{anc}")
            al = entry.get("arch_layer")
            if al not in valid_layers:
                errors.append(f"invalid_or_missing_arch_layer:{eid}:{al}")
            else:
                layer_distribution[al] += 1
        elif st in ("MERGED_INTO", "FALSIFIED_BY"):
            if not entry.get("disposition_target"):
                errors.append(f"missing_disposition_target:{eid}")
        else:
            errors.append(f"invalid_status:{eid}:{st}")

    # Full Git history union check for URLs and arXiv IDs (zero silent evaporation)
    git_union_urls: set[str] = set()
    git_union_arxiv: set[str] = set()
    commits = subprocess.check_output(
        ["git", "log", "--pretty=format:%h", "--", target_rel], cwd=REPO_ROOT, text=True
    ).splitlines()
    for sha in commits:
        hist_txt = subprocess.check_output(["git", "show", f"{sha}:{target_rel}"], cwd=REPO_ROOT, text=True)
        git_union_urls.update(re.findall(r"https?://[^\s\)\]>]+", hist_txt))
        git_union_arxiv.update(re.findall(r"arXiv:(\d{4}\.\d{4,5})", hist_txt))

    head_urls = set(re.findall(r"https?://[^\s\)\]>]+", target_txt))
    head_arxiv = set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", target_txt))
    evaporated_urls = sorted(git_union_urls - head_urls)
    evaporated_arxiv = sorted(git_union_arxiv - head_arxiv)
    if evaporated_urls:
        errors.append(f"evaporated_historical_urls:{len(evaporated_urls)}")
    if evaporated_arxiv:
        errors.append(f"evaporated_historical_arxiv:{len(evaporated_arxiv)}")

    comp_ids = {e.get("id") for e in manifest.get("entries", []) if e.get("domain") == "computation_theory_core" and e.get("status") == "ACTIVE"}
    required_comp_ids = {
        "COMP-BOHM-JACOPINI", "COMP-CHURCH-TURING", "COMP-ALPERN-SCHNEIDER",
        "COMP-DEONTIC-LOGIC", "COMP-ASSUME-GUARANTEE", "COMP-CONCURRENCY-CALM-FLP",
        "COMP-CONTINUOUS-GPAC-BSS",
    }
    missing_comp = sorted(required_comp_ids - comp_ids)
    if missing_comp:
        errors.append(f"missing_computation_theory_core:{','.join(missing_comp)}")

    return {
        "ok": len(errors) == 0,
        "manifest_version": manifest.get("manifest_version"),
        "active_theories_verified": active_checked,
        "layer_distribution": layer_distribution,
        "git_commits_audited": len(commits),
        "evaporated_urls": evaporated_urls,
        "evaporated_arxiv": evaporated_arxiv,
        "errors": errors,
    }


def audit_executable_world_engine() -> dict:
    """Rule 33 / Step 5.5 (v28.0): Independent property-based, real CSP/MUS, and adversarial gate (zero mocks)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.brain.world_kernel_engine import (
        BOTH, NEITHER, REFUTE, SUPPORT, BitemporalFactLedger, ChisholmDeonticEngine, ChisholmNormSpec,
        DependencyGraph, EvidenceBit, FiniteDomainCSPSolver, FourDTimeCut, NamedConstraint,
    )
    from src.brain.five_axis_microworld import (
        DEFAULT_FOUR_LAYER_PROFILES, desugar_intent_essence_to_msk, validate_epoch_migration,
        validate_external_user_protocol, validate_layer_dag, validate_layered_kernel_packet,
    )

    vals = (SUPPORT, REFUTE, BOTH, NEITHER)
    # 1. Exhaustive Arieli-Avron bilattice laws
    bilattice_ok = (
        all(a.weak_implies(a).s == 1 for a in vals)
        and all(a.strong_implies(b) == b.neg().strong_implies(a.neg()) for a in vals for b in vals)
        and BOTH.conflict_guard(REFUTE) == (BOTH, True)
        and EvidenceBit.exists_eval([BOTH], domain_complete=False) == BOTH
        and EvidenceBit.forall_eval([], domain_complete=True) == NEITHER
    )

    # 2. Real CSP/MUS/MCS solver (non-additive 3-constraint allocation conflict)
    solver = FiniteDomainCSPSolver()
    domains = {"slot_a": [1, 2], "slot_b": [1, 2]}
    c1 = NamedConstraint("a_eq_1", ("slot_a",), lambda a: a["slot_a"] == 1)
    c2 = NamedConstraint("b_eq_1", ("slot_b",), lambda a: a["slot_b"] == 1)
    c3 = NamedConstraint("disjoint", ("slot_a", "slot_b"), lambda a: a["slot_a"] != a["slot_b"])
    c4 = NamedConstraint("benign_bound", ("slot_a",), lambda a: a["slot_a"] <= 2)
    mus_res = solver.extract_mus_and_safe_subset(domains, [c1, c2, c3, c4])
    sat_after_relax = solver.solve(domains, [c1, c3, c4])
    csp_ok = (
        mus_res["status"] == "UNSAT"
        and set(mus_res["unsat_core"]) == {"a_eq_1", "b_eq_1", "disjoint"}
        and len(mus_res["safe_subset"]) == 3
        and sat_after_relax["status"] == "SAT"
        and sat_after_relax["witness"]["slot_a"] != sat_after_relax["witness"]["slot_b"]
    )

    # 3. Real positive + negative dependency closure invalidation
    dg = DependencyGraph()
    dg.facts["price"] = 100
    dg.record_derivation("quote", 100, reads=["price"], neg_reads=["override_discount"])
    dg.record_derivation("unrelated", 1, reads=["other"])
    inv = dg.mutate_fact("override_discount", 0.9)
    dep_ok = inv["invalidated"] == ["quote"] and dg.derived["unrelated"]["valid"]

    # 4. Adversarial hiding bypass check (snake_case + CJK) & diamond DAG topology
    pkt = desugar_intent_essence_to_msk({"telos": "checkout", "boundary": "no_oversell", "premise": "p", "intervention": "vram_4gb_hard_cap", "authority": "owner", "grounding": "显存"})
    adv_hiding_ok = (not validate_layered_kernel_packet(pkt, "L0_INTENT", step_budget_remaining=1)["ok"]) and pkt["Judgment"]["status"]["tag"] == "gap"
    diamond_profiles = {
        **DEFAULT_FOUR_LAYER_PROFILES,
        "L1_CONSENSUS": {"forbidden_lower_tokens": frozenset({"cuda", "vram"})},
        "L1_KERNEL": {"forbidden_lower_tokens": frozenset({"raft", "wal"})},
    }
    dag_ok = validate_layer_dag([("L0_INTENT", "L1_CONSENSUS"), ("L0_INTENT", "L1_KERNEL"), ("L1_CONSENSUS", "L3_EXECUTION"), ("L1_KERNEL", "L3_EXECUTION")], profiles=diamond_profiles)["ok"]

    # 5. Real 4D Bitemporal Ledger (non-retroactivity + late obs)
    ledger = BitemporalFactLedger()
    ledger.register_norm("price_floor", 1, effective_from=1, effective_to=10, recorded_at=1, predicate=lambda v: v >= 108)
    ledger.register_norm("price_floor", 2, effective_from=10, effective_to=None, recorded_at=10, predicate=lambda v: v >= 120)
    ledger.append_fact("order_paid", 108, valid_from=5, recorded_at=5, generation=1)
    q4d = ledger.query_at(FourDTimeCut(5, 12, 10, 12), "order_paid", norm_id="price_floor")
    bitemporal_4d_ok = bool(
        q4d.retroactive_norm_detected
        and q4d.compliant_at_fact_time == SUPPORT
        and q4d.compliant_at_norm_time == REFUTE
        and q4d.primary_verdict == SUPPORT
    )

    # 6. Real Chisholm CTD Engine + R3 hard-redline dominance
    ch_spec = ChisholmNormSpec(
        norm_id="delivery_ctd",
        primary_obligation="deliver_on_time",
        compliant_followup="send_receipt",
        contrary_to_duty_compensation="issue_delay_voucher",
        hard_redline_guards=(NamedConstraint("no_forge_audit", ("forge_audit",), lambda a: not a["forge_audit"], hard=True),),
        soft_preferences=(NamedConstraint("low_cost", ("cost",), lambda a: a["cost"] <= 10, hard=False, priority=1_000_000),),
    )
    ch_eng = ChisholmDeonticEngine()
    ch_ctd = ch_eng.evaluate(ch_spec, {"forge_audit": False, "cost": 50}, primary_observed=REFUTE, compensation_observed=SUPPORT)
    ch_r3 = ch_eng.evaluate(ch_spec, {"forge_audit": True, "cost": 0}, primary_observed=SUPPORT, followup_observed=SUPPORT)
    chisholm_ctd_r3_ok = bool(
        ch_ctd.state == "CTD_COMPENSATION_ACTIVE"
        and ch_ctd.explosion_prevented
        and "send_receipt" in ch_ctd.suppressed_obligations
        and ch_r3.state == "HARD_REDLINE_HALT"
    )

    # 7. Cross-Epoch migration + External AI User protocol
    ep_compat = validate_epoch_migration(
        {"epoch": 1, "fields": {"id": {"type": "str", "required": True}}, "invariants": ["R1"]},
        {"epoch": 2, "fields": {"id": {"type": "str", "required": True}, "tag": {"type": "str", "required": True, "default": "v2"}}, "invariants": ["R1"]},
        [{"id": "a1"}],
    )
    ep_break = validate_epoch_migration(
        {"epoch": 1, "fields": {"id": {"type": "str", "required": True}}, "invariants": ["R1"]},
        {"epoch": 2, "fields": {"id": {"type": "str", "required": True}}, "invariants": []},
        [{"id": "a1"}],
    )
    ext_ok = validate_external_user_protocol(
        {"zero_ai_inside_compiler": True, "ai_role": "EXTERNAL_USER_ONLY"},
        {"message": "业务契约校验通过"},
    )
    ext_bad = validate_external_user_protocol(
        {"zero_ai_inside_compiler": False, "ai_role": "EXTERNAL_USER_ONLY"},
        {"message": "hir_arena overflow"},
    )
    epoch_ext_ok = bool(
        ep_compat["decision"] == "MIGRATE_COMPAT"
        and ep_break["decision"] == "HALT_BREAKING"
        and ext_ok["ok"]
        and (not ext_bad["ok"])
    )

    return {
        "ok": bool(bilattice_ok and csp_ok and dep_ok and adv_hiding_ok and dag_ok and bitemporal_4d_ok and chisholm_ctd_r3_ok and epoch_ext_ok),
        "bilattice_laws_ok": bilattice_ok,
        "real_csp_mus_ok": csp_ok,
        "dep_closure_invalidation_ok": dep_ok,
        "adversarial_hiding_and_diamond_dag_ok": bool(adv_hiding_ok and dag_ok),
        "bitemporal_4d_ok": bitemporal_4d_ok,
        "chisholm_ctd_r3_ok": chisholm_ctd_r3_ok,
        "epoch_migration_and_ext_user_ok": epoch_ext_ok,
    }




def audit_v33_convergence_closure() -> dict:
    """v33.0 convergence closure gate: verifies C-like 5-primitive core closure lock,
    DLF formal BNF & boundary spec, 5-Gate bitemporal/Chisholm adjudication, 9-MD lock,
    and manifest v33.0 upgrade."""
    l1_txt = (REPO_ROOT / "docs" / "whitepapers" / "world_cognition_unified_model.md").read_text(encoding="utf-8")
    const_txt = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
    mpath = REPO_ROOT / "docs" / "architecture" / "THEORY_PRESERVATION_MANIFEST.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    entries = {e.get("id"): e for e in manifest.get("entries", [])}
    errors: list[str] = []

    md_locked = len([p for p in REPO_ROOT.rglob("*.md") if ".git" not in p.parts]) == 9
    if not md_locked:
        errors.append("md_file_count_not_locked_at_9")

    zero_ai_const_ok = all(
        tok in const_txt for tok in ("zero_ai_inside_compiler: true", "EXTERNAL_USER_ONLY", "零 AI 纯确定性编译铁律")
    )
    if not zero_ai_const_ok:
        errors.append("zero_ai_constitution_lock_missing")

    core_tokens = (
        "类 C 5-Primitive 内核封闭定理",
        "收敛于控制结构而非开放业务领域",
        "反例压力测试集",
        "core_closure_locked",
    )
    core_entry = entries.get("COMP-C-LIKE-5PRIM-CORE-CLOSURE", {})
    core_closure_locked = bool(
        all(t in l1_txt for t in core_tokens)
        and core_entry.get("status") == "ACTIVE"
        and core_entry.get("arch_layer") == "LAYER_C_L1_DETERMINISTIC_KERNEL"
        and md_locked
        and zero_ai_const_ok
    )
    if not core_closure_locked:
        errors.append("core_closure_locked_failed")

    dlf_tokens = (
        "DLF 形式化 BNF 与充要判定边界",
        "DLF_Program ::=",
        "TransitionDecl ::=",
        "EffectAuthClause ::=",
        "ExecutableDeterministic",
        "PureDeclarativeContract",
        "HaltedOutsideDlf",
    )
    dlf_entry = entries.get("EXEC-DLF-FORMAL-BNF-BOUNDARY", {})
    dlf_formal_spec_present = bool(
        all(t in l1_txt for t in dlf_tokens)
        and dlf_entry.get("status") == "ACTIVE"
        and dlf_entry.get("arch_layer") == "LAYER_D_L2_L3_PHYSICAL_EXECUTION"
    )
    if not dlf_formal_spec_present:
        errors.append("dlf_formal_spec_present_failed")

    five_gate_tokens = (
        "5-Gate 双时态与 Chisholm 归属裁决",
        "FIVE_GATE_ADJUDICATION",
        "M1_STANDARD_CALCULUS_LIB",
        "BitemporalFactLedger",
        "ChisholmDeonticEngine",
    )
    fg_entry = entries.get("CALC-5GATE-BITEMPORAL-CHISHOLM-ADJUDICATION", {})
    five_gate_adjudication_present = bool(
        all(t in l1_txt for t in five_gate_tokens)
        and fg_entry.get("status") == "ACTIVE"
        and fg_entry.get("arch_layer") == "LAYER_C_L1_DETERMINISTIC_KERNEL"
    )
    if not five_gate_adjudication_present:
        errors.append("five_gate_adjudication_present_failed")

    active_cnt = sum(1 for e in manifest.get("entries", []) if e.get("status") == "ACTIVE")
    raw_ver = str(manifest.get("manifest_version", "v0")).lstrip("v")
    major_ver = int(raw_ver.split(".")[0]) if raw_ver.split(".")[0].isdigit() else 0
    manifest_v33_upgraded = bool(major_ver >= 33 and active_cnt >= 82)
    if not manifest_v33_upgraded:
        errors.append(f"manifest_v33_upgrade_failed:{manifest.get('manifest_version')}:{active_cnt}")

    return {
        "ok": bool(
            core_closure_locked
            and dlf_formal_spec_present
            and five_gate_adjudication_present
            and manifest_v33_upgraded
            and not errors
        ),
        "core_closure_locked": core_closure_locked,
        "dlf_formal_spec_present": dlf_formal_spec_present,
        "five_gate_adjudication_present": five_gate_adjudication_present,
        "manifest_v33_upgraded": manifest_v33_upgraded,
        "md_file_count_locked_at_9": md_locked,
        "errors": errors,
    }



def audit_v34_unification_and_lean_synthesis() -> dict:
    """v34.0 conceptual unification, M0/M1 vs L0-L3 orthogonal map & Lean 4 de Bruijn synthesis gate."""
    l1_txt = (REPO_ROOT / "docs" / "whitepapers" / "world_cognition_unified_model.md").read_text(encoding="utf-8")
    const_txt = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
    mpath = REPO_ROOT / "docs" / "architecture" / "THEORY_PRESERVATION_MANIFEST.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    entries = {e.get("id"): e for e in manifest.get("entries", [])}
    errors: list[str] = []

    md_locked = len([p for p in REPO_ROOT.rglob("*.md") if ".git" not in p.parts]) == 9
    if not md_locked:
        errors.append("md_file_count_not_locked_at_9")

    ortho_tokens = (
        "### 3.1G",
        "正交坐标清晰分离",
        "L0–L3",
        "空间架构栈",
        "M0 / M1",
        "演化代际",
        "v33.0→v34.0",
    )
    ortho_entry = entries.get("ARCH-CONCEPTUAL-UNIFICATION-ORTHOGONAL-MAP", {})
    orthogonal_map_verified = bool(
        all(t in l1_txt for t in ortho_tokens)
        and ortho_entry.get("status") == "ACTIVE"
        and ortho_entry.get("arch_layer") == "LAYER_A_META_WORLD"
    )
    if not orthogonal_map_verified:
        errors.append("orthogonal_map_verification_failed")

    triad_tokens = (
        "极简三联统一压缩视图",
        "内核本体",
        "物理边界",
        "离散有限 + 离线立法 + 诚实停机",
        "运行与演化护栏",
        "日常写 `.gift` 完全无需背诵",
    )
    triad_unification_verified = bool(all(t in l1_txt for t in triad_tokens))
    if not triad_unification_verified:
        errors.append("triad_unification_verification_failed")

    lean_tokens = (
        "Lean 4 第一性原理悟性融合",
        "de Bruijn",
        "kernel_checker",
        "PTXLean4",
        "非空真性",
        "证书可复验",
        "依赖类型",
        "宇宙多态",
        "Mathlib",
    )
    lean_entry = entries.get("EXEC-LEAN4-DEBRUIJN-CHECKER-SYNTHESIS", {})
    lean4_debruijn_synthesis_verified = bool(
        all(t in l1_txt for t in lean_tokens)
        and lean_entry.get("status") == "ACTIVE"
        and lean_entry.get("arch_layer") == "LAYER_D_L2_L3_PHYSICAL_EXECUTION"
    )
    if not lean4_debruijn_synthesis_verified:
        errors.append("lean4_debruijn_synthesis_verification_failed")

    const_v34_tokens = (
        "v34.0",
        "极简三联",
        "正交坐标",
        "de Bruijn",
        "kernel_checker",
        "PTXLean4",
        "非空真性",
        "依赖类型",
        "Mathlib",
        "zero_ai_inside_compiler: true",
        "EXTERNAL_USER_ONLY",
        "v34_unification_and_lean_synthesis",
    )
    constitution_v34_synced = bool(all(t in const_txt for t in const_v34_tokens))
    if not constitution_v34_synced:
        errors.append("constitution_v34_sync_failed")

    active_cnt = sum(1 for e in manifest.get("entries", []) if e.get("status") == "ACTIVE")
    raw_ver = str(manifest.get("manifest_version", "v0")).lstrip("v")
    major_ver = int(raw_ver.split(".")[0]) if raw_ver.split(".")[0].isdigit() else 0
    manifest_v34_upgraded = bool(major_ver >= 34 and active_cnt >= 84)
    if not manifest_v34_upgraded:
        errors.append(f"manifest_v34_upgrade_failed:{manifest.get('manifest_version')}:{active_cnt}")

    return {
        "ok": bool(
            md_locked
            and orthogonal_map_verified
            and triad_unification_verified
            and lean4_debruijn_synthesis_verified
            and constitution_v34_synced
            and manifest_v34_upgraded
            and not errors
        ),
        "orthogonal_map_verified": orthogonal_map_verified,
        "triad_unification_verified": triad_unification_verified,
        "lean4_debruijn_synthesis_verified": lean4_debruijn_synthesis_verified,
        "constitution_v34_synced": constitution_v34_synced,
        "manifest_v34_upgraded": manifest_v34_upgraded,
        "md_file_count_locked_at_9": md_locked,
        "errors": errors,
    }



def audit_v35_wheel_first_and_pl_landscape() -> dict:
    """v35.0 Wheel-First Anti-NIH loop, Lean 4 tiered boundary correction & 10-benchmark PL landscape gate."""
    l1_txt = (REPO_ROOT / "docs" / "whitepapers" / "world_cognition_unified_model.md").read_text(encoding="utf-8")
    const_txt = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
    mpath = REPO_ROOT / "docs" / "architecture" / "THEORY_PRESERVATION_MANIFEST.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    entries = {e.get("id"): e for e in manifest.get("entries", [])}
    errors: list[str] = []

    md_locked = len([p for p in REPO_ROOT.rglob("*.md") if ".git" not in p.parts]) == 9
    if not md_locked:
        errors.append("md_file_count_not_locked_at_9")

    lean_tiered_tokens = (
        "日常业务编译热路径",
        "离线元证明",
        "交互式 Tactic",
        "Refinement Types",
        "Verus",
        "Dafny",
    )
    sec_31g = l1_txt.split("### 3.1G", 1)[1].split("### 3.1H", 1)[0] if ("### 3.1G" in l1_txt and "### 3.1H" in l1_txt) else ""
    rule34_sec = const_txt.split("## Rule 34:", 1)[1] if "## Rule 34:" in const_txt else ""
    lean4_tiered_correction_verified = bool(
        all(t in sec_31g for t in lean_tiered_tokens)
        and all(t in rule34_sec for t in lean_tiered_tokens)
        and "坚决拒绝（3 项）" not in sec_31g
        and "坚决拒绝依赖类型" not in rule34_sec
    )
    if not lean4_tiered_correction_verified:
        errors.append("lean4_tiered_correction_failed")

    const_v35_tokens = (
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
        "zero_ai_inside_compiler: true",
        "EXTERNAL_USER_ONLY",
        "v35_wheel_first_and_pl_landscape",
    )
    wheel_first_constitution_synced = bool(all(t in const_txt for t in const_v35_tokens))
    if not wheel_first_constitution_synced:
        errors.append("wheel_first_constitution_sync_failed")

    pl_tokens = (
        "### 3.1H",
        "轮子优先（Wheel-First）五步工作循环与反 NIH 铁律",
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
        "v34.0→v35.0",
    )
    pl_landscape_ten_benchmarks_verified = bool(all(t in l1_txt for t in pl_tokens))
    if not pl_landscape_ten_benchmarks_verified:
        errors.append("pl_landscape_ten_benchmarks_failed")

    expected_v35_entries = {
        "WORKFLOW-WHEEL-FIRST-ANTI-NIH-LAW": "LAYER_A_META_WORLD",
        "SURF-RUST-PURE-SUBSET-BRIDGE-SYN": "LAYER_B_L0_INTENT_SURFACE",
        "COMP-PL-FORMAL-TEN-BENCHMARK-LANDSCAPE": "LAYER_C_L1_DETERMINISTIC_KERNEL",
    }
    entries_ok = all(
        entries.get(eid, {}).get("status") == "ACTIVE" and entries.get(eid, {}).get("arch_layer") == layer
        for eid, layer in expected_v35_entries.items()
    )
    active_cnt = sum(1 for e in manifest.get("entries", []) if e.get("status") == "ACTIVE")
    raw_ver = str(manifest.get("manifest_version", "v0")).lstrip("v")
    major_ver = int(raw_ver.split(".")[0]) if raw_ver.split(".")[0].isdigit() else 0
    manifest_v35_upgraded = bool(major_ver >= 35 and active_cnt >= 87 and entries_ok)
    if not manifest_v35_upgraded:
        errors.append(f"manifest_v35_upgrade_failed:{manifest.get('manifest_version')}:{active_cnt}:{entries_ok}")

    return {
        "ok": bool(
            md_locked
            and lean4_tiered_correction_verified
            and wheel_first_constitution_synced
            and pl_landscape_ten_benchmarks_verified
            and manifest_v35_upgraded
            and not errors
        ),
        "lean4_tiered_correction_verified": lean4_tiered_correction_verified,
        "wheel_first_constitution_synced": wheel_first_constitution_synced,
        "pl_landscape_ten_benchmarks_verified": pl_landscape_ten_benchmarks_verified,
        "manifest_v35_upgraded": manifest_v35_upgraded,
        "md_file_count_locked_at_9": md_locked,
        "errors": errors,
    }


def audit_cross_repo_governance_sync(sec_repo_root: Path | None = None, allow_missing: bool = False) -> dict:
    """Cross-repo governance sync gate verifying sec_repo M0_BOUNDARY_AXIOMS.md,
    BOUNDARY_MANIFEST.json (Rust-Only M1+ & JS M0 Freeze), and CLI protocols."""
    import os
    sec_root = sec_repo_root or Path(os.environ.get("SEC_REPO_ROOT", str(REPO_ROOT.parent / "sec_repo")))
    const_text = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")
    ai_const_errors = [
        f"ai_constitution_missing:{ctok}"
        for ctok in ("Rust-Only", "零 AI 纯确定性编译铁律", "EXTERNAL_USER_ONLY")
        if ctok not in const_text
    ]
    if not sec_root.exists():
        errs = list(ai_const_errors)
        if not allow_missing:
            errs.append("sec_repo_not_found")
        return {
            "ok": len(errs) == 0,
            "mode": "standalone_ai_repo_only",
            "sec_repo_found": False,
            "sec_repo_root": str(sec_root),
            "ai_constitution_zero_ai_enforced": len(ai_const_errors) == 0,
            "errors": errs,
        }
    errors = []
    axioms_path = sec_root / "crates" / "gift-author-kernel" / "M0_BOUNDARY_AXIOMS.md"
    bm_path = sec_root / "BOUNDARY_MANIFEST.json"
    main_rs_path = sec_root / "crates" / "gift-author-kernel" / "src" / "main.rs"
    const_text = (REPO_ROOT / "SYSTEM_CONSTITUTION.md").read_text(encoding="utf-8")

    for req_p in (axioms_path, bm_path, main_rs_path):
        if not req_p.exists():
            errors.append(f"missing_sec_file:{req_p.name}")

    if not errors:
        ax_text = axioms_path.read_text(encoding="utf-8")
        for tok in ("mathcal{A}_1", "R1–R6", "Rust-Only", "FROZEN_COMPAT_ONLY", "Zero-AI-Inside-Compiler Iron Law", "EXTERNAL_USER_ONLY"):
            if tok not in ax_text:
                errors.append(f"sec_axioms_missing:{tok}")
        for ctok in ("Rust-Only", "零 AI 纯确定性编译铁律", "EXTERNAL_USER_ONLY"):
            if ctok not in const_text:
                errors.append(f"ai_constitution_missing:{ctok}")

        bm = json.loads(bm_path.read_text(encoding="utf-8"))
        pol = bm.get("target_evolution_policy", {})
        if pol.get("m0_js_policy") != "FROZEN_COMPAT_ONLY":
            errors.append("sec_manifest_m0_js_policy_not_frozen")
        if pol.get("m1_primary_target") != "rust_only":
            errors.append("sec_manifest_m1_primary_target_not_rust_only")
        if pol.get("m1_js_expansion") != "FORBIDDEN":
            errors.append("sec_manifest_m1_js_expansion_not_forbidden")
        if pol.get("zero_ai_inside_compiler") is not True:
            errors.append("sec_manifest_zero_ai_inside_compiler_not_true")
        if pol.get("ai_role") != "EXTERNAL_USER_ONLY":
            errors.append("sec_manifest_ai_role_not_external_user_only")
        if pol.get("architecture_graph_projection") != "DETERMINISTIC_BTREE_MERMAID":
            errors.append("sec_manifest_graph_projection_not_deterministic")

        main_text = main_rs_path.read_text(encoding="utf-8")
        for flag in ("--emit-m0", "--emit-m0-rust-only", "--compile-gift"):
            if flag not in main_text:
                errors.append(f"sec_cli_missing_flag:{flag}")

    return {
        "ok": len(errors) == 0,
        "sec_repo_found": True,
        "sec_repo_root": str(sec_root),
        "rust_only_m1_enforced": len(errors) == 0,
        "zero_ai_enforced": len(errors) == 0,
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Sovereign AI Brain CTO Repo Doctor ({REPO_VERSION})")
    ap.add_argument("--run-tests", action="store_true", help="Also run full 3-tier unittest suite")
    ap.add_argument("--sec-repo", type=str, default=None, help="Path to sibling sec_repo for cross-repo governance sync")
    ap.add_argument("--allow-missing-sec-repo", action="store_true", help="Allow skipping cross-repo check when sec_repo is absent (standalone default)")
    ap.add_argument("--require-sec-repo", action="store_true", help="Hard-fail if sibling sec_repo is absent")
    ap.add_argument("--compile-adr", action="store_true", help="Compile ARCHITECTURE_DECISIONS.md from ARCHITECTURE_GRAPH.json")
    ap.add_argument("--audit-table", action="store_true", help="Print zero-processing full closure audit table directly from ARCHITECTURE_GRAPH.json")
    ap.add_argument("--trace", type=str, default=None, help="Reverse-lookup any ADR, class, test, or arXiv ID in ARCHITECTURE_GRAPH.json")
    ap.add_argument("--check-arxiv", nargs="+", default=None, help="Pre-ingestion arXiv ID deduplication gate against all existing refs (Rule 25)")
    ap.add_argument("--audit-freshness", action="store_true", help="Run live online 3-pillar freshness audit (GitHub API + HF Hub API + arXiv ratio, Rule 27)")
    args = ap.parse_args()

    if args.compile_adr:
        data = json.loads((REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_GRAPH.json").read_text(encoding="utf-8"))
        out = compile_adr_markdown(data)
        (REPO_ROOT / "docs" / "architecture" / "ARCHITECTURE_DECISIONS.md").write_text(out, encoding="utf-8")
        print("Compiled docs/architecture/ARCHITECTURE_DECISIONS.md from ARCHITECTURE_GRAPH.json")
        return 0

    if args.audit_table:
        return print_audit_table()

    if args.trace:
        return trace_query(args.trace)

    if args.check_arxiv:
        return check_arxiv_duplicates(args.check_arxiv)

    ast_res = audit_zero_dead_code()
    ssot_res = audit_ssot_governance()
    fresh_res = audit_three_pillar_freshness(online=args.audit_freshness)
    slim_res = audit_anti_drift_and_slimming()
    theory_res = audit_theory_preservation_manifest()
    engine_res = audit_executable_world_engine()
    v33_res = audit_v33_convergence_closure()
    v34_res = audit_v34_unification_and_lean_synthesis()
    v35_res = audit_v35_wheel_first_and_pl_landscape()
    allow_missing_sec = True if (args.allow_missing_sec_repo or not args.require_sec_repo) else False
    cross_repo_res = audit_cross_repo_governance_sync(Path(args.sec_repo) if args.sec_repo else None, allow_missing=allow_missing_sec)
    tests_ok = True
    if args.run_tests:
        proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"], cwd=REPO_ROOT)
        tests_ok = proc.returncode == 0

    report = {
        "repo_version": REPO_VERSION,
        "ast_zero_dead_code": ast_res,
        "ssot_governance": ssot_res,
        "freshness_and_anti_rot": fresh_res,
        "anti_drift_and_slimming": slim_res,
        "theory_preservation_manifest": theory_res,
        "executable_world_engine": engine_res,
        "v33_convergence_closure": v33_res,
        "v34_unification_and_lean_synthesis": v34_res,
        "v35_wheel_first_and_pl_landscape": v35_res,
        "cross_repo_governance_sync": cross_repo_res,
        "tests_passed": tests_ok if args.run_tests else "skipped",
        "overall_healthy": bool(ast_res["ok"] and ssot_res["ok"] and fresh_res["ok"] and slim_res["ok"] and theory_res["ok"] and engine_res["ok"] and v33_res["ok"] and v34_res["ok"] and v35_res["ok"] and cross_repo_res["ok"] and tests_ok),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["overall_healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
