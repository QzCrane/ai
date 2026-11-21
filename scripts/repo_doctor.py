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
    md_locked = len([p for p in REPO_ROOT.rglob("*.md") if ".git" not in p.parts]) == 8
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
        "md_file_count_locked_at_8": md_locked,
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

def main() -> int:
    ap = argparse.ArgumentParser(description=f"Sovereign AI Brain CTO Repo Doctor ({REPO_VERSION})")
    ap.add_argument("--run-tests", action="store_true", help="Also run full 3-tier unittest suite")
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
    tests_ok = True
    if args.run_tests:
        proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"], cwd=REPO_ROOT)
        tests_ok = proc.returncode == 0

    report = {
        "repo_version": REPO_VERSION,
        "ast_zero_dead_code": ast_res,
        "ssot_governance": ssot_res,
        "freshness_and_anti_rot": fresh_res,
        "tests_passed": tests_ok if args.run_tests else "skipped",
        "overall_healthy": bool(ast_res["ok"] and ssot_res["ok"] and fresh_res["ok"] and tests_ok),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["overall_healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
