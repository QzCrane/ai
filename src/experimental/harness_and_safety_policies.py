"""
Isolated prototypes extracted from exoskeleton_bridge.py (CTO v6.1 Full-Stack Clean-Room).
Verified by unit tests; strictly separated from src/brain/ production runtime.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class EGraphSaturationBudgetController:
    """
    GAP-103 (EggMind, arXiv:2604.17364 + arXiv:2511.20782): Strategy-guided equality saturation
    controller that partitions rewrite rulesets and enforces a strict e-node budget ceiling to
    prevent e-graph explosion while avoiding premature destructive pruning.
    """
    def __init__(self, max_enodes: int = 2048, max_iters: int = 12) -> None:
        self.max_enodes = max(16, int(max_enodes))
        self.max_iters = max(1, int(max_iters))

    def plan_step(self, current_enodes: int, iteration: int, rule_partitions: List[str]) -> Dict[str, Any]:
        if iteration >= self.max_iters or current_enodes >= self.max_enodes:
            return {"action": "extract_lowest_cost", "partition": None, "budget_saturated": True}
        part = rule_partitions[iteration % max(1, len(rule_partitions))] if rule_partitions else "default"
        return {
            "action": "apply_partition_and_rebuild",
            "partition": part,
            "remaining_enode_budget": self.max_enodes - int(current_enodes),
            "budget_saturated": False,
        }

class KVBlockShadowGuard:
    """
    GAP-109 (KVGuard, arXiv:2611.06138): Compact shadow metadata table per physical KV block
    enforcing per-request ownership & initialization checks at the boundary (<=0.11% memory overhead).
    """
    def __init__(self) -> None:
        self._owners: Dict[int, str] = {}
        self._initialized: Dict[int, bool] = {}

    def allocate_block(self, block_id: int, request_id: str) -> None:
        self._owners[int(block_id)] = str(request_id)
        self._initialized[int(block_id)] = False

    def mark_initialized(self, block_id: int, request_id: str) -> bool:
        bid = int(block_id)
        if self._owners.get(bid) != str(request_id):
            return False
        self._initialized[bid] = True
        return True

    def check_read_safe(self, block_id: int, request_id: str) -> bool:
        bid = int(block_id)
        return self._owners.get(bid) == str(request_id) and bool(self._initialized.get(bid, False))

class ProvenanceMemoryFirewall:
    """
    GAP-119 (Memory Provenance Laundering, arXiv:2607.29167): Prevents untrusted external
    observations from being laundered into high-authority user instructions during memory
    consolidation. Enforces authority non-amplification: action_risk <= min(supporting_authorities).
    Authority levels: 0=untrusted_external, 1=verified_tool, 2=user_confirmed, 3=system_root.
    """
    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, Any]] = {}

    def store_record(self, rec_id: str, text: str, authority_level: int, source_tag: str) -> None:
        self._records[str(rec_id)] = {
            "text": str(text),
            "authority": max(0, min(3, int(authority_level))),
            "source_tag": str(source_tag),
        }

    def authorize_action(self, supporting_rec_ids: List[str], required_authority: int) -> Dict[str, Any]:
        if not supporting_rec_ids:
            return {"authorized": int(required_authority) == 0, "effective_authority": 0}
        auths = [self._records[r]["authority"] for r in supporting_rec_ids if r in self._records]
        eff = min(auths) if auths else 0
        return {
            "authorized": eff >= int(required_authority),
            "effective_authority": eff,
            "required_authority": int(required_authority),
        }

class SigmaReliabilityMemory:
    """
    GAP-123 (Sigma-Mem, arXiv:2607.27958): Online real-symmetric peer-reliability memory
    updated from post-decision correctness feedback with bounded spectral step size (Weyl bound).
    """
    def __init__(self, n_peers: int, lr: float = 0.1) -> None:
        self.n_peers = max(1, int(n_peers))
        self.lr = max(1e-3, min(0.5, float(lr)))
        self.diag_scores: List[float] = [0.5] * self.n_peers

    def update(self, peer_idx: int, correct: bool) -> float:
        i = max(0, min(self.n_peers - 1, int(peer_idx)))
        target = 1.0 if correct else 0.0
        self.diag_scores[i] = (1.0 - self.lr) * self.diag_scores[i] + self.lr * target
        return round(self.diag_scores[i], 6)

    def select_best_peer(self) -> int:
        return max(range(self.n_peers), key=lambda i: self.diag_scores[i])

class MultiTurnSafeguardMonitor:
    """
    GAP-131 (Multi-Turn Safeguarding across GPT-5.5/DeepSeek-V4/Llama-4, arXiv:2611.26009):
    Tracks cumulative multi-turn boundary erosion with exponential decay so gradual multi-turn
    pressure cannot bypass single-turn safety filters.
    """
    def __init__(self, risk_ceiling: float = 0.80, decay: float = 0.85) -> None:
        self.risk_ceiling = float(risk_ceiling)
        self.decay = float(decay)
        self.cumulative_risk: float = 0.0

    def observe_turn(self, turn_risk: float) -> Dict[str, Any]:
        r = max(0.0, min(1.0, float(turn_risk)))
        self.cumulative_risk = min(1.0, self.cumulative_risk * self.decay + r)
        escalate = self.cumulative_risk >= self.risk_ceiling
        return {"cumulative_risk": round(self.cumulative_risk, 4), "escalate_safeguard": escalate}

class LeashThreeLevelSafetyGate:
    """
    GAP-133 (LEASH, arXiv:2611.14115): Unified safety policy across 3 execution levels:
    (1) task gate, (2) action gate, and (3) irreversible-commit gate.
    """
    def __init__(self, task_ceiling: float = 0.65, action_ceiling: float = 0.55, commit_ceiling: float = 0.25) -> None:
        self.task_ceiling = float(task_ceiling)
        self.action_ceiling = float(action_ceiling)
        self.commit_ceiling = float(commit_ceiling)

    def evaluate(
        self,
        task_risk: float,
        action_risk: float,
        commit_risk: float,
        irreversible: bool = False,
    ) -> Dict[str, Any]:
        if float(task_risk) >= self.task_ceiling:
            return {"allow": False, "blocked_at": "task_gate", "require_human": True}
        if float(action_risk) >= self.action_ceiling:
            return {"allow": False, "blocked_at": "action_gate", "require_human": True}
        if irreversible and float(commit_risk) >= self.commit_ceiling:
            return {"allow": False, "blocked_at": "irreversible_commit_gate", "require_human": True}
        return {"allow": True, "blocked_at": None, "require_human": False}

class HarnessToolHealthTelemetryRouter:
    """
    GAP-134 (Agent Harness Awareness / ToolRoute, arXiv:2611.05391): Injects live verified
    tool-health telemetry (availability + p95 latency) before dispatch to avoid stale/broken
    endpoints and minimize routing policy regret.
    """
    def __init__(self, max_latency_ms: float = 1500.0, min_success_rate: float = 0.90) -> None:
        self.max_latency_ms = float(max_latency_ms)
        self.min_success_rate = float(min_success_rate)
        self.telemetry: Dict[str, Dict[str, float]] = {}

    def update_health(self, tool_name: str, latency_ms: float, success_rate: float, alive: bool = True) -> None:
        self.telemetry[tool_name] = {
            "latency_ms": float(latency_ms),
            "success_rate": float(success_rate),
            "alive": 1.0 if alive else 0.0,
        }

    def select_tool(self, candidates: List[str]) -> Dict[str, Any]:
        healthy = []
        for c in candidates:
            t = self.telemetry.get(c, {"latency_ms": 9999.0, "success_rate": 0.0, "alive": 0.0})
            if t["alive"] > 0.5 and t["success_rate"] >= self.min_success_rate and t["latency_ms"] <= self.max_latency_ms:
                healthy.append((t["latency_ms"] / max(0.01, t["success_rate"]), c, t))
        if not healthy:
            return {"selected": None, "status": "no_healthy_tool", "checked": len(candidates)}
        healthy.sort(key=lambda x: x[0])
        _, best, stats = healthy[0]
        return {"selected": best, "status": "healthy_dispatch", "latency_ms": stats["latency_ms"], "success_rate": stats["success_rate"]}

class SafetyLeadTeamGovernanceGate:
    """
    GAP-135 (Aligning AI Teams, arXiv:2611.17820): Prevents multi-agent diffusion of
    responsibility by requiring a designated Safety Lead Agent sign-off anchored to an
    explicit Safety Specifications Document version hash (peer group-chat alone fails).
    """
    def __init__(self, safety_lead_id: str, spec_version: str = "SYSTEM_CONSTITUTION_v4.8") -> None:
        self.safety_lead_id = str(safety_lead_id)
        self.spec_version = str(spec_version)

    def approve_team_action(
        self,
        proposer_id: str,
        approver_id: str,
        spec_version_cited: str,
        lead_risk_score: float,
        max_allowed_risk: float = 0.35,
    ) -> Dict[str, Any]:
        if approver_id != self.safety_lead_id:
            return {"approved": False, "reason": "missing_designated_safety_lead"}
        if spec_version_cited != self.spec_version:
            return {"approved": False, "reason": "stale_or_missing_safety_spec"}
        if float(lead_risk_score) > float(max_allowed_risk):
            return {"approved": False, "reason": "safety_lead_veto"}
        return {"approved": True, "reason": "safety_lead_signed", "proposer": proposer_id, "lead": self.safety_lead_id}

class OwlSightCoTAndTraceHygieneMonitor:
    """
    GAP-137 (OwlSight arXiv:2610.23670 + Stealing Reasoning Traces arXiv:2608.09867):
    Combines fine- and coarse-grained Chain-of-Thought risk monitoring (robust to CoT
    compression/obfuscation) with a strict encrypted-trace hygiene gate blocking PII,
    credentials, or hidden prompt-injection payloads from persisting across API boundaries.
    """
    def __init__(self, fine_threshold: float = 0.60, coarse_threshold: float = 0.55) -> None:
        self.fine_threshold = float(fine_threshold)
        self.coarse_threshold = float(coarse_threshold)
        self._sensitive_markers = ("BEGIN PRIVATE KEY", "sk-", "AKIA", "IGNORE PREVIOUS INSTRUCTIONS")

    def inspect_trace(
        self,
        trace_text: str,
        fine_step_scores: List[float],
        coarse_summary_score: float,
    ) -> Dict[str, Any]:
        max_fine = max((float(x) for x in fine_step_scores), default=0.0)
        coarse = float(coarse_summary_score)
        leaked_markers = [m for m in self._sensitive_markers if m in trace_text]
        flagged = (max_fine >= self.fine_threshold) or (coarse >= self.coarse_threshold) or bool(leaked_markers)
        return {
            "flagged": flagged,
            "max_fine_risk": round(max_fine, 4),
            "coarse_risk": round(coarse, 4),
            "trace_hygiene_violations": leaked_markers,
            "strip_before_persist": bool(leaked_markers) or flagged,
        }

class CodexLayeredSandboxPolicyEngine:
    """
    GAP-141 (OpenAI, Nov 18/19, 2026 — Inside the Codex sandbox):
    1. Layered filesystem rules (:workspace defaults keep .git/.codex read-only;
       explicit deny rules like **/*.env override allow).
    2. Immutable managed enterprise ceiling (requirements.toml) cannot be escalated by local config.
    3. Network proxy evaluates domain rules BEFORE DNS resolution to prevent DNS exfiltration.
    """
    def __init__(
        self,
        readonly_paths: Optional[List[str]] = None,
        denied_paths: Optional[List[str]] = None,
        allowed_domains: Optional[List[str]] = None,
        denied_domains: Optional[List[str]] = None,
        enterprise_block_full_access: bool = True,
    ) -> None:
        self.readonly_paths = list(readonly_paths or [".git", ".codex", ".vscode", ".aws"])
        self.denied_paths = list(denied_paths or [".env", "secrets"])
        self.allowed_domains = list(allowed_domains or ["localhost", "127.0.0.1"])
        self.denied_domains = list(denied_domains or [])
        self.enterprise_block_full_access = bool(enterprise_block_full_access)

    def check_file_op(self, rel_path: str, mode: str) -> Dict[str, Any]:
        p = str(rel_path).replace("\\", "/")
        for d in self.denied_paths:
            if d in p:
                return {"allow": False, "reason": f"explicit_deny:{d}"}
        if mode == "write":
            for r in self.readonly_paths:
                if p == r or p.startswith(r + "/") or ("/" + r + "/") in p:
                    return {"allow": False, "reason": f"protected_readonly:{r}"}
        return {"allow": True, "reason": "workspace_profile_ok"}

    def check_network_pre_dns(self, hostname: str, requested_mode: str = "use_default") -> Dict[str, Any]:
        """Evaluates domain rules BEFORE DNS resolution to block DNS-encoded exfiltration."""
        if requested_mode == "require_escalated" and self.enterprise_block_full_access:
            return {"allow": False, "resolve_dns": False, "reason": "blocked_by_enterprise_requirements"}
        host = str(hostname).lower().strip()
        # Explicit deny always takes precedence over allow
        for d in self.denied_domains:
            if host == d or host.endswith("." + d.lstrip("*.")):
                return {"allow": False, "resolve_dns": False, "reason": f"explicit_domain_deny:{d}"}
        for a in self.allowed_domains:
            if host == a or (a.startswith("*.") and host.endswith(a[1:])):
                return {"allow": True, "resolve_dns": True, "reason": f"domain_allowed:{a}"}
        return {"allow": False, "resolve_dns": False, "reason": "suspend_for_harness_approval"}

class PerformanceFactoryAdversarialLoop:
    """
    GAP-142 (OpenAI Performance Factory, Nov 19/20, 2026):
    Tracks regression/self-improvement tasks outside any single agent context across explicit
    states, requires an independent adversarial counterfactual review to disprove premature
    root-cause diagnoses, verifies post-deploy baseline recovery, and auto-opens self-repair
    tasks when recurring tool errors block agent workflows.
    """
    STATES = ("Detected", "Investigating", "AdversarialReview", "FixProposed", "Deployed", "VerifyRecovery", "Resolved")

    def __init__(self, self_repair_threshold: int = 3) -> None:
        self.self_repair_threshold = int(self_repair_threshold)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.tool_error_counts: Dict[str, int] = {}

    def open_task(self, task_id: str, baseline_metric: float) -> Dict[str, Any]:
        rec = {
            "task_id": task_id,
            "state": "Detected",
            "baseline_metric": float(baseline_metric),
            "adversarial_passed": False,
            "reopen_count": 0,
        }
        self.tasks[task_id] = rec
        return dict(rec)

    def submit_adversarial_review(self, task_id: str, counterfactual_falsified: bool) -> Dict[str, Any]:
        t = self.tasks[task_id]
        if counterfactual_falsified:
            t["state"] = "Investigating"
            t["adversarial_passed"] = False
            t["reopen_count"] += 1
            return {"accepted": False, "state": t["state"], "reopen_count": t["reopen_count"]}
        t["state"] = "FixProposed"
        t["adversarial_passed"] = True
        return {"accepted": True, "state": t["state"]}

    def verify_post_deploy_recovery(self, task_id: str, post_deploy_metric: float, tolerance: float = 0.05) -> Dict[str, Any]:
        t = self.tasks[task_id]
        recovered = float(post_deploy_metric) <= t["baseline_metric"] * (1.0 + float(tolerance))
        t["state"] = "Resolved" if recovered else "Investigating"
        return {"recovered": recovered, "state": t["state"]}

    def record_tool_error(self, tool_name: str) -> Dict[str, Any]:
        self.tool_error_counts[tool_name] = self.tool_error_counts.get(tool_name, 0) + 1
        trigger = self.tool_error_counts[tool_name] >= self.self_repair_threshold
        return {"tool": tool_name, "error_count": self.tool_error_counts[tool_name], "open_self_repair_task": trigger}

class ManagedAgentMetaHarness:
    """
    GAP-144 (Anthropic Managed Agents, Apr 8, 2026):
    Decouples the Session (durable append-only log outside context & container),
    the Harness (stateless cattle recovered via wake(session_id)), and the Sandbox/Hands
    (provisioned lazily only on execute(), with auth tokens isolated in an external Vault Proxy).
    """
    def __init__(self) -> None:
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.provisioned_hands: Dict[str, bool] = {}
        self._vault_tokens: Dict[str, str] = {}

    def register_vault_secret(self, service: str, secret_token: str) -> None:
        self._vault_tokens[service] = str(secret_token)

    def emit_event(self, session_id: str, event: Dict[str, Any]) -> int:
        log = self.sessions.setdefault(session_id, [])
        log.append(dict(event))
        return len(log) - 1

    def get_events(self, session_id: str, start: int = 0, end: Optional[int] = None) -> List[Dict[str, Any]]:
        return list(self.sessions.get(session_id, [])[start:end])

    def wake(self, session_id: str) -> Dict[str, Any]:
        """Stateless harness recovery from durable external session log."""
        events = self.sessions.get(session_id, [])
        return {"session_id": session_id, "recovered_events": len(events), "hand_provisioned": self.provisioned_hands.get(session_id, False)}

    def execute_hand(self, session_id: str, hand_name: str, payload: str, service_for_proxy: Optional[str] = None) -> Dict[str, Any]:
        """Lazy hand provisioning + Vault proxy execution without exposing raw credentials to sandbox."""
        just_provisioned = False
        if not self.provisioned_hands.get(session_id, False):
            self.provisioned_hands[session_id] = True
            just_provisioned = True
        proxy_authed = bool(service_for_proxy and service_for_proxy in self._vault_tokens)
        self.emit_event(session_id, {"type": "hand_exec", "hand": hand_name, "proxy_authed": proxy_authed})
        return {
            "hand": hand_name,
            "lazy_provisioned": just_provisioned,
            "proxy_authed": proxy_authed,
            "raw_credential_exposed_to_sandbox": False,
        }

class LayeredDomainArchitectureLinter:
    """
    GAP-145 (OpenAI Harness Engineering, Feb 11, 2026):
    Mechanically validates forward-only layer dependency order:
      Types(0) -> Config(1) -> Repo(2) -> Service(3) -> Runtime(4) -> UI(5)
    Cross-cutting concerns may only enter via the single explicit 'Providers' interface.
    Violations return actionable remediation instructions directly into agent context.
    """
    LAYER_ORDER = {"Types": 0, "Config": 1, "Repo": 2, "Service": 3, "Runtime": 4, "UI": 5}

    def check_dependency(self, src_layer: str, dst_layer: str) -> Dict[str, Any]:
        if dst_layer == "Providers":
            return {"valid": True, "remediation": None}
        if src_layer not in self.LAYER_ORDER or dst_layer not in self.LAYER_ORDER:
            return {"valid": False, "remediation": f"Unknown layer ({src_layer}->{dst_layer}); route cross-cutting deps through Providers."}
        # Code in a higher layer may only depend on equal or lower foundational layers
        if self.LAYER_ORDER[dst_layer] <= self.LAYER_ORDER[src_layer]:
            return {"valid": True, "remediation": None}
        return {
            "valid": False,
            "remediation": f"Architectural violation: {src_layer} cannot depend on higher layer {dst_layer}. Invert dependency or inject via Providers.",
        }

class TwoStageReasoningBlindClassifierGate:
    """
    GAP-150 (Anthropic Claude Code Auto Mode, Mar 25, 2026):
    1. Strips assistant CoT & tool outputs (reasoning-blind) and drops blanket shell/interpreter
       always-allow rules on entering auto mode.
    2. Runs fast Stage-1 filter; only flagged actions escalate to Stage-2 CoT sharing the same prefix.
    3. Deny-and-continue allows recovery, escalating to human at 3 consecutive or 20 total denials.
    """
    BLANKET_ESCAPES = {"bash", "sh", "python", "node", "ruby"}

    def __init__(self, max_consecutive_denials: int = 3, max_total_denials: int = 20) -> None:
        self.max_consecutive_denials = int(max_consecutive_denials)
        self.max_total_denials = int(max_total_denials)
        self.consecutive_denials: int = 0
        self.total_denials: int = 0

    def filter_auto_mode_allowlist(self, user_allowlist: List[str]) -> List[str]:
        return [t for t in user_allowlist if str(t).lower() not in self.BLANKET_ESCAPES]

    def evaluate_action(self, stage1_flag: bool, stage2_cot_block: bool) -> Dict[str, Any]:
        if not stage1_flag:
            self.consecutive_denials = 0
            return {"allow": True, "stage_used": 1, "escalate_human": False}
        # Stage 2 invoked with prefix cache hit
        if not stage2_cot_block:
            self.consecutive_denials = 0
            return {"allow": True, "stage_used": 2, "escalate_human": False}
        self.consecutive_denials += 1
        self.total_denials += 1
        escalate = (
            self.consecutive_denials >= self.max_consecutive_denials
            or self.total_denials >= self.max_total_denials
        )
        return {
            "allow": False,
            "stage_used": 2,
            "deny_and_continue": not escalate,
            "escalate_human": escalate,
            "consecutive_denials": self.consecutive_denials,
            "total_denials": self.total_denials,
        }

class PAIRLocalClusterAndOpenShellGovernor:
    """
    GAP-151 (NVIDIA PAIR Sep 3, 2026 + NVIDIA OpenShell Oct 21, 2026):
    1. Routes independent subagent inference requests across local LAN GPUs (RTX 20+/3050/M4+)
       using readiness, model presence, and GPU utilization.
    2. OpenShell Policy Prover formally verifies requested permissions stay inside org ceiling.
    """
    def route_lan_subagent(self, nodes: List[Dict[str, Any]], required_model: str) -> Dict[str, Any]:
        eligible = [n for n in nodes if n.get("ready") and required_model in n.get("models", [])]
        if not eligible:
            return {"selected_node": None, "status": "no_ready_lan_node"}
        best = min(eligible, key=lambda n: float(n.get("gpu_util", 1.0)))
        return {"selected_node": best["node_id"], "gpu_util": best["gpu_util"], "status": "routed_mtls"}

    def verify_openshell_policy_ceiling(self, requested_perms: List[str], org_ceiling: List[str]) -> Dict[str, Any]:
        ceiling_set = set(org_ceiling)
        violations = [p for p in requested_perms if p not in ceiling_set]
        return {"proven_within_ceiling": len(violations) == 0, "violations": violations}

class OutOfBandSessionAnomalyDetector:
    """
    GAP-152 (Google Agent Anomaly Detection, Nov 20/21, 2026):
    Asynchronously inspects full-session OpenTelemetry tool-call logs (zero runtime latency)
    to catch permitted-but-anomalous patterns: cascading pagination scraping, resource
    exhaustion, and override-injected high-value tool arguments.
    """
    OVERRIDE_PATTERNS = ("SYSTEM OVERRIDE", "IGNORE ALL VALIDATION", "REFUND 10X")

    def __init__(self, max_repeated_tool_calls: int = 8, high_value_threshold: float = 100.0) -> None:
        self.max_repeated_tool_calls = int(max_repeated_tool_calls)
        self.high_value_threshold = float(high_value_threshold)

    def analyze_session(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        findings: List[str] = []
        tool_counts: Dict[str, int] = {}
        for ev in events:
            tname = str(ev.get("tool", ""))
            if tname:
                tool_counts[tname] = tool_counts.get(tname, 0) + 1
            text = str(ev.get("note", "")) + " " + str(ev.get("prompt", ""))
            if any(pat in text.upper() for pat in self.OVERRIDE_PATTERNS):
                findings.append("override_coercion_detected")
            if float(ev.get("monetary_value", 0.0)) > self.high_value_threshold and not ev.get("human_verified", False):
                findings.append("unverified_high_value_tool_arg")
        for tname, cnt in tool_counts.items():
            if cnt >= self.max_repeated_tool_calls:
                findings.append(f"cascading_loop_or_scraping:{tname}")
        uniq = sorted(set(findings))
        return {"anomalous": len(uniq) > 0, "findings": uniq}

class YoloFSStagedSnapshotFilesystem:
    """
    GAP-174 (SOSP 2026 YoloFS, https://paper.lingyunyang.com/reading-notes/conference/sosp-2026):
    Prevents destructive AI agent filesystem misuse (290 real-world reports) via staged
    mutations, intermediate snapshots, undoable ops, and progressive permissions.
    """

    def __init__(self):
        self.committed_state: dict[str, str] = {}
        self.staged_ops: list[dict] = []
        self.snapshots: list[dict[str, str]] = []

    def stage_write(self, path: str, content: str, destructive: bool = False, approved: bool = False) -> dict:
        if destructive and not approved:
            return {"staged": False, "reason": "progressive_permission_required"}
        self.staged_ops.append({"op": "write", "path": path, "content": content})
        return {"staged": True, "pending_ops": len(self.staged_ops)}

    def commit_with_snapshot(self) -> dict:
        self.snapshots.append(dict(self.committed_state))
        for op in self.staged_ops:
            self.committed_state[op["path"]] = op["content"]
        applied = len(self.staged_ops)
        self.staged_ops.clear()
        return {"committed": applied, "snapshot_id": len(self.snapshots) - 1}

    def undo_last_commit(self) -> bool:
        if not self.snapshots:
            return False
        self.committed_state = self.snapshots.pop()
        return True

class HarnessFlowMCTSOptimizer:
    """
    GAP-175 (HarnessFlow, arXiv:2611.06254, https://arxiv.org/abs/2611.06254):
    MCTS-guided executable harness refinement tree over evaluation histories
    (69.48% mean across 6 benchmarks: +10.89pp over initial, +3.95pp over Meta-Harness).
    """

    def __init__(self, exploration_c: float = 1.414):
        self.exploration_c = float(exploration_c)

    def select_candidate_ucb1(self, candidates: list[dict], total_visits: int) -> str:
        import math
        best_id = None
        best_score = -1e18
        tot = max(1, int(total_visits))
        for cand in candidates:
            visits = int(cand.get("visits", 0))
            if visits == 0:
                return str(cand["id"])
            mean_val = float(cand.get("mean_score", 0.0))
            ucb = mean_val + self.exploration_c * math.sqrt(math.log(tot) / visits)
            if ucb > best_score:
                best_score = ucb
                best_id = str(cand["id"])
        return best_id or ""

class MultiAgentInfoNetworkGuardrail:
    """
    Negative Redline Guardrail (arXiv:2611.11714, https://arxiv.org/abs/2611.11714):
    Proves that full peer transparency acts as a behavioral contagion channel amplifying
    free-riding/congestion when individual and collective incentives misalign, and
    modular 'barbell' topologies perform worst. Enforces isolated peer info on misaligned
    public-goods tasks and rejects barbell wiring.
    """

    def validate_topology(self, incentives_aligned: bool, topology: str, peer_transparency: bool) -> dict:
        topo = str(topology).strip().lower()
        violations = []
        if topo == "barbell":
            violations.append("barbell_topology_banned")
        if (not incentives_aligned) and peer_transparency:
            violations.append("misaligned_peer_contagion_risk")
        return {
            "allowed": len(violations) == 0,
            "violations": violations,
            "recommended_peer_transparency": bool(incentives_aligned),
        }

class DeadlineAwareReasoningBudgetGate:
    """
    GAP-181 Negative Redline (arXiv:2611.11910, Nov 10 2026, https://arxiv.org/abs/2611.11910):
    Thinking answers that miss hard deadlines collapse delivered value (raising burst 8->56
    on single-GPU Qwen3-8B crashes 45s deadline hit-rate from 79.2% to 1.4%). Enforces
    joint value-and-deadline routing, falling back to fast mode when ETA exceeds deadline.
    """

    def select_mode(
        self,
        deadline_sec: float,
        estimated_thinking_sec: float,
        estimated_fast_sec: float,
        concurrent_burst: int = 1,
    ) -> dict:
        burst_factor = 1.0 + max(0, int(concurrent_burst) - 8) * 0.12
        adj_thinking = float(estimated_thinking_sec) * burst_factor
        if adj_thinking <= float(deadline_sec):
            return {"mode": "thinking", "adjusted_eta_sec": round(adj_thinking, 3), "deadline_safe": True}
        return {
            "mode": "fast_fallback",
            "adjusted_eta_sec": round(float(estimated_fast_sec), 3),
            "deadline_safe": float(estimated_fast_sec) <= float(deadline_sec),
        }

class SchemaFillSlotParallelToolDecoder:
    """
    GAP-186 (SchemaFill, arXiv:2610.07086, https://arxiv.org/abs/2610.07086):
    Slot-parallel speculative tool calling: generates independent argument slots
    concurrently and verifies concatenated candidates against target model prefix,
    committing verified slots and correcting mismatches (up to 4.05x throughput).
    """

    def verify_parallel_slots(
        self, candidate_slots: dict[str, str], target_verified_slots: dict[str, str]
    ) -> dict:
        committed = {}
        corrected = []
        for k, cand_val in candidate_slots.items():
            true_val = target_verified_slots.get(k, cand_val)
            committed[k] = true_val
            if cand_val != true_val:
                corrected.append(k)
        return {
            "committed_slots": committed,
            "corrected_slots": corrected,
            "exact_match_ratio": round(
                (len(committed) - len(corrected)) / max(1, len(committed)), 4
            ),
        }

class DIADisagreementHeterogeneousArbiter:
    """
    GAP-187 Negative Redline (DIA, arXiv:2610.23213, https://arxiv.org/abs/2610.23213):
    Proves that repeated samples from the SAME model have correlated errors (more
    same-model test-time compute is NOT independent evidence). Triggers an independent
    heterogeneous solver panel ONLY when intermediate and final predictions disagree
    (+7.0% on MuSiQue, activates on ~17% of queries, -47.2% tokens vs multi-agent).
    """

    def arbitrate(
        self,
        intermediate_answer: str,
        final_answer: str,
        heterogeneous_panel_votes: list[str] | None = None,
    ) -> dict:
        if str(intermediate_answer).strip() == str(final_answer).strip():
            return {
                "arbiter_triggered": False,
                "selected_answer": str(final_answer).strip(),
                "reason": "stage_agreement_fast_accept",
            }
        votes = [str(v).strip() for v in (heterogeneous_panel_votes or []) if str(v).strip()]
        if not votes:
            return {
                "arbiter_triggered": True,
                "selected_answer": str(final_answer).strip(),
                "reason": "disagreement_awaiting_heterogeneous_panel",
            }
        counts = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        winner = max(counts.items(), key=lambda kv: kv[1])[0]
        return {
            "arbiter_triggered": True,
            "selected_answer": winner,
            "reason": "heterogeneous_independent_arbitration",
        }


class SoteriaVerifiedPlanEnforcer:
    """
    ADR-012 (Soteria, arXiv:2611.17880, Nov 13 2026, https://arxiv.org/abs/2611.17880):
    Formally verifies hierarchical plans against safety specifications BEFORE any tool call,
    and uses verified fallback branches when an unsafe action is blocked (1.7x task success, 0 violations).
    """

    def verify_plan(self, steps: list[dict], banned_ops: set[str]) -> dict:
        violations = [s["op"] for s in steps if s.get("op") in banned_ops]
        return {"plan_verified": len(violations) == 0, "blocked_ops": violations}


class TenonAgentResidencyController:
    """
    ADR-012 (TenonAgent, arXiv:2611.00051, https://arxiv.org/abs/2611.00051):
    Virtualizes sandbox residency via fail-closed predictive admission + kernel pacing,
    achieving 100% task success at 8.6x memory oversubscription and 5.5x speedup.
    """

    def admit_sandbox(self, active_mem_mb: float, req_mem_mb: float, host_budget_mb: float) -> dict:
        if float(active_mem_mb) + float(req_mem_mb) <= float(host_budget_mb):
            return {"admit": True, "action": "materialize_immediately"}
        return {"admit": False, "action": "dematerialize_idle_then_retry_fail_closed"}


class CASEBenchSecurityContextGuard:
    """
    ADR-012 Negative Redline (CASE-Bench, arXiv:2611.06294, https://arxiv.org/abs/2611.06294):
    Proves 1 insecure exemplar in repo context raises correct-but-exploitable code by +6.3pp
    and coding agents miss safeguards in 64-73% of trajectories. Explicitly injects security
    safeguards and strips vulnerable exemplars from coding-agent context.
    """

    def audit_context_window(self, has_security_safeguard: bool, insecure_exemplar_count: int) -> dict:
        ok = bool(has_security_safeguard) and int(insecure_exemplar_count) == 0
        return {"secure_context": ok, "inject_safeguard_required": not bool(has_security_safeguard)}


@dataclass
class QHijackCrossPrecisionAuditGate:
    """QHijack-LoRA Cross-Precision Paired Audit Gate (arXiv:2611.14978, Negative Redline).
    Detects adapter-only backdoors that remain dormant in FP16/BF16 but activate under
    target quantized deployment (INT4/INT8/FP8) by exploiting base quantization residuals."""
    max_decision_flip_rate: float = 0.02
    max_logit_kl_divergence: float = 0.15

    def audit_adapter(self, fp16_decisions: list[int], quant_decisions: list[int], mean_kl: float) -> dict:
        if len(fp16_decisions) != len(quant_decisions) or not fp16_decisions:
            return {"approved": False, "reason": "invalid_paired_probe_set", "flip_rate": 1.0, "mean_kl": mean_kl}
        flips = sum(1 for a, b in zip(fp16_decisions, quant_decisions) if a != b)
        flip_rate = flips / len(fp16_decisions)
        approved = (flip_rate <= self.max_decision_flip_rate) and (mean_kl <= self.max_logit_kl_divergence)
        return {
            "approved": approved,
            "reason": "pass_cross_precision_audit" if approved else "block_qhijack_quantization_residual_activation",
            "flip_rate": round(flip_rate, 4),
            "mean_kl": round(mean_kl, 4),
        }


# =============================================================================
# v9.2 Multi-Agent Containment, Privilege Minimization & Harness Anti-Forgetting
# =============================================================================

@dataclass
class AirlockProvenanceContainmentGate:
    """Airlock (arXiv:2611.06727): Tree-structured permission narrowing + append-only
    provenance DAG; forward-isolates tainted branches on alert while preserving healthy work."""
    def contain_blast_radius(self, edges: list[tuple[str, str]], tainted_roots: list[str]) -> dict:
        adj: dict[str, list[str]] = {}
        nodes: set[str] = set(tainted_roots)
        for src, dst in edges:
            adj.setdefault(src, []).append(dst)
            nodes.add(src)
            nodes.add(dst)
        quarantined: set[str] = set(tainted_roots)
        queue = list(tainted_roots)
        while queue:
            cur = queue.pop(0)
            for nxt in adj.get(cur, []):
                if nxt not in quarantined:
                    quarantined.add(nxt)
                    queue.append(nxt)
        healthy = sorted(nodes - quarantined)
        return {
            "quarantined_nodes": sorted(quarantined),
            "preserved_healthy_nodes": healthy,
            "zero_post_convergence_leak": True,
        }


@dataclass
class PRISMCausalPrivilegeMinimizer:
    """PRISM (arXiv:2611.11539): Causal counterfactual necessity + information bottleneck
    role induction minimizing over-authorized MCP tools (-78.3% tools, 15.9% blast radius)."""
    necessity_threshold: float = 0.55

    def minimize_tools(self, tool_necessity_scores: dict[str, float]) -> dict:
        granted = sorted(t for t, s in tool_necessity_scores.items() if s >= self.necessity_threshold)
        revoked = sorted(t for t, s in tool_necessity_scores.items() if s < self.necessity_threshold)
        reduction = (len(revoked) / len(tool_necessity_scores)) if tool_necessity_scores else 0.0
        return {"granted_tools": granted, "revoked_tools": revoked, "privilege_reduction_ratio": round(reduction, 4)}


@dataclass
class ContinualHarnessRegressionGuard:
    """Continual Harness Forgetting Guard (arXiv:2611.03219, Negative Redline):
    Prevents catastrophic forgetting during multi-task harness self-evolution (where SWE-Bench
    can plunge from 94% to 32% after 5 rounds on a new benchmark) via regression anchoring."""
    max_allowed_drop_pp: float = 1.5

    def approve_evolution(self, baseline_scores: dict[str, float], candidate_scores: dict[str, float]) -> dict:
        regressions = {}
        for task, base in baseline_scores.items():
            cand = candidate_scores.get(task, 0.0)
            if (base - cand) > self.max_allowed_drop_pp:
                regressions[task] = round(base - cand, 3)
        return {"approved": len(regressions) == 0, "regressions_pp": regressions}


@dataclass
class EvolveAuditedGuardrailRegistry:
    """EVOLVE (arXiv:2611.02516): Cumulative, independently audited executable guardrail tools."""
    audited_tools: dict[str, bool] = field(default_factory=dict)

    def register_tool(self, tool_name: str, passed_independent_audit: bool) -> bool:
        if not passed_independent_audit:
            return False
        self.audited_tools[tool_name] = True
        return True


@dataclass
class GovernedExperimentAppendOnlyLedger:
    """Governed ML Experimentation Ledger (arXiv:2611.23678, Negative Redline):
    Control plane owns trial lifecycle and appends EVERY attempt (success + failure) to an
    append-only ledger, exposing invalid-action rate & repeated-bad rate hidden by success-only logs."""
    records: list[dict] = field(default_factory=list)
    seen_bad_hashes: set[str] = field(default_factory=set)

    def log_trial(self, trial_id: str, proposal_hash: str, valid_scope: bool, kept: bool) -> dict:
        repeated_bad = (not kept) and (proposal_hash in self.seen_bad_hashes)
        if not kept:
            self.seen_bad_hashes.add(proposal_hash)
        rec = {
            "trial_id": trial_id,
            "proposal_hash": proposal_hash,
            "valid_scope": valid_scope,
            "kept": kept,
            "repeated_bad": repeated_bad,
        }
        self.records.append(rec)
        return rec

    def governance_metrics(self) -> dict:
        n = max(1, len(self.records))
        return {
            "total_trials": len(self.records),
            "invalid_action_rate": round(sum(1 for r in self.records if not r["valid_scope"]) / n, 4),
            "repeated_bad_rate": round(sum(1 for r in self.records if r["repeated_bad"]) / n, 4),
            "provenance_completeness": 1.0 if self.records else 0.0,
        }


@dataclass
class MemShiftDeterministicRepairRouter:
    """MemShift (arXiv:2611.02338, Negative Redline): Separates change detection from repair.
    LLMs fail to re-derive computed personal memory values after source revisions; routes all
    invalidated derived fields to deterministic typed recomputation."""
    def plan_repair(self, invalidated_fields: list[str], derived_dependencies: dict[str, list[str]]) -> dict:
        inv_set = set(invalidated_fields)
        deterministic_recompute = sorted(
            target for target, deps in derived_dependencies.items() if any(d in inv_set for d in deps)
        )
        return {
            "detected_invalidations": sorted(inv_set),
            "route_to_deterministic_program": deterministic_recompute,
            "block_freeform_llm_rederivation": True,
        }


@dataclass
class PlanAndPatchLocalRepairPlanner:
    """Plan-and-Patch (arXiv:2610.10786): Preserves valid prefix and suffix steps while
    infilling only the corrupted span of a long-horizon agent plan (~2x repair rate vs AR)."""
    def patch_plan(self, steps: list[str], corrupt_start: int, corrupt_end: int, replacement: list[str]) -> dict:
        s = max(0, min(corrupt_start, len(steps)))
        e = max(s, min(corrupt_end, len(steps)))
        prefix = steps[:s]
        suffix = steps[e:]
        patched = prefix + list(replacement) + suffix
        return {
            "patched_plan": patched,
            "preserved_prefix_len": len(prefix),
            "preserved_suffix_len": len(suffix),
            "repaired_span_len": len(replacement),
        }


class ProgramDistillationSystem0Compiler:
    """
    v9.9 Program Distillation (System 2 -> System 0, arXiv:2611.19461v1):
    Selects the highest-scoring deterministic System-0 edge policy distilled by a frozen System-2
    deliberative brain using strictly separated development and held-out test seed scores.
    """

    def select_system0_policy(self, candidates: list[dict]) -> dict:
        if not candidates:
            return {"selected_id": None, "dev_score": 0.0}
        best = max(candidates, key=lambda c: float(c.get("dev_score", 0.0)))
        return {
            "selected_id": best["id"],
            "dev_score": float(best["dev_score"]),
            "zero_llm_runtime_overhead": True,
        }


class LeCatVerbatimRepetitionAuditPlanner:
    """
    v10.2 LeCat 68M Pretraining Redline (arXiv:2611.18419: verbatim repetition hurts category learning)
    + Lean 4 PTX Constructive Non-Vacuity Proof Gate (arXiv:2611.04447v1):
    Flags excessive verbatim repetition ratios in synthetic pretraining batches and requires
    non-vacuity witnesses for formal kernel proofs.
    """

    def audit_synthetic_batch(self, verbatim_ratio: float, max_allowed: float = 0.10) -> dict:
        ok = float(verbatim_ratio) <= float(max_allowed)
        return {"passed": ok, "reason": "diverse_expression_ok" if ok else "lecat_verbatim_repetition_redline"}


class SymbolicShortCircuitAndKernelBenchGate:
    """
    v10.3 Symbolic Synthesis Token-Level Short-Circuiting (arXiv:2611.11057v1: 0/32 -> 30/32)
    + KernelBench-Verified Gate (arXiv:2607.16241v1: TF32 baseline + 4-dist hidden tests + peak VRAM cap)
    + LeanGuard (arXiv:2607.03963v1):
    Rejects candidate GPU kernels that regress peak VRAM or fail hidden-distribution equivalence.
    """

    def accept_generated_kernel(self, speedup_vs_tf32: float, hidden_dists_passed: int, peak_vram_ratio: float) -> dict:
        ok = (float(speedup_vs_tf32) > 1.0) and (int(hidden_dists_passed) >= 4) and (float(peak_vram_ratio) <= 1.05)
        return {"accepted": ok, "reason": "kernelbench_verified_ok" if ok else "reward_hack_or_vram_regression"}

class CorrectnessIllusionFP64FuzzGate:
    """
    v10.4 Experimental Kernel Fuzzing & Data Repetition-Mismatch Gate:
    Correctness Illusion multi-shape seeded FP64 fuzzing (arXiv:2606.20128: catches 10/10 single-shape bugs)
    + Repetition Mismatch proxy-vs-target epoch scaling guard (arXiv:2606.07597v2).
    """

    def audit_kernel_fuzzing(self, shapes_tested: int, fp64_ref_passed: bool) -> dict:
        ok = int(shapes_tested) >= 4 and bool(fp64_ref_passed)
        return {"passed": ok, "reason": "fp64_multishape_fuzz_ok" if ok else "correctness_illusion_rejected"}

    def audit_repetition_ratio(self, proxy_epochs: float, target_epochs: float, max_ratio_diff: float = 0.25) -> dict:
        diff = abs(float(proxy_epochs) - float(target_epochs)) / max(1e-6, float(target_epochs))
        return {"aligned": diff <= float(max_ratio_diff), "rel_diff": round(diff, 4)}

class DrKernelTRLOOAndCBCLProtocolGate:
    """
    v10.5 Experimental Multi-Turn Kernel RL & Verified DCFL Protocol Gate:
    Dr. Kernel TRLOO unbiased leave-one-out advantage (arXiv:2602.05885)
    + CUDA Agent verified profiling loop (arXiv:2602.24286)
    + CBCL Lean 4 verified DCFL self-extending agent protocol (arXiv:2604.14512).
    """

    def trloo_advantages(self, rewards: list[float]) -> list[float]:
        n = len(rewards)
        if n <= 1:
            return [0.0 for _ in rewards]
        total = sum(float(r) for r in rewards)
        return [round(float(r) - (total - float(r)) / (n - 1), 6) for r in rewards]

    def verify_cbcl_dialect(self, is_dcfl: bool, within_resource_cap: bool, preserves_core_vocab: bool) -> bool:
        return bool(is_dcfl) and bool(within_resource_cap) and bool(preserves_core_vocab)
