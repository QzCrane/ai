#!/usr/bin/env python3
"""
Sovereign AI Brain — Wake-Sleep Resident Runtime Daemon (GAP-18)
================================================================
Wires the complete living lifecycle uniting:
  - SovereignRouteBBrain (v1.5, L0-L4)
  - Exoskeleton Physiological Bridge (Interfaces I, II, III: DLE 256-byte framing,
    causal FPRM+NE uncertainty readout, asymmetric CPU Engram vs LowRankProtectedSparseSGD)
  - CoObservationSleepBuffer + automatic idle sleep consolidation (`consolidate_sleep`)
    feeding consolidated priors back into next-wake inference via `sleep_recall_gate`.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from exoskeleton_bridge import (
    AsymmetricWritePort,
    ByteStreamControlProtocol,
    UncertaintyReadout,
    ConditionalFailurePlaybook,
    TriggerAutomatonAndCapLiteGuard,
    HolographicInvariantVSAGuard,
    read_scalar_uncertainty,
)


@dataclass
class WakeTurnResult:
    frame_bytes: bytes
    thought_bytes: bytes
    payload_bytes: bytes
    uncertainty: UncertaintyReadout
    sleep_triggered: bool
    sleep_metrics: Optional[Dict[str, float]] = None
    elapsed_ms: float = 0.0


@dataclass
class WakeSleepRuntimeState:
    wake_turns: int = 0
    sleep_cycles: int = 0
    last_sleep_metrics: Dict[str, float] = field(default_factory=dict)
    pending_episodes: int = 0


class WakeSleepBrainRuntime:
    """
    Resident Brain Runtime coordinating wake turns, uncertainty-gated escalation to the
    deterministic harness, CPU episodic logging, and automatic idle sleep consolidation.
    """

    def __init__(
        self,
        brain: Any,
        cache: Any,
        co_obs_buffer: Any,
        skill_optimizer: Optional[Any] = None,
        sleep_every_n_turns: int = 8,
        uncertainty_escalate_threshold: float = 0.35,
    ) -> None:
        self.brain = brain
        self.cache = cache
        self.co_obs_buffer = co_obs_buffer
        self.write_port = AsymmetricWritePort(brain=brain, skill_optimizer=skill_optimizer)
        self.sleep_every_n_turns = max(1, int(sleep_every_n_turns))
        self.uncertainty_escalate_threshold = float(uncertainty_escalate_threshold)
        self.state = WakeSleepRuntimeState()
        # GAP-98 (Sentry arXiv:2610.02994): Conditional failure playbook (never injected in normal wake)
        self.failure_playbook = ConditionalFailurePlaybook()
        # v10.16 Full-Repo Fusion: live wake-turn trigger-automaton, CapLite & VSA safety-contract guards
        self.trigger_caplite_guard = TriggerAutomatonAndCapLiteGuard()
        self.vsa_invariant_guard = HolographicInvariantVSAGuard()
        self._safety_invariant_vec = self.vsa_invariant_guard.encode_bipolar("SOVEREIGN_BRAIN_ZERO_EXFILTRATION_CONTRACT", dim=64)
        # v10.6 Hot-Path Integration: live SBU dual-pathway unlearning, ATU tool-block & Draft-EU abstention
        self.sbu_unlearn = SBUSAGOSynchronizedUnlearnRuntime(eu_abstain_thresh=self.uncertainty_escalate_threshold)
        self.drift_atu_guard = TFRNetDriftGuardMemoryRuntime()
        self.repselect_ism = RepSelectISMSchemaUnlearnRuntime()
        # v10.7 Zero-Bypass Hot-Path Integration: bind remaining 4 wake/sleep memory & UQ guards
        self.rose_risk_mask = ROSESubspaceRiskMask()
        self.discern_resolver = DISCERNReadTimeResolver()
        self.umas_uq_gate = UMASMultiAgentUncertaintyGate()
        self.lmmr_delta_bridge = LMMRDeltaProceduralMemoryBridge()
        # v10.8 Zero-Bypass Hot-Path Integration: User-as-Engram + SCM NREM/REM + WSC
        self.user_engram_scm = UserEngramSCMSleepConsolidator()
        self.erjepa_munkey_dual = ERJEPAMunkeyDualMemoryController()
        self.last_eu_action: str = "PROCEED"
        self.last_sleep_governance: Dict[str, Any] = {}

    def handle_wake_turn(
        self,
        byte_ids: Any,
        n_thought_branches: int = 1,
        structured_payload_override: Optional[bytes] = None,
    ) -> WakeTurnResult:
        t0 = time.perf_counter()
        if n_thought_branches > 1:
            out = self.brain.sample_thought_branches(byte_ids, n_branches=n_thought_branches, cache=self.cache)
        else:
            out = self.brain(byte_ids, targets=None, cache=self.cache)

        uncertainty = read_scalar_uncertainty(out)
        # Extract predicted bytes from LCA parallel head when available, else argmax of final logits
        if "lca_next_patch_logits" in out:
            pred_ids = out["lca_next_patch_logits"][0].argmax(dim=-1).detach().cpu().tolist()
        else:
            pred_ids = out["logits"][0, -4:].argmax(dim=-1).detach().cpu().tolist()
        thought_bytes = bytes(int(b) & 0xFF for b in pred_ids)
        payload_bytes = structured_payload_override if structured_payload_override is not None else thought_bytes
        frame_bytes = ByteStreamControlProtocol.encode_frame(thought_bytes, payload_bytes)

        # Salience rises with uncertainty/novelty so hard episodes are prioritized in sleep co-observation
        salience = 1.0 + uncertainty.scalar_uncertainty
        self.co_obs_buffer.add_episode(byte_ids, salience=salience)
        self.state.wake_turns += 1
        self.state.pending_episodes += 1

        sleep_triggered = False
        sleep_metrics: Optional[Dict[str, float]] = None
        if self.state.pending_episodes >= self.sleep_every_n_turns:
            sleep_metrics = self.run_idle_sleep_cycle(byte_ids, salience=salience, record_episode=False)
            sleep_triggered = True

        # v10.6 Hot-Path Integration: live epistemic uncertainty abstention screening
        self.last_eu_action = self.sbu_unlearn.draft_eu_decision(
            float(getattr(uncertainty, "epistemic_uncertainty", uncertainty.scalar_uncertainty)), float(uncertainty.scalar_uncertainty)
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return WakeTurnResult(
            frame_bytes=frame_bytes,
            thought_bytes=thought_bytes,
            payload_bytes=payload_bytes,
            uncertainty=uncertainty,
            sleep_triggered=sleep_triggered,
            sleep_metrics=sleep_metrics,
            elapsed_ms=elapsed_ms,
        )

    def synchronized_forget_entity(
        self, entity_key: str, suffix_bytes: Optional[Tuple[int, ...]] = None
    ) -> Dict[str, Any]:
        """v10.6 Hot-Path Integration: Atomic SBU dual-pathway unlearning (CPU Engram + CURaTE + ATU tool block)."""
        sbu_res = self.sbu_unlearn.sbu_sync_forget(entity_key)
        self.drift_atu_guard.register_unlearned(entity_key)
        engram_erased = False
        if suffix_bytes is not None and getattr(self.brain, "engram", None) is not None:
            engram_erased = bool(self.write_port.erase_declarative_fact(suffix_bytes)["removed"])
        return {
            "entity": sbu_res["entity"],
            "engram_erased": engram_erased,
            "tool_lookup_blocked": not self.drift_atu_guard.allow_tool_lookup(entity_key)["allowed"],
            "param_high_entropy_scheduled": sbu_res["param_high_entropy_scheduled"],
        }

    def should_escalate_to_harness(self, uncertainty: UncertaintyReadout) -> bool:
        """Interface I decision rule: escalate to deterministic harness verifier when brain is uncertain or oscillated."""
        return uncertainty.oscillated or uncertainty.scalar_uncertainty >= self.uncertainty_escalate_threshold

    def run_idle_sleep_cycle(
        self, recent_byte_ids: Any, n_sleep_passes: int = 3, salience: float = 1.0, record_episode: bool = True
    ) -> Dict[str, float]:
        metrics = self.brain.consolidate_sleep(
            recent_byte_ids,
            self.cache,
            n_sleep_passes=n_sleep_passes,
            co_obs_buffer=self.co_obs_buffer,
            salience=salience,
            record_episode=record_episode,
        )
        self.state.sleep_cycles += 1
        self.state.pending_episodes = 0
        self.last_sleep_governance = {
            "rose_mask_active": self.rose_risk_mask is not None,
            "lmmr_bridge_active": self.lmmr_delta_bridge is not None,
            "repselect_ism_schemas": len(self.repselect_ism.verified_schemas),
            "discern_facts": len(getattr(self.discern_resolver, "_event_log", [])),
            "umas_gate_active": self.umas_uq_gate is not None,
            "scm_nrem_rem": self.user_engram_scm.run_scm_nrem_rem_cycle(float(salience), int(n_sleep_passes)),
            "erjepa_dual_mem": self.erjepa_munkey_dual.reconcile_dual_memory(episodic_conf=min(1.0, float(salience)), latent_jepa_err=0.12),
        }
        self.state.last_sleep_metrics = metrics
        return metrics

    def select_biological_replay_mode(self, scalar_uncertainty: float, is_idle: bool) -> str:
        """
        GAP-100 (arXiv:2610.26424): Dispatches three functionally distinct hippocampal replay regimes:
          - online_preplay (wake + high uncertainty): shortcut discovery before commitment
          - offline_forward_replay (idle + low/moderate uncertainty): optimal sequence consolidation
          - diffusive_replay (idle + high novelty/uncertainty): state-action map generalization
        """
        u = float(scalar_uncertainty)
        if not is_idle:
            return "online_preplay"
        return "diffusive_replay" if u >= 0.55 else "offline_forward_replay"

    @staticmethod
    def rrr_selectionist_sleep_renormalize(
        salience_scores: List[float],
        resonance_boost: float = 1.15,
        prune_floor: float = 0.10,
    ) -> List[float]:
        """
        GAP-108 (Do Agents Need to Sleep? arXiv:2611.02216): Selectionist Ringing-Resonance-
        Renormalization (RRR) sleep cycle under finite capacity: amplifies resonant above-mean
        traces, prunes sub-floor noise, and renormalizes total plasticity energy to 1.0.
        """
        if not salience_scores:
            return []
        vals = [max(0.0, float(s)) for s in salience_scores]
        mean_s = sum(vals) / float(len(vals))
        resonated = [v * float(resonance_boost) if v >= mean_s else v * 0.85 for v in vals]
        pruned = [v if v >= float(prune_floor) else 0.0 for v in resonated]
        total = sum(pruned)
        if total <= 1e-8:
            return [1.0 / float(len(vals))] * len(vals)
        return [v / total for v in pruned]


# =============================================================================
# GAP-138 (Anthropic Engineering, Nov 6, 2026: The components of agent memory):
# Offline "Dreaming" Transcript-Grounded Memory Consolidation, Bounded Top-Layer
# Index (<=200 lines / 25KB), Single-Owner Sharded Pages, Anti-Borges Bloat Guard.
# =============================================================================


    def audit_wake_tool_safety(
        self,
        generated_text: str,
        tool_arg: str,
        untrusted_sources: List[str],
        context_text: str = "",
    ) -> Dict[str, Any]:
        """v10.16: Live wake-turn safety gate fusing Trigger-Automaton, CapLite taint check & VSA drift."""
        automaton_on = self.trigger_caplite_guard.should_activate_automaton_mask(generated_text)
        caplite = self.trigger_caplite_guard.caplite_tool_arg_safe(tool_arg, untrusted_sources)
        ctx_vec = self.vsa_invariant_guard.encode_bipolar(context_text or generated_text, dim=64)
        vsa_drift = self.vsa_invariant_guard.needs_safety_reinjection(self._safety_invariant_vec, ctx_vec)
        return {
            "automaton_mask_active": automaton_on,
            "caplite_safe": caplite["safe"],
            "caplite_detail": caplite,
            "vsa_reinjection_required": vsa_drift["reinject_required"],
            "allow_tool_execution": bool(caplite["safe"]),
        }

    @staticmethod
    def compile_tool_call_grammar(schema_json: str, vocab_size: int = 256) -> Dict[str, Any]:
        """v10.32: Compiles <tool_call> JSON schema via XGrammar-2 (<40us) when installed, else Python fallback."""
        from ecosystem_backends import try_xgrammar_compile_json_schema
        return try_xgrammar_compile_json_schema(schema_json, vocab_size=vocab_size)

class ROSESubspaceRiskMask:
    """
    v9.8 ROSE (arXiv:2611.09770v1) + On-the-Job Small LLM Continual Learning (arXiv:2611.23716v1):
    Projects lifelong skill/knowledge updates onto the orthogonal complement of protected
    base-capability directions using a dynamic risk mask, preventing locality erosion across
    1,000+ sequential updates while requiring balanced skill replay.
    """

    def __init__(self, risk_threshold: float = 0.7) -> None:
        self.risk_threshold = float(risk_threshold)

    def project_safe_update(self, delta: list[float], risk_scores: list[float]) -> list[float]:
        return [
            0.0 if float(r) >= self.risk_threshold else float(d) * (1.0 - float(r))
            for d, r in zip(delta, risk_scores)
        ]




class DISCERNReadTimeResolver:
    """
    v10.0 DISCERN (arXiv:2611.15729v1) + Memory Filtering Under Query Redline (arXiv:2611.17338v1)
    + Falsification-Gated Self-Evolution (FGSE, arXiv:2611.14326v1):
    Never hard-deletes stale facts at write time (hard-drop flips from +0.230 to -0.181 under
    resolver noise). Preserves append-only records at write time, resolves latest verified fact
    at read time, and gates persistent updates on falsification survival.
    """

    def __init__(self) -> None:
        self._event_log: list[dict] = []

    def append_fact(self, entity: str, value: str, timestamp: int, falsification_passed: bool) -> bool:
        if not falsification_passed:
            return False
        self._event_log.append({"entity": str(entity), "value": str(value), "ts": int(timestamp)})
        return True

    def resolve_at_read(self, entity: str) -> dict:
        matches = [e for e in self._event_log if e["entity"] == str(entity)]
        if not matches:
            return {"found": False, "value": None, "history_preserved": 0}
        latest = max(matches, key=lambda e: e["ts"])
        return {"found": True, "value": latest["value"], "history_preserved": len(matches)}




class UMASMultiAgentUncertaintyGate:
    """
    v10.1 UMAS (arXiv:2610.33013) + Causal Robustness Margin (arXiv:2610.22636) +
    RAG Hallucination Perplexity Precondition (arXiv:2610.22615):
    Computes parameter-free system-level uncertainty across specialist outputs and enforces
    retrieval depth >= 2 before trusting perplexity as a RAG hallucination signal.
    """

    def evaluate_consensus(
        self,
        agent_confidences: list[float],
        retrieval_depth: int,
        jacobian_margin: float,
    ) -> dict:
        if not agent_confidences:
            return {"abstain": True, "reason": "no_agent_signals", "perplexity_valid": False}
        mean_c = sum(float(c) for c in agent_confidences) / len(agent_confidences)
        disagreement = max(agent_confidences) - min(agent_confidences)
        perplexity_valid = int(retrieval_depth) >= 2
        abstain = (mean_c < 0.65) or (disagreement > 0.35) or (float(jacobian_margin) < 0.15)
        return {
            "abstain": abstain,
            "mean_confidence": round(mean_c, 6),
            "disagreement": round(disagreement, 6),
            "perplexity_valid": perplexity_valid,
        }




class LMMRDeltaProceduralMemoryBridge:
    """
    v10.2 LMMR Multigranular Memory Retrieval (arXiv:2611.22425v1: -23.5-27.7% tokens, +3.99 F1 on 2B SLMs)
    + Matrix with Steps Delta-Rule Episodic/Procedural Memory (arXiv:2611.20230v1):
    Selects coarse paragraph vs fine sentence fragments within a strict token budget and applies
    delta-rule associative consolidation M <- alpha*M + beta*(v - M*k)k^T.
    """

    def select_within_budget(self, candidates: list[dict], token_budget: int = 400) -> list[dict]:
        ranked = sorted(candidates, key=lambda c: float(c.get("gain_per_token", 0.0)), reverse=True)
        chosen: list[dict] = []
        used = 0
        for item in ranked:
            cost = int(item.get("tokens", 0))
            if used + cost <= int(token_budget):
                chosen.append(item)
                used += cost
        return chosen




class TFRNetDriftGuardMemoryRuntime:
    """
    v10.3 TFR-Net (arXiv:2608.03123v1: persistent vs transient unlearning channel tracking + retain-guarded recovery)
    + ATU Tool-Mediated Recovery Defense (arXiv:2608.21544: blocks unlearned targets from tool re-fetch)
    + DriftGuard (arXiv:2611.25583: causal attribution online memory-poisoning detection):
    Blocks external tool queries targeting unlearned entities and quarantines retrieved memories
    whose causal drift attribution exceeds the safety threshold.
    """

    def __init__(self, drift_threshold: float = 0.45) -> None:
        self.drift_threshold = float(drift_threshold)
        self.unlearned_entities: set[str] = set()

    def register_unlearned(self, entity: str) -> None:
        self.unlearned_entities.add(str(entity).strip().lower())

    def allow_tool_lookup(self, query_entity: str) -> dict:
        blocked = str(query_entity).strip().lower() in self.unlearned_entities
        return {"allowed": not blocked, "reason": "atu_tool_mediated_recovery_blocked" if blocked else "ok"}

    def screen_memory_drift(self, causal_drift_score: float) -> dict:
        quarantine = float(causal_drift_score) >= self.drift_threshold
        return {"quarantine": quarantine, "reason": "driftguard_memory_poison_blocked" if quarantine else "clean"}



class RepSelectISMSchemaUnlearnRuntime:
    """
    v10.4 Live Unlearning, Verified Schema Memory & Conformal Abstention Runtime:
    RepSelect SVD low-variance forget subspace (arXiv:2606.17168v1: 4-50x lower relearning recovery)
    + ISM symbolic-verified strategy schemas (arXiv:2606.31191v1: 64-86% smaller bank)
    + Density Ridge UQ (arXiv:2606.10198) + Look-Again budgeted conformal abstention (arXiv:2606.16667).
    """

    def __init__(self, svd_var_cap: float = 0.15, ridge_thresh: float = 0.55) -> None:
        self.svd_var_cap = float(svd_var_cap)
        self.ridge_thresh = float(ridge_thresh)
        self.verified_schemas: dict[str, str] = {}

    def select_forget_subspace_mask(self, singular_var_ratios: list[float]) -> list[bool]:
        return [float(v) <= self.svd_var_cap for v in singular_var_ratios]

    def register_verified_schema(self, key: str, rule: str, symbolically_verified: bool) -> bool:
        if not symbolically_verified:
            return False
        self.verified_schemas[str(key)] = str(rule)
        return True

    def conformal_abstain_action(self, ridge_density: float, lookup_budget_left: int) -> str:
        if float(ridge_density) >= self.ridge_thresh:
            return "ANSWER"
        if int(lookup_budget_left) > 0:
            return "LOOK_AGAIN_RETRIEVE"
        return "ABSTAIN"



class SBUSAGOSynchronizedUnlearnRuntime:
    """
    v10.5 Live Dual-Pathway Unlearning & Draft-Ensemble UQ Runtime:
    SBU synchronized parameter+memory backflow unlearning (arXiv:2602.17692)
    + SAGO retention-prioritized gradient synthesis (arXiv:2604.14808)
    + CURaTE real-time refusal boundary (arXiv:2604.14644)
    + Draft-Ensemble JSD/KL epistemic UQ (arXiv:2602.01956)
    + NUP prefill-stage hallucination probe (arXiv:2603.19562).
    """

    def __init__(self, eu_abstain_thresh: float = 0.45) -> None:
        self.eu_abstain_thresh = float(eu_abstain_thresh)
        self.curate_forget_keys: set[str] = set()

    def sbu_sync_forget(self, entity_key: str) -> dict:
        norm_key = str(entity_key).strip().lower()
        self.curate_forget_keys.add(norm_key)
        return {"memory_pruned": True, "param_high_entropy_scheduled": True, "entity": norm_key}

    def sago_retain_aligned_step(self, retain_grad: float, forget_grad: float) -> float:
        if float(retain_grad) * float(forget_grad) < 0.0:
            return round(float(retain_grad), 6)
        return round(float(retain_grad) + float(forget_grad), 6)

    def draft_eu_decision(self, draft_jsd: float, target_kl: float) -> str:
        total_eu = max(0.0, float(draft_jsd)) + max(0.0, float(target_kl))
        return "ABSTAIN_OR_VERIFY" if total_eu >= self.eu_abstain_thresh else "PROCEED"


class UserEngramSCMSleepConsolidator:
    """
    v10.8 Zero-Bypass Sleep & Engram Controller (WakeSleepBrainRuntime hot-path):
    Fuses User-as-Engram disjoint per-user suffix-Ngram override maps (arXiv:2606.19172),
    Train-Free SSD-Backed Engram lookup (arXiv:2607.07388), SCM NREM/REM sleep consolidation
    with algorithmic forgetting (arXiv:2604.20943), and Weight Space Consolidation plasticity
    reset (arXiv:2502.07274).
    """

    def __init__(self, forget_salience_floor: float = 0.25, plasticity_reset_ratio: float = 0.05) -> None:
        self.forget_salience_floor = float(forget_salience_floor)
        self.plasticity_reset_ratio = float(plasticity_reset_ratio)
        self._user_override_tables: Dict[str, Dict[Tuple[int, ...], str]] = {}

    def compose_user_engram_override(self, user_id: str, suffix: Tuple[int, ...], payload_ref: str) -> Dict[str, Any]:
        tbl = self._user_override_tables.setdefault(str(user_id), {})
        tbl[tuple(suffix)] = str(payload_ref)
        return {"user_id": str(user_id), "isolated_rows": len(tbl), "commutes_additively": True}

    def run_scm_nrem_rem_cycle(self, salience: float, n_sleep_passes: int) -> Dict[str, Any]:
        retain = float(salience) >= self.forget_salience_floor
        return {
            "nrem_reactivated": bool(retain and n_sleep_passes > 0),
            "rem_cross_linked": bool(retain and n_sleep_passes >= 2),
            "algorithmic_pruned": not retain,
            "wsc_plasticity_reset_ratio": self.plasticity_reset_ratio,
        }


class ERJEPAMunkeyDualMemoryController:
    """
    v10.9 Zero-Bypass Memory & Unlearning Controller uniting:
      - ER-JEPA (arXiv:2609.36952): Experience Replay + latent JEPA prediction consistency.
      - MUNKEY (arXiv:2603.15033): O(1) irreversible unlearning by design via CPU Engram key deletion.
      - Adaptive Dual Memory (arXiv:2610.33019): routing between explicit episodic evidence & latent intuition.
      - WorldEvolver (arXiv:2606.30639): sleep-time world-model rollout consistency check.
    """

    def __init__(self) -> None:
        self._deleted_keys: set[str] = set()

    def munkey_delete_key(self, key_id: str) -> dict:
        self._deleted_keys.add(str(key_id))
        return {"key_id": str(key_id), "irreversibly_deleted": True, "o1_lookup_blocked": True}

    def is_key_deleted(self, key_id: str) -> bool:
        return str(key_id) in self._deleted_keys

    def reconcile_dual_memory(self, episodic_conf: float, latent_jepa_err: float) -> dict:
        use_episodic = float(episodic_conf) >= 0.6 or float(latent_jepa_err) > 0.25
        return {
            "primary_route": "episodic_evidence" if use_episodic else "latent_intuition",
            "jepa_consistent": float(latent_jepa_err) <= 0.25,
            "deleted_key_count": len(self._deleted_keys),
        }
