"""
Isolated prototypes extracted from wake_sleep_runtime.py (CTO v6.1 Full-Stack Clean-Room).
Verified by unit tests; strictly separated from src/brain/ production runtime.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

class HierarchicalRetentionMemory:
    """
    GAP-118 (WMT arXiv:2608.20631 + LiveMem arXiv:2608.02515): Hierarchical task/subtask memory
    with dynamic retention scores, event-based folding of completed trajectories, and decay of
    low-utility or poisoned entries (-32.8% prompt tokens, +9.97pp GAIA-Text).
    """
    def __init__(self, decay_rate: float = 0.85, prune_threshold: float = 0.20) -> None:
        self.decay_rate = float(decay_rate)
        self.prune_threshold = float(prune_threshold)
        self._items: Dict[str, Dict[str, Any]] = {}

    def upsert(self, key: str, content: str, retention_score: float = 1.0, folded: bool = False) -> None:
        self._items[str(key)] = {"content": str(content), "score": float(retention_score), "folded": bool(folded)}

    def fold_completed(self, key: str, summary: str) -> bool:
        if key not in self._items:
            return False
        self._items[key]["content"] = str(summary)
        self._items[key]["folded"] = True
        return True

    def decay_and_prune(self) -> int:
        pruned = 0
        for k in list(self._items.keys()):
            self._items[k]["score"] *= self.decay_rate
            if self._items[k]["score"] < self.prune_threshold:
                del self._items[k]
                pruned += 1
        return pruned

    def active_keys(self) -> List[str]:
        return list(self._items.keys())

class DreamingMemoryConsolidator:
    """
    GAP-138 (Anthropic, Nov 6, 2026 — The components of agent memory):
    1. Offline dreaming validates in-session memory notes against raw session transcripts
       to correct drift/hallucinations before future sessions trust stale notes.
    2. Enforces bounded top-layer injected index (<=200 lines & <=25KB), demoting
       verbose details to lower-tier sharded pages with lightweight pointers.
    3. Single-owner sharded entity pages prevent concurrent multi-agent overwrites
       (Anthropic empirical jump: 41/57 = 72% -> 57/57 = 100% fact survival).
    4. Anti-Borges bloat guard ensures compiled memory never exceeds raw source bytes.
    """
    def __init__(self, max_index_lines: int = 200, max_index_bytes: int = 25 * 1024) -> None:
        self.max_index_lines = int(max_index_lines)
        self.max_index_bytes = int(max_index_bytes)
        self.entity_pages: Dict[str, Dict[str, Any]] = {}
        self.top_index: List[str] = []

    def write_sharded_entity(self, entity_id: str, owner_agent: str, verbatim_facts: List[str], raw_source_bytes: int) -> Dict[str, Any]:
        existing = self.entity_pages.get(entity_id)
        if existing is not None and existing["owner"] != owner_agent:
            return {"written": False, "reason": "single_owner_conflict", "owner": existing["owner"]}
        compiled = "\n".join(verbatim_facts)
        compiled_bytes = len(compiled.encode("utf-8"))
        # Anti-Borges guard: compiled wiki must not exceed raw source size
        if raw_source_bytes > 0 and compiled_bytes > raw_source_bytes:
            return {"written": False, "reason": "borges_bloat_rejected", "compiled_bytes": compiled_bytes, "raw_source_bytes": raw_source_bytes}
        self.entity_pages[entity_id] = {
            "owner": owner_agent,
            "facts": list(verbatim_facts),
            "verified": False,
            "bytes": compiled_bytes,
        }
        return {"written": True, "entity_id": entity_id, "fact_count": len(verbatim_facts), "bytes": compiled_bytes}

    def dream_consolidate(self, session_transcript: str) -> Dict[str, Any]:
        """Offline sleep-time dreaming: verify facts against raw transcript & build bounded top index."""
        corrected = 0
        demoted = 0
        index_lines: List[str] = []
        used_bytes = 0
        for eid, page in sorted(self.entity_pages.items()):
            grounded = [f for f in page["facts"] if f in session_transcript]
            if len(grounded) < len(page["facts"]):
                corrected += (len(page["facts"]) - len(grounded))
                page["facts"] = grounded
            page["verified"] = True
            pointer_line = f"- [{eid}] ({len(page['facts'])} verified facts) -> page://{eid}"
            line_bytes = len((pointer_line + "\n").encode("utf-8"))
            if len(index_lines) < self.max_index_lines and (used_bytes + line_bytes) <= self.max_index_bytes:
                index_lines.append(pointer_line)
                used_bytes += line_bytes
            else:
                demoted += 1
        self.top_index = index_lines
        return {
            "corrected_stale_facts": corrected,
            "top_index_lines": len(self.top_index),
            "top_index_bytes": used_bytes,
            "demoted_to_lower_tier": demoted,
        }

class MEMENTOCompactionAndBacktrackingController:
    """
    GAP-173 (MEMENTO, arXiv:2611.00244, https://arxiv.org/abs/2611.00244):
    Learned context management for long-context processing under a strict working-window
    cap (<= 20,000 tokens for 2.5M-token books; raises Qwen3-8B from 36.9% to 67.8%).
    Combines hindsight-note compaction, test-time context folding, and confidence-based
    backtracking when answer confidence falls below threshold.
    """

    def __init__(self, max_window_tokens: int = 20000, backtrack_conf_threshold: float = 0.65):
        self.max_window_tokens = max(512, int(max_window_tokens))
        self.backtrack_conf_threshold = float(backtrack_conf_threshold)
        self.checkpoints: list[dict] = []

    def step_fold_or_backtrack(
        self,
        current_tokens: int,
        hindsight_notes: list[str],
        answer_confidence: float,
    ) -> dict:
        action = "keep"
        folded_tokens = int(current_tokens)
        if current_tokens > self.max_window_tokens:
            self.checkpoints.append({"tokens": current_tokens, "notes_count": len(hindsight_notes)})
            folded_tokens = min(self.max_window_tokens, max(512, len(hindsight_notes) * 128))
            action = "compact_and_fold"
        if answer_confidence < self.backtrack_conf_threshold and self.checkpoints:
            action = "confidence_backtrack"
        return {
            "action": action,
            "active_tokens": folded_tokens,
            "within_window_cap": folded_tokens <= self.max_window_tokens,
            "checkpoints_stored": len(self.checkpoints),
        }

class OPSDWeightTextMemoryRouter:
    """
    GAP-177 (Route to Remember / OPSD, arXiv:2611.05636, Nov 3 2026, https://arxiv.org/abs/2611.05636):
    On-Policy Self-Distillation (OPSD) weight memory paired with a verify-then-regenerate
    router that serves high-confidence queries directly from weight memory (zero extra tokens)
    and falls back to text memory retrieval only when verification fails.
    """

    def __init__(self, verify_threshold: float = 0.80):
        self.verify_threshold = float(verify_threshold)

    def route_query(self, weight_answer_conf: float, text_memory_available: bool = True) -> dict:
        if float(weight_answer_conf) >= self.verify_threshold:
            return {"source": "weight_memory", "regenerate_with_text": False, "saved_context_tokens": True}
        return {
            "source": "text_memory_fallback" if text_memory_available else "weight_memory_low_conf",
            "regenerate_with_text": bool(text_memory_available),
            "saved_context_tokens": False,
        }

class TrainingFreeMemoryLifecycleGate:
    """
    GAP-178 (Training-Free Memory Lifecycle, arXiv:2611.07631, Nov 6 2026, https://arxiv.org/abs/2611.07631):
    Principle: attribute usefulness after use (with UCB exploration bonus) and screen for
    evidence preservation before offline merge/compression (+11.92pp recall@5 over bare cosine,
    0 observed losses in 100 audited merges, zero extra online write LLM call).
    """

    def __init__(self, ucb_c: float = 0.5):
        self.ucb_c = float(ucb_c)

    def score_memory_ucb(self, usefulness_mean: float, exposure_count: int, total_queries: int) -> float:
        import math
        n = max(1, int(exposure_count))
        t = max(1, int(total_queries))
        return float(usefulness_mean) + self.ucb_c * math.sqrt(math.log(t + 1.0) / n)

    def screen_merge_preservation(self, required_evidence_ids: list[str], merged_evidence_ids: list[str]) -> dict:
        missing = sorted(set(required_evidence_ids) - set(merged_evidence_ids))
        return {"approve_merge": len(missing) == 0, "missing_evidence": missing}

class KnowinglyWrongBeliefConsistencyGuard:
    """
    Negative Redline Guard (arXiv:2611.05287, Nov 3 2026, https://arxiv.org/abs/2611.05287):
    Training/editing an LLM against its own internal beliefs causes misaligned answers to
    surge from 4% to >60% (undetectable by content filters or persona vectors). Blocks bare
    forced assertions that conflict with high-confidence base beliefs unless grounded in
    coherent verifiable evidence.
    """

    def audit_knowledge_update(
        self, base_belief_prob: float, target_fact_contradicts_base: bool, has_coherent_evidence: bool
    ) -> dict:
        high_conflict = bool(target_fact_contradicts_base) and float(base_belief_prob) >= 0.75
        if high_conflict and not has_coherent_evidence:
            return {"allow_update": False, "reason": "misalignment_risk_forced_belief_conflict"}
        return {"allow_update": True, "reason": "consistent_or_evidenced"}



@dataclass
class ShapleyCompactMemoryPruner:
    """Compact Audio-Visual Memory / Query-Agnostic Shapley Token Pruner (arXiv:2611.23085).
    Training-free Monte Carlo Shapley importance pruning supporting up to 80% token reduction
    on 3B-4B on-device models with accuracy gains and decoding speedup."""
    keep_ratio: float = 0.20

    def prune_indices(self, shapley_scores: list[float]) -> list[int]:
        if not shapley_scores:
            return []
        k = max(1, int(round(len(shapley_scores) * self.keep_ratio)))
        top_idx = sorted(range(len(shapley_scores)), key=lambda i: shapley_scores[i], reverse=True)[:k]
        return sorted(top_idx)
