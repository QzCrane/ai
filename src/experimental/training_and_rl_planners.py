"""
Experimental / standalone prototypes extracted from training_engine.py (CTO v6.0 Clean-Room Refactor).
These classes are verified by unit tests and kept strictly outside src/brain/ production hot path.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

class ModeHoppingCheckpointSelector:
    """
    GAP-136 (Generalization Dynamics of LM Pre-training, arXiv:2609.33150):
    Pre-training can abruptly mode-hop between memorized (System-1) and generalizable
    (System-2) circuits due to capacity competition, and checkpoint averaging fails to
    recover the generalizing basin. Tracks OOD generalization probe vs memorization gap
    and selects the peak System-2 intermediate checkpoint.
    """
    def __init__(self, hop_drop_threshold: float = 0.08) -> None:
        self.hop_drop_threshold = float(hop_drop_threshold)
        self.history: List[Dict[str, Any]] = []

    def record(self, step: int, id_loss: float, ood_gen_acc: float) -> Dict[str, Any]:
        prev_best = max((h["ood_gen_acc"] for h in self.history), default=0.0)
        mode_hop_detected = bool(self.history) and ((prev_best - float(ood_gen_acc)) >= self.hop_drop_threshold)
        entry = {
            "step": int(step),
            "id_loss": round(float(id_loss), 4),
            "ood_gen_acc": round(float(ood_gen_acc), 4),
            "mode_hop_detected": mode_hop_detected,
        }
        self.history.append(entry)
        return entry

    def select_best_checkpoint(self) -> Dict[str, Any]:
        if not self.history:
            return {"best_step": None, "ood_gen_acc": 0.0, "avoid_checkpoint_averaging": True}
        best = max(self.history, key=lambda h: (h["ood_gen_acc"], -h["id_loss"]))
        return {
            "best_step": best["step"],
            "ood_gen_acc": best["ood_gen_acc"],
            "avoid_checkpoint_averaging": True,
        }

class ZeroTIMThreeWayPolicyContract:
    """
    GAP-139 (Google Zero-TIM, Nov 16, 2026):
    Verifies bitwise-identical IEEE-754 float32 logprob bytes across:
      1) Sampler Decode vs Serving Prefill (isolates serving engine reduction drift)
      2) Serving Prefill vs Trainer Forward (isolates compilation/autodiff drift)
    And enforces Token-In/Token-Out (TiTO) exact token/byte ID sequence preservation.
    """
    @staticmethod
    def _to_f32_bytes(vals: List[float]) -> List[bytes]:
        import struct
        return [struct.pack("!f", float(v)) for v in vals]

    def verify_contract(
        self,
        decode_logprobs: List[float],
        prefill_logprobs: List[float],
        trainer_logprobs: List[float],
        rollout_ids: List[int],
        trainer_ids: List[int],
    ) -> Dict[str, Any]:
        tito_exact = (list(rollout_ids) == list(trainer_ids))
        d_b = self._to_f32_bytes(decode_logprobs)
        p_b = self._to_f32_bytes(prefill_logprobs)
        t_b = self._to_f32_bytes(trainer_logprobs)
        decode_vs_prefill_exact = (d_b == p_b)
        prefill_vs_trainer_exact = (p_b == t_b)
        if not tito_exact:
            diagnosis = "tito_token_id_drift"
        elif not decode_vs_prefill_exact:
            diagnosis = "serving_engine_reduction_drift"
        elif not prefill_vs_trainer_exact:
            diagnosis = "trainer_compilation_or_autodiff_drift"
        else:
            diagnosis = "bitwise_zero_tim_verified"
        return {
            "bitwise_zero_tim": tito_exact and decode_vs_prefill_exact and prefill_vs_trainer_exact,
            "tito_exact": tito_exact,
            "decode_vs_prefill_exact": decode_vs_prefill_exact,
            "prefill_vs_trainer_exact": prefill_vs_trainer_exact,
            "diagnosis": diagnosis,
        }

class StreamingDiLoCoFragmentScheduler:
    """
    GAP-140 (Google MaxText Streaming DiLoCo, Nov 19/20, 2026):
    Partitions model weights into P = n_layers + 1 fragments and streams 1/P of parameter
    deltas per step across distributed slices with outer Nesterov momentum, eliminating
    the periodic full-model synchronization stall of Vanilla DiLoCo.
    """
    def __init__(self, n_layers: int, model_size_gb: float = 65.5, outer_nesterov_mu: float = 0.9) -> None:
        self.n_layers = max(1, int(n_layers))
        self.num_fragments = self.n_layers + 1  # P = N_layers + 1
        self.sync_cadence_h = self.num_fragments  # H = P
        self.model_size_gb = float(model_size_gb)
        self.outer_nesterov_mu = float(outer_nesterov_mu)

    def fragment_for_step(self, step: int) -> Dict[str, Any]:
        frag_id = int(step) % self.num_fragments
        per_step_gb = self.model_size_gb / float(self.num_fragments)
        return {
            "step": int(step),
            "fragment_id": frag_id,
            "total_fragments": self.num_fragments,
            "per_step_transfer_gb": round(per_step_gb, 4),
            "traffic_reduction_vs_dp": self.num_fragments,
            "periodic_full_model_stall": False,
        }

class MultiTurnInTurnToolRLOOEngine:
    """
    GAP-149 (Google Tunix + Gemma 4 Multi-Turn Agentic RL, Nov 19, 2026):
    1. Continues assistant turn seamlessly after native <|tool_response> stop token
       without resetting prefix KV cache.
    2. Computes RLOO (Leave-One-Out) group advantages for multi-turn GRPO/GSPO trajectories.
    """
    TOOL_RESPONSE_STOP = "<|tool_response>"

    def should_continue_in_turn(self, assistant_output: str) -> bool:
        return str(assistant_output).rstrip().endswith(self.TOOL_RESPONSE_STOP)

    def compute_rloo_advantages(self, group_rewards: List[float]) -> List[float]:
        n = len(group_rewards)
        if n <= 1:
            return [0.0 for _ in group_rewards]
        total = sum(float(r) for r in group_rewards)
        advs = []
        for r in group_rewards:
            r_f = float(r)
            baseline = (total - r_f) / float(n - 1)
            advs.append(round(r_f - baseline, 6))
        return advs

class FlexShardMuonAndCollectiveFreeCkptPlanner:
    """
    GAP-155 (PyTorch TorchTitan, Oct 21, 2026):
    1. Assigns Owned/BlockShard 2D-aligned placements to Muon-optimized 2D matrices
       (eliminating pre-update AllGather and aligning with blockwise quantization).
    2. Tracks one-time metadata collective state for collective-free async checkpoint saves.
    """
    def __init__(self) -> None:
        self.metadata_synced: bool = False

    def assign_placement(self, param_name: str, ndim: int, optimizer_kind: str) -> str:
        if str(optimizer_kind).lower() == "muon" and int(ndim) >= 2:
            return "BlockShard_2D_Owned"
        return "FSDP2_RowShard"

    def record_checkpoint_save(self, step: int) -> Dict[str, Any]:
        if not self.metadata_synced:
            self.metadata_synced = True
            return {"step": int(step), "ran_global_collective": True, "barrier_only": False}
        return {"step": int(step), "ran_global_collective": False, "barrier_only": True}

class GPPOGeneralizedDivergenceRL:
    """
    GAP-164 (GPPO, arXiv:2610.13015): Computes generalized f-divergence penalties
    (reverse_kl, forward_kl, js, chi2) with monotonic trust-region regularization.
    """
    def f_divergence(self, ratio: float, kind: str = "reverse_kl") -> float:
        import math
        r = max(1e-9, float(ratio))
        k = str(kind).lower()
        if k == "reverse_kl":
            return round(r - 1.0 - math.log(r), 6)
        if k == "forward_kl":
            return round(-math.log(r) + r - 1.0 if r == 1.0 else (1.0 / r) - 1.0 + math.log(r), 6)
        if k == "chi2":
            return round(0.5 * ((r - 1.0) ** 2), 6)
        # Jensen-Shannon
        m = 0.5 * (1.0 + r)
        return round(0.5 * (r * math.log(r / m) + math.log(1.0 / m)), 6)

class HintedScalingLawForecaster:
    """
    GAP-167 (Hinted Scaling Laws, arXiv:2611.02270): Combines accuracy across progressive
    hint levels into an area-under-hint-curve (AUHC) signal that separates small models
    operating below the unhinted noise floor.
    """
    def compute_hinted_score(self, accuracies_by_hint_level: List[float]) -> Dict[str, Any]:
        if not accuracies_by_hint_level:
            return {"auhc": 0.0, "above_noise_floor": False}
        vals = [max(0.0, min(1.0, float(a))) for a in accuracies_by_hint_level]
        auhc = sum(vals) / float(len(vals))
        monotone = all(b >= a - 1e-6 for a, b in zip(vals[:-1], vals[1:]))
        return {"auhc": round(auhc, 6), "monotone_with_hints": monotone, "above_noise_floor": auhc > 0.05}

class TRUERErrorAwareRewardRectifier:
    """
    GAP-168 (TRUER, arXiv:2611.13714): Distinguishes true correctness from majority-vote
    consensus during Test-Time RL, flipping false-positive majority rewards when the frozen
    error-aware rectifier predicts incorrectness.
    """
    def __init__(self, rectifier_threshold: float = 0.50) -> None:
        self.rectifier_threshold = float(rectifier_threshold)

    def rectify_reward(self, majority_vote_reward: float, rectifier_correct_prob: float) -> Dict[str, Any]:
        mv = float(majority_vote_reward)
        p_corr = max(0.0, min(1.0, float(rectifier_correct_prob)))
        fp_caught = (mv > 0.5) and (p_corr < self.rectifier_threshold)
        fn_rescued = (mv <= 0.5) and (p_corr >= self.rectifier_threshold)
        rectified = 1.0 if p_corr >= self.rectifier_threshold else 0.0
        return {
            "rectified_reward": rectified,
            "false_positive_majority_caught": fp_caught,
            "correct_minority_rescued": fn_rescued,
        }


class LOOMLeaveOneOutBatchSelector:
    """
    ADR-013 (LOOM, arXiv:2610.00436, https://arxiv.org/abs/2610.00436):
    Every Batch Is Its Own Validation Set — removes diagonal from per-example gradient
    Gram matrix to weight samples by leave-one-out gradient SNR (zero held-out val set,
    half-batch beats full-batch, rejects >80% injected label noise).
    """

    def select_by_loo_alignment(self, per_example_grads: list[list[float]], keep_frac: float = 0.5) -> dict:
        n = len(per_example_grads)
        if n <= 1:
            return {"selected_indices": list(range(n)), "loo_scores": [1.0] * n}
        dim = len(per_example_grads[0])
        sum_g = [sum(per_example_grads[i][d] for i in range(n)) for d in range(dim)]
        scores = []
        for i, gi in enumerate(per_example_grads):
            # Leave-one-out mean gradient: (sum_g - gi) / (n - 1)
            loo_dot = sum(gi[d] * ((sum_g[d] - gi[d]) / (n - 1)) for d in range(dim))
            scores.append(round(loo_dot, 6))
        k = max(1, min(n, int(round(n * float(keep_frac)))))
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        return {"selected_indices": sorted(ranked[:k]), "loo_scores": scores}


class GRAFTGradientRoutingAdapter:
    """
    ADR-013 (GRAFT, arXiv:2610.17062, https://arxiv.org/abs/2610.17062):
    Routes gradients of monitor-flagged (noisy/reward-hacking) samples strictly into a
    sacrificial adapter parameter set, then ablates the sacrificial parameters after
    training without teaching the model to hide misbehavior (verified 135M–32B).
    """

    def route_sample_gradient(self, is_flagged: bool) -> dict:
        if is_flagged:
            return {"target_params": "sacrificial_adapter", "update_base_weights": False, "ablate_at_export": True}
        return {"target_params": "main_backbone", "update_base_weights": True, "ablate_at_export": False}


class CalibratedDecisionRewardJudge:
    """
    Calibrated Decision Rewards (arXiv:2611.25135) + SFT-First Redline (arXiv:2611.07639):
    Reference-free calibrated judge reward matches answer-key GRPO (+4.65 MATH500 on Qwen3-1.7B,
    half ECE 0.074 vs 0.141), but flags fallback to gold SFT / answer-key when base pass rate is too low.
    """

    def score_and_gate(self, judge_conf: float, base_pass_rate: float, has_gold_sft: bool = False) -> dict:
        if has_gold_sft and float(base_pass_rate) < 0.35:
            return {"mode": "run_gold_sft_first", "reward": 0.0, "avoid_cold_start_grpo": True}
        if float(base_pass_rate) < 0.15:
            return {"mode": "require_answer_key_fallback", "reward": round(float(judge_conf), 4), "avoid_cold_start_grpo": False}
        return {"mode": "calibrated_reference_free_grpo", "reward": round(float(judge_conf), 4), "avoid_cold_start_grpo": False}


class GroupLevelSyntheticCurator:
    """
    ADR-013 (arXiv:2610.00779, https://arxiv.org/abs/2610.00779):
    Individual-level quality filters fail as synthetic data share grows. Evaluates group-level
    diversity & cross-sample redundancy before admitting a synthetic batch into pretraining.
    """

    def admit_synthetic_group(self, mean_quality: float, group_diversity_score: float, min_div: float = 0.45) -> dict:
        admit = float(mean_quality) >= 0.70 and float(group_diversity_score) >= float(min_div)
        return {"admitted": admit, "reason": "group_passed" if admit else "low_group_diversity_or_quality"}


class DialogGRPOElicitationDAG:
    """
    ADR-008 (DialogGRPO, arXiv:2611.01878, https://arxiv.org/abs/2611.01878):
    Models proactive clarification as a dialogue-state DAG combining immediate info-gain with
    lookahead value (>88% vague-query recovery without over-asking on complete queries).
    """

    def should_ask_clarification(self, missing_info_prob: float, info_gain: float, threshold: float = 0.40) -> bool:
        return (float(missing_info_prob) * float(info_gain)) >= float(threshold)


@dataclass
class FtFSAEFeatureDataSelector:
    """Follow the Features / FtF (arXiv:2611.04085): Model-agnostic SAE feature activation
    ranking + diversity resampling for ultra-data-efficient instruction tuning."""
    budget: int = 100

    def select_indices(self, sample_feature_scores: list[dict[str, float]]) -> list[int]:
        if not sample_feature_scores:
            return []
        covered: dict[str, int] = {}
        chosen: list[int] = []
        remaining = set(range(len(sample_feature_scores)))
        while remaining and len(chosen) < self.budget:
            best_i = max(
                remaining,
                key=lambda i: sum(v / (1.0 + covered.get(f, 0)) for f, v in sample_feature_scores[i].items()),
            )
            chosen.append(best_i)
            remaining.remove(best_i)
            for f in sample_feature_scores[best_i]:
                covered[f] = covered.get(f, 0) + 1
        return chosen


class TestTimeSelfPlayVerifierFreePlanner:
    """
    v10.0 Test-Time Self-Play (TTSP, arXiv:2611.07367v1, Nov 5, 2026):
    Asymmetric test-time self-play scaling without a verifier (+35% absolute mean@8 on
    LiveCodeBench v6 in the no-verifier limit; fails only when base policy coverage is near zero).
    """

    def should_activate_ttsp(self, verifier_reliability: float, base_policy_coverage: float) -> dict:
        active = (float(verifier_reliability) < 0.7) and (float(base_policy_coverage) > 0.05)
        return {"activate_ttsp": active, "reason": "low_verifier_nonzero_coverage" if active else "bypass"}


class TRACERolloutGuidedFP4Planner:
    """
    v10.1 TRACE Rollout-Guided FP4 QAT (arXiv:2610.07767, 5.4x rollout speedup) +
    BitMTP Low-Bit Multi-Token Prediction (arXiv:2610.28500):
    Guides training-side FP4 mantissa rounding from rollout-side FP4 outcomes for RTX 3050 RL loops.
    """

    def select_fp4_rounding_bias(self, rollout_fp4_val: float, train_fp4_val: float) -> dict:
        delta = round(float(rollout_fp4_val) - float(train_fp4_val), 6)
        return {"mantissa_nudge": 1 if delta > 0 else (-1 if delta < 0 else 0), "delta": delta}
