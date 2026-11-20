"""
Autonomous R&D Expansion:
  1. Non-Stationary Concept Drift Detection (CUSUM) + Online Dirichlet-Multinomial
     Likelihood Calibration for POMDP Belief Exoskeleton (Solving Dynamic-World Drift).
  2. Bitemporal Provenance-Aware Fact Memory with Contradiction Coexistence &
     Authority Resolution (Solving Hard Problem #8: Memory Lifecycle).
"""
from __future__ import annotations

import json
import os
import numpy as np


# ==============================================================================
# PART 1: Non-Stationary Concept Drift & Online Likelihood Calibration (POMDP)
# ==============================================================================
def run_nonstationary_drift_experiment(num_episodes: int = 300, drift_point: int = 150, seed: int = 42) -> dict:
    """
    At episode 0..149: Probe A ('fast_sensor') is 90% accurate, Probe B ('slow_sensor') is 86% accurate.
    At episode 150..299 (SUDDEN CONCEPT DRIFT / SENSOR DEGRADATION):
      Probe A degrades to 50% (pure coin-flip noise!), while Probe B remains 86% accurate.
    Compare:
      - Static POMDP (never updates sensor likelihood matrix O)
      - Drift-Aware Online-Calibrated POMDP (Dirichlet-Multinomial decay + CUSUM alarm)
    """
    rng = np.random.default_rng(seed)
    U = np.array([[100.0, -120.0], [-120.0, 100.0], [-5.0, -5.0]])  # 2 states, 3 actions (Fix0, Fix1, Abstain)
    prior = np.array([0.5, 0.5])
    probe_costs = {"probe_A": 5.0, "probe_B": 8.0}

    true_L_pre = {
        "probe_A": np.array([[0.90, 0.10], [0.10, 0.90]]),
        "probe_B": np.array([[0.86, 0.14], [0.14, 0.86]]),
    }
    true_L_post = {
        "probe_A": np.array([[0.50, 0.50], [0.50, 0.50]]),  # Degraded to pure noise!
        "probe_B": np.array([[0.86, 0.14], [0.14, 0.86]]),
    }

    def eval_episode(b_init: np.ndarray, est_L: dict[str, np.ndarray], actual_L: dict[str, np.ndarray], true_s: int):
        b = b_init.copy()
        cost = 0.0
        obs_log = []
        for _ in range(3):
            cur_u = float(np.max(U @ b))
            best_p, best_voi = None, 0.0
            for pname, L in est_L.items():
                p_o = b @ L
                fut_u = 0.0
                for o in range(2):
                    if p_o[o] > 1e-12:
                        bp = (b * L[:, o]) / p_o[o]
                        fut_u += float(p_o[o]) * float(np.max(U @ bp))
                voi = fut_u - cur_u - probe_costs[pname]
                if voi > best_voi:
                    best_voi, best_p = voi, pname
            if best_p is None:
                break
            o_val = int(rng.choice(2, p=actual_L[best_p][true_s]))
            b = (b * est_L[best_p][:, o_val])
            b /= max(float(np.sum(b)), 1e-12)
            cost += probe_costs[best_p]
            obs_log.append((best_p, o_val))
        act = int(np.argmax(U @ b))
        return float(U[act, true_s]) - cost, act == true_s, (act < 2 and act != true_s), obs_log

    # Static POMDP
    static_post_u, static_post_cat = [], 0
    # Adaptive Drift-Aware POMDP (Dirichlet-Multinomial with exponential forgetting gamma=0.92 + CUSUM)
    counts = {
        "probe_A": np.array([[9.0, 1.0], [1.0, 9.0]]),
        "probe_B": np.array([[8.6, 1.4], [1.4, 8.6]]),
    }
    gamma_decay = 0.92
    cusum_A = 0.0
    drift_detected_ep = None
    adapt_post_u, adapt_post_cat = [], 0

    for ep in range(num_episodes):
        actual_L = true_L_pre if ep < drift_point else true_L_post
        true_s = int(rng.integers(0, 2))

        # Static run
        u_s, _, cat_s, _ = eval_episode(prior, true_L_pre, actual_L, true_s)
        if ep >= drift_point:
            static_post_u.append(u_s)
            static_post_cat += int(cat_s)

        # Adaptive run
        est_L = {
            p: counts[p] / np.sum(counts[p], axis=1, keepdims=True)
            for p in counts
        }
        u_a, _, cat_a, obs_log = eval_episode(prior, est_L, actual_L, true_s)
        if ep >= drift_point:
            adapt_post_u.append(u_a)
            adapt_post_cat += int(cat_a)

        # Delayed ground-truth feedback arrives -> update Dirichlet counts & CUSUM drift detector
        for pname, o_val in obs_log:
            counts[pname] = gamma_decay * counts[pname] + 0.05
            counts[pname][true_s, o_val] += 1.0
            if pname == "probe_A":
                # Surprise residual: expected error rate 0.10; accumulate excess error
                err_ind = 1.0 if o_val != true_s else 0.0
                cusum_A = max(0.0, cusum_A + (err_ind - 0.20))
                if cusum_A > 3.0 and drift_detected_ep is None:
                    drift_detected_ep = ep

    n_post = num_episodes - drift_point
    return {
        "drift_onset_episode": drift_point,
        "cusum_drift_detected_episode": drift_detected_ep,
        "detection_delay_episodes": (drift_detected_ep - drift_point) if drift_detected_ep is not None else None,
        "post_drift_static_pomdp": {
            "mean_net_utility": round(float(np.mean(static_post_u)), 2),
            "catastrophic_error_rate": round(static_post_cat / n_post, 4),
        },
        "post_drift_adaptive_pomdp": {
            "mean_net_utility": round(float(np.mean(adapt_post_u)), 2),
            "catastrophic_error_rate": round(adapt_post_cat / n_post, 4),
        },
        "learned_probe_A_accuracy_post_drift": round(
            float(counts["probe_A"][0, 0] / np.sum(counts["probe_A"][0])), 3
        ),
    }


# ==============================================================================
# PART 2: Bitemporal Provenance-Aware Fact Memory & Contradiction Coexistence
# ==============================================================================
class BitemporalFactMemory:
    """
    Solves Hard Problem #8 (Memory Lifecycle):
      - Distinguishes Temporal Updates (valid_from/valid_to) from Contradictions.
      - Allows conflicting claims from different sources to coexist with explicit
        authority ranking and provenance instead of silent overwrite.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def assert_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: int,
        valid_to: int | None,
        source: str,
        authority_rank: int,
    ) -> None:
        # Close older open-ended record from the SAME or lower authority if strictly superseded in time
        for rec in self.records:
            if (
                rec["subject"] == subject
                and rec["predicate"] == predicate
                and rec["valid_to"] is None
                and valid_from > rec["valid_from"]
                and rec["obj"] != obj
                and authority_rank >= rec["authority_rank"]
            ):
                rec["valid_to"] = valid_from - 1
        self.records.append(
            {
                "subject": subject,
                "predicate": predicate,
                "obj": obj,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "source": source,
                "authority_rank": authority_rank,
            }
        )

    def query_at_time(self, subject: str, predicate: str, query_time: int) -> dict:
        active = [
            r
            for r in self.records
            if r["subject"] == subject
            and r["predicate"] == predicate
            and r["valid_from"] <= query_time
            and (r["valid_to"] is None or query_time <= r["valid_to"])
        ]
        distinct_objs = sorted({r["obj"] for r in active})
        has_conflict = len(distinct_objs) > 1
        best = max(active, key=lambda r: r["authority_rank"]) if active else None
        return {
            "query_time": query_time,
            "resolved_best_answer": best["obj"] if best else None,
            "best_source": best["source"] if best else None,
            "has_coexisting_contradiction": has_conflict,
            "all_active_claims": active,
        }


def run_bitemporal_memory_test() -> dict:
    mem = BitemporalFactMemory()
    # Temporal evolution: protocol version changed in 2026
    mem.assert_fact("CoreBus", "default_solver", "ClampedLSTSQ", valid_from=2024, valid_to=None, source="v1_spec", authority_rank=5)
    mem.assert_fact("CoreBus", "default_solver", "FISTA_NNLS", valid_from=2026, valid_to=None, source="v2_audit_spec", authority_rank=9)
    # Simultaneous contradiction in 2026 from an untrusted low-authority blog post
    mem.assert_fact("CoreBus", "default_solver", "RandomHash", valid_from=2026, valid_to=None, source="unverified_blog", authority_rank=2)

    q_2025 = mem.query_at_time("CoreBus", "default_solver", 2025)
    q_2026 = mem.query_at_time("CoreBus", "default_solver", 2026)
    return {
        "historical_query_2025": {
            "answer": q_2025["resolved_best_answer"],
            "conflict": q_2025["has_coexisting_contradiction"],
        },
        "current_query_2026": {
            "answer": q_2026["resolved_best_answer"],
            "best_source": q_2026["best_source"],
            "conflict_flagged": q_2026["has_coexisting_contradiction"],
            "coexisting_claims_count": len(q_2026["all_active_claims"]),
        },
    }


if __name__ == "__main__":
    out = {
        "nonstationary_drift_pomdp": run_nonstationary_drift_experiment(),
        "bitemporal_contradiction_memory": run_bitemporal_memory_test(),
    }
    print(json.dumps(out, indent=2))
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "drift_and_bitemporal_report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
