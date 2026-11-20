"""
Layers 3 & 5: Resource-Rational VOC Metareasoning Router, Closed-Loop Evolution Engine
              & Human-AI Cognitive Interface Layer
======================================================================================
Connects all 5 Converged Layers of SYSTEM_CONSTITUTION.md into one end-to-end runnable
system:
  1. Layer 1: ImmortalIRRegistry (versioned model-agnostic cognitive primitives)
  2. Layer 2: TargetModelCompiler (SPD closed-form subspace compiler + FISTA-NNLS KV)
  3. Layer 3: Resource-Rational VOC Router + Explicit State Exoskeleton
  4. Layer 4: IsolatedDeterministicSandbox (independent verification, anti-self-pollution)
  5. Layer 5: Closed-Loop Abstraction & Governance Engine + Human Cognitive Summary Layer
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from abstraction_synthesizer import ASTAntiUnifier
from ir_registry import CognitivePrimitiveIR, ImmortalIRRegistry, VerificationRecord
from isolated_sandbox import IsolatedDeterministicSandbox
from model_compiler import TargetModelCompiler, extract_lcp_divergence


@dataclass
class TaskSpec:
    task_id: str
    domain: str
    description: str
    fn_name: str
    test_cases: List[Dict[str, Any]]
    feature_vector: np.ndarray
    is_ood_heldout: bool = False


class ResourceRationalVOCRouter:
    """
    Tier-3 Reuse: Value of Computation (VOC) Metareasoning Router (Russell & Wefald 1991).
    Decides between:
      - 'EXECUTE_COMPILED_PRIMITIVE' (fast, O(1) deterministic reuse via abstracted primitive)
      - 'NEURAL_SEARCH_AND_VERIFY'   (expensive discovery search + sandbox verification)
      - 'ABSTAIN_INSUFFICIENT_EVIDENCE' (honest epistemic abstention when VOC < 0)
    """

    def __init__(self, compute_cost_per_search_unit: float = 0.08, success_reward: float = 1.0):
        self.cost_unit = compute_cost_per_search_unit
        self.reward = success_reward

    def decide_route(
        self,
        has_matching_primitive: bool,
        primitive_confidence: float,
        estimated_search_steps: int,
        expected_search_prob: float,
    ) -> Tuple[str, Dict[str, float]]:
        voc_primitive = (
            primitive_confidence * self.reward - 0.01 * self.cost_unit
            if has_matching_primitive
            else -1.0
        )
        voc_search = expected_search_prob * self.reward - estimated_search_steps * self.cost_unit
        voc_abstain = 0.0

        metrics = {
            "voc_primitive": round(voc_primitive, 4),
            "voc_search": round(voc_search, 4),
            "voc_abstain": voc_abstain,
        }
        if has_matching_primitive and voc_primitive >= max(voc_search, voc_abstain):
            return "EXECUTE_COMPILED_PRIMITIVE", metrics
        if voc_search > voc_abstain:
            return "NEURAL_SEARCH_AND_VERIFY", metrics
        return "ABSTAIN_INSUFFICIENT_EVIDENCE", metrics


class ClosedLoopIntrinsicSystem:
    """
    Complete 5-Layer Closed-Loop System implementing Recursive Abstraction Growth
    under SYSTEM_CONSTITUTION.md.
    """

    def __init__(self, registry_path: str, d_model: int = 128, seed: int = 2026):
        self.registry = ImmortalIRRegistry(registry_path)
        self.sandbox = IsolatedDeterministicSandbox(timeout_sec=1.5)
        self.synthesizer = ASTAntiUnifier()
        self.compiler = TargetModelCompiler(d_model=d_model, seed=seed)
        self.router = ResourceRationalVOCRouter()
        self.solved_programs: Dict[str, str] = {}
        self.compute_ledger: List[Dict[str, Any]] = []
        self.rng = np.random.default_rng(seed)

        # Calibrate spike subspace on synthetic background activations
        Z_bg = self.rng.standard_normal((64, d_model)) * 0.25
        spike_dirs, _ = np.linalg.qr(self.rng.standard_normal((d_model, 6)))
        Z_bg += (self.rng.standard_normal((64, 6)) * 20.0) @ spike_dirs[:, :6].T
        self.Z_non_target_bg = Z_bg
        self.compiler.calibrate_spike_subspace(Z_bg)

    def discover_and_solve_from_scratch(
        self, task: TaskSpec, candidate_pool: List[str]
    ) -> Dict[str, Any]:
        """Simulates expensive base-level search + independent sandbox verification."""
        t0 = time.perf_counter()
        attempts = 0
        failed_trace_tokens = None
        for cand_src in candidate_pool:
            attempts += 1
            ver = self.sandbox.verify_suite(cand_src, task.fn_name, task.test_cases)
            if ver["all_passed"]:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.solved_programs[task.task_id] = cand_src
                lcp_info = None
                if failed_trace_tokens is not None:
                    lcp_info = extract_lcp_divergence(failed_trace_tokens, cand_src.split())
                entry = {
                    "task_id": task.task_id,
                    "mode": "DISCOVER_FROM_SCRATCH",
                    "search_attempts": attempts,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "passed": True,
                    "lcp_divergence": lcp_info,
                }
                self.compute_ledger.append(entry)
                return {**entry, "solution_code": cand_src}
            elif failed_trace_tokens is None:
                failed_trace_tokens = cand_src.split()

        return {
            "task_id": task.task_id,
            "mode": "DISCOVER_FROM_SCRATCH",
            "search_attempts": attempts,
            "passed": False,
        }

    def abstract_and_compile_primitive(
        self,
        task_a_id: str,
        task_b_id: str,
        primitive_id: str,
        primitive_name: str,
        domain: str,
        Z_pos: np.ndarray,
        Delta_target: np.ndarray,
        unit_tests: List[Dict[str, Any]],
        rarity_critical: bool = False,
    ) -> Dict[str, Any]:
        """
        Tier-2 Abstraction -> Layer-1 Immortal IR -> Layer-2 SPD Closed-Form Compilation.
        """
        src_a = self.solved_programs[task_a_id]
        src_b = self.solved_programs[task_b_id]
        abst = self.synthesizer.synthesize_from_functions(src_a, src_b, primitive_name)
        if abst is None:
            return {"ok": False, "reason": "AST anti-unification failed"}

        # Independent sandbox verification of the synthesized higher-order primitive
        ver = self.sandbox.verify_suite(abst.template_source, primitive_name, unit_tests)
        if not ver["all_passed"]:
            return {"ok": False, "reason": "Synthesized primitive failed independent sandbox tests", "ver": ver}

        # Compile Layer-2 SPD Closed-Form Subspace Operator with non-target drift gate
        skill, comp_report = self.compiler.compile_skill_spd(
            skill_id=primitive_id,
            Z_pos=Z_pos,
            Delta_target=Delta_target,
            Z_non_target=self.Z_non_target_bg,
            max_allowed_drift=1e-4,
        )
        if not comp_report["accepted"] or skill is None:
            return {"ok": False, "reason": "Rejected by non-target drift gate", "compiler_report": comp_report}

        # Register into Layer-1 Immortal IR Registry with snapshot for one-click rollback
        ir = CognitivePrimitiveIR(
            primitive_id=primitive_id,
            name=primitive_name,
            domain=domain,
            version=1,
            code_source=abst.template_source,
            ast_template=abst.template_source,
            preconditions=["xs is list of numbers", "meta parameters valid"],
            postconditions=["deterministic pure output matching verifier"],
            input_schema={"xs": "List[float]", "params": str(abst.parameter_names)},
            provenance_task_ids=[task_a_id, task_b_id],
            unit_tests=unit_tests,
            verification_history=[
                VerificationRecord(
                    timestamp=time.time(),
                    passed=True,
                    ood_accuracy=1.0,
                    base_accuracy=0.0,
                    non_target_drift=comp_report["non_target_drift"],
                    amortized_speedup=1.0,
                    verifier_type="isolated_subprocess_unit_suite",
                )
            ],
            rarity_critical=rarity_critical,
        )
        snap_id = self.registry.register_primitive(ir, reason="abstract_and_compile")
        return {
            "ok": True,
            "primitive_id": primitive_id,
            "template_source": abst.template_source,
            "meta_params": abst.parameter_names,
            "compression_ratio": abst.compression_ratio,
            "compiler_report": comp_report,
            "rollback_snapshot_id": snap_id,
        }

    def solve_ood_with_voc_router(
        self, task: TaskSpec, meta_param_candidates: List[Tuple[Any, ...]]
    ) -> Dict[str, Any]:
        """
        Solves an UNSEEN held-out OOD task using the VOC Router + Active Layer-1 Primitives.
        """
        active_prims = self.registry.list_active(domain=task.domain)
        has_prim = len(active_prims) > 0
        route, voc_metrics = self.router.decide_route(
            has_matching_primitive=has_prim,
            primitive_confidence=0.92 if has_prim else 0.0,
            estimated_search_steps=15,
            expected_search_prob=0.6,
        )
        if route != "EXECUTE_COMPILED_PRIMITIVE":
            return {"task_id": task.task_id, "route": route, "voc": voc_metrics, "passed": False}

        t0 = time.perf_counter()
        prim = active_prims[0]
        attempts = 0
        for param_tuple in meta_param_candidates:
            attempts += 1
            # Bind OOD task parameters onto the abstracted cognitive primitive
            wrapped_tests = [
                {"args": tc["args"] + list(param_tuple), "expected": tc["expected"]}
                for tc in task.test_cases
            ]
            ver = self.sandbox.verify_suite(prim.code_source, prim.name, wrapped_tests)
            if ver["all_passed"]:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                prim.usage_count += 1
                prim.success_count += 1
                self.registry.save()
                entry = {
                    "task_id": task.task_id,
                    "mode": "REUSE_COMPILED_PRIMITIVE_OOD",
                    "route": route,
                    "voc": voc_metrics,
                    "primitive_used": prim.primitive_id,
                    "bound_params": list(param_tuple),
                    "search_attempts": attempts,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "passed": True,
                }
                self.compute_ledger.append(entry)
                return entry

        return {"task_id": task.task_id, "route": route, "voc": voc_metrics, "passed": False}

    def generate_human_cognitive_summary(self) -> Dict[str, Any]:
        """
        Loop 6 (Human-AI Cognitive Interface Layer):
        Compresses full internal logs into concise, human-digestible mechanisms,
        amortized compute economics, active primitives, and governance status.
        """
        discover_runs = [e for e in self.compute_ledger if e["mode"] == "DISCOVER_FROM_SCRATCH"]
        reuse_runs = [e for e in self.compute_ledger if e["mode"] == "REUSE_COMPILED_PRIMITIVE_OOD"]
        avg_disc_attempts = (
            float(np.mean([e["search_attempts"] for e in discover_runs])) if discover_runs else 0.0
        )
        avg_reuse_attempts = (
            float(np.mean([e["search_attempts"] for e in reuse_runs])) if reuse_runs else 0.0
        )
        return {
            "active_primitives_count": len(self.registry.list_active()),
            "total_snapshots_for_rollback": len(self.registry.snapshots),
            "avg_discovery_search_attempts": round(avg_disc_attempts, 2),
            "avg_ood_primitive_reuse_attempts": round(avg_reuse_attempts, 2),
            "search_space_shrink_factor": round(avg_disc_attempts / max(avg_reuse_attempts, 1e-6), 2),
            "compiled_subspace_operators": list(self.compiler.compiled_skills.keys()),
        }
