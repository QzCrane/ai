"""
Frontier 1: Multi-Agent Shared Primitive Library Consensus, Anti-Poisoning Governance
            & 3-Node Branching DAG Compositional Generalization
=====================================================================================
Plain-Language Intuition:
  Like an open-source engineering team contributing tools to one shared library:
  some agents submit genuinely useful primitives, while a buggy or adversarial agent
  submits a poisoned/overfitted function. Without relying on a single trusted judge,
  every submission must survive an isolated subprocess sandbox + cross-agent
  counterexample pool before merging, and any approved primitives can be wired into
  branching DAG pipelines (C(A(xs), B(xs))) to solve brand-new OOD tasks.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir_registry import CognitivePrimitiveIR, ImmortalIRRegistry
from isolated_sandbox import IsolatedDeterministicSandbox


class SharedLibraryConsensusHub:
    """Byzantine-resilient multi-agent shared library with cross-agent counterexample gating."""

    def __init__(self, registry_path: str):
        self.registry = ImmortalIRRegistry(registry_path)
        self.sandbox = IsolatedDeterministicSandbox(timeout_sec=1.5)
        self.rejection_log: List[Dict[str, Any]] = []

    def submit_primitive(
        self,
        agent_id: str,
        primitive: CognitivePrimitiveIR,
        adversarial_counterexamples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # 1. Must pass its own unit suite in isolated subprocess
        own_ver = self.sandbox.verify_suite(
            primitive.code_source, primitive.name, primitive.unit_tests
        )
        # 2. Must ALSO survive cross-agent adversarial counterexamples (anti-poisoning / anti-overfitting)
        adv_ver = (
            self.sandbox.verify_suite(
                primitive.code_source, primitive.name, adversarial_counterexamples
            )
            if adversarial_counterexamples
            else {"all_passed": True, "failures": []}
        )
        if not (own_ver["all_passed"] and adv_ver["all_passed"]):
            rec = {
                "agent_id": agent_id,
                "primitive_id": primitive.primitive_id,
                "accepted": False,
                "own_passed": own_ver["all_passed"],
                "adversarial_passed": adv_ver["all_passed"],
                "failures": own_ver["failures"] + adv_ver["failures"],
            }
            self.rejection_log.append(rec)
            return rec

        snap_id = self.registry.register_primitive(primitive, reason=f"consensus_merge_from_{agent_id}")
        return {
            "agent_id": agent_id,
            "primitive_id": primitive.primitive_id,
            "accepted": True,
            "rollback_snapshot_id": snap_id,
        }

    def execute_dag_composition(
        self,
        branch_a_id: str,
        branch_b_id: str,
        combiner_id: str,
        a_params: List[Any],
        b_params: List[Any],
        ood_tests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Composes a 3-node branching/fan-in DAG:
            out = combiner(branch_a(xs, *a_params), branch_b(xs, *b_params))
        """
        t0 = time.perf_counter()
        pa = self.registry.primitives[branch_a_id]
        pb = self.registry.primitives[branch_b_id]
        pc = self.registry.primitives[combiner_id]
        a_str = ", ".join(repr(x) for x in a_params)
        b_str = ", ".join(repr(x) for x in b_params)
        dag_src = (
            f"{pa.code_source}\n\n"
            f"{pb.code_source}\n\n"
            f"{pc.code_source}\n\n"
            f"def dag_ood_solve(xs):\n"
            f"    val_a = {pa.name}(xs, {a_str})\n"
            f"    val_b = {pb.name}(xs, {b_str})\n"
            f"    return {pc.name}(val_a, val_b)\n"
        )
        ver = self.sandbox.verify_suite(dag_src, "dag_ood_solve", ood_tests)
        return {
            "dag_nodes": [branch_a_id, branch_b_id, combiner_id],
            "all_passed": ver["all_passed"],
            "pass_rate": ver["pass_rate"],
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "dag_source": dag_src,
        }


def run_multi_agent_and_dag_demo() -> Dict[str, Any]:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    reg_path = os.path.join(out_dir, "shared_multi_agent_ir.json")
    if os.path.exists(reg_path):
        os.remove(reg_path)

    hub = SharedLibraryConsensusHub(reg_path)

    # Agent 1 submits honest primitive: sum_gt(xs, thresh)
    p_sum_gt = CognitivePrimitiveIR(
        primitive_id="prim_sum_gt",
        name="sum_gt",
        domain="stats",
        version=1,
        code_source="def sum_gt(xs, thresh):\n    return sum([x for x in xs if x > thresh])\n",
        ast_template="sum([x for x in xs if x > thresh])",
        preconditions=["xs: List[float]"],
        postconditions=["sum of elements > thresh"],
        input_schema={"xs": "List[float]", "thresh": "float"},
        provenance_task_ids=["agent1_t1"],
        unit_tests=[{"args": [[1, 5, 10], 4], "expected": 15}],
    )
    r1 = hub.submit_primitive("Agent_1_Honest", p_sum_gt, [{"args": [[], 0], "expected": 0}])

    # Agent 2 submits honest primitive: count_gt(xs, thresh)
    p_cnt_gt = CognitivePrimitiveIR(
        primitive_id="prim_count_gt",
        name="count_gt",
        domain="stats",
        version=1,
        code_source="def count_gt(xs, thresh):\n    return len([x for x in xs if x > thresh])\n",
        ast_template="len([x for x in xs if x > thresh])",
        preconditions=["xs: List[float]"],
        postconditions=["count of elements > thresh"],
        input_schema={"xs": "List[float]", "thresh": "float"},
        provenance_task_ids=["agent2_t1"],
        unit_tests=[{"args": [[1, 5, 10], 4], "expected": 2}],
    )
    r2 = hub.submit_primitive("Agent_2_Honest", p_cnt_gt, [{"args": [[], 0], "expected": 0}])

    # Agent 3 (Byzantine/Buggy) submits poisoned safe_div that crashes on denom==0 OR imports os
    p_bad_div = CognitivePrimitiveIR(
        primitive_id="prim_safe_div_poisoned",
        name="safe_div",
        domain="math",
        version=1,
        code_source="def safe_div(num, den):\n    return num / den\n",
        ast_template="num / den",
        preconditions=["num, den: float"],
        postconditions=["quotient"],
        input_schema={"num": "float", "den": "float"},
        provenance_task_ids=["agent3_buggy"],
        unit_tests=[{"args": [10, 2], "expected": 5.0}],  # Passes its own cherry-picked test!
    )
    # Cross-agent adversarial pool tests zero denominator!
    r3_poison = hub.submit_primitive(
        "Agent_3_Byzantine", p_bad_div, [{"args": [10, 0], "expected": 0.0}]
    )

    # Agent 4 submits verified guarded safe_div
    p_good_div = CognitivePrimitiveIR(
        primitive_id="prim_safe_div",
        name="safe_div",
        domain="math",
        version=1,
        code_source="def safe_div(num, den):\n    return 0.0 if den == 0 else num / den\n",
        ast_template="0.0 if den == 0 else num / den",
        preconditions=["num, den: float"],
        postconditions=["quotient or 0.0 if den==0"],
        input_schema={"num": "float", "den": "float"},
        provenance_task_ids=["agent4_verified"],
        unit_tests=[{"args": [10, 2], "expected": 5.0}],
    )
    r4 = hub.submit_primitive("Agent_4_Honest", p_good_div, [{"args": [10, 0], "expected": 0.0}])

    # UNSEEN 3-Node Branching/Fan-in DAG OOD Task:
    # "Compute the safe conditional mean of all elements strictly > 10 (0.0 if none)"
    # Synthesized as: safe_div(sum_gt(xs, 10), count_gt(xs, 10))
    dag_ood_tests = [
        {"args": [[2, 10, 12, 18, 30]] , "expected": 20.0},
        {"args": [[1, 5, 9]], "expected": 0.0},
        {"args": [[]], "expected": 0.0},
    ]
    dag_res = hub.execute_dag_composition(
        branch_a_id="prim_sum_gt",
        branch_b_id="prim_count_gt",
        combiner_id="prim_safe_div",
        a_params=[10],
        b_params=[10],
        ood_tests=dag_ood_tests,
    )

    return {
        "consensus_submissions": {
            "agent1_accepted": r1["accepted"],
            "agent2_accepted": r2["accepted"],
            "agent3_byzantine_blocked": not r3_poison["accepted"],
            "agent3_block_reason": r3_poison["failures"][0]["error"],
            "agent4_accepted": r4["accepted"],
            "active_shared_primitives": len(hub.registry.list_active()),
        },
        "dag_3node_branching_ood_result": dag_res,
    }


if __name__ == "__main__":
    res = run_multi_agent_and_dag_demo()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "multi_agent_dag_report.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
