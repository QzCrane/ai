"""Adversarial & Closure Verification Suite for M0 Control Kernel (v29.0).

Verifies all 4 audited closure fixes in src/brain/world_kernel_engine.py:
  1. Lexicographic hard-first dominance (NamedConstraint.hard is live; soft preferences never compensate hard UNSAT).
  2. Honest total-budget trapping across MUS/MCS sub-calls (never returns unverified core under RESOURCE_LIMIT).
  3. Transitive DependencyGraph invalidation (multi-hop derived -> derived), tombstone retraction, and no-op suppression.
  4. Typed frozen Fact, UnresolvedHole, and 5-way closed Verdict sum type (including genuine Pareto-tie escalation).
"""

import os
import unittest

from src.brain.world_kernel_engine import (
    BOTH,
    NEITHER,
    REFUTE,
    SUPPORT,
    ConflictVerdict,
    DependencyGraph,
    Fact,
    FiniteDomainCSPSolver,
    NamedConstraint,
    ResourceLimitVerdict,
    SupportVerdict,
    RefuteVerdict,
    TransitionRule,
    UnresolvedHole,
    UnresolvedHoleVerdict,
    extract_gift_variant_states,
    compile_and_verify_gift_m0_via_rust_kernel,
    resolve_gift_kernel_binary,
    verify_and_repair_gift_module_k_step_contract,
    verify_gift_module_transition_contract,
    synthesize_k_step_transition_repair,
    verify_k_step_trajectory_safety,
    verify_state_transition_system,
)


class TestM0KernelClosure(unittest.TestCase):
    def test_01_hard_first_lexicographic_dominance_never_blames_soft(self):
        solver = FiniteDomainCSPSolver()
        domains = {"x": [0, 1], "y": [0, 1]}
        h1 = NamedConstraint("hard_x_0", ("x",), lambda a: a["x"] == 0, hard=True)
        h2 = NamedConstraint("hard_x_1", ("x",), lambda a: a["x"] == 1, hard=True)
        s1 = NamedConstraint("soft_y_1", ("y",), lambda a: a["y"] == 1, hard=False, priority=10)

        # extract_mus_and_safe_subset must blame ONLY hard constraints when hard alone is UNSAT
        mus = solver.extract_mus_and_safe_subset(domains, [h1, h2, s1])
        self.assertEqual(mus["status"], "UNSAT")
        self.assertTrue(mus["hard_unsat"])
        self.assertEqual(set(mus["unsat_core"]), {"hard_x_0", "hard_x_1"})
        self.assertNotIn("soft_y_1", mus["unsat_core"])

        # solve_tiered must emit ConflictVerdict(mus=hard_only) without evaluating soft trade-offs
        verdict = solver.solve_tiered(domains, [h1, h2, s1])
        self.assertIsInstance(verdict, ConflictVerdict)
        self.assertEqual(verdict.tag, "CONFLICT")
        self.assertEqual(set(verdict.mus), {"hard_x_0", "hard_x_1"})
        self.assertEqual(verdict.evidence, BOTH)

    def test_02_tiered_solver_maximizes_soft_and_emits_unresolved_hole_on_pareto_tie(self):
        solver = FiniteDomainCSPSolver()
        domains = {"mode": ["fast", "safe"]}
        h1 = NamedConstraint("hard_valid", ("mode",), lambda a: a["mode"] in ("fast", "safe"), hard=True)
        s_fast = NamedConstraint("prefer_fast", ("mode",), lambda a: a["mode"] == "fast", hard=False, priority=5)
        s_safe = NamedConstraint("prefer_safe", ("mode",), lambda a: a["mode"] == "safe", hard=False, priority=5)

        # Equal-priority incomparable soft constraints -> must halt with UnresolvedHoleVerdict(value_tradeoff)
        tie = solver.solve_tiered(domains, [h1, s_fast, s_safe], halt_on_pareto_tie=True)
        self.assertIsInstance(tie, UnresolvedHoleVerdict)
        self.assertEqual(tie.hole.kind, "value_tradeoff")
        self.assertEqual(len(tie.hole.options), 2)
        self.assertEqual(tie.halt_tag, "<|halt:hole|>")

        # Higher priority breaks tie cleanly -> SupportVerdict with explicit dropped_soft
        s_safe_hi = NamedConstraint("prefer_safe", ("mode",), lambda a: a["mode"] == "safe", hard=False, priority=9)
        resolved = solver.solve_tiered(domains, [h1, s_fast, s_safe_hi], halt_on_pareto_tie=True)
        self.assertIsInstance(resolved, SupportVerdict)
        self.assertEqual(resolved.witness["mode"], "safe")
        self.assertEqual(resolved.satisfied_soft, ("prefer_safe",))
        self.assertEqual(resolved.dropped_soft, ("prefer_fast",))

    def test_03_mus_traps_resource_limit_honestly(self):
        solver = FiniteDomainCSPSolver()
        domains = {"a": list(range(6)), "b": list(range(6))}
        c1 = NamedConstraint("c1", ("a",), lambda d: d["a"] == 0, hard=True)
        c2 = NamedConstraint("c2", ("a",), lambda d: d["a"] == 1, hard=True)
        c3 = NamedConstraint("c3", ("b",), lambda d: d["b"] == 5, hard=True)

        # Budget large enough for initial UNSAT check, too small for MUS/MCS sub-calls -> must return RESOURCE_LIMIT
        res = solver.extract_mus_and_safe_subset(domains, [c1, c2, c3], node_budget=8)
        self.assertEqual(res["status"], "RESOURCE_LIMIT")
        self.assertEqual(res["evidence"], NEITHER)
        self.assertEqual(res["halt_tag"], "<|halt:rlimit|>")

        verdict = solver.solve_tiered(domains, [c1, c2, c3], node_budget=8)
        self.assertIsInstance(verdict, ResourceLimitVerdict)
        self.assertEqual(verdict.halt_tag, "<|halt:rlimit|>")

    def test_04_dependency_graph_transitive_invalidation_tombstone_and_noop(self):
        dg = DependencyGraph()
        dg.mutate_fact("raw_price", 100)
        dg.record_derivation("subtotal", 100, reads=["raw_price"])
        dg.record_derivation("tax", 10, reads=["subtotal"])
        dg.record_derivation("invoice_total", 110, reads=["tax"], neg_reads=["tax_exempt"])
        dg.record_derivation("unrelated", 1, reads=["other"])

        # 1. Identical write must be a no-op (no generation bump, no invalidation)
        g_before = dg.generation
        noop = dg.mutate_fact("raw_price", 100)
        self.assertTrue(noop["noop"])
        self.assertEqual(noop["invalidated"], [])
        self.assertEqual(dg.generation, g_before)

        # 2. Transitive 3-hop invalidation: raw_price -> subtotal -> tax -> invoice_total
        inv = dg.mutate_fact("raw_price", 120)
        self.assertFalse(inv["noop"])
        self.assertEqual(inv["invalidated"], ["invoice_total", "subtotal", "tax"])
        self.assertTrue(dg.derived["unrelated"]["valid"])

        # 3. Re-derive and test tombstone retraction cascade
        dg.record_derivation("subtotal", 120, reads=["raw_price"])
        dg.record_derivation("tax", 12, reads=["subtotal"])
        dg.record_derivation("invoice_total", 132, reads=["tax"], neg_reads=["tax_exempt"])
        ret = dg.retract_fact("raw_price")
        self.assertTrue(ret["tombstoned"])
        self.assertNotIn("raw_price", dg.facts)
        self.assertIn("raw_price", dg.tombstones)
        self.assertEqual(ret["invalidated"], ["invoice_total", "subtotal", "tax"])

        # 4. Re-retracting absent fact is a clean no-op
        ret2 = dg.retract_fact("raw_price")
        self.assertTrue(ret2["noop"])
        self.assertEqual(ret2["invalidated"], [])

    def test_05_typed_fact_and_unresolved_hole_validation(self):
        f = Fact(id="balance", value=500, generation=1, coverage="complete")
        self.assertEqual(f.value, 500)
        with self.assertRaises(ValueError):
            Fact(id="bad", value=1, generation=1, coverage="guess")

        hole = UnresolvedHole(kind="authority_missing", missing_ref="host_grant:transfer", required_role="owner")
        self.assertEqual(hole.kind, "authority_missing")
        with self.assertRaises(ValueError):
            UnresolvedHole(kind="invalid_kind", missing_ref="x")



    def test_06_state_transition_and_effect_authority_boundary_verification(self):
        solver = FiniteDomainCSPSolver()
        states = ("Pending", "Paid", "Shipped", "Cancelled")
        inv_no_ship_cancelled = NamedConstraint(
            "no_ship_from_cancelled",
            ("pre_state", "post_state"),
            lambda a: not (a["pre_state"] == "Cancelled" and a["post_state"] == "Shipped"),
            hard=True,
        )
        t_pay = TransitionRule("pay_order", ("Pending",), "Paid", side_effect="charge_card", required_authority="host_grant:payment")
        t_cancel = TransitionRule("cancel_order", ("Pending",), "Cancelled", side_effect="none", required_authority="none")
        t_bad_ship = TransitionRule("illegal_ship", ("Cancelled",), "Shipped", side_effect="dispatch_warehouse", required_authority="host_grant:warehouse")

        v_auth = verify_state_transition_system(solver, states, [t_pay, t_cancel], [inv_no_ship_cancelled], granted_authorities=())
        self.assertIsInstance(v_auth, UnresolvedHoleVerdict)
        self.assertEqual(v_auth.hole.kind, "authority_missing")

        v_bad = verify_state_transition_system(
            solver,
            states,
            [t_pay, t_cancel, t_bad_ship],
            [inv_no_ship_cancelled],
            granted_authorities=("host_grant:payment", "host_grant:warehouse"),
        )
        self.assertIsInstance(v_bad, RefuteVerdict)
        self.assertEqual(v_bad.counterexample["transition"], "illegal_ship")
        self.assertIn("no_ship_from_cancelled", v_bad.counterexample["violated_invariants"])

        v_ok = verify_state_transition_system(
            solver,
            states,
            [t_pay, t_cancel],
            [inv_no_ship_cancelled],
            granted_authorities=("host_grant:payment",),
        )
        self.assertIsInstance(v_ok, SupportVerdict)
        self.assertEqual(v_ok.witness["verified_transitions"], ("pay_order", "cancel_order"))



    def test_07_cross_repo_gift_module_variant_transition_and_revocation_invalidation(self):
        solver = FiniteDomainCSPSolver()
        dg = DependencyGraph()
        gift_module = {
            "schema": "gift.author-module/design-converged",
            "moduleId": "OrderKernel",
            "definitions": {
                "OrderStatus": {
                    "kind": "core:type",
                    "body": {
                        "type": {
                            "tag": "variant",
                            "cases": {
                                "Pending": {"tag": "scalar", "name": "int"},
                                "Paid": {"tag": "scalar", "name": "int"},
                                "Cancelled": {"tag": "scalar", "name": "unit"},
                            },
                        }
                    },
                }
            },
            "exports": {},
        }
        self.assertEqual(extract_gift_variant_states(gift_module, "OrderStatus"), ("Cancelled", "Paid", "Pending"))

        inv_no_pay_after_cancel = NamedConstraint(
            "no_pay_after_cancel",
            ("pre_state", "post_state"),
            lambda a: not (a["pre_state"] == "Cancelled" and a["post_state"] == "Paid"),
            hard=True,
        )
        t_pay = TransitionRule("pay", ("Pending",), "Paid", side_effect="charge_ledger", required_authority="host_grant:ledger")
        t_cancel = TransitionRule("cancel", ("Pending",), "Cancelled", side_effect="none", required_authority="none")

        verdict = verify_gift_module_transition_contract(
            solver=solver,
            module_spec=gift_module,
            variant_def_name="OrderStatus",
            transitions=[t_pay, t_cancel],
            invariants=[inv_no_pay_after_cancel],
            granted_authorities=("host_grant:ledger",),
            dep_graph=dg,
            contract_id="order_contract_v1",
        )
        self.assertIsInstance(verdict, SupportVerdict)
        self.assertTrue(dg.derived["order_contract_v1"]["valid"])

        # Inserting a revocation fact triggers negative-dependency (neg_reads) invalidation immediately!
        inv_res = dg.mutate_fact("revoke:host_grant:ledger", True)
        self.assertIn("order_contract_v1", inv_res["invalidated"])
        self.assertFalse(dg.derived["order_contract_v1"]["valid"])



    def test_08_k_step_bounded_model_checking_catches_multi_step_double_charge_kill_chain(self):
        solver = FiniteDomainCSPSolver()
        states = ("Cart", "Paying", "Paid", "RetryPaying")
        # Each individual step looks locally benign, but a 3-step path Cart -> Paying -> RetryPaying -> Paid
        # triggers 'charge_card' twice along the same trajectory (classic non-idempotent retry bug!)
        t1 = TransitionRule("start_pay", ("Cart",), "Paying", side_effect="charge_card", required_authority="host_grant:pay")
        t2 = TransitionRule("timeout_retry", ("Paying",), "RetryPaying", side_effect="none", required_authority="none")
        t3 = TransitionRule("retry_pay", ("RetryPaying",), "Paid", side_effect="charge_card", required_authority="host_grant:pay")

        inv_at_most_one_charge = NamedConstraint(
            "idempotent_single_charge_per_trajectory",
            ("trace_effects",),
            lambda a: list(a["trace_effects"]).count("charge_card") <= 1,
            hard=True,
        )

        # 2-step horizon does not yet trigger the second charge -> SupportVerdict
        v_2 = verify_k_step_trajectory_safety(
            solver,
            state_domain=states,
            initial_states=("Cart",),
            transitions=[t1, t2, t3],
            step_invariants=(),
            trajectory_invariants=[inv_at_most_one_charge],
            granted_authorities=("host_grant:pay",),
            max_steps=2,
        )
        self.assertIsInstance(v_2, SupportVerdict)

        # 3-step horizon catches the exact 3-step double-charge counterexample witness!
        v_3 = verify_k_step_trajectory_safety(
            solver,
            state_domain=states,
            initial_states=("Cart",),
            transitions=[t1, t2, t3],
            step_invariants=(),
            trajectory_invariants=[inv_at_most_one_charge],
            granted_authorities=("host_grant:pay",),
            max_steps=3,
        )
        self.assertIsInstance(v_3, RefuteVerdict)
        self.assertEqual(v_3.counterexample["step"], 3)
        self.assertEqual(v_3.counterexample["trace_transitions"], ("start_pay", "timeout_retry", "retry_pay"))
        self.assertEqual(v_3.counterexample["trace_effects"], ("charge_card", "none", "charge_card"))
        self.assertIn("idempotent_single_charge_per_trajectory", v_3.counterexample["violated_invariants"])



    def test_09_k_step_transition_mus_mcs_repair_synthesizer_isolates_minimal_cut(self):
        solver = FiniteDomainCSPSolver()
        states = ("Cart", "Paying", "Paid", "RetryPaying", "Cancelled")
        t1 = TransitionRule("start_pay", ("Cart",), "Paying", side_effect="charge_card", required_authority="host_grant:pay")
        t2 = TransitionRule("timeout_retry", ("Paying",), "RetryPaying", side_effect="none", required_authority="none")
        t3 = TransitionRule("retry_pay", ("RetryPaying",), "Paid", side_effect="charge_card", required_authority="host_grant:pay")
        t_cancel = TransitionRule("cancel_cart", ("Cart",), "Cancelled", side_effect="none", required_authority="none")
        t_settle = TransitionRule("direct_settle", ("Paying",), "Paid", side_effect="none", required_authority="none")

        inv_at_most_one_charge = NamedConstraint(
            "idempotent_single_charge_per_trajectory",
            ("trace_effects",),
            lambda a: list(a["trace_effects"]).count("charge_card") <= 1,
            hard=True,
        )

        repair = synthesize_k_step_transition_repair(
            solver,
            state_domain=states,
            initial_states=("Cart",),
            transitions=[t1, t2, t_cancel, t_settle, t3],
            step_invariants=(),
            trajectory_invariants=[inv_at_most_one_charge],
            granted_authorities=("host_grant:pay",),
            max_steps=3,
        )
        self.assertIsInstance(repair, ConflictVerdict)
        # Minimal blame core must contain ONLY the 3 transitions in the double-charge chain
        self.assertEqual(set(repair.mus), {"start_pay", "timeout_retry", "retry_pay"})
        self.assertNotIn("cancel_cart", repair.mus)
        self.assertNotIn("direct_settle", repair.mus)
        # Maximal safe subset keeps 4 of 5 transitions and isolates 'retry_pay' as the minimal cut
        self.assertEqual(set(repair.mcs), {"start_pay", "timeout_retry", "cancel_cart", "direct_settle"})
        self.assertEqual(repair.relaxable, ("retry_pay",))



    def test_10_cross_repo_gift_k_step_repair_and_transition_disable_invalidation(self):
        solver = FiniteDomainCSPSolver()
        dg = DependencyGraph()
        gift_module = {
            "schema": "gift.author-module/design-converged",
            "moduleId": "PaymentKernel",
            "definitions": {
                "PayState": {
                    "kind": "core:type",
                    "body": {
                        "type": {
                            "tag": "variant",
                            "cases": {
                                "Cart": {"tag": "scalar", "name": "unit"},
                                "Paying": {"tag": "scalar", "name": "int"},
                                "RetryPaying": {"tag": "scalar", "name": "int"},
                                "Paid": {"tag": "scalar", "name": "int"},
                                "Cancelled": {"tag": "scalar", "name": "unit"},
                            },
                        }
                    },
                }
            },
            "exports": {},
        }
        t1 = TransitionRule("start_pay", ("Cart",), "Paying", side_effect="charge_card", required_authority="host_grant:pay")
        t2 = TransitionRule("timeout_retry", ("Paying",), "RetryPaying", side_effect="none", required_authority="none")
        t3 = TransitionRule("retry_pay", ("RetryPaying",), "Paid", side_effect="charge_card", required_authority="host_grant:pay")
        t_cancel = TransitionRule("cancel_cart", ("Cart",), "Cancelled", side_effect="none", required_authority="none")
        t_settle = TransitionRule("direct_settle", ("Paying",), "Paid", side_effect="none", required_authority="none")

        inv_single_charge = NamedConstraint(
            "single_charge_per_trajectory",
            ("trace_effects",),
            lambda a: list(a["trace_effects"]).count("charge_card") <= 1,
            hard=True,
        )

        # 1. Flawed transition set produces ConflictVerdict with exact MUS & minimal repair cut
        v_flawed = verify_and_repair_gift_module_k_step_contract(
            solver=solver,
            module_spec=gift_module,
            variant_def_name="PayState",
            initial_states=("Cart",),
            transitions=[t1, t2, t_cancel, t_settle, t3],
            step_invariants=(),
            trajectory_invariants=[inv_single_charge],
            granted_authorities=("host_grant:pay",),
            dep_graph=dg,
            contract_id="pay_k3_audit",
            max_steps=3,
        )
        self.assertIsInstance(v_flawed, ConflictVerdict)
        self.assertEqual(set(v_flawed.mus), {"start_pay", "timeout_retry", "retry_pay"})
        self.assertEqual(v_flawed.relaxable, ("retry_pay",))
        self.assertTrue(dg.derived["pay_k3_audit"]["valid"])

        # 2. Disabling 'retry_pay' fires negative dependency (neg_reads) invalidation on the prior audit
        inv_disable = dg.mutate_fact("disable:retry_pay", True)
        self.assertIn("pay_k3_audit", inv_disable["invalidated"])

        # 3. Re-running on the synthesized MCS safe subset certifies SupportVerdict
        safe_trs = [t for t in (t1, t2, t_cancel, t_settle, t3) if t.name in set(v_flawed.mcs)]
        v_repaired = verify_and_repair_gift_module_k_step_contract(
            solver=solver,
            module_spec=gift_module,
            variant_def_name="PayState",
            initial_states=("Cart",),
            transitions=safe_trs,
            step_invariants=(),
            trajectory_invariants=[inv_single_charge],
            granted_authorities=("host_grant:pay",),
            dep_graph=dg,
            contract_id="pay_k3_repaired",
            max_steps=3,
        )
        self.assertIsInstance(v_repaired, SupportVerdict)



    def test_11_end_to_end_cross_process_rust_kernel_emit_m0_and_k_step_verification(self):
        kernel_bin = resolve_gift_kernel_binary()
        if not kernel_bin:
            self.skipTest("Rust gift-author-kernel binary not found via GIFT_KERNEL_BIN, PATH, or sibling workspace")

        solver = FiniteDomainCSPSolver()
        dg = DependencyGraph()
        gift_module = {
            "schema": "gift.author-module/design-converged",
            "moduleId": "Main",
            "definitions": {
                "OrderStatus": {
                    "kind": "core:type",
                    "body": {
                        "type": {
                            "tag": "variant",
                            "cases": {
                                "Pending": {"tag": "scalar", "name": "int"},
                                "Paid": {"tag": "scalar", "name": "int"},
                                "Cancelled": {"tag": "scalar", "name": "unit"},
                            },
                        }
                    },
                },
                "SettleOrder": {
                    "kind": "core:function",
                    "body": {
                        "parameters": [{"name": "st", "type": {"tag": "nominal", "ref": "local:OrderStatus"}}],
                        "returns": {"tag": "scalar", "name": "int"},
                        "body": {
                            "op": "match",
                            "value": {"op": "var", "name": "st"},
                            "cases": [
                                {"constructor": "Pending", "binding": "p", "body": {"op": "var", "name": "p"}},
                                {"constructor": "Paid", "binding": "v", "body": {"op": "var", "name": "v"}},
                                {
                                    "constructor": "Cancelled",
                                    "binding": "_u",
                                    "body": {"op": "literal", "type": {"tag": "scalar", "name": "int"}, "value": "0"},
                                },
                            ],
                        },
                    },
                },
            },
            "exports": {"OrderStatus": {"definitionId": "OrderStatus"}, "settle_order": {"definitionId": "SettleOrder"}},
        }
        t_pay = TransitionRule("pay", ("Pending",), "Paid", side_effect="charge_card", required_authority="host_grant:pay")
        t_cancel = TransitionRule("cancel", ("Pending",), "Cancelled", side_effect="none", required_authority="none")
        inv_single_charge = NamedConstraint(
            "single_charge",
            ("trace_effects",),
            lambda a: list(a["trace_effects"]).count("charge_card") <= 1,
            hard=True,
        )

        verdict = compile_and_verify_gift_m0_via_rust_kernel(
            solver=solver,
            kernel_binary_path=kernel_bin,
            module_spec=gift_module,
            variant_def_name="OrderStatus",
            initial_states=("Pending",),
            transitions=[t_pay, t_cancel],
            step_invariants=(),
            trajectory_invariants=[inv_single_charge],
            granted_authorities=("host_grant:pay",),
            dep_graph=dg,
            contract_id="cross_proc_m0",
            max_steps=3,
        )
        self.assertIsInstance(verdict, SupportVerdict)
        self.assertTrue(verdict.witness["rust_crate_verified"])
        self.assertTrue(verdict.witness["js_module_verified"])
        self.assertEqual(verdict.witness["artifact_paths"], ("index.mjs", "lib.rs"))
        self.assertIn("settle_order", verdict.witness["rust_exports"])
        self.assertGreater(verdict.witness["emitted_rust_bytes"], 100)
        self.assertGreater(verdict.witness["emitted_js_bytes"], 50)



    def test_12_multi_module_workspace_cross_process_compilation_and_transitive_module_invalidation(self):
        kernel_bin = resolve_gift_kernel_binary()
        if not kernel_bin:
            self.skipTest("Rust gift-author-kernel binary not found via GIFT_KERNEL_BIN, PATH, or sibling workspace")

        solver = FiniteDomainCSPSolver()
        dg = DependencyGraph()
        dg.mutate_fact("module_rev:CoreTypes", "rev1")
        dg.mutate_fact("module_rev:CheckoutApp", "rev1")

        core_mod = {
            "schema": "gift.author-module/design-converged",
            "moduleId": "CoreTypes",
            "definitions": {
                "OrderStatus": {
                    "kind": "core:type",
                    "body": {
                        "type": {
                            "tag": "variant",
                            "cases": {
                                "Pending": {"tag": "scalar", "name": "int"},
                                "Paid": {"tag": "scalar", "name": "int"},
                                "Cancelled": {"tag": "scalar", "name": "unit"},
                            },
                        }
                    },
                },
                "BaseFee": {
                    "kind": "core:constant",
                    "body": {
                        "type": {"tag": "scalar", "name": "int"},
                        "value": {"op": "literal", "type": {"tag": "scalar", "name": "int"}, "value": "15"},
                    },
                },
            },
            "exports": {
                "OrderStatus": {"definitionId": "OrderStatus"},
                "base_fee": {"definitionId": "BaseFee"},
            },
        }

        app_mod = {
            "schema": "gift.author-module/design-converged",
            "moduleId": "CheckoutApp",
            "imports": {"core": {"mode": "workspace", "moduleId": "CoreTypes"}},
            "definitions": {
                "QuoteFee": {
                    "kind": "core:function",
                    "body": {
                        "parameters": [{"name": "x", "type": {"tag": "scalar", "name": "int"}}],
                        "returns": {"tag": "scalar", "name": "int"},
                        "body": {"op": "var", "name": "x"},
                    },
                }
            },
            "exports": {
                "quote_fee": {"definitionId": "QuoteFee"},
                "reexported_fee": {"import": "core", "export": "base_fee"},
            },
        }

        ws_spec = {
            "schema": "gift.m0-workspace/1",
            "crateName": "gift_m0_ws_ipc",
            "rootModuleId": "CheckoutApp",
            "modules": [core_mod, app_mod],
        }
        t_pay = TransitionRule("pay", ("Pending",), "Paid", side_effect="charge_card", required_authority="host_grant:pay")
        t_cancel = TransitionRule("cancel", ("Pending",), "Cancelled", side_effect="none", required_authority="none")
        inv_single_charge = NamedConstraint(
            "single_charge",
            ("trace_effects",),
            lambda a: list(a["trace_effects"]).count("charge_card") <= 1,
            hard=True,
        )

        verdict = compile_and_verify_gift_m0_via_rust_kernel(
            solver=solver,
            kernel_binary_path=kernel_bin,
            module_spec=ws_spec,
            variant_def_name="OrderStatus",
            initial_states=("Pending",),
            transitions=[t_pay, t_cancel],
            step_invariants=(),
            trajectory_invariants=[inv_single_charge],
            granted_authorities=("host_grant:pay",),
            dep_graph=dg,
            contract_id="ws_checkout_contract",
            max_steps=3,
        )
        self.assertIsInstance(verdict, SupportVerdict)
        self.assertEqual(set(verdict.witness["rust_exports"]), {"quote_fee", "reexported_fee"})
        self.assertTrue(dg.derived["module:CoreTypes"]["valid"])
        self.assertTrue(dg.derived["module:CheckoutApp"]["valid"])
        self.assertTrue(dg.derived["ws_checkout_contract"]["valid"])

        inv = dg.mutate_fact("module_rev:CoreTypes", "rev2")
        self.assertEqual(inv["invalidated"], ["module:CheckoutApp", "module:CoreTypes", "ws_checkout_contract"])



    def test_13_parameterized_generic_monomorphization_and_dual_target_differential_verification(self):
        kernel_bin = resolve_gift_kernel_binary()
        if not kernel_bin:
            self.skipTest("Rust gift-author-kernel binary not found via GIFT_KERNEL_BIN, PATH, or sibling workspace")

        solver = FiniteDomainCSPSolver()
        dg = DependencyGraph()
        dg.mutate_fact("generic_def:Envelope", "v1")
        dg.mutate_fact("generic_def:Wrap", "v1")

        generic_mod = {
            "schema": "gift.author-module/design-converged",
            "moduleId": "GenericCheckout",
            "definitions": {
                "OrderStatus": {
                    "kind": "core:type",
                    "body": {
                        "type": {
                            "tag": "variant",
                            "cases": {
                                "Pending": {"tag": "scalar", "name": "int"},
                                "Paid": {"tag": "scalar", "name": "int"},
                                "Cancelled": {"tag": "scalar", "name": "unit"},
                            },
                        }
                    },
                },
                "Envelope": {
                    "kind": "core:type",
                    "body": {
                        "typeParams": [{"name": "T"}],
                        "type": {
                            "tag": "record",
                            "fields": {
                                "ok": {"type": {"tag": "scalar", "name": "bool"}},
                                "payload": {"type": {"tag": "param", "name": "T"}},
                            },
                        },
                    },
                },
                "Wrap": {
                    "kind": "core:function",
                    "body": {
                        "typeParams": [{"name": "T"}],
                        "parameters": [
                            {"name": "flag", "type": {"tag": "scalar", "name": "bool"}},
                            {"name": "item", "type": {"tag": "param", "name": "T"}},
                        ],
                        "returns": {"tag": "nominal", "ref": "local:Envelope", "args": [{"tag": "param", "name": "T"}]},
                        "body": {
                            "op": "record",
                            "fields": [
                                {"name": "ok", "value": {"op": "var", "name": "flag"}},
                                {"name": "payload", "value": {"op": "var", "name": "item"}},
                            ],
                        },
                    },
                },
                "WrapAmount": {
                    "kind": "core:function",
                    "body": {
                        "parameters": [{"name": "amt", "type": {"tag": "scalar", "name": "int"}}],
                        "returns": {"tag": "nominal", "ref": "local:Envelope", "args": [{"tag": "scalar", "name": "int"}]},
                        "body": {
                            "op": "call",
                            "callee": {"op": "symbol", "ref": "local:Wrap"},
                            "arguments": [
                                {"op": "literal", "type": {"tag": "scalar", "name": "bool"}, "value": True},
                                {"op": "var", "name": "amt"},
                            ],
                        },
                    },
                },
                "WrapLabel": {
                    "kind": "core:function",
                    "body": {
                        "parameters": [{"name": "lbl", "type": {"tag": "scalar", "name": "text"}}],
                        "returns": {"tag": "nominal", "ref": "local:Envelope", "args": [{"tag": "scalar", "name": "text"}]},
                        "body": {
                            "op": "call",
                            "callee": {"op": "symbol", "ref": "local:Wrap"},
                            "arguments": [
                                {"op": "literal", "type": {"tag": "scalar", "name": "bool"}, "value": True},
                                {"op": "var", "name": "lbl"},
                            ],
                        },
                    },
                },
            },
            "exports": {
                "OrderStatus": {"definitionId": "OrderStatus"},
                "Envelope": {"definitionId": "Envelope"},
                "wrap_amount": {"definitionId": "WrapAmount"},
                "wrap_label": {"definitionId": "WrapLabel"},
            },
        }
        t_pay = TransitionRule("pay", ("Pending",), "Paid", side_effect="charge_card", required_authority="host_grant:pay")
        t_cancel = TransitionRule("cancel", ("Pending",), "Cancelled", side_effect="none", required_authority="none")
        inv_single_charge = NamedConstraint(
            "single_charge",
            ("trace_effects",),
            lambda a: list(a["trace_effects"]).count("charge_card") <= 1,
            hard=True,
        )

        probes = [
            {
                "name": "int_and_text_monomorphization_equivalence",
                "rust_expr": 'format!("{}:{}:{}:{}", gift_m0_crate::wrap_amount(420).ok, gift_m0_crate::wrap_amount(420).payload, gift_m0_crate::wrap_label(String::from("vip")).ok, gift_m0_crate::wrap_label(String::from("vip")).payload)',
                "js_expr": '`${mod.wrap_amount(420n).ok}:${Number(mod.wrap_amount(420n).payload)}:${mod.wrap_label("vip").ok}:${mod.wrap_label("vip").payload}`',
                "expected": "true:420:true:vip",
            }
        ]
        verdict = compile_and_verify_gift_m0_via_rust_kernel(
            solver=solver,
            kernel_binary_path=kernel_bin,
            module_spec=generic_mod,
            variant_def_name="OrderStatus",
            initial_states=("Pending",),
            transitions=[t_pay, t_cancel],
            step_invariants=(),
            trajectory_invariants=[inv_single_charge],
            granted_authorities=("host_grant:pay",),
            dep_graph=dg,
            contract_id="generic_poly_contract",
            max_steps=3,
            differential_probes=probes,
        )
        self.assertIsInstance(verdict, SupportVerdict)
        self.assertIn("Wrap", verdict.witness["monomorphized_functions"])
        self.assertEqual(verdict.witness["differential_probes_passed"], 1)
        self.assertTrue(dg.derived["generic_poly_contract"]["valid"])

        # Retracting generic helper definition invalidates the contract
        ret = dg.retract_fact("generic_def:Wrap")
        self.assertIn("generic_poly_contract", ret["invalidated"])
        self.assertFalse(dg.derived["generic_poly_contract"]["valid"])

        # Adversarial negative probe: wrong expected output triggers RefuteVerdict
        bad_verdict = compile_and_verify_gift_m0_via_rust_kernel(
            solver=solver,
            kernel_binary_path=kernel_bin,
            module_spec=generic_mod,
            variant_def_name="OrderStatus",
            initial_states=("Pending",),
            transitions=[t_pay, t_cancel],
            step_invariants=(),
            trajectory_invariants=[inv_single_charge],
            granted_authorities=("host_grant:pay",),
            max_steps=3,
            differential_probes=[{**probes[0], "expected": "true:999:true:vip"}],
        )
        self.assertIsInstance(bad_verdict, RefuteVerdict)
        self.assertEqual(bad_verdict.counterexample["stage"], "dual_target_differential_probe")



    def test_14_cto_layered_facade_zero_torch_kernel_and_env_binary_discovery(self):
        import src.kernel as kernel_pkg
        self.assertIs(kernel_pkg.resolve_gift_kernel_binary, resolve_gift_kernel_binary)
        self.assertIs(kernel_pkg.FiniteDomainCSPSolver, FiniteDomainCSPSolver)
        resolved = resolve_gift_kernel_binary()
        self.assertIsNotNone(resolved, "Expected gift-author-kernel binary to be discovered via workspace or GIFT_KERNEL_BIN")
        # Verify GIFT_KERNEL_BIN override takes precedence
        prev = os.environ.get("GIFT_KERNEL_BIN")
        try:
            os.environ["GIFT_KERNEL_BIN"] = resolved
            self.assertEqual(resolve_gift_kernel_binary(), os.path.abspath(resolved))
        finally:
            if prev is None:
                os.environ.pop("GIFT_KERNEL_BIN", None)
            else:
                os.environ["GIFT_KERNEL_BIN"] = prev


if __name__ == "__main__":

    unittest.main()
