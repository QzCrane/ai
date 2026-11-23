"""
v29.0 M0 Decidable Finite-Domain Bilattice, Tiered CSP/MUS Solver & Transitive Dependency Kernel.

HONEST M0 SCOPE & EXCLUSION BOUNDARY:
This module implements ONLY the strictly decidable finite-domain discrete control spine:
  1. EvidenceBit: True Belnap-Dunn Bilattice with BOTH truth-order (conj/disj/neg)
     AND knowledge-order (consensus/gullibility), Arieli-Avron weak & strong implications
     (satisfying reflexivity, MP, transitivity, contraposition), explicit non-collapsing
     conflict_guard(), and bug-free non-amplifying forall_eval/exists_eval.
  2. M0 Typed Carriers & 5-Way Closed Verdict Sum Type:
     Fact, NamedConstraint (hard/soft tier + priority), UnresolvedHole, and
     Verdict = SupportVerdict | RefuteVerdict | ConflictVerdict | UnresolvedHoleVerdict | ResourceLimitVerdict.
  3. FiniteDomainCSPSolver: Real backtracking finite-domain constraint solver with
     metered total node budgets (trapped across MUS/MCS sub-calls), 2-tier lexicographic
     hard-first dominance (solve_tiered), Minimal Unsatisfiable Subsets (MUS / UNSAT Core),
     and Maximal Satisfiable Subsets (MCS / safe_subset).
  4. DependencyGraph: Real transitive positive (reads) + negative (neg_reads) dependency
     closure with tombstone retraction (retract_fact) and no-op write suppression.

Explicitly excluded from M0 (A_out): continuous/stochastic dynamics, in-step open-world
ontology mutation, pragmatic/embodied game equilibria, and self-authorization.
"""

from __future__ import annotations

from collections import deque
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


VALID_HOLE_KINDS = frozenset({
    "observable_ro",
    "epistemic_intervention",
    "value_tradeoff",
    "authority_missing",
    "binding_missing",
})


@dataclass(frozen=True)
class EvidenceBit:
    """True Belnap-Dunn four-valued bilattice element (s, r) in {0, 1}^2."""

    s: int
    r: int

    def __post_init__(self) -> None:
        if self.s not in (0, 1) or self.r not in (0, 1):
            raise ValueError(f"EvidenceBit bits must be 0 or 1, got ({self.s}, {self.r})")

    @property
    def label(self) -> str:
        return {(1, 0): "SUPPORT", (0, 1): "REFUTE", (1, 1): "BOTH", (0, 0): "NEITHER"}[(self.s, self.r)]

    # --- Truth order (F <_t {N, B} <_t T) ---
    def neg(self) -> "EvidenceBit":
        return EvidenceBit(self.r, self.s)

    def conj(self, other: "EvidenceBit") -> "EvidenceBit":
        return EvidenceBit(self.s & other.s, self.r | other.r)

    def disj(self, other: "EvidenceBit") -> "EvidenceBit":
        return EvidenceBit(self.s | other.s, self.r & other.r)

    # --- Knowledge order (N <_k {T, F} <_k B) ---
    def consensus(self, other: "EvidenceBit") -> "EvidenceBit":
        """Knowledge-order meet (otimes): retains only evidence agreed by both sources."""
        return EvidenceBit(self.s & other.s, self.r & other.r)

    def gullibility(self, other: "EvidenceBit") -> "EvidenceBit":
        """Knowledge-order join (oplus): merges evidence from both sources (may yield BOTH)."""
        return EvidenceBit(self.s | other.s, self.r | other.r)

    # --- Arieli-Avron standard bilattice implications (algebraically sound) ---
    def weak_implies(self, consequent: "EvidenceBit") -> "EvidenceBit":
        """Arieli-Avron weak implication (a >_w b): satisfies Reflexivity, MP, Transitivity, Deduction."""
        return consequent if self.s == 1 else SUPPORT

    def strong_implies(self, consequent: "EvidenceBit") -> "EvidenceBit":
        """Arieli-Avron strong implication: (a >_w b) /\\_t (~b >_w ~a). Satisfies Contraposition & <=_t laws."""
        return self.weak_implies(consequent).conj(consequent.neg().weak_implies(self.neg()))

    def conflict_guard(self, consequent: "EvidenceBit") -> Tuple["EvidenceBit", bool]:
        """Explicit conflict-preserving guard without collapsing BOTH into SUPPORT or REFUTE."""
        if self == BOTH:
            return BOTH, True
        if self == REFUTE:
            return SUPPORT, False
        if self == SUPPORT:
            return consequent, (consequent == BOTH)
        # NEITHER: cannot amplify ignorance into classical truth
        return (SUPPORT if consequent == SUPPORT else (BOTH if consequent == BOTH else NEITHER)), (consequent == BOTH)

    @staticmethod
    def forall_eval(items: Sequence["EvidenceBit"], domain_complete: bool) -> "EvidenceBit":
        """Non-amplifying universal quantifier: preserves BOTH/REFUTE, never vacuous-truths empty domain."""
        if not items:
            return NEITHER
        acc = SUPPORT
        for it in items:
            acc = acc.conj(it)
        if not domain_complete and acc == SUPPORT:
            return NEITHER
        return acc

    @staticmethod
    def exists_eval(items: Sequence["EvidenceBit"], domain_complete: bool) -> "EvidenceBit":
        """Non-amplifying existential quantifier: preserves BOTH/SUPPORT, never erases conflict."""
        if not items:
            return REFUTE if domain_complete else NEITHER
        acc = REFUTE
        for it in items:
            acc = acc.disj(it)
        if not domain_complete and acc == REFUTE:
            return NEITHER
        return acc


SUPPORT = EvidenceBit(1, 0)
REFUTE = EvidenceBit(0, 1)
BOTH = EvidenceBit(1, 1)
NEITHER = EvidenceBit(0, 0)


@dataclass(frozen=True)
class Fact:
    """Immutable generation-stamped M0 fact carrier with tombstone & coverage markers."""

    id: str
    value: Any
    generation: int
    coverage: str = "complete"
    tombstone: bool = False
    rev_digest: str = ""

    def __post_init__(self) -> None:
        if self.coverage not in ("complete", "partial"):
            raise ValueError(f"Fact coverage must be 'complete' or 'partial', got {self.coverage!r}")


@dataclass(frozen=True)
class UnresolvedHole:
    """First-class typed M0 halting carrier for ambiguity, missing observations, or Pareto value trade-offs."""

    kind: str
    missing_ref: str
    required_role: str = "human"
    options: Tuple[Tuple[str, ...], ...] = ()
    details: str = ""

    def __post_init__(self) -> None:
        if self.kind not in VALID_HOLE_KINDS:
            raise ValueError(f"Invalid UnresolvedHole kind {self.kind!r}; expected one of {sorted(VALID_HOLE_KINDS)}")


@dataclass(frozen=True)
class SupportVerdict:
    witness: Dict[str, Any]
    dep_closure: Tuple[str, ...] = ()
    satisfied_soft: Tuple[str, ...] = ()
    dropped_soft: Tuple[str, ...] = ()
    evidence: EvidenceBit = SUPPORT
    nodes_visited: int = 0
    tag: str = "SUPPORT"


@dataclass(frozen=True)
class RefuteVerdict:
    counterexample: Dict[str, Any]
    dep_closure: Tuple[str, ...] = ()
    evidence: EvidenceBit = REFUTE
    nodes_visited: int = 0
    tag: str = "REFUTE"


@dataclass(frozen=True)
class ConflictVerdict:
    kind: str  # "evidence_both" | "unsat_core"
    mus: Tuple[str, ...]
    mcs: Tuple[str, ...]
    relaxable: Tuple[str, ...] = ()
    evidence: EvidenceBit = BOTH
    nodes_visited: int = 0
    halt_tag: str = "<|halt:bottom|>"
    tag: str = "CONFLICT"


@dataclass(frozen=True)
class UnresolvedHoleVerdict:
    hole: UnresolvedHole
    evidence: EvidenceBit = NEITHER
    nodes_visited: int = 0
    halt_tag: str = "<|halt:hole|>"
    tag: str = "UNRESOLVED_HOLE"


@dataclass(frozen=True)
class ResourceLimitVerdict:
    budget_kind: str = "nodes"
    nodes_visited: int = 0
    evidence: EvidenceBit = NEITHER
    halt_tag: str = "<|halt:rlimit|>"
    tag: str = "RESOURCE_LIMIT"


Verdict = Union[
    SupportVerdict,
    RefuteVerdict,
    ConflictVerdict,
    UnresolvedHoleVerdict,
    ResourceLimitVerdict,
]


@dataclass(frozen=True)
class NamedConstraint:
    """A real executable constraint predicate over variable assignments with hard/soft tiering."""

    name: str
    variables: Tuple[str, ...]
    predicate: Callable[[Dict[str, Any]], bool]
    hard: bool = True
    priority: int = 0


class FiniteDomainCSPSolver:
    """Real backtracking finite-domain CSP solver with metered budget, 2-tier hard-first dominance, MUS & MCS."""

    def solve(
        self,
        domains: Dict[str, Sequence[Any]],
        constraints: Sequence[NamedConstraint],
        node_budget: int = 10000,
    ) -> Dict[str, Any]:
        vars_order = sorted(domains.keys(), key=lambda v: len(domains[v]))
        nodes_visited = 0
        assignment: Dict[str, Any] = {}

        def _consistent(assigned: Dict[str, Any]) -> bool:
            for c in constraints:
                if all(v in assigned for v in c.variables):
                    if not c.predicate(assigned):
                        return False
            return True

        def _backtrack(idx: int) -> Optional[Dict[str, Any]]:
            nonlocal nodes_visited
            if idx == len(vars_order):
                return dict(assignment)
            var = vars_order[idx]
            for val in domains[var]:
                nodes_visited += 1
                if nodes_visited > node_budget:
                    return None
                assignment[var] = val
                if _consistent(assignment):
                    sol = _backtrack(idx + 1)
                    if sol is not None:
                        return sol
                    if nodes_visited > node_budget:
                        return None
                del assignment[var]
            return None

        sol = _backtrack(0)
        if sol is not None:
            return {
                "status": "SAT",
                "evidence": SUPPORT,
                "witness": sol,
                "nodes_visited": nodes_visited,
            }
        if nodes_visited > node_budget:
            return {
                "status": "RESOURCE_LIMIT",
                "evidence": NEITHER,
                "halt_tag": "<|halt:rlimit|>",
                "nodes_visited": nodes_visited,
            }
        return {
            "status": "UNSAT",
            "evidence": REFUTE,
            "nodes_visited": nodes_visited,
        }

    def extract_mus_and_safe_subset(
        self,
        domains: Dict[str, Sequence[Any]],
        constraints: Sequence[NamedConstraint],
        node_budget: int = 10000,
    ) -> Dict[str, Any]:
        """Deletion-based MUS & hard-prioritized MCS with strict total budget trapping (Rule R3 & R6)."""
        total_nodes = 0

        def _metered_solve(cs: Sequence[NamedConstraint]) -> Dict[str, Any]:
            nonlocal total_nodes
            rem = max(0, node_budget - total_nodes)
            if rem == 0:
                return {
                    "status": "RESOURCE_LIMIT",
                    "evidence": NEITHER,
                    "halt_tag": "<|halt:rlimit|>",
                    "nodes_visited": 0,
                }
            res = self.solve(domains, cs, node_budget=rem)
            total_nodes += int(res.get("nodes_visited", 0))
            return res

        def _rlimit_result() -> Dict[str, Any]:
            return {
                "status": "RESOURCE_LIMIT",
                "evidence": NEITHER,
                "halt_tag": "<|halt:rlimit|>",
                "nodes_visited": total_nodes,
            }

        # Rule R3 (Lexicographic Hard-First Dominance): if hard constraints alone are UNSAT,
        # never blame soft constraints for a hard contradiction.
        hard_cs = [c for c in constraints if c.hard]
        soft_cs = [c for c in constraints if not c.hard]
        target_for_mus: Sequence[NamedConstraint] = constraints

        if hard_cs and soft_cs:
            hard_check = _metered_solve(hard_cs)
            if hard_check["status"] == "RESOURCE_LIMIT":
                return _rlimit_result()
            if hard_check["status"] == "UNSAT":
                target_for_mus = hard_cs

        base = _metered_solve(target_for_mus)
        if base["status"] == "RESOURCE_LIMIT":
            return _rlimit_result()
        if base["status"] != "UNSAT":
            base["nodes_visited"] = total_nodes
            return base

        # Deletion-based MUS extraction: drop c iff remaining constraints stay UNSAT; trap RESOURCE_LIMIT!
        mus: List[NamedConstraint] = list(target_for_mus)
        i = 0
        while i < len(mus):
            trial = mus[:i] + mus[i + 1 :]
            res = _metered_solve(trial)
            if res["status"] == "RESOURCE_LIMIT":
                return _rlimit_result()
            if res["status"] == "UNSAT":
                mus = trial
            else:
                i += 1

        # Hard-first greedy maximal satisfiable subset (MCS / safe_subset)
        ordered = hard_cs + sorted(soft_cs, key=lambda c: c.priority, reverse=True)
        safe: List[NamedConstraint] = []
        for c in ordered:
            res = _metered_solve(safe + [c])
            if res["status"] == "RESOURCE_LIMIT":
                return _rlimit_result()
            if res["status"] == "SAT":
                safe.append(c)

        safe_names = {s.name for s in safe}
        return {
            "status": "UNSAT",
            "evidence": REFUTE,
            "unsat_core": [c.name for c in mus],
            "safe_subset": [c.name for c in safe],
            "relaxed_candidates": [c.name for c in constraints if c.name not in safe_names],
            "hard_unsat": bool(hard_cs and target_for_mus is hard_cs),
            "nodes_visited": total_nodes,
        }

    def solve_tiered(
        self,
        domains: Dict[str, Sequence[Any]],
        constraints: Sequence[NamedConstraint],
        node_budget: int = 10000,
        halt_on_pareto_tie: bool = True,
    ) -> Verdict:
        """M0 2-tier lexicographic solver returning a closed 5-way Verdict.

        Tier 1 (hard=True): non-compensable invariants. UNSAT -> ConflictVerdict(mus, mcs).
        Tier 2 (hard=False): soft preferences maximized inside hard-feasible space.
        If halt_on_pareto_tie=True and >=2 incomparable maximal soft subsets tie on max priority,
        emits UnresolvedHoleVerdict(kind='value_tradeoff') for explicit human resolution.
        """
        hard_cs = [c for c in constraints if c.hard]
        soft_cs = sorted([c for c in constraints if not c.hard], key=lambda c: c.priority, reverse=True)
        total_nodes = 0

        def _rem() -> int:
            return max(0, node_budget - total_nodes)

        # Tier 1: Hard constraints must be jointly satisfied first
        hard_res = self.solve(domains, hard_cs, node_budget=_rem())
        total_nodes += int(hard_res.get("nodes_visited", 0))
        if hard_res["status"] == "RESOURCE_LIMIT":
            return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)
        if hard_res["status"] == "UNSAT":
            mus_res = self.extract_mus_and_safe_subset(domains, hard_cs, node_budget=_rem())
            total_nodes += int(mus_res.get("nodes_visited", 0))
            if mus_res["status"] == "RESOURCE_LIMIT":
                return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)
            return ConflictVerdict(
                kind="unsat_core",
                mus=tuple(mus_res["unsat_core"]),
                mcs=tuple(mus_res["safe_subset"]),
                relaxable=tuple(mus_res["relaxed_candidates"]),
                nodes_visited=total_nodes,
            )

        if not soft_cs:
            return SupportVerdict(witness=hard_res["witness"], nodes_visited=total_nodes)

        # Fast path: all soft constraints jointly satisfiable with hard constraints
        all_res = self.solve(domains, hard_cs + soft_cs, node_budget=_rem())
        total_nodes += int(all_res.get("nodes_visited", 0))
        if all_res["status"] == "RESOURCE_LIMIT":
            return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)
        if all_res["status"] == "SAT":
            return SupportVerdict(
                witness=all_res["witness"],
                satisfied_soft=tuple(c.name for c in soft_cs),
                dropped_soft=(),
                nodes_visited=total_nodes,
            )

        # Tier 2 conflict among soft constraints: detect genuine Pareto tie at same priority
        if halt_on_pareto_tie and len(soft_cs) <= 12:
            best_score: Optional[Tuple[int, int]] = None
            pareto_sets: List[Tuple[Tuple[str, ...], Dict[str, Any]]] = []
            # Search subsets in decreasing size/priority order (bounded by node_budget)
            n = len(soft_cs)
            for mask in range((1 << n) - 1, -1, -1):
                subset = [soft_cs[j] for j in range(n) if (mask >> j) & 1]
                score = (sum(c.priority for c in subset), len(subset))
                if best_score is not None and score < best_score:
                    continue
                if _rem() == 0:
                    return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)
                res = self.solve(domains, hard_cs + subset, node_budget=_rem())
                total_nodes += int(res.get("nodes_visited", 0))
                if res["status"] == "RESOURCE_LIMIT":
                    return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)
                if res["status"] == "SAT":
                    names = tuple(c.name for c in subset)
                    if best_score is None or score > best_score:
                        best_score = score
                        pareto_sets = [(names, res["witness"])]
                    elif score == best_score:
                        pareto_sets.append((names, res["witness"]))

            if len(pareto_sets) > 1:
                return UnresolvedHoleVerdict(
                    hole=UnresolvedHole(
                        kind="value_tradeoff",
                        missing_ref=",".join(c.name for c in soft_cs),
                        required_role="human",
                        options=tuple(p[0] for p in pareto_sets),
                        details="Multiple incomparable maximal soft-constraint subsets tie inside hard-feasible space.",
                    ),
                    nodes_visited=total_nodes,
                )
            if len(pareto_sets) == 1:
                chosen_names, witness = pareto_sets[0]
                chosen_set = set(chosen_names)
                return SupportVerdict(
                    witness=witness,
                    satisfied_soft=chosen_names,
                    dropped_soft=tuple(c.name for c in soft_cs if c.name not in chosen_set),
                    nodes_visited=total_nodes,
                )

        # Greedy fallback for large soft sets or halt_on_pareto_tie=False
        chosen: List[NamedConstraint] = []
        last_witness = hard_res["witness"]
        for c in soft_cs:
            if _rem() == 0:
                return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)
            res = self.solve(domains, hard_cs + chosen + [c], node_budget=_rem())
            total_nodes += int(res.get("nodes_visited", 0))
            if res["status"] == "RESOURCE_LIMIT":
                return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)
            if res["status"] == "SAT":
                chosen.append(c)
                last_witness = res["witness"]

        chosen_names = tuple(c.name for c in chosen)
        chosen_set = set(chosen_names)
        return SupportVerdict(
            witness=last_witness,
            satisfied_soft=chosen_names,
            dropped_soft=tuple(c.name for c in soft_cs if c.name not in chosen_set),
            nodes_visited=total_nodes,
        )


class DependencyGraph:
    """Transitive positive (reads) + negative (neg_reads) dependency closure with tombstone retraction."""

    def __init__(self) -> None:
        self.generation: int = 1
        self.facts: Dict[str, Any] = {}
        self.fact_records: Dict[str, Fact] = {}
        self.tombstones: Dict[str, int] = {}
        self.derived: Dict[str, Dict[str, Any]] = {}

    def get_fact(self, fact_key: str) -> Optional[Fact]:
        return self.fact_records.get(fact_key)

    def record_derivation(
        self,
        key: str,
        value: Any,
        reads: Iterable[str],
        neg_reads: Iterable[str] = (),
    ) -> None:
        self.derived[key] = {
            "value": value,
            "reads": set(reads),
            "neg_reads": set(neg_reads),
            "generation": self.generation,
            "valid": True,
        }

    def _propagate_invalidation(self, initial_keys: Iterable[str]) -> List[str]:
        """BFS transitive closure over derived -> derived positive reads."""
        invalidated: List[str] = []
        queue: deque[str] = deque(initial_keys)
        visited: Set[str] = set(initial_keys)
        while queue:
            cur = queue.popleft()
            for dkey, entry in self.derived.items():
                if not entry["valid"] or dkey in visited:
                    continue
                if cur in entry["reads"]:
                    entry["valid"] = False
                    invalidated.append(dkey)
                    visited.add(dkey)
                    queue.append(dkey)
        return invalidated

    def mutate_fact(self, fact_key: str, new_value: Any) -> Dict[str, Any]:
        """Mutate or insert a fact, skip no-op writes, bump generation, and transitively invalidate."""
        existed = fact_key in self.facts
        if existed and self.facts[fact_key] == new_value:
            return {
                "generation": self.generation,
                "fact_key": fact_key,
                "invalidated": [],
                "noop": True,
            }

        self.facts[fact_key] = new_value
        self.tombstones.pop(fact_key, None)
        self.generation += 1
        self.fact_records[fact_key] = Fact(id=fact_key, value=new_value, generation=self.generation)

        direct_invalidated: List[str] = []
        for dkey, entry in self.derived.items():
            if not entry["valid"]:
                continue
            hit_pos = fact_key in entry["reads"]
            hit_neg = (not existed) and (fact_key in entry["neg_reads"])
            if hit_pos or hit_neg:
                entry["valid"] = False
                direct_invalidated.append(dkey)

        cascaded = self._propagate_invalidation(direct_invalidated)
        return {
            "generation": self.generation,
            "fact_key": fact_key,
            "invalidated": sorted(direct_invalidated + cascaded),
            "noop": False,
        }

    def retract_fact(self, fact_key: str) -> Dict[str, Any]:
        """Retract a fact with a monotonic tombstone and transitively invalidate dependent derivations."""
        existed = fact_key in self.facts
        if not existed:
            return {
                "generation": self.generation,
                "fact_key": fact_key,
                "tombstoned": fact_key in self.tombstones,
                "invalidated": [],
                "noop": True,
            }

        del self.facts[fact_key]
        self.generation += 1
        self.tombstones[fact_key] = self.generation
        self.fact_records[fact_key] = Fact(id=fact_key, value=None, generation=self.generation, tombstone=True)

        direct_invalidated: List[str] = []
        for dkey, entry in self.derived.items():
            if not entry["valid"]:
                continue
            if fact_key in entry["reads"]:
                entry["valid"] = False
                direct_invalidated.append(dkey)

        cascaded = self._propagate_invalidation(direct_invalidated)
        return {
            "generation": self.generation,
            "fact_key": fact_key,
            "tombstoned": True,
            "tombstone_generation": self.generation,
            "invalidated": sorted(direct_invalidated + cascaded),
            "noop": False,
        }


@dataclass(frozen=True)
class TransitionRule:
    """M0 Primitive #2: Declarative state-transition & side-effect/authority boundary rule."""

    name: str
    from_states: Tuple[str, ...]
    to_state: str
    side_effect: str = "none"
    required_authority: str = "none"


def verify_state_transition_system(
    solver: FiniteDomainCSPSolver,
    state_domain: Sequence[str],
    transitions: Sequence[TransitionRule],
    invariants: Sequence[NamedConstraint],
    granted_authorities: Iterable[str] = (),
    node_budget: int = 10000,
) -> Verdict:
    """Verify an explicit state-transition system against M0 authority boundaries (Rule A4/R5)
    and post-state invariants (Primitive #1 & #2)."""
    granted = set(granted_authorities) | {"none"}
    state_set = set(state_domain)
    total_nodes = 0

    for tr in transitions:
        if tr.required_authority not in granted:
            return UnresolvedHoleVerdict(
                hole=UnresolvedHole(
                    kind="authority_missing",
                    missing_ref=f"{tr.name}:{tr.required_authority}",
                    required_role=tr.required_authority,
                    details=f"Transition {tr.name!r} triggers side_effect={tr.side_effect!r} without granted authority {tr.required_authority!r}.",
                ),
                nodes_visited=total_nodes,
            )
        if tr.to_state not in state_set or any(s not in state_set for s in tr.from_states):
            return RefuteVerdict(
                counterexample={"transition": tr.name, "from_states": tr.from_states, "to_state": tr.to_state},
                dep_closure=(tr.name,),
                nodes_visited=total_nodes,
            )

    for tr in transitions:
        for pre in tr.from_states:
            rem = max(0, node_budget - total_nodes)
            if rem == 0:
                return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)
            step_domains = {"pre_state": [pre], "post_state": [tr.to_state], "side_effect": [tr.side_effect]}
            verdict = solver.solve_tiered(step_domains, invariants, node_budget=rem)
            total_nodes += verdict.nodes_visited
            if isinstance(verdict, ResourceLimitVerdict):
                return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)
            if isinstance(verdict, ConflictVerdict):
                return RefuteVerdict(
                    counterexample={
                        "transition": tr.name,
                        "pre_state": pre,
                        "post_state": tr.to_state,
                        "side_effect": tr.side_effect,
                        "violated_invariants": verdict.mus,
                    },
                    dep_closure=verdict.mus,
                    nodes_visited=total_nodes,
                )
            if isinstance(verdict, UnresolvedHoleVerdict):
                return verdict

    return SupportVerdict(
        witness={"verified_transitions": tuple(t.name for t in transitions), "state_count": len(state_set)},
        dep_closure=tuple(c.name for c in invariants),
        nodes_visited=total_nodes,
    )


def extract_gift_variant_states(module_spec: Dict[str, Any], variant_def_name: str) -> Tuple[str, ...]:
    """Extract the closed finite state domain from a `gift.author-module/design-converged`
    or `gift.m0-workspace/1` variant type definition (`core:type` -> `tag == 'variant'`)."""
    modules = module_spec.get("modules", [module_spec]) if module_spec.get("schema") == "gift.m0-workspace/1" else [module_spec]
    for mod in modules:
        defs = mod.get("definitions", {})
        if variant_def_name in defs:
            t_body = defs[variant_def_name].get("body", {}).get("type", {})
            if t_body.get("tag") != "variant" or not isinstance(t_body.get("cases"), dict) or not t_body["cases"]:
                raise ValueError(f"Definition {variant_def_name!r} is not a non-empty closed Gift variant type.")
            return tuple(sorted(t_body["cases"].keys()))
    raise ValueError(f"Variant definition {variant_def_name!r} not found in Gift module or workspace.")


def verify_gift_module_transition_contract(
    solver: FiniteDomainCSPSolver,
    module_spec: Dict[str, Any],
    variant_def_name: str,
    transitions: Sequence[TransitionRule],
    invariants: Sequence[NamedConstraint],
    granted_authorities: Iterable[str] = (),
    dep_graph: Optional[DependencyGraph] = None,
    contract_id: str = "gift_m0_contract",
    node_budget: int = 10000,
) -> Verdict:
    """Cross-Repo M0 Bridge: Verify a Gift module's closed variant state machine against
    declarative TransitionRules, authority boundaries (A4/R5), and hard/soft invariants,
    while recording causal dependency edges (`DepEdge`) for incremental invalidation (Primitive #4)."""
    if module_spec.get("schema") not in ("gift.author-module/design-converged", "gift.m0-workspace/1"):
        return UnresolvedHoleVerdict(
            hole=UnresolvedHole(
                kind="binding_missing",
                missing_ref="schema:gift.author-module/design-converged",
                details="Input module does not carry the design-converged Gift module schema.",
            ),
            nodes_visited=0,
        )
    state_domain = extract_gift_variant_states(module_spec, variant_def_name)
    verdict = verify_state_transition_system(
        solver=solver,
        state_domain=state_domain,
        transitions=transitions,
        invariants=invariants,
        granted_authorities=granted_authorities,
        node_budget=node_budget,
    )
    if dep_graph is not None:
        reads = [f"gift_type:{variant_def_name}"] + [f"transition:{t.name}" for t in transitions] + [f"inv:{c.name}" for c in invariants]
        neg_reads = [f"revoke:{a}" for a in granted_authorities]
        dep_graph.record_derivation(contract_id, verdict.tag, reads=reads, neg_reads=neg_reads)
    return verdict


def verify_k_step_trajectory_safety(
    solver: FiniteDomainCSPSolver,
    state_domain: Sequence[str],
    initial_states: Sequence[str],
    transitions: Sequence[TransitionRule],
    step_invariants: Sequence[NamedConstraint],
    trajectory_invariants: Sequence[NamedConstraint] = (),
    granted_authorities: Iterable[str] = (),
    max_steps: int = 3,
    node_budget: int = 10000,
) -> Verdict:
    """Bounded k-step Model Checking & Multi-Step Counterexample Witness Hunter (M0 Primitives #1-#5).

    Explores all reachable state trajectories up to horizon `max_steps` from `initial_states`
    under `transitions`, strictly metered by `node_budget` (Rule R6):
      - Any reachable transition requiring an ungranted authority halts with
        UnresolvedHoleVerdict(kind='authority_missing', details=exact trace prefix).
      - Any reachable step or multi-step prefix violating `step_invariants` or
        `trajectory_invariants` (which inspect `trace_states` and `trace_effects`)
        halts immediately with RefuteVerdict containing the shortest multi-step counterexample witness.
      - Exceeding `node_budget` aborts honestly with ResourceLimitVerdict.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    granted = set(granted_authorities) | {"none"}
    state_set = set(state_domain)
    total_nodes = 0

    for s0 in initial_states:
        if s0 not in state_set:
            return RefuteVerdict(
                counterexample={"invalid_initial_state": s0, "state_domain": tuple(state_domain)},
                dep_closure=("initial_states",),
                nodes_visited=total_nodes,
            )

    by_from: Dict[str, List[TransitionRule]] = {s: [] for s in state_domain}
    for tr in transitions:
        if tr.to_state not in state_set or any(s not in state_set for s in tr.from_states):
            return RefuteVerdict(
                counterexample={"transition": tr.name, "from_states": tr.from_states, "to_state": tr.to_state},
                dep_closure=(tr.name,),
                nodes_visited=total_nodes,
            )
        for pre in tr.from_states:
            by_from[pre].append(tr)

    # BFS queue of (current_state, trace_states, trace_transitions, trace_effects)
    queue: deque[Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]] = deque(
        (s0, (s0,), (), ()) for s0 in initial_states
    )
    trajectories_checked = 0

    while queue:
        cur_state, trace_states, trace_trs, trace_effs = queue.popleft()
        if len(trace_trs) >= max_steps:
            continue
        for tr in by_from.get(cur_state, ()):
            total_nodes += 1
            if total_nodes > node_budget:
                return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)

            next_states = trace_states + (tr.to_state,)
            next_trs = trace_trs + (tr.name,)
            next_effs = trace_effs + (tr.side_effect,)

            if tr.required_authority not in granted:
                return UnresolvedHoleVerdict(
                    hole=UnresolvedHole(
                        kind="authority_missing",
                        missing_ref=f"{tr.name}:{tr.required_authority}",
                        required_role=tr.required_authority,
                        details=f"Reachable at step {len(next_trs)} via trace {next_trs}: missing {tr.required_authority!r}.",
                    ),
                    nodes_visited=total_nodes,
                )

            rem = max(0, node_budget - total_nodes)
            if rem == 0:
                return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)

            step_domains = {
                "pre_state": [cur_state],
                "post_state": [tr.to_state],
                "side_effect": [tr.side_effect],
                "step": [len(next_trs)],
                "trace_states": [next_states],
                "trace_transitions": [next_trs],
                "trace_effects": [next_effs],
            }
            all_invs = list(step_invariants) + list(trajectory_invariants)
            verdict = solver.solve_tiered(step_domains, all_invs, node_budget=rem)
            total_nodes += verdict.nodes_visited
            if isinstance(verdict, ResourceLimitVerdict):
                return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)
            if isinstance(verdict, ConflictVerdict):
                return RefuteVerdict(
                    counterexample={
                        "step": len(next_trs),
                        "trace_states": next_states,
                        "trace_transitions": next_trs,
                        "trace_effects": next_effs,
                        "violated_invariants": verdict.mus,
                    },
                    dep_closure=verdict.mus,
                    nodes_visited=total_nodes,
                )
            if isinstance(verdict, UnresolvedHoleVerdict):
                return verdict

            trajectories_checked += 1
            queue.append((tr.to_state, next_states, next_trs, next_effs))

    return SupportVerdict(
        witness={"max_steps": max_steps, "trajectories_checked": trajectories_checked},
        dep_closure=tuple(c.name for c in list(step_invariants) + list(trajectory_invariants)),
        nodes_visited=total_nodes,
    )


def synthesize_k_step_transition_repair(
    solver: FiniteDomainCSPSolver,
    state_domain: Sequence[str],
    initial_states: Sequence[str],
    transitions: Sequence[TransitionRule],
    step_invariants: Sequence[NamedConstraint],
    trajectory_invariants: Sequence[NamedConstraint] = (),
    granted_authorities: Iterable[str] = (),
    max_steps: int = 3,
    node_budget: int = 10000,
) -> Verdict:
    """Transition-Level MUS/MCS Counterexample Repair Synthesizer (M0 Primitives #1-#5).

    1. Runs `verify_k_step_trajectory_safety` under a shared total `node_budget` (Rule R6).
    2. If safe (`SupportVerdict`), unresolved (`UnresolvedHoleVerdict`), or budget-exhausted
       (`ResourceLimitVerdict`), returns that verdict immediately.
    3. If a k-step counterexample trajectory is found (`RefuteVerdict`):
       - Extracts the exact transition-level Minimal Unsatisfiable Subset (`mus`): a minimal
         subset of transitions that still triggers a k-step invariant violation.
       - Extracts a greedy Maximal Safe Transition Subset (`mcs` / `safe_subset`) that satisfies
         all step and trajectory invariants up to `max_steps`, plus `relaxable` (the minimal
         transitions to gate, make idempotent, or disable to restore safety).
       - Returns `ConflictVerdict(kind='unsat_core', mus=..., mcs=..., relaxable=...)`.
    """
    total_nodes = 0

    def _run_bmc(tr_sub: Sequence[TransitionRule]) -> Verdict:
        nonlocal total_nodes
        rem = max(0, node_budget - total_nodes)
        if rem == 0:
            return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=0)
        v = verify_k_step_trajectory_safety(
            solver=solver,
            state_domain=state_domain,
            initial_states=initial_states,
            transitions=tr_sub,
            step_invariants=step_invariants,
            trajectory_invariants=trajectory_invariants,
            granted_authorities=granted_authorities,
            max_steps=max_steps,
            node_budget=rem,
        )
        total_nodes += v.nodes_visited
        return v

    base = _run_bmc(transitions)
    if isinstance(base, ResourceLimitVerdict):
        return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)
    if not isinstance(base, RefuteVerdict):
        return base

    # Seed MUS deletion from the counterexample's active transitions if present, else full set
    ce_trs = set(base.counterexample.get("trace_transitions", ()))
    seed_mus = [t for t in transitions if t.name in ce_trs] if ce_trs else list(transitions)
    mus: List[TransitionRule] = list(seed_mus)
    i = 0
    while i < len(mus):
        trial = mus[:i] + mus[i + 1 :]
        v = _run_bmc(trial)
        if isinstance(v, ResourceLimitVerdict):
            return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)
        if isinstance(v, RefuteVerdict):
            mus = trial
        else:
            i += 1

    # Greedy Maximal Safe Transition Subset (MCS)
    safe: List[TransitionRule] = []
    for t in transitions:
        v = _run_bmc(safe + [t])
        if isinstance(v, ResourceLimitVerdict):
            return ResourceLimitVerdict(budget_kind="nodes", nodes_visited=total_nodes)
        if isinstance(v, SupportVerdict):
            safe.append(t)

    safe_names = {s.name for s in safe}
    return ConflictVerdict(
        kind="unsat_core",
        mus=tuple(t.name for t in mus),
        mcs=tuple(t.name for t in safe),
        relaxable=tuple(t.name for t in transitions if t.name not in safe_names),
        nodes_visited=total_nodes,
    )


def verify_and_repair_gift_module_k_step_contract(
    solver: FiniteDomainCSPSolver,
    module_spec: Dict[str, Any],
    variant_def_name: str,
    initial_states: Sequence[str],
    transitions: Sequence[TransitionRule],
    step_invariants: Sequence[NamedConstraint],
    trajectory_invariants: Sequence[NamedConstraint] = (),
    granted_authorities: Iterable[str] = (),
    dep_graph: Optional[DependencyGraph] = None,
    contract_id: str = "gift_m0_k_step_contract",
    max_steps: int = 3,
    node_budget: int = 10000,
) -> Verdict:
    """End-to-End Cross-Repo M0 Bridge (Primitives #1-#5 Unified):
    1. Extracts the closed finite state domain from a `gift.author-module/design-converged`
       variant type definition (`extract_gift_variant_states`).
    2. Runs `synthesize_k_step_transition_repair` up to horizon `max_steps` under shared
       budget `node_budget` (Rule R6) to either certify multi-step trajectory safety
       (`SupportVerdict`), escalate missing authority (`UnresolvedHoleVerdict`), or
       isolate the exact transition-level MUS blame core + MCS safe transition subset
       (`ConflictVerdict`).
    3. Registers positive (`reads`: variant type, transitions, step & trajectory invariants)
       and negative (`neg_reads`: `revoke:<authority>`, `disable:<transition>`) dependency
       edges in `dep_graph` for automatic transitive invalidation (Rule R4).
    """
    if module_spec.get("schema") not in ("gift.author-module/design-converged", "gift.m0-workspace/1"):
        return UnresolvedHoleVerdict(
            hole=UnresolvedHole(
                kind="binding_missing",
                missing_ref="schema:gift.author-module/design-converged",
                details="Input module does not carry the design-converged Gift module schema.",
            ),
            nodes_visited=0,
        )
    state_domain = extract_gift_variant_states(module_spec, variant_def_name)
    verdict = synthesize_k_step_transition_repair(
        solver=solver,
        state_domain=state_domain,
        initial_states=initial_states,
        transitions=transitions,
        step_invariants=step_invariants,
        trajectory_invariants=trajectory_invariants,
        granted_authorities=granted_authorities,
        max_steps=max_steps,
        node_budget=node_budget,
    )
    if dep_graph is not None:
        all_invs = list(step_invariants) + list(trajectory_invariants)
        reads = (
            [f"gift_type:{variant_def_name}"]
            + [f"transition:{t.name}" for t in transitions]
            + [f"inv:{c.name}" for c in all_invs]
        )
        neg_reads = [f"revoke:{a}" for a in granted_authorities] + [f"disable:{t.name}" for t in transitions]
        dep_graph.record_derivation(contract_id, verdict.tag, reads=reads, neg_reads=neg_reads)
    return verdict


def compile_and_verify_gift_m0_via_rust_kernel(
    solver: FiniteDomainCSPSolver,
    kernel_binary_path: str,
    module_spec: Dict[str, Any],
    variant_def_name: str,
    initial_states: Sequence[str],
    transitions: Sequence[TransitionRule],
    step_invariants: Sequence[NamedConstraint],
    trajectory_invariants: Sequence[NamedConstraint] = (),
    granted_authorities: Iterable[str] = (),
    dep_graph: Optional[DependencyGraph] = None,
    contract_id: str = "gift_rust_m0_pipeline",
    max_steps: int = 3,
    node_budget: int = 10000,
    timeout_seconds: float = 15.0,
    differential_probes: Sequence[Dict[str, str]] = (),
) -> Verdict:
    """True Cross-Process Rust+Python M0 Pipeline (`gift-author-kernel --emit-m0` + `world_kernel_engine`):
    1. Invokes the compiled Rust `gift-author-kernel --emit-m0` binary over bounded stdin/stdout
       to validate grammar, lexical scopes, types, pattern exhaustiveness, and effect purity.
    2. If the Rust kernel rejects the module or halts on an M0BoundaryCategory:
       - `ResourceLimit` -> returns `ResourceLimitVerdict(budget_kind='rust_kernel')`
       - `SemanticRejection` -> returns `RefuteVerdict(counterexample=...)`
       - `OutsideM0Carrier` / `UninterpretedDialect` / `UnresolvedInput` -> returns `UnresolvedHoleVerdict`
    3. If the Rust kernel certifies `verdict.tag == 'complete'` and emits `lib.rs`, runs
       `verify_and_repair_gift_module_k_step_contract` to verify k-step trajectory safety and
       authority boundaries, attaching the emitted Rust source length & exports to the witness.
    """
    payload = json.dumps(module_spec).encode("utf-8")
    try:
        proc = subprocess.run(
            [kernel_binary_path, "--emit-m0"],
            input=payload,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ResourceLimitVerdict(budget_kind="rust_kernel_timeout", nodes_visited=0)

    try:
        report = json.loads(proc.stdout.decode("utf-8"))
    except Exception as exc:
        return RefuteVerdict(
            counterexample={"stage": "rust_kernel_transport", "error": str(exc), "returncode": proc.returncode},
            dep_closure=("gift_author_kernel",),
            nodes_visited=0,
        )

    if proc.returncode != 0 or report.get("schema") == "gift.author-compilation-failure/1":
        diag = report.get("diagnostic", {})
        if diag.get("code") == "ResourceLimit":
            return ResourceLimitVerdict(budget_kind="rust_kernel", nodes_visited=0)
        return RefuteVerdict(
            counterexample={"stage": "rust_kernel_diagnostic", "diagnostic": diag},
            dep_closure=(diag.get("code", "unknown"),),
            nodes_visited=0,
        )

    m0_verdict = report.get("verdict", {})
    res = report.get("result", {})
    if m0_verdict.get("tag") != "complete":
        cats = tuple(m0_verdict.get("categories", ()))
        if "resource_limit" in cats:
            return ResourceLimitVerdict(budget_kind="rust_kernel", nodes_visited=0)
        if "semantic_rejection" in cats:
            return RefuteVerdict(
                counterexample={"stage": "rust_kernel_semantic_rejection", "problems": res.get("problems", [])},
                dep_closure=cats,
                nodes_visited=0,
            )
        return UnresolvedHoleVerdict(
            hole=UnresolvedHole(
                kind="binding_missing",
                missing_ref=",".join(cats) or "rust_m0_boundary",
                details=f"Rust M0 boundary halted with categories={cats}, needs={res.get('needs', [])}",
            ),
            nodes_visited=0,
        )

    verdict = verify_and_repair_gift_module_k_step_contract(
        solver=solver,
        module_spec=module_spec,
        variant_def_name=variant_def_name,
        initial_states=initial_states,
        transitions=transitions,
        step_invariants=step_invariants,
        trajectory_invariants=trajectory_invariants,
        granted_authorities=granted_authorities,
        dep_graph=dep_graph,
        contract_id=contract_id,
        max_steps=max_steps,
        node_budget=node_budget,
    )
    if isinstance(verdict, SupportVerdict):
        artifacts = {a.get("path", ""): a.get("content", "") for a in res.get("artifacts", [])}
        emitted_rs = artifacts.get("lib.rs", "")
        emitted_js = artifacts.get("index.mjs", "")
        exports = tuple(e.get("rust_name", "") for e in res.get("exports", []))
        source_map = res.get("source_map", [])
        def_counts: Dict[str, int] = {}
        for sm in source_map:
            dname = sm.get("definition", "")
            if dname:
                def_counts[dname] = def_counts.get(dname, 0) + 1
        all_mods = module_spec.get("modules", [module_spec]) if module_spec.get("schema") == "gift.m0-workspace/1" else [module_spec]
        generic_defs: List[str] = []
        for mod in all_mods:
            for dname, dval in mod.get("definitions", {}).items():
                if isinstance(dval, dict) and isinstance(dval.get("body"), dict) and dval["body"].get("typeParams"):
                    generic_defs.append(dname)
        mono_fns = tuple(sorted(set(k for k, v in def_counts.items() if v > 1 or k in generic_defs)))

        if differential_probes:
            crate_name = res.get("context", {}).get("crate_name", "gift_m0_crate")
            scratch_base = os.path.dirname(os.path.abspath(kernel_binary_path))
            scratch_dir = os.path.join(scratch_base, f"diff_probe_{os.getpid()}")
            shutil.rmtree(scratch_dir, ignore_errors=True)
            os.makedirs(scratch_dir, exist_ok=True)
            try:
                lib_rs_path = os.path.join(scratch_dir, "lib.rs")
                rlib_path = os.path.join(scratch_dir, f"lib{crate_name}.rlib")
                index_mjs_path = os.path.join(scratch_dir, "index.mjs")
                with open(lib_rs_path, "w", encoding="utf-8") as f:
                    f.write(emitted_rs)
                with open(index_mjs_path, "w", encoding="utf-8") as f:
                    f.write(emitted_js)
                c_lib = subprocess.run(
                    ["rustc", "--edition", "2024", "--crate-type", "lib", "-o", rlib_path, lib_rs_path],
                    capture_output=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                if c_lib.returncode != 0:
                    return RefuteVerdict(
                        counterexample={"stage": "rustc_lib_compile", "stderr": c_lib.stderr.decode("utf-8", errors="replace")},
                        dep_closure=(contract_id, "rustc_compile"),
                        nodes_visited=verdict.nodes_visited,
                    )
                for idx, probe in enumerate(differential_probes):
                    p_name = probe.get("name", f"probe_{idx}")
                    r_expr = probe["rust_expr"]
                    j_expr = probe["js_expr"]
                    expected = probe["expected"]
                    drv_rs = os.path.join(scratch_dir, f"drv_{idx}.rs")
                    drv_bin = os.path.join(scratch_dir, f"drv_{idx}")
                    drv_mjs = os.path.join(scratch_dir, f"drv_{idx}.mjs")
                    with open(drv_rs, "w", encoding="utf-8") as f:
                        f.write(f"extern crate {crate_name};\nfn main() {{ println!(\"{{}}\", {r_expr}); }}\n")
                    with open(drv_mjs, "w", encoding="utf-8") as f:
                        f.write(f"import * as mod from './index.mjs';\nconsole.log(String({j_expr}));\n")
                    c_drv = subprocess.run(
                        ["rustc", "--edition", "2024", "--extern", f"{crate_name}={rlib_path}", "-o", drv_bin, drv_rs],
                        capture_output=True,
                        timeout=timeout_seconds,
                        check=False,
                    )
                    if c_drv.returncode != 0:
                        return RefuteVerdict(
                            counterexample={"stage": "rustc_driver_compile", "probe": p_name, "stderr": c_drv.stderr.decode("utf-8", errors="replace")},
                            dep_closure=(contract_id, p_name),
                            nodes_visited=verdict.nodes_visited,
                        )
                    r_run = subprocess.run([drv_bin], capture_output=True, timeout=timeout_seconds, check=False)
                    j_run = subprocess.run(["node", drv_mjs], capture_output=True, timeout=timeout_seconds, check=False)
                    r_out = r_run.stdout.decode("utf-8", errors="replace").strip()
                    j_out = j_run.stdout.decode("utf-8", errors="replace").strip()
                    if r_run.returncode != 0 or j_run.returncode != 0 or r_out != j_out or r_out != expected:
                        return RefuteVerdict(
                            counterexample={
                                "stage": "dual_target_differential_probe",
                                "probe": p_name,
                                "rust_out": r_out,
                                "js_out": j_out,
                                "expected": expected,
                            },
                            dep_closure=(contract_id, p_name),
                            nodes_visited=verdict.nodes_visited,
                        )
            finally:
                shutil.rmtree(scratch_dir, ignore_errors=True)

        if dep_graph is not None:
            if generic_defs:
                existing_reads = set(dep_graph.derived.get(contract_id, {}).get("reads", ()))
                existing_neg = set(dep_graph.derived.get(contract_id, {}).get("neg_reads", ()))
                dep_graph.record_derivation(
                    contract_id,
                    verdict.tag,
                    reads=existing_reads | {f"generic_def:{g}" for g in generic_defs},
                    neg_reads=existing_neg,
                )
            if module_spec.get("schema") == "gift.m0-workspace/1":
                mods = module_spec.get("modules", [])
                for mod in mods:
                    mid = mod.get("moduleId", "")
                    imports = mod.get("imports", {})
                    upstreams = [f"module:{imp.get('moduleId')}" for imp in imports.values() if isinstance(imp, dict) and imp.get("moduleId")]
                    dep_graph.record_derivation(f"module:{mid}", "verified", reads=[f"module_rev:{mid}"] + upstreams)
                root_mid = module_spec.get("rootModuleId") or (mods[-1].get("moduleId", "") if mods else "")
                if root_mid:
                    existing_reads = set(dep_graph.derived.get(contract_id, {}).get("reads", ()))
                    existing_neg = set(dep_graph.derived.get(contract_id, {}).get("neg_reads", ()))
                    dep_graph.record_derivation(
                        contract_id,
                        verdict.tag,
                        reads=existing_reads | {f"module:{root_mid}"},
                        neg_reads=existing_neg,
                    )
        return SupportVerdict(
            witness={
                **verdict.witness,
                "rust_crate_verified": bool(emitted_rs),
                "js_module_verified": bool(emitted_js),
                "artifact_paths": tuple(sorted(artifacts.keys())),
                "rust_exports": exports,
                "monomorphized_functions": mono_fns,
                "differential_probes_passed": len(differential_probes),
                "emitted_rust_bytes": len(emitted_rs),
                "emitted_js_bytes": len(emitted_js),
            },
            dep_closure=verdict.dep_closure,
            satisfied_soft=verdict.satisfied_soft,
            dropped_soft=verdict.dropped_soft,
            nodes_visited=verdict.nodes_visited,
        )
    return verdict


def resolve_gift_kernel_binary(explicit_path: Optional[str] = None) -> Optional[str]:
    """Resolve the `gift-author-kernel` CLI binary without hardcoding sibling layout:
    1. Explicit path argument;
    2. `GIFT_KERNEL_BIN` environment variable (recommended cross-repo CI/runtime contract);
    3. `gift-author-kernel` on system `PATH`;
    4. Standard sibling workspace build outputs (`sec_repo/target/debug`, etc.).
    """
    candidates: List[str] = []
    if explicit_path:
        candidates.append(explicit_path)
    env_bin = os.environ.get("GIFT_KERNEL_BIN")
    if env_bin:
        candidates.append(env_bin)
    which_bin = shutil.which("gift-author-kernel")
    if which_bin:
        candidates.append(which_bin)
    here = os.path.dirname(os.path.abspath(__file__))
    workspace_parent = os.path.abspath(os.path.join(here, "..", "..", ".."))
    for rel in (
        ("sec_repo", "target", "debug", "gift-author-kernel"),
        ("sec_repo", "target", "release", "gift-author-kernel"),
        ("sec_repo", "crates", "gift-author-kernel", "target", "debug", "gift-author-kernel"),
        ("sec_repo", ".tmp", "gift-author-kernel-target", "debug", "gift-author-kernel"),
    ):
        candidates.append(os.path.join(workspace_parent, *rel))
    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return os.path.abspath(cand)
    return None


def _require_exact_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an exact int (not bool/float), got {value!r}")
    return value


@dataclass(frozen=True)
class FourDTimeCut:
    """4D Bitemporal evaluation cut: (fact_time, obs_time, norm_time, judge_time)."""

    fact_time: int
    obs_time: int
    norm_time: int
    judge_time: int

    def __post_init__(self) -> None:
        _require_exact_int("fact_time", self.fact_time)
        _require_exact_int("obs_time", self.obs_time)
        _require_exact_int("norm_time", self.norm_time)
        _require_exact_int("judge_time", self.judge_time)
        if self.judge_time < self.obs_time:
            raise ValueError(
                f"Causal violation: judge_time ({self.judge_time}) < obs_time ({self.obs_time})"
            )


@dataclass(frozen=True)
class BitemporalFactEntry:
    """Append-only immutable bitemporal fact or tombstone record."""

    fact_id: str
    value: Any
    valid_from: int
    valid_to: Optional[int]
    recorded_at: int
    generation: int
    tombstone: bool = False
    seq: int = 0

    def covers_fact_time(self, t_fact: int) -> bool:
        return self.valid_from <= t_fact and (self.valid_to is None or t_fact < self.valid_to)


@dataclass(frozen=True)
class NormRuleVersion:
    """Immutable versioned normative predicate with explicit effective window & recording time."""

    norm_id: str
    version: int
    effective_from: int
    effective_to: Optional[int]
    recorded_at: int
    predicate: Callable[[Any], bool]
    hard: bool = True
    description: str = ""

    def active_at(self, t_norm: int, t_obs: int) -> bool:
        return (
            self.recorded_at <= t_obs
            and self.effective_from <= t_norm
            and (self.effective_to is None or t_norm < self.effective_to)
        )


@dataclass(frozen=True)
class FourDQueryResult:
    """Structured 4D bitemporal evaluation output preserving non-retroactivity & late-obs visibility."""

    cut: FourDTimeCut
    fact_id: str
    fact_status: str
    fact_value: Any
    fact_evidence: EvidenceBit
    resurrection_blocked: bool
    reconciliation_required: bool
    late_observation_detected: bool
    late_observations: Tuple[BitemporalFactEntry, ...]
    norm_id: Optional[str]
    norm_at_fact_version: Optional[int]
    norm_at_cut_version: Optional[int]
    compliant_at_fact_time: EvidenceBit
    compliant_at_norm_time: EvidenceBit
    retroactive_norm_detected: bool
    norm_temporal_status: str
    primary_verdict: EvidenceBit


class BitemporalFactLedger:
    """Append-only 4D bitemporal fact & norm ledger (zero silent history overwrite)."""

    def __init__(self) -> None:
        self._seq: int = 0
        self._fact_log: List[BitemporalFactEntry] = []
        self._norm_log: List[NormRuleVersion] = []

    @property
    def fact_history(self) -> Tuple[BitemporalFactEntry, ...]:
        return tuple(self._fact_log)

    def append_fact(
        self,
        fact_id: str,
        value: Any,
        valid_from: int,
        recorded_at: int,
        generation: int,
        valid_to: Optional[int] = None,
        tombstone: bool = False,
    ) -> BitemporalFactEntry:
        _require_exact_int("valid_from", valid_from)
        _require_exact_int("recorded_at", recorded_at)
        _require_exact_int("generation", generation)
        if valid_to is not None:
            _require_exact_int("valid_to", valid_to)
            if valid_to <= valid_from:
                raise ValueError(f"valid_to ({valid_to}) must be > valid_from ({valid_from})")
        for prev in self._fact_log:
            if (
                prev.fact_id == fact_id
                and prev.valid_from == valid_from
                and prev.valid_to == valid_to
                and prev.generation == generation
                and prev.recorded_at == recorded_at
            ):
                if prev.value != value or prev.tombstone != tombstone:
                    raise ValueError(
                        f"Silent history overwrite forbidden for {fact_id!r} at gen={generation}, recorded_at={recorded_at}"
                    )
                return prev
        self._seq += 1
        entry = BitemporalFactEntry(
            fact_id=fact_id,
            value=None if tombstone else value,
            valid_from=valid_from,
            valid_to=valid_to,
            recorded_at=recorded_at,
            generation=generation,
            tombstone=tombstone,
            seq=self._seq,
        )
        self._fact_log.append(entry)
        return entry

    def register_norm(
        self,
        norm_id: str,
        version: int,
        effective_from: int,
        recorded_at: int,
        predicate: Callable[[Any], bool],
        effective_to: Optional[int] = None,
        hard: bool = True,
        description: str = "",
    ) -> NormRuleVersion:
        _require_exact_int("version", version)
        _require_exact_int("effective_from", effective_from)
        _require_exact_int("recorded_at", recorded_at)
        if effective_to is not None:
            _require_exact_int("effective_to", effective_to)
            if effective_to <= effective_from:
                raise ValueError("effective_to must be > effective_from")
        for prev in self._norm_log:
            if prev.norm_id == norm_id and prev.version == version:
                raise ValueError(f"Duplicate norm version forbidden: {norm_id}@v{version}")
        nv = NormRuleVersion(
            norm_id=norm_id,
            version=version,
            effective_from=effective_from,
            effective_to=effective_to,
            recorded_at=recorded_at,
            predicate=predicate,
            hard=hard,
            description=description,
        )
        self._norm_log.append(nv)
        return nv

    def _resolve_norm(self, norm_id: str, t_norm: int, t_obs: int) -> Optional[NormRuleVersion]:
        cands = [n for n in self._norm_log if n.norm_id == norm_id and n.active_at(t_norm, t_obs)]
        if not cands:
            return None
        return max(cands, key=lambda n: (n.version, n.effective_from, n.recorded_at))

    def query_at(
        self,
        cut: FourDTimeCut,
        fact_id: str,
        norm_id: Optional[str] = None,
        dep_graph: Optional[DependencyGraph] = None,
        derivation_key: Optional[str] = None,
    ) -> FourDQueryResult:
        matching_fact_time = [
            e for e in self._fact_log if e.fact_id == fact_id and e.covers_fact_time(cut.fact_time)
        ]
        visible = [e for e in matching_fact_time if e.recorded_at <= cut.obs_time]
        late_obs = tuple(
            e for e in matching_fact_time if cut.obs_time < e.recorded_at <= cut.judge_time
        )

        fact_status = "ABSENT"
        fact_value: Any = None
        fact_evidence = NEITHER
        resurrection_blocked = False
        reconciliation_required = False

        if visible:
            max_gen = max(e.generation for e in visible)
            top_entries = [e for e in visible if e.generation == max_gen]
            tombstones = [e for e in top_entries if e.tombstone]
            live_top = [e for e in top_entries if not e.tombstone]

            if tombstones:
                fact_status = "TOMBSTONED"
                fact_evidence = REFUTE
                tomb_rec_at = min(t.recorded_at for t in tombstones)
                late_lower_live = [
                    e for e in visible
                    if (not e.tombstone) and e.generation < max_gen and e.recorded_at > tomb_rec_at
                ]
                if late_lower_live or live_top:
                    resurrection_blocked = True
                    reconciliation_required = True
            else:
                distinct_vals = []
                for e in live_top:
                    if e.value not in distinct_vals:
                        distinct_vals.append(e.value)
                if len(distinct_vals) > 1:
                    fact_status = "CONFLICT"
                    fact_value = tuple(distinct_vals)
                    fact_evidence = BOTH
                else:
                    fact_status = "ACTIVE"
                    fact_value = distinct_vals[0]
                    fact_evidence = SUPPORT

        norm_fact: Optional[NormRuleVersion] = None
        norm_cut: Optional[NormRuleVersion] = None
        comp_fact = NEITHER
        comp_cut = NEITHER
        retroactive_norm_detected = False
        norm_status = "NO_NORM_REQUESTED"

        if norm_id is not None:
            norm_fact = self._resolve_norm(norm_id, cut.fact_time, cut.obs_time)
            norm_cut = self._resolve_norm(norm_id, cut.norm_time, cut.obs_time)
            if fact_status == "CONFLICT":
                comp_fact = BOTH
                comp_cut = BOTH
                norm_status = "FACT_CONFLICT"
            elif fact_status != "ACTIVE":
                norm_status = "INDETERMINATE_NO_ACTIVE_FACT"
            else:
                if norm_fact is not None:
                    comp_fact = SUPPORT if norm_fact.predicate(fact_value) else REFUTE
                if norm_cut is not None:
                    comp_cut = SUPPORT if norm_cut.predicate(fact_value) else REFUTE

                if norm_fact is None and norm_cut is None:
                    norm_status = "INDETERMINATE_NO_NORM"
                elif cut.norm_time != cut.fact_time and comp_fact == SUPPORT and comp_cut == REFUTE:
                    retroactive_norm_detected = True
                    norm_status = "RETROACTIVE_NORM_DRIFT"
                elif cut.norm_time != cut.fact_time and comp_fact == REFUTE and comp_cut == SUPPORT:
                    retroactive_norm_detected = True
                    norm_status = "RETROACTIVE_RELAXATION"
                elif comp_fact == SUPPORT and comp_cut == SUPPORT:
                    norm_status = "CONSISTENT_COMPLIANT"
                elif comp_fact == REFUTE and comp_cut == REFUTE:
                    norm_status = "CONSISTENT_VIOLATION"
                else:
                    norm_status = "PARTIAL_NORM_COVERAGE"

        primary_verdict = comp_fact if norm_id is not None else fact_evidence

        if dep_graph is not None and derivation_key is not None:
            reads = [f"4d_fact:{fact_id}@t={cut.fact_time},obs<={cut.obs_time}"]
            if norm_id is not None:
                reads.append(f"4d_norm:{norm_id}@t={cut.fact_time}")
            dep_graph.record_derivation(
                derivation_key,
                primary_verdict.label,
                reads=reads,
                neg_reads=[f"tombstone:{fact_id}", f"late_obs:{fact_id}"],
            )

        return FourDQueryResult(
            cut=cut,
            fact_id=fact_id,
            fact_status=fact_status,
            fact_value=fact_value,
            fact_evidence=fact_evidence,
            resurrection_blocked=resurrection_blocked,
            reconciliation_required=reconciliation_required,
            late_observation_detected=len(late_obs) > 0,
            late_observations=late_obs,
            norm_id=norm_id,
            norm_at_fact_version=norm_fact.version if norm_fact else None,
            norm_at_cut_version=norm_cut.version if norm_cut else None,
            compliant_at_fact_time=comp_fact,
            compliant_at_norm_time=comp_cut,
            retroactive_norm_detected=retroactive_norm_detected,
            norm_temporal_status=norm_status,
            primary_verdict=primary_verdict,
        )


@dataclass(frozen=True)
class ChisholmNormSpec:
    """Stratified Chisholm Contrary-To-Duty (CTD) norm specification."""

    norm_id: str
    primary_obligation: str
    compliant_followup: str
    contrary_to_duty_compensation: str
    hard_redline_guards: Tuple[NamedConstraint, ...]
    soft_preferences: Tuple[NamedConstraint, ...] = ()


@dataclass(frozen=True)
class ChisholmEvaluationResult:
    """Explosion-free stratified deontic verdict enforcing R3 hard-redline dominance."""

    norm_id: str
    state: str
    active_obligations: Tuple[str, ...]
    suppressed_obligations: Tuple[str, ...]
    explosion_prevented: bool
    hard_redline_violated: Tuple[str, ...]
    verdict: Verdict


class ChisholmDeonticEngine:
    """Stratified Chisholm CTD Deontic Automaton with R3 Hard-Redline Dominance."""

    def __init__(self, solver: Optional[FiniteDomainCSPSolver] = None) -> None:
        self._solver = solver or FiniteDomainCSPSolver()

    def evaluate(
        self,
        spec: ChisholmNormSpec,
        action_assignment: Dict[str, Any],
        primary_observed: EvidenceBit,
        followup_observed: EvidenceBit = NEITHER,
        compensation_observed: EvidenceBit = NEITHER,
        candidate_domains: Optional[Dict[str, Sequence[Any]]] = None,
        node_budget: int = 2000,
    ) -> ChisholmEvaluationResult:
        violated_redlines = tuple(
            c.name
            for c in spec.hard_redline_guards
            if all(v in action_assignment for v in c.variables) and not c.predicate(action_assignment)
        )
        if violated_redlines:
            return ChisholmEvaluationResult(
                norm_id=spec.norm_id,
                state="HARD_REDLINE_HALT",
                active_obligations=(),
                suppressed_obligations=(
                    spec.primary_obligation,
                    spec.compliant_followup,
                    spec.contrary_to_duty_compensation,
                ),
                explosion_prevented=True,
                hard_redline_violated=violated_redlines,
                verdict=ConflictVerdict(
                    kind="unsat_core",
                    mus=violated_redlines,
                    mcs=(),
                    relaxable=tuple(c.name for c in spec.soft_preferences),
                ),
            )

        if candidate_domains is not None:
            hard_guards = [
                NamedConstraint(c.name, c.variables, c.predicate, hard=True, priority=c.priority)
                for c in spec.hard_redline_guards
            ]
            soft_prefs = [
                NamedConstraint(c.name, c.variables, c.predicate, hard=False, priority=c.priority)
                for c in spec.soft_preferences
            ]
            csp_verdict = self._solver.solve_tiered(
                candidate_domains,
                hard_guards + soft_prefs,
                node_budget=node_budget,
            )
            if isinstance(csp_verdict, ConflictVerdict):
                return ChisholmEvaluationResult(
                    norm_id=spec.norm_id,
                    state="HARD_REDLINE_HALT",
                    active_obligations=(),
                    suppressed_obligations=(
                        spec.primary_obligation,
                        spec.compliant_followup,
                        spec.contrary_to_duty_compensation,
                    ),
                    explosion_prevented=True,
                    hard_redline_violated=csp_verdict.mus,
                    verdict=csp_verdict,
                )

        if primary_observed == BOTH:
            return ChisholmEvaluationResult(
                norm_id=spec.norm_id,
                state="EVIDENCE_CONFLICT",
                active_obligations=(spec.primary_obligation,),
                suppressed_obligations=(spec.compliant_followup, spec.contrary_to_duty_compensation),
                explosion_prevented=True,
                hard_redline_violated=(),
                verdict=ConflictVerdict(kind="evidence_both", mus=(spec.primary_obligation,), mcs=()),
            )

        if primary_observed == NEITHER:
            return ChisholmEvaluationResult(
                norm_id=spec.norm_id,
                state="INDETERMINATE_PRIMARY",
                active_obligations=(spec.primary_obligation,),
                suppressed_obligations=(spec.compliant_followup, spec.contrary_to_duty_compensation),
                explosion_prevented=True,
                hard_redline_violated=(),
                verdict=UnresolvedHoleVerdict(
                    hole=UnresolvedHole(
                        kind="observable_ro",
                        missing_ref=f"{spec.norm_id}:{spec.primary_obligation}",
                        details="Primary obligation observation missing; refusing premature CTD activation.",
                    )
                ),
            )

        if primary_observed == SUPPORT:
            if followup_observed == BOTH:
                v: Verdict = ConflictVerdict(kind="evidence_both", mus=(spec.compliant_followup,), mcs=())
            elif followup_observed == SUPPORT:
                v = SupportVerdict(
                    witness={"branch": "compliant_followup", "fulfilled": spec.compliant_followup},
                    dep_closure=(spec.primary_obligation, spec.compliant_followup),
                )
            elif followup_observed == REFUTE:
                v = RefuteVerdict(
                    counterexample={"branch": "compliant_followup", "violated": spec.compliant_followup},
                    dep_closure=(spec.primary_obligation, spec.compliant_followup),
                )
            else:
                v = UnresolvedHoleVerdict(
                    hole=UnresolvedHole(
                        kind="observable_ro",
                        missing_ref=f"{spec.norm_id}:{spec.compliant_followup}",
                        details="Primary obligation fulfilled; awaiting compliant_followup observation.",
                    )
                )
            return ChisholmEvaluationResult(
                norm_id=spec.norm_id,
                state="COMPLIANT_FOLLOWUP_ACTIVE",
                active_obligations=(spec.compliant_followup,),
                suppressed_obligations=(spec.contrary_to_duty_compensation,),
                explosion_prevented=True,
                hard_redline_violated=(),
                verdict=v,
            )

        if compensation_observed == BOTH:
            v_ctd: Verdict = ConflictVerdict(
                kind="evidence_both", mus=(spec.contrary_to_duty_compensation,), mcs=()
            )
        elif compensation_observed == SUPPORT:
            v_ctd = SupportVerdict(
                witness={
                    "branch": "ctd_compensation",
                    "primary_breached": spec.primary_obligation,
                    "compensation_fulfilled": spec.contrary_to_duty_compensation,
                },
                dep_closure=(spec.primary_obligation, spec.contrary_to_duty_compensation),
            )
        elif compensation_observed == REFUTE:
            v_ctd = RefuteVerdict(
                counterexample={
                    "branch": "ctd_compensation",
                    "primary_breached": spec.primary_obligation,
                    "compensation_violated": spec.contrary_to_duty_compensation,
                },
                dep_closure=(spec.primary_obligation, spec.contrary_to_duty_compensation),
            )
        else:
            v_ctd = UnresolvedHoleVerdict(
                hole=UnresolvedHole(
                    kind="observable_ro",
                    missing_ref=f"{spec.norm_id}:{spec.contrary_to_duty_compensation}",
                    details="Primary obligation breached (~a); CTD compensation active and awaiting observation.",
                )
            )

        return ChisholmEvaluationResult(
            norm_id=spec.norm_id,
            state="CTD_COMPENSATION_ACTIVE",
            active_obligations=(spec.contrary_to_duty_compensation,),
            suppressed_obligations=(spec.compliant_followup,),
            explosion_prevented=True,
            hard_redline_violated=(),
            verdict=v_ctd,
        )
