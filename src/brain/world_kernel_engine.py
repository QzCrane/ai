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
        if dep_graph is not None and module_spec.get("schema") == "gift.m0-workspace/1":
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
                "emitted_rust_bytes": len(emitted_rs),
                "emitted_js_bytes": len(emitted_js),
            },
            dep_closure=verdict.dep_closure,
            satisfied_soft=verdict.satisfied_soft,
            dropped_soft=verdict.dropped_soft,
            nodes_visited=verdict.nodes_visited,
        )
    return verdict
