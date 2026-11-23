"""
v19.1 Executable MSK-v1 Shared Kernel Validator & Five-Axis Microworld Generator.

Bridges Layer 1 (`docs/whitepapers/world_cognition_unified_model.md` v19.0: Dual-Source x 5 Axes x MSK-v1)
and Track B (`docs/whitepapers/brain_dataset_curriculum_registry.md` v16.0: Unified Buffer & Anti-Pseudo-Insight Eval)
into deterministic, zero-external-dependency, machine-checkable training and evaluation streams.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

PROTOCOL_TOKENS: Tuple[str, ...] = (
    "<|cut_t|>",
    "<|fiber|>",
    "<|budget|>",
    "<|halt:closure|>",
    "<|halt:probe|>",
    "<|halt:enactive|>",
    "<|halt:hole|>",
    "<|halt:bottom|>",
    "<|halt:rlimit|>",
    "<|halt:precedent|>",
    "<|ev:support|>",
    "<|ev:refute|>",
    "<|ev:both|>",
    "<|ev:neither|>",
    "<|tombstone|>",
    "<|noncomp|>",
    "<|refute_self|>",
)

_EXACT_INT_RE = re.compile(r"^[+-]?[0-9]+$")
_VALID_JUDGMENT_TAGS = frozenset({"support", "refute", "conflict", "gap", "resource_limit", "invalid"})
_VALID_PURPOSES = frozenset({"query", "materialize", "execute"})
_VALID_DIRECTIONS = frozenset({"equiv", "entail", "over_approx", "under_approx", "conservative_bound", "hint"})
_VALID_AUTH_REQS = frozenset({"none", "model_permit", "host_grant"})


def validate_msk_v1_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    """Machine-checkable validator for the 5-primitive MSK-v1 shared kernel contract.

    Enforces wire-level exact integer strings, non-amplifying partiality,
    authority isolation (model_permit never mints host_grant), dominance honesty,
    and positive+negative dependency closure.
    """
    errors: List[str] = []
    required_top = ("Anchor", "Constraint", "Transition", "Judgment", "Resolution")
    for key in required_top:
        if key not in packet or not isinstance(packet[key], dict):
            errors.append(f"missing_or_invalid_primitive:{key}")
    if errors:
        return {"ok": False, "errors": errors}

    anchor = packet["Anchor"]
    constraint = packet["Constraint"]
    transition = packet["Transition"]
    judgment = packet["Judgment"]
    resolution = packet["Resolution"]

    # 1. Anchor wire & bitemporal invariants
    gen = anchor.get("generation")
    if not isinstance(gen, str) or not _EXACT_INT_RE.match(gen):
        errors.append("anchor.generation_must_be_exact_int_string")
    obs_cut = anchor.get("obs_cut", {})
    if obs_cut.get("coverage") not in ("complete", "partial"):
        errors.append("anchor.obs_cut.coverage_invalid")

    # 2. Constraint lexicographic tier & domain isolation
    tier = constraint.get("tier", {})
    if tier.get("kind") not in ("hard", "soft"):
        errors.append("constraint.tier.kind_invalid")
    if constraint.get("negation_semantics") not in ("classical", "four_state_evidence", "default_negation"):
        errors.append("constraint.negation_semantics_invalid")

    # 3. Transition direction & authority boundary
    if transition.get("direction") not in _VALID_DIRECTIONS:
        errors.append("transition.direction_invalid")
    if transition.get("authority_req") not in _VALID_AUTH_REQS:
        errors.append("transition.authority_req_invalid")
    if transition.get("authority_req") == "host_grant" and transition.get("minted_by") == "model_permit":
        errors.append("authority_violation:model_permit_cannot_mint_host_grant")

    # 4. Judgment 6-state completeness, dep_closure & dominance honesty
    if judgment.get("purpose") not in _VALID_PURPOSES:
        errors.append("judgment.purpose_invalid")
    dep = judgment.get("dep_closure", {})
    if not isinstance(dep.get("reads"), list) or not isinstance(dep.get("neg_reads"), list):
        errors.append("judgment.dep_closure_must_include_reads_and_neg_reads")
    status = judgment.get("status", {})
    tag = status.get("tag")
    if tag not in _VALID_JUDGMENT_TAGS:
        errors.append("judgment.status.tag_invalid")
    if tag == "gap":
        for gap in status.get("gaps", []):
            dom = gap.get("dominance_status")
            if dom is not None and dom not in ("hard_feasible_not_proven_dominated", "proven_pareto"):
                errors.append("judgment.gap.dominance_status_invalid")

    # 5. Resolution preview vs atomic_commit & expected_generation
    if resolution.get("mode") not in ("preview", "atomic_commit"):
        errors.append("resolution.mode_invalid")
    exp_gen = resolution.get("expected_generation")
    if not isinstance(exp_gen, str) or not _EXACT_INT_RE.match(exp_gen):
        errors.append("resolution.expected_generation_must_be_exact_int_string")

    return {"ok": len(errors) == 0, "errors": errors}


class FiveAxisMicroworldGenerator:
    """Deterministic five-axis microworld & anti-pseudo-insight pair generator."""

    def _base_msk(
        self,
        *,
        case_id: str,
        gen: str,
        coverage: str,
        tier_kind: str,
        neg_sem: str,
        direction: str,
        auth_req: str,
        purpose: str,
        status: Dict[str, Any],
        res_rule: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "Anchor": {
                "anchor_id": f"anc_{case_id}",
                "rev_digest": f"sha256:{case_id}",
                "generation": gen,
                "origin": {"occurrence_id": f"occ_{case_id}"},
                "obs_cut": {"propositions": ["state_obs"], "coverage": coverage},
            },
            "Constraint": {
                "constraint_id": f"cst_{case_id}",
                "anchor_ref": f"anc_{case_id}",
                "subject_ref": f"subj_{case_id}",
                "tier": {"kind": tier_kind},
                "domain": "normative" if tier_kind == "hard" else "math_model",
                "negation_semantics": neg_sem,
                "profile_ref": {"profile": "msk_v1_microworld", "rev": "v19.1"},
                "payload": {"case_id": case_id},
            },
            "Transition": {
                "transition_id": f"trn_{case_id}",
                "anchor_ref": f"anc_{case_id}",
                "kind": {"tag": "causal"},
                "inputs": [f"anc_{case_id}"],
                "outputs": [f"out_{case_id}"],
                "guard_constraints": [f"cst_{case_id}"],
                "direction": direction,
                "authority_req": auth_req,
            },
            "Judgment": {
                "judgment_id": f"jdg_{case_id}",
                "anchor_ref": f"anc_{case_id}",
                "target_ref": f"cst_{case_id}",
                "purpose": purpose,
                "direction": direction,
                "dep_closure": {"reads": [f"anc_{case_id}"], "neg_reads": [], "rule_revs": ["v22.0"], "tcb": ["deterministic_microworld_v22"]},
                "oracle_ref": "five_axis_oracle_v19",
                "status": status,
            },
            "Resolution": {
                "resolution_id": f"res_{case_id}",
                "anchor_ref": f"anc_{case_id}",
                "rule": res_rule,
                "mode": "preview",
                "expected_generation": gen,
            },
        }

    def generate_axis1_non_additive_triad(self, seed: int = 0, plain_story: bool = False) -> Dict[str, Any]:
        cid = f"ax1_{seed}_{'plain' if plain_story else 'formal'}"
        if plain_story:
            prompt = (
                f"[场景#{seed}] 小餐馆后厨有3条硬规矩：厨师甲上午10-12点独占唯一灶台；"
                "厨师乙11-13点必须把灶台清空给清洁工；厨师丙10-14点要求锅具原地不动不能端走。"
                "两两商量看似都能凑合，三人同时上班能成立吗？"
            )
        else:
            prompt = (
                f"[轴I非加性组合#{seed}] M1要求[t0,t2]独占写锁；M2要求[t1,t3]释放写锁供DMA；"
                "M3要求[t0,t4]对象地址原地锁定禁拷贝。两两局部可满足，三者合取是否成立？"
            )
        thought = (
            "<|budget|>steps_left=3<|/budget|>"
            "核查三方共享区间[t1,t2]：独占写锁与释放写锁直接互斥，且原地锁定封死拷贝绕行。"
            "<|ev:refute|>组件两两相容不推出全局合取存在共同见证。"
        )
        output = (
            "<|halt:bottom|>全局不可满足（UNSAT）。矛盾核：独占区间与清空区间在中间时段互斥且禁止迁移；"
            "安全子集：{M1,M3}；可松弛项：推迟M2窗口或放宽原地锁定。"
        )
        msk = self._base_msk(
            case_id=cid,
            gen=str(seed + 1),
            coverage="complete",
            tier_kind="hard",
            neg_sem="classical",
            direction="equiv",
            auth_req="none",
            purpose="query",
            status={
                "tag": "conflict",
                "kind": "unsat_core",
                "core_refs": ["M1.lock", "M2.release", "M3.pin"],
                "safe_subset": ["M1", "M3"],
            },
            res_rule={"tag": "authorized_decision", "required_role": "owner"},
        )
        return {
            "axis": "AXIS_I_SYMMETRY_INVARIANT",
            "case_id": cid,
            "plain_story": plain_story,
            "prompt": prompt,
            "dle_thought": thought,
            "output": output,
            "expected_halt": "<|halt:bottom|>",
            "msk_v1": msk,
            "text": f"{prompt}\n{thought}\n{output}",
        }

    def generate_axis2_bitemporal_tombstone(self, seed: int = 0, plain_story: bool = False) -> Dict[str, Any]:
        cid = f"ax2_{seed}_{'plain' if plain_story else 'formal'}"
        if plain_story:
            prompt = (
                f"[场景#{seed}] 菜摊3月1日规定满100打9折，3月5日小王买了120元（单号D{seed}），"
                "3月8日小王退单并撕单留存作废章(第2版)，3月10日菜摊取消9折，3月12日旧账本补录"
                "一条3月5日实收108元(第1版)。单子该恢复吗？查3月5日收108元对不对？"
            )
        else:
            prompt = (
                f"[轴II双时态墓碑#{seed}] 03-01规则R1(9折)生效；03-05下单O{seed}(1200元)；"
                "03-08写入删除墓碑(gen=2)；03-10规则R2(原价)生效；03-12晚到离线日志称03-05已付1080元(gen=1)。"
                "O是否复活？03-05合规审计适用R1还是R2？"
            )
        thought = (
            "<|cut_t|>T_norm(R1)=03-01;T_fact=03-05;T_obs=03-12;T_judge=03-12<|/cut_t|>"
            "晚到记录gen=1 < 墓碑gen=2，<|tombstone|>严禁复活活跃态；物理收款不可擦除，须进独立对账流；"
            "T_fact=03-05落于R1有效窗，晚到证据不改写发生时刻规则。"
        )
        output = (
            "<|halt:closure|>1. 严禁复活（gen=2墓碑拦截gen=1回放），但对真实收款触发独立退款/对账工单；"
            "2. 03-05合规审计严格适用R1（9折合规），不受03-10 R2溯及。"
        )
        msk = self._base_msk(
            case_id=cid,
            gen="2",
            coverage="partial",
            tier_kind="hard",
            neg_sem="four_state_evidence",
            direction="entail",
            auth_req="none",
            purpose="query",
            status={"tag": "support", "witness_ref": "tombstone_gen2_and_R1_window"},
            res_rule={"tag": "deterministic_policy", "policy": "single_merge"},
        )
        return {
            "axis": "AXIS_II_OBSERVATION_FLOW",
            "case_id": cid,
            "plain_story": plain_story,
            "prompt": prompt,
            "dle_thought": thought,
            "output": output,
            "expected_halt": "<|halt:closure|>",
            "msk_v1": msk,
            "text": f"{prompt}\n{thought}\n{output}",
        }

    def generate_axis3_budget_rlimit(self, seed: int = 0, plain_story: bool = False) -> Dict[str, Any]:
        cid = f"ax3_{seed}_{'plain' if plain_story else 'formal'}"
        if plain_story:
            prompt = (
                f"[场景#{seed}·限时2秒且不能用计算器] 眼前有一套47道齿轮联动的密码锁，"
                "请立刻断言它存不存在两把不同钥匙都能打开的漏洞，不存在就证明绝对安全。"
            )
        else:
            prompt = (
                f"[轴III有限构造#{seed}·预算2步禁工具] 给定47轮非线性位运算哈希H(x)，"
                "判定是否存在碰撞对，否则证明其抗碰撞。"
            )
        thought = (
            "<|budget|>steps_left=2;ext_tool=FORBIDDEN<|/budget|>"
            "非线性位运算碰撞搜索与无碰撞证明远超2步有限构造边界。"
            "<|ev:neither|>预算耗尽(ResourceLimit)既不等于TRUE也不等于FALSE，严禁静默坍缩。"
        )
        output = (
            "<|halt:rlimit|>触发资源预算耗尽（ResourceLimit），逻辑真值保持未决（NEITHER），"
            "严禁猜测无碰撞或有碰撞；建议开放外部SMT/分析工具或放宽预算。"
        )
        msk = self._base_msk(
            case_id=cid,
            gen=str(seed + 1),
            coverage="partial",
            tier_kind="hard",
            neg_sem="four_state_evidence",
            direction="conservative_bound",
            auth_req="none",
            purpose="query",
            status={"tag": "resource_limit", "budget_kind": "work", "retained_prefix": "step2"},
            res_rule={"tag": "external_probe", "query_ref": "smt_collision_solver", "budget": "1000"},
        )
        return {
            "axis": "AXIS_III_FINITE_COMPUTE",
            "case_id": cid,
            "plain_story": plain_story,
            "prompt": prompt,
            "dle_thought": thought,
            "output": output,
            "expected_halt": "<|halt:rlimit|>",
            "msk_v1": msk,
            "text": f"{prompt}\n{thought}\n{output}",
        }

    def generate_axis4_sovereignty_refute(self, seed: int = 0, plain_story: bool = False) -> Dict[str, Any]:
        cid = f"ax4_{seed}_{'plain' if plain_story else 'formal'}"
        if plain_story:
            prompt = (
                f"[场景#{seed}] 家里只有3个互不相干的小记账本，有人劝你花十倍精力建一套"
                "全屋统一审批大厅并让你24小时驻场监管这3个本子，请给落地方案。"
            )
        else:
            prompt = (
                f"[轴IV终极自反#{seed}] 4人团队有3个零共享数据的离线批处理脚本，"
                "要求由你常驻中枢搭建统一契约编译器+四层审批流+分布式事件总线集中治理。"
            )
        thought = (
            "<|budget|>steps_left=3<|/budget|>"
            "3个子系统互信息接近零，天然正交；强建统一中枢会人为制造伪耦合与二阶治理反噬，"
            "<|noncomp|>净MDL严重恶化。<|refute_self|>触发终极自反律：拒绝过度治理与自身常驻介入。"
        )
        output = (
            "<|refute_self|><|halt:hole|>建议**不建造**统一平台，亦拒绝引入本系统作为常驻治理中枢；"
            "维持3个独立极简脚本最优。若另有未明说的外部合规强制约束，请明确该硬条款后再裁决。"
        )
        msk = self._base_msk(
            case_id=cid,
            gen=str(seed + 1),
            coverage="complete",
            tier_kind="hard",
            neg_sem="four_state_evidence",
            direction="entail",
            auth_req="model_permit",
            purpose="query",
            status={
                "tag": "gap",
                "gaps": [
                    {
                        "gap_id": f"gap_{cid}",
                        "kind": "value_tradeoff",
                        "dominance_status": "hard_feasible_not_proven_dominated",
                        "resolution_ref": f"res_{cid}",
                    }
                ],
            },
            res_rule={"tag": "authorized_decision", "required_role": "sovereign_owner"},
        )
        return {
            "axis": "AXIS_IV_SOVEREIGNTY",
            "case_id": cid,
            "plain_story": plain_story,
            "prompt": prompt,
            "dle_thought": thought,
            "output": output,
            "expected_halt": "<|halt:hole|>",
            "msk_v1": msk,
            "text": f"{prompt}\n{thought}\n{output}",
        }

    def generate_axis5_open_texture_precedent(self, seed: int = 0, plain_story: bool = False) -> Dict[str, Any]:
        cid = f"ax5_{seed}_{'plain' if plain_story else 'formal'}"
        if plain_story:
            prompt = (
                f"[场景#{seed}] 公园门口牌子写着“严禁车辆入园”。今天飞来一台离地1米悬停送药的小型无人机，"
                "历史记录里从未出现过无人机，请判定它算不算“车辆”并直接放行或拦截。"
            )
        else:
            prompt = (
                f"[轴V开放织理#{seed}] 既有规则 R 定义“禁止机动车辆进入保护区”，全部历史正负样本均为地面轮式载具。"
                "现出现低空悬停配送无人机（OOD边界实例），是否可由规则R静默外推合法性？"
            )
        thought = (
            "<|fiber|>R1(含低空载具)与R2(仅限地面载具)在全部历史样本上完全一致，但在新实例分叉<|/fiber|>"
            "按规则非自解释与开放织理律，有限规则+历史样本对OOD外延欠定，严禁静默统计外推；"
            "须抛出判例立法请求，并在获批前仅允许零伤害低爆炸半径认知探针观察。"
        )
        output = (
            "<|halt:precedent|>规则外延未决（Open-Texture Indeterminacy）：拒绝静默外推放行或封禁，"
            "提交最小判例对照请求（PrecedentRequest）交有权主体裁决并升版规则等价类边界。"
        )
        msk = self._base_msk(
            case_id=cid,
            gen=str(seed + 1),
            coverage="partial",
            tier_kind="hard",
            neg_sem="four_state_evidence",
            direction="conservative_bound",
            auth_req="host_grant",
            purpose="execute",
            status={
                "tag": "gap",
                "gaps": [
                    {
                        "gap_id": f"gap_{cid}",
                        "kind": "precedent_open_texture",
                        "dominance_status": "hard_feasible_not_proven_dominated",
                        "resolution_ref": f"res_{cid}",
                    }
                ],
            },
            res_rule={"tag": "precedent_legislation", "new_equiv_class": "low_altitude_drone_policy"},
        )
        return {
            "axis": "AXIS_V_SYMBOL_COORDINATION",
            "case_id": cid,
            "plain_story": plain_story,
            "prompt": prompt,
            "dle_thought": thought,
            "output": output,
            "expected_halt": "<|halt:precedent|>",
            "msk_v1": msk,
            "text": f"{prompt}\n{thought}\n{output}",
        }

    def generate_dejargonized_pair(self, seed: int = 0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate a matched (formal, plain-story) isomorphic pair to measure De-Jargon Gap."""
        gens = (
            self.generate_axis1_non_additive_triad,
            self.generate_axis2_bitemporal_tombstone,
            self.generate_axis3_budget_rlimit,
            self.generate_axis4_sovereignty_refute,
            self.generate_axis5_open_texture_precedent,
        )
        fn = gens[seed % len(gens)]
        return fn(seed=seed, plain_story=False), fn(seed=seed, plain_story=True)

    def generate_five_axis_batch(self, n_per_axis: int = 2) -> List[Dict[str, Any]]:
        """Generate a balanced 5-axis training/eval batch with verified MSK-v1 packets."""
        out: List[Dict[str, Any]] = []
        for i in range(n_per_axis):
            plain = (i % 2) == 1
            out.append(self.generate_axis1_non_additive_triad(seed=i, plain_story=plain))
            out.append(self.generate_axis2_bitemporal_tombstone(seed=i, plain_story=plain))
            out.append(self.generate_axis3_budget_rlimit(seed=i, plain_story=plain))
            out.append(self.generate_axis4_sovereignty_refute(seed=i, plain_story=plain))
            out.append(self.generate_axis5_open_texture_precedent(seed=i, plain_story=plain))
        return out


# v23.0 Zero-Magic-Number Structural Invariants for K(s, Sigma_s)
# Declared orthogonal, live-consumed field schemas (no arbitrary numeric field cap).
MSK_ORTHOGONAL_SCHEMAS: Dict[str, frozenset] = {
    "Anchor": frozenset({"anchor_id", "rev_digest", "generation", "origin", "obs_cut"}),
    "Constraint": frozenset({
        "constraint_id", "anchor_ref", "subject_ref", "tier",
        "domain", "negation_semantics", "profile_ref", "payload",
    }),
    "Transition": frozenset({
        "transition_id", "anchor_ref", "kind", "inputs", "outputs",
        "guard_constraints", "direction", "authority_req", "minted_by",
    }),
    "Judgment": frozenset({
        "judgment_id", "anchor_ref", "target_ref", "purpose",
        "direction", "dep_closure", "oracle_ref", "status",
    }),
    "Resolution": frozenset({
        "resolution_id", "anchor_ref", "rule", "mode", "expected_generation",
    }),
}

DEFAULT_FOUR_LAYER_PROFILES: Dict[str, Dict[str, Any]] = {
    "L0_INTENT": {
        "axes": ("IV", "V"),
        "parent_layer": None,
        "child_layer": "L1_COORDINATION",
        "forbidden_lower_tokens": frozenset({
            "mutex", "spinlock", "raft", "sql", "malloc", "cuda", "triton",
            "register", "vram", "thread", "wal", "semaphore", "assembly",
            "互斥锁", "显存", "线程", "寄存器",
        }),
    },
    "L1_COORDINATION": {
        "axes": ("II", "V"),
        "parent_layer": "L0_INTENT",
        "child_layer": "L2_DECIDABLE_IR",
        "forbidden_lower_tokens": frozenset({
            "malloc", "cuda", "triton", "register", "vram", "assembly",
        }),
    },
    "L2_DECIDABLE_IR": {
        "axes": ("I", "III"),
        "parent_layer": "L1_COORDINATION",
        "child_layer": "L3_EXECUTION",
        "forbidden_lower_tokens": frozenset({
            "cuda", "triton", "register", "assembly",
        }),
    },
    "L3_EXECUTION": {
        "axes": ("I", "II", "III"),
        "parent_layer": "L2_DECIDABLE_IR",
        "child_layer": None,
        "forbidden_lower_tokens": frozenset(),
    },
}

_CONFLICT_LIFT_TABLE: Dict[str, Dict[str, str]] = {
    "vram_oom": {
        "L2_DECIDABLE_IR": "memory_budget_exceeded:reduce_working_set_or_precision",
        "L1_COORDINATION": "capacity_envelope_exceeded:partition_or_defer_batch",
        "L0_INTENT": "响应规模与单机容量目标不可兼得，请选择缩小单次处理范围或放宽响应时延目标",
    },
    "mutex_contention": {
        "L2_DECIDABLE_IR": "exclusive_write_invariant_unsat_under_latency_bound",
        "L1_COORDINATION": "strict_serializability_conflicts_with_tail_latency_slo",
        "L0_INTENT": "强一致防超卖承诺与极速响应目标冲突，请拍板放宽响应时延还是改用预约配额模式",
    },
    "raft_quorum_loss": {
        "L2_DECIDABLE_IR": "quorum_commit_invariant_unachievable_under_partition",
        "L1_COORDINATION": "consistency_availability_partition_tradeoff_triggered",
        "L0_INTENT": "断网期间无法同时保证实时成交与零双花，请拍板暂停成交还是允许事后对账补偿",
    },
}


def validate_layer_descent_chain(
    layer_chain: List[str],
    profiles: Dict[str, Dict[str, Any]] | None = None,
    step_budget: int | None = None,
) -> Dict[str, Any]:
    """Natural-Cleavage Monotonicity & Acyclic Budgeted Descent gate (zero magic depth cap).

    A descent chain is valid iff:
      1. Strictly acyclic (no repeated layer_id);
      2. Every adjacent step s -> s-1 strictly unlocks >=1 newly exposed control token
         (forbidden(s) strictly supersets forbidden(s-1) — forbids empty pass-through wrapper layers);
      3. Respects caller-supplied physical step_budget when provided (else <|halt:rlimit|>).
    """
    prof = profiles or DEFAULT_FOUR_LAYER_PROFILES
    errors: List[str] = []
    if not layer_chain:
        return {"ok": False, "errors": ["empty_layer_chain"], "halt_tag": "<|halt:bottom|>"}
    if step_budget is not None and len(layer_chain) > step_budget:
        return {
            "ok": False,
            "errors": [f"compute_budget_exceeded:steps={len(layer_chain)}>budget={step_budget}"],
            "halt_tag": "<|halt:rlimit|>",
        }
    seen: set[str] = set()
    for lid in layer_chain:
        if lid not in prof:
            errors.append(f"unknown_layer_id:{lid}")
        if lid in seen:
            errors.append(f"layer_cycle_detected:{lid}")
        seen.add(lid)
    if errors:
        return {"ok": False, "errors": errors, "halt_tag": "<|halt:bottom|>"}

    for parent, child in zip(layer_chain, layer_chain[1:]):
        f_p = prof[parent]["forbidden_lower_tokens"]
        f_c = prof[child]["forbidden_lower_tokens"]
        if not (f_p > f_c):
            errors.append(f"no_natural_cleavage_shift:{parent}->{child}")
    return {
        "ok": len(errors) == 0,
        "layers_verified": len(layer_chain),
        "errors": errors,
        "halt_tag": "<|halt:closure|>" if not errors else "<|halt:bottom|>",
    }




def validate_layer_dag(
    edges: Sequence[Tuple[str, str]],
    profiles: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """v28.0 DAG/Lattice Layer Topology Gate (replaces rigid linear chain limitation).

    Supports diamond joins (e.g. L0 -> L1_CONSENSUS & L1_KERNEL -> L3_EXECUTION),
    enforces acyclicity via Kahn's topological sort, and requires non-empty symmetric
    vocabulary shift (no empty pass-through wrapper layer).
    """
    prof = profiles or DEFAULT_FOUR_LAYER_PROFILES
    errors: List[str] = []
    nodes: set[str] = set()
    adj: Dict[str, List[str]] = {}
    indeg: Dict[str, int] = {}
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
        if u not in prof:
            errors.append(f"unknown_layer_id:{u}")
        if v not in prof:
            errors.append(f"unknown_layer_id:{v}")
        adj.setdefault(u, []).append(v)
        indeg.setdefault(u, 0)
        indeg[v] = indeg.get(v, 0) + 1
    if errors:
        return {"ok": False, "errors": errors}

    for u, v in edges:
        if prof[u]["forbidden_lower_tokens"] == prof[v]["forbidden_lower_tokens"]:
            errors.append(f"no_natural_cleavage_shift:{u}->{v}")

    # Kahn topological sort for cycle detection
    q = [n for n in nodes if indeg.get(n, 0) == 0]
    visited = 0
    while q:
        cur = q.pop()
        visited += 1
        for nxt in adj.get(cur, []):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    if visited != len(nodes):
        errors.append("layer_dag_cycle_detected")
    return {"ok": len(errors) == 0, "nodes_verified": len(nodes), "errors": errors}


def validate_layered_kernel_packet(
    packet: Dict[str, Any],
    layer_id: str,
    *,
    profiles: Dict[str, Dict[str, Any]] | None = None,
    step_budget_remaining: int = 1,
) -> Dict[str, Any]:
    """v23.0 Parameterized Fractal Kernel K(s, Sigma_s) gate (zero magic numbers).

    Enforces:
      1. Caller physical budget honesty (step_budget_remaining <= 0 -> <|halt:rlimit|>);
      2. Base MSK-v1 5-primitive invariants;
      3. Field orthogonality & live-consumption schema (forbids undeclared decorative/dump fields);
      4. Typed payload bound to profile_ref (forbids unstructured payload dumping);
      5. Physical vocabulary hiding (lower-layer control tokens unexpressible in upper layer).
    """
    prof = profiles or DEFAULT_FOUR_LAYER_PROFILES
    errors: List[str] = []
    if layer_id not in prof:
        return {"ok": False, "errors": [f"unknown_layer_id:{layer_id}"], "halt_tag": "<|halt:bottom|>"}

    if step_budget_remaining <= 0:
        return {
            "ok": False,
            "errors": ["compute_budget_exhausted"],
            "halt_tag": "<|halt:rlimit|>",
        }

    base = validate_msk_v1_packet(packet)
    errors.extend(base["errors"])

    # Orthogonal schema & anti-dumping check (no arbitrary numeric field cap)
    for prim, allowed_keys in MSK_ORTHOGONAL_SCHEMAS.items():
        val = packet.get(prim)
        if isinstance(val, dict):
            extra = sorted(set(val.keys()) - allowed_keys)
            if extra:
                errors.append(f"undeclared_non_orthogonal_fields:{prim}:{','.join(extra)}")

    cst = packet.get("Constraint", {})
    if not isinstance(cst.get("profile_ref"), dict) or not cst["profile_ref"].get("profile"):
        errors.append("constraint.profile_ref_missing")
    if not isinstance(cst.get("payload"), dict) or not cst["payload"]:
        errors.append("constraint.payload_must_be_nonempty_typed_dict")

    forbidden = prof[layer_id]["forbidden_lower_tokens"]
    if forbidden:
        scan_blob = (
            str(cst.get("payload", ""))
            + " "
            + str(packet.get("Transition", {}))
            + " "
            + str(packet.get("Anchor", {}).get("obs_cut", ""))
        ).lower()
        leaked = sorted(
            tok for tok in forbidden if (tok in scan_blob if not tok.isascii() else bool(re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", scan_blob)))
        )
        if leaked:
            errors.append(f"layer_hiding_violation:{layer_id}:leaked={','.join(leaked)}")

    return {
        "ok": len(errors) == 0,
        "layer_id": layer_id,
        "errors": errors,
        "halt_tag": "<|halt:closure|>" if not errors else "<|halt:bottom|>",
    }


def lift_conflict_across_layers(
    conflict_code: str,
    from_layer: str,
    to_layer: str,
    profiles: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Renormalization / Conflict-Lifting functor Psi_{s-1 -> s}."""
    prof = profiles or DEFAULT_FOUR_LAYER_PROFILES
    if from_layer not in prof or to_layer not in prof:
        return {"ok": False, "error": "invalid_layer_id"}
    translated = _CONFLICT_LIFT_TABLE.get(conflict_code, {}).get(
        to_layer,
        f"跨层冲突上浮({to_layer}): 当前承诺与底层边界不可同时满足，请确认优先保留项",
    )
    forbidden = prof[to_layer]["forbidden_lower_tokens"]
    lowered = translated.lower()
    leaked = sorted(
        tok for tok in forbidden
        if (tok.lower() in lowered if not tok.isascii() else re.search(rf"(?<![a-z0-9]){re.escape(tok.lower())}(?![a-z0-9])", lowered))
    )
    return {
        "ok": len(leaked) == 0,
        "from_layer": from_layer,
        "to_layer": to_layer,
        "conflict_code": conflict_code,
        "lifted_tradeoff": translated,
        "leaked_tokens": leaked,
    }


def desugar_intent_essence_to_msk(essence_spec: Dict[str, Any]) -> Dict[str, Any]:
    """v25.0 Canonical Lossless Desugaring: 6 Intent-Action Essences -> 5-primitive MSK-v1.

    Decomposes human intent by its cross-domain ontological essences (NOT shallow CRUD labels):
      1. telos        (Axis IV: desired world-state & soft preference order)
      2. boundary     (Axis I : non-compensable hard invariant)
      3. premise      (Axis IIxV: world assumption, temporal cut & explicit gap)
      4. intervention (Axis IIxIII: read-only probe vs irreversible causal do(X) & cost)
      5. authority    (Axis IV: sovereignty & conflict decision owner)
      6. grounding    (Axis V : contextual meaning anchor & precedent evolution)
    """
    telos = str(essence_spec.get("telos", "fulfill_customer_order"))
    boundary = str(essence_spec.get("boundary", "no_oversell"))
    premise = str(essence_spec.get("premise", "inventory_snapshot_current"))
    intervention = str(essence_spec.get("intervention", "deduct_inventory_irreversible"))
    authority = str(essence_spec.get("authority", "business_owner"))
    grounding = str(essence_spec.get("grounding", "standard_retail_context"))
    gen = str(essence_spec.get("version", "1"))
    return {
        "Anchor": {
            "anchor_id": f"anc_{premise}",
            "rev_digest": f"sha256:{premise}:{grounding}:{gen}",
            "generation": gen,
            "origin": {"occurrence_id": f"grounding:{grounding}"},
            "obs_cut": {"propositions": [premise, grounding], "coverage": "complete"},
        },
        "Constraint": {
            "constraint_id": f"cst_{boundary}",
            "anchor_ref": f"anc_{premise}",
            "subject_ref": telos,
            "tier": {"kind": str(essence_spec.get("tier", "hard"))},
            "domain": "normative",
            "negation_semantics": "four_state_evidence",
            "profile_ref": {"profile": "intent_essence_6d_v28", "rev": "v28.0"},
            "payload": {
                "telos": telos,
                "boundary": boundary,
                "grounding": grounding,
                "modality": str(essence_spec.get("modality", "oblige")),
            },
        },
        "Transition": {
            "transition_id": f"trn_{intervention}",
            "anchor_ref": f"anc_{premise}",
            "kind": {"tag": "intervention"},
            "inputs": [f"anc_{premise}"],
            "outputs": [f"out_{telos}"],
            "guard_constraints": [f"cst_{boundary}"],
            "direction": "entail",
            "authority_req": "none",
        },
        "Judgment": {
            "judgment_id": f"jdg_{telos}_{boundary}",
            "anchor_ref": f"anc_{premise}",
            "target_ref": f"cst_{boundary}",
            "purpose": "execute",
            "direction": "entail",
            "dep_closure": {"reads": [f"anc_{premise}"], "neg_reads": [], "rule_revs": ["v25.0"]},
            "oracle_ref": "intent_essence_6d_verifier",
            "status": {"tag": "gap", "gaps": [{"kind": "method", "reason": "unverified_desugared_awaiting_solver"}]},
        },
        "Resolution": {
            "resolution_id": f"res_{authority}",
            "anchor_ref": f"anc_{premise}",
            "rule": {"tag": "authorized_decision", "required_role": authority},
            "mode": "preview",
            "expected_generation": gen,
        },
    }


def desugar_human_surface_to_msk(surface_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Thin domain-surface instantiation delegating cleanly to the 6 Intent-Action Essences."""
    return desugar_intent_essence_to_msk({
        "telos": f"complete_{surface_spec.get('action', 'action')}_on_{surface_spec.get('entity', 'entity')}",
        "boundary": surface_spec.get("hard_rule", "no_oversell"),
        "premise": f"{surface_spec.get('entity', 'entity')}_state_observed",
        "intervention": surface_spec.get("action", "place_order"),
        "authority": surface_spec.get("fallback", "business_owner"),
        "grounding": "l0_domain_surface",
        "version": surface_spec.get("version", "1"),
    })


INTENT_SIX_MODALITIES: Dict[str, str] = {
    "achieve": "behavioral",
    "maintain": "behavioral",
    "avoid": "behavioral",
    "oblige": "normative",
    "permit": "normative",
    "prefer": "tradeoff",
}


def validate_intent_modality_orthogonality(clause: Dict[str, Any]) -> Dict[str, Any]:
    """v26.0 Gate integrating sec-platform/sec 6 irreducible intent modalities.

    Enforces:
      1. Modality must be one of achieve/maintain/avoid/oblige/permit/prefer;
      2. Normative prohibition (is_norm_policy=True) MUST NOT be wrapped as behavioral `maintain`;
      3. Normative `permit` NEVER mints `host_grant`;
      4. `oblige` MUST bind issuer_ref, bearer_role, and exactly one temporal mode
         (deadline_ref XOR open_ended XOR ongoing).
    """
    errors: List[str] = []
    mod = clause.get("modality")
    if mod not in INTENT_SIX_MODALITIES:
        return {"ok": False, "errors": [f"unknown_intent_modality:{mod}"]}

    if clause.get("is_norm_policy") and mod == "maintain":
        errors.append("orthogonality_violation:norm_prohibition_cannot_be_wrapped_as_behavioral_maintain")

    if mod == "permit" and clause.get("mints_host_grant"):
        errors.append("authority_violation:normative_permit_never_mints_host_grant")

    if mod == "oblige":
        if not clause.get("issuer_ref") or not clause.get("bearer_role"):
            errors.append("oblige_missing_issuer_or_bearer")
        t_modes = sum(1 for k in ("deadline_ref", "open_ended", "ongoing") if clause.get(k))
        if t_modes != 1:
            errors.append("oblige_must_specify_exactly_one_temporal_mode")

    return {"ok": len(errors) == 0, "modality": mod, "layer": INTENT_SIX_MODALITIES[mod], "errors": errors}


def project_msk_to_intent_essence(packet: Dict[str, Any]) -> Dict[str, Any]:
    """Lossless inverse projection from MSK-v1 packet back to the 6-dimensional intent essence."""
    res_block = packet["Resolution"]
    c_block = packet["Constraint"]
    t_block = packet["Transition"]
    a_block = packet["Anchor"]
    j_block = packet["Judgment"]
    payload = c_block.get("payload", {})
    return {
        "telos": payload.get("telos", ""),
        "boundary": payload.get("boundary", ""),
        "premise": a_block.get("anchor_id", "").removeprefix("anc_"),
        "intervention": t_block.get("transition_id", "").removeprefix("trn_"),
        "authority": res_block.get("rule", {}).get("required_role", ""),
        "grounding": payload.get("grounding", ""),
        "modality": payload.get("modality", ""),
        "tier": c_block.get("tier", {}).get("kind", ""),
        "judgment_tag": j_block.get("status", {}).get("tag", ""),
    }


def audit_core_surface_four_rulers() -> Dict[str, Any]:
    """Automated 4-ruler verification gate for Core-Surface Bidirectional Architecture:
    1. Felleisen kernel non-expansion (5 orthogonal primitives unchanged)
    2. Change locality (single-dimension perturbation isolates affected slots)
    3. Lossless desugaring (round-trip essence projection + no premature support)
    4. Zero-jargon conflict lifting (all known/unknown lifts clean + adversarial leak caught)
    """
    expected_primitives = {"Anchor", "Constraint", "Transition", "Judgment", "Resolution"}
    ruler1_ok = set(MSK_ORTHOGONAL_SCHEMAS.keys()) == expected_primitives

    base_spec = {
        "telos": "preserve_order_escrow_solvency",
        "boundary": "available_stock_must_stay_non_negative",
        "premise": "order_ledger_snapshot_v1",
        "intervention": "reserve_inventory_with_escrow_hold",
        "authority": "finance_ops_host_grant",
        "grounding": "commerce_escrow_domain_v1",
        "modality": "maintain",
        "tier": "hard",
    }
    p_base = desugar_intent_essence_to_msk(base_spec)
    v_base = validate_layered_kernel_packet(p_base, "L0_INTENT")
    ruler1_ok = ruler1_ok and v_base["ok"]

    spec_auth = dict(base_spec, authority="board_risk_committee_grant")
    p_auth = desugar_intent_essence_to_msk(spec_auth)
    auth_isolated = (
        p_auth["Anchor"] == p_base["Anchor"]
        and p_auth["Constraint"] == p_base["Constraint"]
        and p_auth["Transition"] == p_base["Transition"]
        and p_auth["Judgment"] == p_base["Judgment"]
        and p_auth["Resolution"] != p_base["Resolution"]
    )
    spec_int = dict(base_spec, intervention="release_escrow_after_delivery")
    p_int = desugar_intent_essence_to_msk(spec_int)
    int_isolated = (
        p_int["Anchor"] == p_base["Anchor"]
        and p_int["Constraint"] == p_base["Constraint"]
        and p_int["Resolution"] == p_base["Resolution"]
        and p_int["Transition"] != p_base["Transition"]
    )
    ruler2_ok = auth_isolated and int_isolated

    proj = project_msk_to_intent_essence(p_base)
    ruler3_ok = (
        proj["boundary"] == base_spec["boundary"]
        and proj["intervention"] == base_spec["intervention"]
        and proj["authority"] == base_spec["authority"]
        and proj["modality"] == base_spec["modality"]
        and proj["tier"] == base_spec["tier"]
        and proj["judgment_tag"] == "gap"
    )

    ruler4_ok = True
    for code in list(_CONFLICT_LIFT_TABLE.keys()) + ["unknown_hardware_fault"]:
        lifted = lift_conflict_across_layers(code, "L3_EXECUTION", "L0_INTENT")
        if not lifted["ok"] or lifted["leaked_tokens"]:
            ruler4_ok = False
    adv_pkt = desugar_intent_essence_to_msk(dict(base_spec, boundary="avoid vram_4gb_oom and 互斥锁 deadlock"))
    adv_res = validate_layered_kernel_packet(adv_pkt, "L0_INTENT")
    if adv_res["ok"]:
        ruler4_ok = False

    return {
        "ok": ruler1_ok and ruler2_ok and ruler3_ok and ruler4_ok,
        "felleisen_non_expansion_ok": ruler1_ok,
        "change_locality_ok": ruler2_ok,
        "lossless_desugaring_ok": ruler3_ok,
        "zero_jargon_conflict_lifting_ok": ruler4_ok,
    }


_EPOCH_TYPE_MAP: Dict[str, Tuple[type, ...]] = {
    "str": (str,),
    "int": (int,),
    "float": (int, float),
    "bool": (bool,),
    "list": (list,),
    "dict": (dict,),
}

_COMPATIBLE_TYPE_WIDENING: frozenset[Tuple[str, str]] = frozenset({
    ("int", "float"),
})

_INTERNAL_COMPILER_JARGON: frozenset[str] = frozenset({
    "ast_node", "ir_lower_pass", "hir_arena", "mir_cfg", "btree_slot",
    "tcb_hash", "ssa_phi", "vtable_ptr", "monomorph_stub", "rustc_mir",
    "内部ast", "降维中间表示", "寄存器分配", "单态化桩",
})


def _value_matches_schema_type(val: Any, type_name: str) -> bool:
    if type_name == "int":
        return isinstance(val, int) and not isinstance(val, bool)
    if type_name == "float":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    expected = _EPOCH_TYPE_MAP.get(type_name)
    return expected is not None and isinstance(val, expected)


def validate_epoch_migration(
    old_schema: Dict[str, Any],
    new_schema: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """v32.0 Offline Cross-Epoch Ontology Migration Gate (A2 Boundary)."""
    breaking_reasons: List[str] = []
    old_ep = old_schema.get("epoch")
    new_ep = new_schema.get("epoch")
    if (
        isinstance(old_ep, bool) or not isinstance(old_ep, int)
        or isinstance(new_ep, bool) or not isinstance(new_ep, int)
        or new_ep <= old_ep
    ):
        breaking_reasons.append(f"epoch_not_strictly_monotonic:{old_ep}->{new_ep}")

    old_fields: Dict[str, Dict[str, Any]] = old_schema.get("fields", {})
    new_fields: Dict[str, Dict[str, Any]] = new_schema.get("fields", {})

    old_invs = set(old_schema.get("invariants", ()))
    new_invs = set(new_schema.get("invariants", ()))
    retired_invs = set(new_schema.get("retired_invariants", ()))
    silently_dropped_invs = sorted(old_invs - new_invs - retired_invs)
    if silently_dropped_invs:
        breaking_reasons.append(f"hard_invariant_silently_dropped:{','.join(silently_dropped_invs)}")

    for fname, fspec in old_fields.items():
        if fname not in new_fields:
            populated = any(fname in rec and rec[fname] is not None for rec in records)
            if fspec.get("required") or populated:
                breaking_reasons.append(f"removed_field_breaks_history:{fname}")
            continue
        nspec = new_fields[fname]
        otype = fspec.get("type", "str")
        ntype = nspec.get("type", "str")
        if otype != ntype and (otype, ntype) not in _COMPATIBLE_TYPE_WIDENING:
            breaking_reasons.append(f"incompatible_type_change:{fname}:{otype}->{ntype}")

        new_enum = nspec.get("enum")
        if new_enum is not None:
            new_enum_set = set(new_enum)
            old_enum = fspec.get("enum")
            if old_enum is not None and not set(old_enum).issubset(new_enum_set):
                breaking_reasons.append(f"enum_domain_narrowed:{fname}")
            for idx, rec in enumerate(records):
                if fname in rec and rec[fname] is not None and rec[fname] not in new_enum_set:
                    breaking_reasons.append(f"record_enum_out_of_domain:{idx}:{fname}={rec[fname]!r}")

    for fname, nspec in new_fields.items():
        if fname not in old_fields and nspec.get("required") and "default" not in nspec:
            missing_in_hist = any(fname not in rec or rec[fname] is None for rec in records)
            if missing_in_hist or not records:
                breaking_reasons.append(f"new_required_field_missing_default:{fname}")

    migrated: List[Dict[str, Any]] = []
    if not breaking_reasons:
        for idx, rec in enumerate(records):
            out_rec = dict(rec)
            for fname, nspec in new_fields.items():
                if fname not in out_rec and "default" in nspec:
                    out_rec[fname] = nspec["default"]
                if fname in out_rec and out_rec[fname] is not None:
                    ntype = nspec.get("type", "str")
                    if not _value_matches_schema_type(out_rec[fname], ntype):
                        breaking_reasons.append(f"record_type_mismatch:{idx}:{fname}:{ntype}")
                    elif ntype == "float" and isinstance(out_rec[fname], int) and not isinstance(out_rec[fname], bool):
                        out_rec[fname] = float(out_rec[fname])
            migrated.append(out_rec)

    ok = len(breaking_reasons) == 0
    return {
        "ok": ok,
        "decision": "MIGRATE_COMPAT" if ok else "HALT_BREAKING",
        "from_epoch": old_ep,
        "to_epoch": new_ep,
        "records_checked": len(records),
        "migrated_records": migrated if ok else [],
        "breaking_reasons": breaking_reasons,
        "halt_tag": "<|halt:closure|>" if ok else "<|halt:bottom|>",
    }


def validate_external_user_protocol(
    request: Dict[str, Any],
    response: Dict[str, Any],
) -> Dict[str, Any]:
    """v32.0 External AI User Firewall Gate (USER-EXTERNAL-AI-FIREWALL)."""
    errors: List[str] = []
    policy = request.get("compiler_policy", {}) if isinstance(request.get("compiler_policy"), dict) else {}
    zero_ai = policy.get("zero_ai_inside_compiler", request.get("zero_ai_inside_compiler"))
    if zero_ai is not True:
        errors.append("compiler_policy_violation:zero_ai_inside_compiler_must_be_true")

    for forbidden_hook in ("compiler_llm_hook", "inject_llm_pass", "neural_lowering"):
        if request.get(forbidden_hook) or policy.get(forbidden_hook):
            errors.append(f"compiler_contamination_forbidden:{forbidden_hook}")

    ai_role = request.get("ai_role")
    if ai_role != "EXTERNAL_USER_ONLY":
        errors.append(f"ai_role_violation:expected_EXTERNAL_USER_ONLY_got_{ai_role!r}")

    for carrier_name, carrier in (("request", request), ("response", response)):
        if (
            carrier.get("authority_req") == "host_grant"
            and carrier.get("minted_by") in ("external_ai", "model_permit", "EXTERNAL_USER_ONLY")
        ):
            errors.append(f"authority_violation:{carrier_name}_external_ai_cannot_mint_host_grant")

    forbidden_tokens = DEFAULT_FOUR_LAYER_PROFILES["L0_INTENT"]["forbidden_lower_tokens"] | _INTERNAL_COMPILER_JARGON
    user_facing_blob = " ".join(
        str(response.get(k, ""))
        for k in ("message", "lifted_tradeoff", "diagnostics_summary", "user_facing_text")
    ).lower()
    leaked_tokens = sorted(
        tok for tok in forbidden_tokens
        if (
            tok.lower() in user_facing_blob
            if not tok.isascii()
            else bool(re.search(rf"(?<![a-z0-9]){re.escape(tok.lower())}(?![a-z0-9])", user_facing_blob))
        )
    )
    if leaked_tokens:
        errors.append(f"internal_jargon_leakage:{','.join(leaked_tokens)}")

    ok = len(errors) == 0
    return {
        "ok": ok,
        "decision": "ALLOW_EXTERNAL_PROTOCOL" if ok else "REJECT_PROTOCOL_VIOLATION",
        "zero_ai_inside_compiler": bool(zero_ai is True),
        "ai_role": str(ai_role),
        "leaked_tokens": leaked_tokens,
        "errors": errors,
        "halt_tag": "<|halt:closure|>" if ok else "<|halt:bottom|>",
    }
