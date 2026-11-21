"""
Sovereign AI Brain — Declarative Phase & Hardware Configuration Matrix (v9.5).
Pure-Python (zero PyTorch import overhead) single source of truth for switching
between Phase 1 (Pretraining), Phase 2 (Post-Training/RLVR), Phase 3 (Serving/Exoskeleton),
and Scale-Up (Blackwell/Cluster) profiles aligned 1-to-1 with ADR-001..ADR-012.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class DeploymentProfile:
    profile_id: str
    phase: str
    target_hardware: str
    vram_budget_gb: float
    brain_preset: str
    active_adrs: Tuple[str, ...]
    challenger_adrs: Tuple[str, ...] = ()
    feature_flags: Dict[str, bool] = field(default_factory=dict)


DEPLOYMENT_PROFILES: Dict[str, DeploymentProfile] = {
    "rtx3050_4gb_phase1_pretrain": DeploymentProfile(
        profile_id="rtx3050_4gb_phase1_pretrain",
        phase="Phase 1 · 预训练",
        target_hardware="NVIDIA RTX 3050 Laptop (4GB VRAM, Ampere SM86)",
        vram_budget_gb=4.0,
        brain_preset="rtx3050_4gb_65m",
        active_adrs=("ADR-001", "ADR-002", "ADR-003", "ADR-004", "ADR-005", "ADR-007", "ADR-009", "ADR-010"),
        challenger_adrs=("ADR-002", "ADR-007"),
        feature_flags={
            "enable_cpu_engram": True,
            "enable_rank2_deltaproduct": True,
            "enable_arloop_depth_attn": True,
            "enable_lca_multibyte": True,
            "enable_gatesight_kv_prune": False,
            "quantize_momentum_int8": True,
            "use_grad_checkpointing": True,
        },
    ),
    "rtx3050_4gb_phase2_rlvr": DeploymentProfile(
        profile_id="rtx3050_4gb_phase2_rlvr",
        phase="Phase 2 · 后训练/RLVR",
        target_hardware="NVIDIA RTX 3050 Laptop (4GB VRAM, Ampere SM86)",
        vram_budget_gb=4.0,
        brain_preset="rtx3050_4gb_65m",
        active_adrs=("ADR-008", "ADR-009", "ADR-011"),
        challenger_adrs=("ADR-008",),
        feature_flags={
            "use_deletion_tuning_mask": True,
            "use_gppo_divergence": True,
            "use_kappo_single_completion": True,
            "use_zero_tim_contract": True,
            "use_truer_reward_rectifier": True,
        },
    ),
    "rtx3050_4gb_phase3_serving": DeploymentProfile(
        profile_id="rtx3050_4gb_phase3_serving",
        phase="Phase 3 · 推理与外骨骼",
        target_hardware="NVIDIA RTX 3050 Laptop (4GB VRAM, Ampere SM86)",
        vram_budget_gb=4.0,
        brain_preset="rtx3050_4gb_65m_true",
        active_adrs=("ADR-003", "ADR-005", "ADR-006", "ADR-009", "ADR-010", "ADR-011", "ADR-012"),
        challenger_adrs=("ADR-001", "ADR-003", "ADR-005", "ADR-006"),
        feature_flags={
            "enable_gatesight_kv_prune": True,
            "enable_bitnest_spec_decode": True,
            "enable_int8_426bit_gemm": True,
            "enable_opsd_memory_router": True,
            "enable_yolofs_staged_fs": True,
            "enable_dia_heterogeneous_arbiter": True,
            "enable_replay_ssm_checkpoint": True,
            "enable_transactional_kv_guard": True,
        },
    ),
    "blackwell_cluster_scaleup": DeploymentProfile(
        profile_id="blackwell_cluster_scaleup",
        phase="Scale-Up · 集群/Blackwell",
        target_hardware="NVIDIA Blackwell SM100+ Multi-Node Cluster",
        vram_budget_gb=192.0,
        brain_preset="cluster_scale_1b",
        active_adrs=("ADR-001", "ADR-002", "ADR-003", "ADR-004", "ADR-005", "ADR-007", "ADR-008", "ADR-012"),
        challenger_adrs=(),
        feature_flags={
            "enable_nvfp4_microblock": True,
            "enable_streaming_diloco": True,
            "enable_flexshard_muon": True,
            "enable_gatesight_kv_prune": True,
            "enable_replay_ssm_checkpoint": True,
            "enable_transactional_kv_guard": True,
        },
    ),
}


def get_profile(profile_id: str) -> DeploymentProfile:
    if profile_id not in DEPLOYMENT_PROFILES:
        raise KeyError(f"Unknown profile_id={profile_id!r}. Available: {sorted(DEPLOYMENT_PROFILES)}")
    return DEPLOYMENT_PROFILES[profile_id]
