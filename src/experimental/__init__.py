"""
Sovereign AI Brain — Experimental & Phase-Gated Module Registry (CTO v9.4).
Maps every ADR phase directly to its isolated module so code structure and
docs/architecture/ARCHITECTURE_DECISIONS.md stay 100% isomorphic.
"""
from __future__ import annotations

PHASE_ROUTING_REGISTRY = {
    "Phase 2 · 后训练/RLVR (Accepted Backbone)": {
        "module": "src.experimental.training_and_rl_planners",
        "adrs": ["ADR-008"],
        "highlights": [
            "GPPOGeneralizedDivergenceRL",
            "TRUERErrorAwareRewardRectifier",
            "ZeroTIMThreeWayPolicyContract",
            "MultiTurnInTurnToolRLOOEngine",
            "FtFSAEFeatureDataSelector",
        ],
    },
    "Phase 3 · 推理与外骨骼 (Accepted Backbone)": {
        "modules": [
            "src.experimental.memory_and_sleep_planners",
            "src.experimental.harness_and_safety_policies",
            "src.experimental.kernel_and_serving_planners",
        ],
        "adrs": ["ADR-003", "ADR-005", "ADR-007", "ADR-009", "ADR-011", "ADR-012"],
        "highlights": [
            "OPSDWeightTextMemoryRouter",
            "TrainingFreeMemoryLifecycleGate",
            "MEMENTOCompactionAndBacktrackingController",
            "YoloFSStagedSnapshotFilesystem",
            "DIADisagreementHeterogeneousArbiter",
            "SchemaFillSlotParallelToolDecoder",
            "HelionCompileTimeIRHeuristicPlanner",
            "Int8Exact426BitBlockGEMM",
            "FairwayDualLanePrefillScheduler",
            "PrefixCacheMetastabilityBudgeter",
            "QHijackCrossPrecisionAuditGate",
            "ShapleyCompactMemoryPruner",
            "CmuFluidServicePriorityRouter",
            "CISAChunkIndexedSparseAttentionPlanner",
            "TACCoTInterleavedLatentToolScheduler",
            "AirlockProvenanceContainmentGate",
            "PRISMCausalPrivilegeMinimizer",
            "ContinualHarnessRegressionGuard",
            "EvolveAuditedGuardrailRegistry",
            "GovernedExperimentAppendOnlyLedger",
            "MemShiftDeterministicRepairRouter",
            "PlanAndPatchLocalRepairPlanner",
        ],
    },
    "Scale-Up · 集群/Blackwell (Hardware-Gated Backbone)": {
        "modules": [
            "src.experimental.kernel_and_serving_planners",
            "src.experimental.training_and_rl_planners",
        ],
        "adrs": ["ADR-005", "ADR-007"],
        "highlights": [
            "NVFP4MicroBlockQuantizer",
            "NVFP4ThreePathKVCachePlanner",
            "StreamingDiLoCoFragmentScheduler",
            "FlexShardMuonAndCollectiveFreeCkptPlanner",
            "EpiLoRAFusedEpiloguePlanner",
        ],
    },
    "⚔️ 同位待 PK (True A/B Challengers)": {
        "module": "src.experimental.kernel_and_serving_planners",
        "adrs": ["ADR-001", "ADR-002", "ADR-003", "ADR-005"],
        "highlights": [
            "TaSQOneBitKVSpaceTailor",
            "StagQDyadicMultiPrecisionStream",
            "E8LatticeImplicit2BitQAT",
        ],
    },
    "Legacy PoC (Historical Reference Only)": {
        "module": "src.experimental.legacy_numpy_backbone_poc",
        "adrs": [],
        "highlights": ["Early pure-NumPy non-Transformer PoC superseded by PyTorch Route B"],
    },
}
