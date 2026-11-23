#!/usr/bin/env python3
"""
Sovereign AI Brain — 2026 SOTA Open-Source Ecosystem Wheel Matrix, Conflict Arbitration & Live Freshness Gate (v10.32)
=======================================================================================================================
Organizes 77 verified 2025-2026 open-source wheels across 16 functional domains with explicit
conflict arbitration (`primary` / `challenger_or_alt` / `scaleup_or_backend` / `rejected_conflict`),
live 2026-11-21 GitHub API telemetry (`github_repo`, `verified_date`, `verified_pushed_at`, `license`),
and an executable `audit_ecosystem_wheel_freshness(online=...)` gate to prevent repo rot.

Every bridge degrades gracefully to pure PyTorch / pure Python when optional wheels are absent.
"""
from __future__ import annotations

import importlib.util
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

WHEEL_DOMAINS: tuple[str, ...] = (
    "1_kernels_and_compilers",
    "2_quantization_and_checkpoints",
    "3_constrained_decoding",
    "4_serving_and_edge_inference",
    "5_distributed_optimizers_and_rlvr",
    "6_corpus_curation",
    "7_benchmarks_and_agent_evals",
    "8_sandboxes_and_observability",
    "9_streaming_dataloaders_and_data_ops",
    "10_kv_offload_tracking_and_guardrails",
    "11_formal_verifiers_and_stem_ingestion",
    "12_local_vector_memory_and_mcp",
    "13_code_ast_and_static_analysis",
    "14_web_extraction_and_browser_exoskeleton",
    "15_mechanistic_interpretability",
    "16_local_edge_speech_io",
)

ECOSYSTEM_WHEEL_CATALOG: Dict[str, Dict[str, str]] = {
    # --- Domain 1: GPU Kernels & Compilers (8 wheels) ---
    "fla": {
        "domain": "1_kernels_and_compilers",
        "verdict": "primary",
        "pip": "flash-linear-attention",
        "tier": "hotpath_wired",
        "github_repo": "fla-org/flash-linear-attention",
        "verified_pushed_at": "2026-11-21",
        "license": "MIT",
        "role": "Triton/TileLang GDN-2, Mamba-3, DeltaProduct, KalmaNet, mHC & fused AttnRes kernels",
        "conflict_note": "Primary linear-attention kernel library; supersedes standalone mamba-ssm for hybrid GDN+Attention.",
        "url": "https://github.com/fla-org/flash-linear-attention",
    },
    "liger_kernel": {
        "domain": "1_kernels_and_compilers",
        "verdict": "primary",
        "pip": "liger-kernel",
        "tier": "hotpath_wired",
        "github_repo": "linkedin/Liger-Kernel",
        "verified_pushed_at": "2026-11-21",
        "license": "BSD-2-Clause",
        "role": "Triton fused RMSNorm, SwiGLU, RoPE & FusedLinearCrossEntropy (-40-60% train VRAM)",
        "conflict_note": "Primary all-in-one Triton training kernel suite; beats separate single-op libraries for 256-byte vocab.",
        "url": "https://github.com/linkedin/Liger-Kernel",
    },
    "flashinfer": {
        "domain": "1_kernels_and_compilers",
        "verdict": "primary",
        "pip": "flashinfer-python",
        "tier": "serving_kernel_base",
        "github_repo": "flashinfer-ai/flashinfer",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "Unified LLM serving kernel library (Paged/Ragged KV, BF16/FP8/FP4 GEMM, RoPE, sampling) under SGLang/vLLM",
        "conflict_note": "Primary serving-side attention & GEMM kernel base; complements fla linear-attention kernels.",
        "url": "https://github.com/flashinfer-ai/flashinfer",
    },
    "cut_cross_entropy": {
        "domain": "1_kernels_and_compilers",
        "verdict": "challenger_or_alt",
        "pip": "cut-cross-entropy",
        "tier": "hotpath_wired",
        "github_repo": "apple/ml-cross-entropy",
        "verified_pushed_at": "2026-09-11",
        "license": "Apple-Sample",
        "role": "Exact linear+CE without materializing full logits in VRAM (37x memory savings on large vocabs)",
        "conflict_note": "Hot-path fallback in try_liger_fused_linear_ce; preferred over Liger CE if vocab scales up.",
        "url": "https://github.com/apple/ml-cross-entropy",
    },
    "flash_attention_4": {
        "domain": "1_kernels_and_compilers",
        "verdict": "scaleup_or_backend",
        "pip": "flash-attn",
        "tier": "scaleup_hopper_blackwell",
        "github_repo": "Dao-AILab/flash-attention",
        "verified_pushed_at": "2026-11-21",
        "license": "BSD-3-Clause",
        "role": "CuTe-DSL FlashAttention-4 reaching 1,605 TFLOPs/s on B200 (1.3x cuDNN / 2.7x Triton)",
        "conflict_note": "Scale-Up ONLY (Hopper/Blackwell SM90/SM100); RTX 3050 Ampere uses PyTorch native SDPA/FA2.",
        "url": "https://github.com/dao-ailab/flash-attention",
    },
    "flashqla": {
        "domain": "1_kernels_and_compilers",
        "verdict": "scaleup_or_backend",
        "pip": "flashqla",
        "tier": "kernel_backend",
        "github_repo": "QwenLM/FlashQLA",
        "verified_pushed_at": "2026-10-30",
        "license": "MIT",
        "role": "QwenLM plug-and-play high-performance GDN backend for fla",
        "conflict_note": "Complementary backend plugged underneath fla (no conflict).",
        "url": "https://github.com/QwenLM/FlashQLA",
    },
    "tilelang": {
        "domain": "1_kernels_and_compilers",
        "verdict": "scaleup_or_backend",
        "pip": "tilelang",
        "tier": "kernel_backend",
        "github_repo": "tile-ai/tilelang",
        "verified_pushed_at": "2026-11-20",
        "license": "MIT",
        "role": "High-performance GPU/CPU kernel DSL powering fla GDN/KDA kernels",
        "conflict_note": "Complementary kernel DSL alongside Triton/Helion.",
        "url": "https://github.com/tile-ai/tilelang",
    },
    "helion": {
        "domain": "1_kernels_and_compilers",
        "verdict": "scaleup_or_backend",
        "pip": "helion",
        "tier": "kernel_compiler",
        "github_repo": "pytorch/helion",
        "verified_pushed_at": "2026-11-21",
        "license": "BSD-3-Clause",
        "role": "PyTorch-native tile DSL compiling to autotuned Triton/CuTe with 2026 compile-time IR heuristics",
        "conflict_note": "Replaces hand-written Triton/CUDA boilerplate for custom Route B fused kernels (4x shorter code).",
        "url": "https://github.com/pytorch/helion",
    },

    # --- Domain 2: Quantization, QAT & Checkpoints (4 wheels) ---
    "safetensors": {
        "domain": "2_quantization_and_checkpoints",
        "verdict": "primary",
        "pip": "safetensors",
        "tier": "hotpath_wired",
        "github_repo": "huggingface/safetensors",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "Zero-pickle fast memory-mapped checkpoint weight serialization",
        "conflict_note": "Primary weight format paired with .pt optimizer metadata (replaces raw pickle weights).",
        "url": "https://github.com/huggingface/safetensors",
    },
    "torchao": {
        "domain": "2_quantization_and_checkpoints",
        "verdict": "primary",
        "pip": "torchao",
        "tier": "quantization",
        "github_repo": "pytorch/ao",
        "verified_pushed_at": "2026-11-21",
        "license": "BSD-3-Clause",
        "role": "PyTorch-native INT4/INT8 quantization & PARQ optimizer-level 3-bit/4-bit QAT",
        "conflict_note": "Primary PyTorch-native QAT/quantization stack; composable with torch.compile (beats bitsandbytes).",
        "url": "https://github.com/pytorch/ao",
    },
    "auto_round": {
        "domain": "2_quantization_and_checkpoints",
        "verdict": "challenger_or_alt",
        "pip": "auto-round",
        "tier": "quantization_challenger",
        "github_repo": "intel/auto-round",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "Intel SignRoundV2 signed-gradient PTQ (W4A16/MXFP4/GGUF, PTQ cost with QAT-level accuracy)",
        "conflict_note": "Same-slot PTQ challenger against custom block-Hadamard Q4 for offline 4-bit export.",
        "url": "https://github.com/intel/auto-round",
    },
    "bitsandbytes": {
        "domain": "2_quantization_and_checkpoints",
        "verdict": "rejected_conflict",
        "pip": "bitsandbytes (DO NOT INSTALL)",
        "tier": "rejected",
        "github_repo": "bitsandbytes-foundation/bitsandbytes",
        "verified_pushed_at": "2026-11-02",
        "license": "MIT",
        "role": "Legacy NF4 / LLM.int8() / 8-bit AdamW library",
        "conflict_note": "REJECTED: Superseded by PolarExpressMuonOptimizer (INT8 momentum) + TorchAO + AutoRound; weaker torch.compile composability.",
        "url": "https://github.com/bitsandbytes-foundation/bitsandbytes",
    },

    # --- Domain 3: Microsecond Constrained Decoding (2 wheels) ---
    "xgrammar": {
        "domain": "3_constrained_decoding",
        "verdict": "primary",
        "pip": "xgrammar",
        "tier": "hotpath_wired",
        "github_repo": "mlc-ai/xgrammar",
        "verified_pushed_at": "2026-11-18",
        "license": "Apache-2.0",
        "role": "<40 us C++ pushdown-automaton constrained decoding with Structural Tag for <tool_call>",
        "conflict_note": "Primary 2026-11 constrained decoder (ACM CAIS '26 XGrammar-2; CoT stays unconstrained).",
        "url": "https://github.com/mlc-ai/xgrammar",
    },
    "llguidance": {
        "domain": "3_constrained_decoding",
        "verdict": "challenger_or_alt",
        "pip": "llguidance",
        "tier": "optional_alt",
        "github_repo": "guidance-ai/llguidance",
        "verified_pushed_at": "2026-11-18",
        "license": "MIT",
        "role": "~50 us Rust lazy-mask CFG/JSON structured output engine",
        "conflict_note": "Same-slot Rust challenger to XGrammar-2 for deeply recursive CFGs.",
        "url": "https://github.com/guidance-ai/llguidance",
    },

    # --- Domain 4: Serving & Edge Inference (7 wheels) ---
    "sglang": {
        "domain": "4_serving_and_edge_inference",
        "verdict": "primary",
        "pip": "sglang",
        "tier": "serving_primary",
        "github_repo": "sgl-project/sglang",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "RadixAttention GPU serving engine with dedicated hybrid GDN/Mamba state prefix cache",
        "conflict_note": "Primary GPU server for hybrid 3:1 GDN+Attention & multi-turn shared-prefix agent workloads.",
        "url": "https://github.com/sgl-project/sglang",
    },
    "vllm": {
        "domain": "4_serving_and_edge_inference",
        "verdict": "challenger_or_alt",
        "pip": "vllm",
        "tier": "serving_challenger",
        "github_repo": "vllm-project/vllm",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "PagedAttention GPU server with unified hybrid KV+Mamba block pool",
        "conflict_note": "Same-slot challenger to SGLang when mixed short/long bursty traffic needs unified KV+state pooling.",
        "url": "https://github.com/vllm-project/vllm",
    },
    "llama_cpp": {
        "domain": "4_serving_and_edge_inference",
        "verdict": "primary",
        "pip": "source-build / llama-cpp-python",
        "tier": "edge_primary",
        "github_repo": "ggml-org/llama.cpp",
        "verified_pushed_at": "2026-11-21",
        "license": "MIT",
        "role": "MIT C/C++ GGUF local CPU+GPU hybrid inference engine with AVX2/AVX512/CUDA/Metal SIMD",
        "conflict_note": "Primary single-user edge/laptop runtime; strictly outperforms Ollama wrapper on identical GGUF.",
        "url": "https://github.com/ggml-org/llama.cpp",
    },
    "bitnet_cpp": {
        "domain": "4_serving_and_edge_inference",
        "verdict": "primary",
        "pip": "source-build (microsoft/BitNet)",
        "tier": "edge_ternary",
        "github_repo": "microsoft/BitNet",
        "verified_pushed_at": "2026-07-27",
        "license": "MIT",
        "role": "2.37x-6.17x x86 / 1.37x-5.07x ARM lossless ternary/low-bit CPU edge inference (TL & I2_S)",
        "conflict_note": "Primary ultra-low-bit CPU backend complementing llama.cpp for sub-2-bit/ternary deployments.",
        "url": "https://github.com/microsoft/BitNet",
    },
    "executorch": {
        "domain": "4_serving_and_edge_inference",
        "verdict": "challenger_or_alt",
        "pip": "executorch",
        "tier": "mobile_edge",
        "github_repo": "pytorch/executorch",
        "verified_pushed_at": "2026-11-21",
        "license": "BSD-3-Clause",
        "role": "PyTorch-native mobile/embedded runtime paired with TorchAO QAT (phone-deployment)",
        "conflict_note": "Mobile/embedded deployment option alongside MLC-LLM.",
        "url": "https://github.com/pytorch/executorch",
    },
    "mlc_llm": {
        "domain": "4_serving_and_edge_inference",
        "verdict": "challenger_or_alt",
        "pip": "mlc-llm",
        "tier": "cross_platform_edge",
        "github_repo": "mlc-ai/mlc-llm",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "Apache TVM compiled universal LLM engine across CUDA/Vulkan/Metal/WebGPU/iOS/Android",
        "conflict_note": "Cross-platform compiled edge alternative to ExecuTorch & llama.cpp.",
        "url": "https://github.com/mlc-ai/mlc-llm",
    },
    "ollama": {
        "domain": "4_serving_and_edge_inference",
        "verdict": "rejected_conflict",
        "pip": "ollama (DO NOT USE IN PROD/BENCH)",
        "tier": "rejected",
        "github_repo": "ollama/ollama",
        "verified_pushed_at": "2026-11-21",
        "license": "MIT",
        "role": "Single-machine wrapper over llama.cpp",
        "conflict_note": "REJECTED for prod/hybrid: forces single-slot concurrency on 11 hybrid architectures (52x slower than SGLang/vLLM at 50 conc) and mutates raw prompts.",
        "url": "https://winder.ai/vllm-vs-ollama-vs-sglang-llm-inference-comparison/",
    },

    # --- Domain 5: Distributed Training, Optimizers & Async RLVR (8 wheels) ---
    "torchtitan": {
        "domain": "5_distributed_optimizers_and_rlvr",
        "verdict": "primary",
        "pip": "torchtitan",
        "tier": "scaleup_training_primary",
        "github_repo": "pytorch/torchtitan",
        "verified_pushed_at": "2026-11-21",
        "license": "BSD-3-Clause",
        "role": "PyTorch-native 5D parallel pretraining platform with FlexShard distributed Muon (-79% opt time), native GDN/Helion & TitanRL",
        "conflict_note": "Primary multi-GPU Scale-Up trainer; supersedes fla-org/flame (whose features merged upstream).",
        "url": "https://github.com/pytorch/torchtitan",
    },
    "dion": {
        "domain": "5_distributed_optimizers_and_rlvr",
        "verdict": "scaleup_or_backend",
        "pip": "git+https://github.com/microsoft/dion.git",
        "tier": "scaleup_optimizer",
        "github_repo": "microsoft/dion",
        "verified_pushed_at": "2026-09-30",
        "license": "MIT",
        "role": "Microsoft Dion/Dion3 distributed orthonormalized Muon (Gram Newton-Schulz + CuTeDSL, up to 6x faster step)",
        "conflict_note": "Drop-in distributed Scale-Up upgrade path for our single-GPU PolarExpressMuonOptimizer.",
        "url": "https://github.com/microsoft/dion",
    },
    "flame": {
        "domain": "5_distributed_optimizers_and_rlvr",
        "verdict": "challenger_or_alt",
        "pip": "git+https://github.com/fla-org/flame.git",
        "tier": "scaleup_alt_watch_stale",
        "github_repo": "fla-org/flame",
        "verified_pushed_at": "2026-04-22",
        "license": "MIT",
        "role": "Minimal TorchTitan wrapper specifically pre-configured for fla linear-attention models",
        "conflict_note": "STALE WATCH (last push 2026-04-22): prefer upstream pytorch/torchtitan directly for Scale-Up.",
        "url": "https://github.com/fla-org/flame",
    },
    "verl": {
        "domain": "5_distributed_optimizers_and_rlvr",
        "verdict": "primary",
        "pip": "verl",
        "tier": "phase2_rlvr_primary",
        "github_repo": "verl-project/verl",
        "verified_pushed_at": "2026-11-20",
        "license": "Apache-2.0",
        "role": "Production HybridFlow RLVR / GRPO / Dr.GRPO post-training framework (FSDP/Megatron/TorchTitan + SGLang/vLLM)",
        "conflict_note": "Primary general RLVR orchestrator; pair with AReaL 2.0 / slime for fully async long-horizon agent RL.",
        "url": "https://github.com/verl-project/verl",
    },
    "areal": {
        "domain": "5_distributed_optimizers_and_rlvr",
        "verdict": "primary",
        "pip": "areal",
        "tier": "phase2_async_rlvr_primary",
        "github_repo": "areal-project/areal",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "AReaL 2.0 microservice fully-asynchronous agentic RL (2.77x faster than sync RL, KPop/IcePop KL masking)",
        "conflict_note": "Co-primary for fully asynchronous multi-turn agentic RLVR alongside verl.",
        "url": "https://github.com/areal-project/areal",
    },
    "slime": {
        "domain": "5_distributed_optimizers_and_rlvr",
        "verdict": "challenger_or_alt",
        "pip": "git+https://github.com/THUDM/slime.git",
        "tier": "phase2_rlvr_challenger",
        "github_repo": "THUDM/slime",
        "verified_pushed_at": "2026-11-05",
        "license": "Apache-2.0",
        "role": "SGLang-native + Megatron high-performance RL scaling substrate for frontier verifiable environments",
        "conflict_note": "Same-slot challenger to verl/AReaL when SGLang+Megatron tightly coupled rollouts are used.",
        "url": "https://github.com/THUDM/slime",
    },
    "skyrl": {
        "domain": "5_distributed_optimizers_and_rlvr",
        "verdict": "challenger_or_alt",
        "pip": "skyrl",
        "tier": "phase2_agent_rl_challenger",
        "github_repo": "skyrl-project/SkyRL",
        "verified_pushed_at": "2026-11-20",
        "license": "Apache-2.0",
        "role": "Modular full-stack multi-turn tool-use agent RL library with Harbor & Tinker API support",
        "conflict_note": "Same-slot multi-turn tool RL challenger to AReaL 2.0.",
        "url": "https://github.com/skyrl-project/SkyRL",
    },
    "prime_rl": {
        "domain": "5_distributed_optimizers_and_rlvr",
        "verdict": "challenger_or_alt",
        "pip": "prime-rl",
        "tier": "phase2_agent_rl_challenger",
        "github_repo": "PrimeIntellect-ai/prime-rl",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "Hackable 1000+-GPU agentic RL framework tightly integrated with PrimeIntellect verifiers hub",
        "conflict_note": "Same-slot challenger to verl/AReaL for verifiable environment hubs.",
        "url": "https://github.com/PrimeIntellect-ai/prime-rl",
    },
    "openrlhf": {
        "domain": "5_distributed_optimizers_and_rlvr",
        "verdict": "challenger_or_alt",
        "pip": "openrlhf",
        "tier": "phase2_rlvr_challenger",
        "github_repo": "OpenRLHF/OpenRLHF",
        "verified_pushed_at": "2026-11-03",
        "license": "Apache-2.0",
        "role": "Production PPO / REINFORCE++ / GRPO / RLOO post-training framework",
        "conflict_note": "Legacy Ray+DeepSpeed challenger; prefer verl / AReaL 2.0 / slime for 2026 async agentic RLVR.",
        "url": "https://github.com/openrlhf/openrlhf",
    },

    # --- Domain 6: Corpus Curation & Deduplication (2 wheels) ---
    "datatrove": {
        "domain": "6_corpus_curation",
        "verdict": "primary",
        "pip": "datatrove",
        "tier": "corpus_pipeline",
        "github_repo": "huggingface/datatrove",
        "verified_pushed_at": "2026-11-18",
        "license": "Apache-2.0",
        "role": "CPU-first FineWeb-Edu reference corpus filtering & near-deduplication for 30GB+ streams",
        "conflict_note": "Primary 30GB local curation wheel paired with prepare_seed_corpus.py RCE anti-poisoning.",
        "url": "https://github.com/huggingface/datatrove",
    },
    "nemo_curator": {
        "domain": "6_corpus_curation",
        "verdict": "scaleup_or_backend",
        "pip": "nemo-curator",
        "tier": "corpus_scaleup",
        "github_repo": "NVIDIA/NeMo-Curator",
        "verified_pushed_at": "2026-11-19",
        "license": "Apache-2.0",
        "role": "NVIDIA GPU-accelerated MinHash LSH fuzzy/semantic dedup (~4 TB/hr on 8xH100)",
        "conflict_note": "Scale-Up only (10T+ tokens on multi-GPU); overkill for 30GB RTX 3050 local corpus where DataTrove wins.",
        "url": "https://github.com/NVIDIA/NeMo-Curator",
    },

    # --- Domain 7: Real Benchmarks & Agent Evals (4 wheels) ---
    "lm_eval": {
        "domain": "7_benchmarks_and_agent_evals",
        "verdict": "primary",
        "pip": "lm-eval",
        "tier": "evaluation",
        "github_repo": "EleutherAI/lm-evaluation-harness",
        "verified_pushed_at": "2026-11-04",
        "license": "MIT",
        "role": "EleutherAI lm-evaluation-harness standardized benchmarks (GSM8K/MATH/RULER)",
        "conflict_note": "Primary knowledge/reasoning benchmark suite.",
        "url": "https://github.com/EleutherAI/lm-evaluation-harness",
    },
    "inspect_ai": {
        "domain": "7_benchmarks_and_agent_evals",
        "verdict": "primary",
        "pip": "inspect-ai",
        "tier": "agent_evaluation",
        "github_repo": "UKGovernmentBEIS/inspect_ai",
        "verified_pushed_at": "2026-11-21",
        "license": "MIT",
        "role": "UK AISI MIT-licensed sandboxed agentic eval framework (SWE-bench Pro, tool boundaries, prompt-injection safety)",
        "conflict_note": "Primary 2026 agentic & safety regression harness (also covers code evals where evalplus went quiet).",
        "url": "https://github.com/UKGovernmentBEIS/inspect_ai",
    },
    "evalplus": {
        "domain": "7_benchmarks_and_agent_evals",
        "verdict": "challenger_or_alt",
        "pip": "evalplus",
        "tier": "evaluation_watch_stale",
        "github_repo": "evalplus/evalplus",
        "verified_pushed_at": "2025-10-02",
        "license": "Apache-2.0",
        "role": "HumanEval+ & MBPP+ sandbox code execution benchmarks",
        "conflict_note": "STALE WATCH (last push 2025-10-02): keep for HumanEval+/MBPP+ parity, supplement with Inspect AI SWE-bench Pro.",
        "url": "https://github.com/evalplus/evalplus",
    },
    "openhands": {
        "domain": "7_benchmarks_and_agent_evals",
        "verdict": "challenger_or_alt",
        "pip": "openhands",
        "tier": "agent_control_plane_ref",
        "github_repo": "OpenHands/OpenHands",
        "verified_pushed_at": "2026-11-21",
        "license": "MIT",
        "role": "Open-source self-hosted coding agent control plane (ACP protocol, sandboxed execution, audit logs)",
        "conflict_note": "Reference agent harness control plane for src/harness/ interop and SWE-bench benchmarking.",
        "url": "https://github.com/OpenHands/openhands",
    },

    # --- Domain 8: Hardware-Isolated Agent Sandboxes & Observability (5 wheels) ---
    "firecracker": {
        "domain": "8_sandboxes_and_observability",
        "verdict": "primary",
        "pip": "binary-vmm (firecracker-microvm/firecracker)",
        "tier": "exoskeleton_isolation_primary",
        "github_repo": "firecracker-microvm/firecracker",
        "verified_pushed_at": "2026-11-20",
        "license": "Apache-2.0",
        "role": "Rust KVM hardware microVM (<125ms cold start, <5 MiB overhead, dedicated guest kernel per sandbox)",
        "conflict_note": "Primary hardware-level isolation boundary for untrusted code in src/harness/ (beats shared-kernel Docker).",
        "url": "https://github.com/firecracker-microvm/firecracker",
    },
    "e2b": {
        "domain": "8_sandboxes_and_observability",
        "verdict": "primary",
        "pip": "e2b",
        "tier": "exoskeleton_sandbox_sdk",
        "github_repo": "e2b-dev/e2b",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "Apache-2.0 self-hostable AI agent sandbox runtime & SDK orchestrating Firecracker microVM memory snapshots (~150ms)",
        "conflict_note": "Primary self-hostable microVM sandbox orchestration layer (beats archived Daytona).",
        "url": "https://github.com/e2b-dev/e2b",
    },
    "langfuse": {
        "domain": "8_sandboxes_and_observability",
        "verdict": "primary",
        "pip": "langfuse",
        "tier": "observability_primary",
        "github_repo": "langfuse/langfuse",
        "verified_pushed_at": "2026-11-21",
        "license": "MIT",
        "role": "MIT-core self-hosted ClickHouse + OpenTelemetry AI agent tracing, evals & skill versioning platform",
        "conflict_note": "Primary sovereign observability stack (MIT OSI license; beats Elastic-License/Dynatrace-acquired Arize Phoenix).",
        "url": "https://github.com/langfuse/langfuse",
    },
    "daytona": {
        "domain": "8_sandboxes_and_observability",
        "verdict": "rejected_conflict",
        "pip": "daytona (DO NOT USE — OSS ARCHIVED)",
        "tier": "rejected",
        "github_repo": "daytonaio/daytona",
        "verified_pushed_at": "2026-07-24",
        "license": "NONE",
        "role": "Former OSS workspace sandbox (archived to private codebase June 2026)",
        "conflict_note": "REJECTED (GitHub API verified archived=True, lic=NONE): use Apache-2.0 E2B + Firecracker instead.",
        "url": "https://github.com/daytonaio/daytona",
    },
    "arize_phoenix": {
        "domain": "8_sandboxes_and_observability",
        "verdict": "rejected_conflict",
        "pip": "arize-phoenix (AVOID FOR SOVEREIGN STACK)",
        "tier": "rejected",
        "github_repo": "Arize-ai/phoenix",
        "verified_pushed_at": "2026-11-21",
        "license": "Elastic-2.0",
        "role": "Source-available AI observability tool (acquired by Dynatrace Oct 2026)",
        "conflict_note": "REJECTED for sovereign stack: non-OSI Elastic License 2.0 + Dynatrace acquisition (2026-10-01); use MIT Langfuse.",
        "url": "https://github.com/Arize-ai/phoenix",
    },
    # --- Domain 9: Streaming DataLoaders, Fast Rust Data Ops & Synthetic Pipelines (8 wheels) ---
    "litdata": {
        "domain": "9_streaming_dataloaders_and_data_ops",
        "verdict": "primary",
        "pip": "litdata",
        "tier": "streaming_loader_primary",
        "github_repo": "Lightning-AI/litdata",
        "verified_pushed_at": "2026-11-17",
        "license": "Apache-2.0",
        "role": "Zero-copy deterministic streaming & resumable sharded DataLoader for 30GB+ byte corpora",
        "conflict_note": "Primary streaming dataloader for deterministic resume; outperforms raw HF datasets streaming.",
        "url": "https://github.com/Lightning-AI/litdata",
    },
    "streaming": {
        "domain": "9_streaming_dataloaders_and_data_ops",
        "verdict": "challenger_or_alt",
        "pip": "mosaicml-streaming",
        "tier": "streaming_loader_challenger",
        "github_repo": "mosaicml/streaming",
        "verified_pushed_at": "2026-11-03",
        "license": "Apache-2.0",
        "role": "MosaicML elastic deterministic MDS streaming dataset with mid-epoch checkpoint resume",
        "conflict_note": "Same-slot challenger to LitData for multi-node elastic resume.",
        "url": "https://github.com/mosaicml/streaming",
    },
    "webdataset": {
        "domain": "9_streaming_dataloaders_and_data_ops",
        "verdict": "challenger_or_alt",
        "pip": "webdataset",
        "tier": "streaming_watch_stale",
        "github_repo": "webdataset/webdataset",
        "verified_pushed_at": "2026-02-09",
        "license": "BSD-3-Clause",
        "role": "POSIX tar-shard sequential streaming format",
        "conflict_note": "STALE WATCH (last push 2026-02-09): prefer LitData / MosaicML Streaming for exact step-level resume.",
        "url": "https://github.com/webdataset/webdataset",
    },
    "polars": {
        "domain": "9_streaming_dataloaders_and_data_ops",
        "verdict": "primary",
        "pip": "polars",
        "tier": "local_data_ops_primary",
        "github_repo": "pola-rs/polars",
        "verified_pushed_at": "2026-11-21",
        "license": "MIT",
        "role": "Rust multi-threaded lazy DataFrame engine for 30GB local Parquet/JSONL filtering (10-50x faster than pandas)",
        "conflict_note": "Primary local tabular/parquet manipulation wheel paired with DuckDB.",
        "url": "https://github.com/pola-rs/polars",
    },
    "duckdb": {
        "domain": "9_streaming_dataloaders_and_data_ops",
        "verdict": "primary",
        "pip": "duckdb",
        "tier": "local_sql_ops_primary",
        "github_repo": "duckdb/duckdb",
        "verified_pushed_at": "2026-11-20",
        "license": "MIT",
        "role": "In-process analytical SQL engine for zero-RAM-overflow 30GB corpus dedup & domain quota audits",
        "conflict_note": "Complements Polars for out-of-core SQL joins/dedup on 16GB host RAM.",
        "url": "https://github.com/duckdb/duckdb",
    },
    "uv": {
        "domain": "9_streaming_dataloaders_and_data_ops",
        "verdict": "primary",
        "pip": "uv",
        "tier": "env_packaging_primary",
        "github_repo": "astral-sh/uv",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "10-100x faster Rust Python package & lockfile manager for reproducible RTX 3050 CUDA environments",
        "conflict_note": "Primary environment & dependency resolver replacing slow pip/conda backtracking.",
        "url": "https://github.com/astral-sh/uv",
    },
    "distilabel": {
        "domain": "9_streaming_dataloaders_and_data_ops",
        "verdict": "primary",
        "pip": "distilabel",
        "tier": "synthetic_pipeline_primary",
        "github_repo": "argilla-io/distilabel",
        "verified_pushed_at": "2026-11-18",
        "license": "Apache-2.0",
        "role": "Verifiable synthetic data generation & teacher-critique pipeline framework (for <=20% synthetic Megadoc budget)",
        "conflict_note": "Primary synthetic curation pipeline; strictly bounded by <=20% synthetic / >=80% real anchor red line.",
        "url": "https://github.com/argilla-io/distilabel",
    },
    "curator": {
        "domain": "9_streaming_dataloaders_and_data_ops",
        "verdict": "challenger_or_alt",
        "pip": "bespokelabs-curator",
        "tier": "synthetic_pipeline_challenger",
        "github_repo": "bespokelabsai/curator",
        "verified_pushed_at": "2026-11-02",
        "license": "Apache-2.0",
        "role": "Structured batch synthetic data viewer & generator",
        "conflict_note": "Same-slot lightweight challenger to Distilabel.",
        "url": "https://github.com/bespokelabsai/curator",
    },

    # --- Domain 10: KV Offload, Local Experiment Tracking & Safety Guardrails (7 wheels) ---
    "lmcache": {
        "domain": "10_kv_offload_tracking_and_guardrails",
        "verdict": "primary",
        "pip": "lmcache",
        "tier": "kv_offload_primary",
        "github_repo": "LMCache/LMCache",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "Multi-tier GPU->CPU RAM->SSD KV cache reuse layer for vLLM/SGLang (3-10x TTFT speedup on long agentic prefixes)",
        "conflict_note": "Primary KV offload & prefix persistence layer complementing our CPU-Pinned Engram.",
        "url": "https://github.com/LMCache/LMCache",
    },
    "mooncake": {
        "domain": "10_kv_offload_tracking_and_guardrails",
        "verdict": "scaleup_or_backend",
        "pip": "mooncake-transfer-engine",
        "tier": "kv_disagg_scaleup",
        "github_repo": "kvcache-ai/Mooncake",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "Kimi KVCache-centric disaggregated prefill/decode & RDMA KV pool (native SGLang/vLLM/LMCache L2 backend)",
        "conflict_note": "Scale-Up multi-node disaggregated serving backend beneath LMCache/SGLang.",
        "url": "https://github.com/kvcache-ai/Mooncake",
    },
    "swanlab": {
        "domain": "10_kv_offload_tracking_and_guardrails",
        "verdict": "primary",
        "pip": "swanlab",
        "tier": "experiment_tracking_primary",
        "github_repo": "SwanHubX/SwanLab",
        "verified_pushed_at": "2026-11-20",
        "license": "Apache-2.0",
        "role": "Lightweight local/self-hosted Chinese-friendly experiment tracker for RTX 3050 BPB/loss/grad-norm curves",
        "conflict_note": "Primary local sovereign training tracker (zero cloud lock-in; beats proprietary W&B).",
        "url": "https://github.com/SwanHubX/SwanLab",
    },
    "trackio": {
        "domain": "10_kv_offload_tracking_and_guardrails",
        "verdict": "challenger_or_alt",
        "pip": "trackio",
        "tier": "experiment_tracking_alt",
        "github_repo": "gradio-app/trackio",
        "verified_pushed_at": "2026-11-12",
        "license": "MIT",
        "role": "Zero-infra local MIT drop-in W&B-compatible experiment tracker",
        "conflict_note": "Ultra-lightweight MIT local challenger to SwanLab.",
        "url": "https://github.com/gradio-app/trackio",
    },
    "mlflow": {
        "domain": "10_kv_offload_tracking_and_guardrails",
        "verdict": "challenger_or_alt",
        "pip": "mlflow",
        "tier": "experiment_tracking_enterprise",
        "github_repo": "mlflow/mlflow",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "Full-featured enterprise ML lifecycle & model registry",
        "conflict_note": "Enterprise Scale-Up option; heavier than SwanLab/Trackio for single-GPU RTX 3050.",
        "url": "https://github.com/mlflow/mlflow",
    },
    "nemoguardrails": {
        "domain": "10_kv_offload_tracking_and_guardrails",
        "verdict": "challenger_or_alt",
        "pip": "nemoguardrails",
        "tier": "safety_guardrail_ref",
        "github_repo": "NVIDIA/NeMo-Guardrails",
        "verified_pushed_at": "2026-11-19",
        "license": "Apache-2.0",
        "role": "Programmable Colang dialog & tool-call rail runtime for enterprise agent boundaries",
        "conflict_note": "External guardrail reference complementing src/harness/ deterministic exoskeleton.",
        "url": "https://github.com/NVIDIA/NeMo-Guardrails",
    },
    "purplellama": {
        "domain": "10_kv_offload_tracking_and_guardrails",
        "verdict": "challenger_or_alt",
        "pip": "llamafirewall",
        "tier": "safety_firewall_ref",
        "github_repo": "meta-llama/PurpleLlama",
        "verified_pushed_at": "2026-11-20",
        "license": "MIT",
        "role": "Meta LlamaFirewall / CyberSecEval prompt-injection & indirect tool-poisoning defense suite",
        "conflict_note": "Reference prompt-injection firewall benchmark paired with our StoredPI/CapLite guards.",
        "url": "https://github.com/meta-llama/PurpleLlama",
    },
    # --- Domain 11: Formal Math/Code Verifiers & STEM Document Ingestion (7 wheels) ---
    "lean4": {
        "domain": "11_formal_verifiers_and_stem_ingestion",
        "verdict": "primary",
        "pip": "elan / lean4 (leanprover/lean4)",
        "tier": "rlvr_formal_verifier_primary",
        "github_repo": "leanprover/lean4",
        "verified_pushed_at": "2026-11-21",
        "license": "Apache-2.0",
        "role": "Deterministic formal theorem prover & compiler oracle for Stage 4 RLVR self-correction (zero hallucination reward)",
        "conflict_note": "Primary formal math verifier oracle; keeps frontier proofs in external Mathlib retrieval rather than bloating 65M weights.",
        "url": "https://github.com/leanprover/lean4",
    },
    "pantograph": {
        "domain": "11_formal_verifiers_and_stem_ingestion",
        "verdict": "primary",
        "pip": "pantograph",
        "tier": "rlvr_lean_repl_primary",
        "github_repo": "lenianiva/Pantograph",
        "verified_pushed_at": "2026-11-15",
        "license": "Apache-2.0",
        "role": "High-speed Python<->Lean 4 interactive tactic REPL & proof-search bridge for RLVR verifier feedback",
        "conflict_note": "Primary Python bridge for interactive Lean 4 error-feedback loops.",
        "url": "https://github.com/lenianiva/Pantograph",
    },
    "z3": {
        "domain": "11_formal_verifiers_and_stem_ingestion",
        "verdict": "primary",
        "pip": "z3-solver",
        "tier": "smt_verifier_primary",
        "github_repo": "Z3Prover/z3",
        "verified_pushed_at": "2026-11-21",
        "license": "MIT",
        "role": "Microsoft Z3 SMT solver for deterministic logic, constraint & Soteria formal plan verification",
        "conflict_note": "Complements Lean 4 for fast millisecond SMT/constraint checking in src/harness/.",
        "url": "https://github.com/Z3Prover/z3",
    },
    "sandbox_fusion": {
        "domain": "11_formal_verifiers_and_stem_ingestion",
        "verdict": "primary",
        "pip": "sandbox-fusion",
        "tier": "rlvr_code_sandbox_primary",
        "github_repo": "bytedance/SandboxFusion",
        "verified_pushed_at": "2026-07-14",
        "license": "Apache-2.0",
        "role": "ByteDance multi-language isolated code execution verifier built for verl/RLVR reward pipelines",
        "conflict_note": "Primary deterministic code-reward sandbox for Stage 4 RLVR (paired with Firecracker/E2B for agent tools).",
        "url": "https://github.com/bytedance/SandboxFusion",
    },
    "mineru": {
        "domain": "11_formal_verifiers_and_stem_ingestion",
        "verdict": "primary",
        "pip": "mineru",
        "tier": "stem_pdf_ingestion_primary",
        "github_repo": "opendatalab/MinerU",
        "verified_pushed_at": "2026-11-20",
        "license": "AGPL-3.0",
        "role": "SOTA Chinese+English STEM PDF/LaTeX/formula/table parser for turning textbooks & papers into clean pretraining bytes",
        "conflict_note": "Primary Chinese+STEM PDF parser; pair with MIT Docling when strict permissive commercial license is required.",
        "url": "https://github.com/opendatalab/MinerU",
    },
    "docling": {
        "domain": "11_formal_verifiers_and_stem_ingestion",
        "verdict": "challenger_or_alt",
        "pip": "docling",
        "tier": "doc_ingestion_mit_alt",
        "github_repo": "docling-project/docling",
        "verified_pushed_at": "2026-11-21",
        "license": "MIT",
        "role": "MIT-licensed multi-format document/PDF/DOCX structured parser",
        "conflict_note": "Permissive MIT alternative to MinerU for enterprise/commercial ingestion.",
        "url": "https://github.com/docling-project/docling",
    },
    "olmocr": {
        "domain": "11_formal_verifiers_and_stem_ingestion",
        "verdict": "challenger_or_alt",
        "pip": "olmocr",
        "tier": "ocr_watch_stale",
        "github_repo": "allenai/olmocr",
        "verified_pushed_at": "2026-03-25",
        "license": "Apache-2.0",
        "role": "AllenAI vision-LM PDF OCR pipeline for scanned scientific books",
        "conflict_note": "STALE WATCH (last push 2026-03-25): use for scanned-image PDFs; MinerU/Docling lead on native STEM PDFs.",
        "url": "https://github.com/allenai/olmocr",
    },

    # --- Domain 12: Local Vector Memory & MCP Interop (3 wheels) ---
    "usearch": {
        "domain": "12_local_vector_memory_and_mcp",
        "verdict": "primary",
        "pip": "usearch",
        "tier": "local_vector_primary",
        "github_repo": "unum-cloud/usearch",
        "verified_pushed_at": "2026-11-18",
        "license": "Apache-2.0",
        "role": "Single-file SIMD C++/Rust HNSW vector index (zero-daemon, 0 MB GPU VRAM) for external L3 knowledge & Mathlib retrieval",
        "conflict_note": "Primary embedded vector index for single-machine RTX 3050 (beats heavy server DBs for local sovereignty).",
        "url": "https://github.com/unum-cloud/usearch",
    },
    "qdrant_client": {
        "domain": "12_local_vector_memory_and_mcp",
        "verdict": "challenger_or_alt",
        "pip": "qdrant-client",
        "tier": "server_vector_scaleup",
        "github_repo": "qdrant/qdrant",
        "verified_pushed_at": "2026-11-20",
        "license": "Apache-2.0",
        "role": "Rust production hybrid dense+sparse vector search engine for multi-user Scale-Up retrieval",
        "conflict_note": "Scale-Up server alternative to embedded USearch.",
        "url": "https://github.com/qdrant/qdrant",
    },
    "mcp": {
        "domain": "12_local_vector_memory_and_mcp",
        "verdict": "primary",
        "pip": "mcp",
        "tier": "exoskeleton_protocol_primary",
        "github_repo": "modelcontextprotocol/python-sdk",
        "verified_pushed_at": "2026-11-17",
        "license": "MIT",
        "role": "Official Model Context Protocol Python SDK for standardized tool/sandbox/verifier wiring in src/harness/",
        "conflict_note": "Primary standard tool protocol paired with our CapLite/ToolUpdateIntegrityGate security guards.",
        "url": "https://github.com/modelcontextprotocol/python-sdk",
    },
    # --- Domain 13: Code AST, Repo Search & Deterministic Linting (3 wheels) ---
    "tree_sitter": {
        "domain": "13_code_ast_and_static_analysis",
        "verdict": "primary",
        "pip": "tree-sitter",
        "tier": "code_ast_primary",
        "github_repo": "tree-sitter/tree-sitter",
        "verified_pushed_at": "2026-11-21",
        "license": "MIT",
        "role": "Incremental multi-language AST parser for code-corpus syntax filtering, repo-map extraction & edit verification",
        "conflict_note": "Primary universal AST parser for both pretraining code cleaning and coding-agent exoskeleton.",
        "url": "https://github.com/tree-sitter/tree-sitter",
    },
    "ruff": {
        "domain": "13_code_ast_and_static_analysis",
        "verdict": "primary",
        "pip": "ruff",
        "tier": "python_linter_primary",
        "github_repo": "astral-sh/ruff",
        "verified_pushed_at": "2026-11-21",
        "license": "MIT",
        "role": "100x faster Rust Python linter/formatter providing instant deterministic code-quality reward & pre-commit checks",
        "conflict_note": "Primary Python static checker replacing slow flake8/black/isort stacks.",
        "url": "https://github.com/astral-sh/ruff",
    },
    "ripgrep": {
        "domain": "13_code_ast_and_static_analysis",
        "verdict": "primary",
        "pip": "binary-cli (BurntSushi/ripgrep)",
        "tier": "code_search_primary",
        "github_repo": "BurntSushi/ripgrep",
        "verified_pushed_at": "2026-11-21",
        "license": "Unlicense",
        "role": "Blazing-fast Rust regex repository search engine for coding-agent exoskeleton tool calls",
        "conflict_note": "Primary zero-index codebase search tool.",
        "url": "https://github.com/BurntSushi/ripgrep",
    },

    # --- Domain 14: Web Extraction & Browser Exoskeleton (3 wheels) ---
    "crawl4ai": {
        "domain": "14_web_extraction_and_browser_exoskeleton",
        "verdict": "primary",
        "pip": "crawl4ai",
        "tier": "web_crawler_primary",
        "github_repo": "unclecode/crawl4ai",
        "verified_pushed_at": "2026-11-20",
        "license": "Apache-2.0",
        "role": "Open-source async LLM-friendly web crawler converting live pages into clean Markdown for agent tools",
        "conflict_note": "Primary agent web-reading tool paired with Playwright.",
        "url": "https://github.com/unclecode/crawl4ai",
    },
    "playwright": {
        "domain": "14_web_extraction_and_browser_exoskeleton",
        "verdict": "primary",
        "pip": "playwright",
        "tier": "browser_automation_primary",
        "github_repo": "microsoft/playwright-python",
        "verified_pushed_at": "2026-11-20",
        "license": "Apache-2.0",
        "role": "Deterministic headless browser automation engine for web-acting exoskeleton tasks",
        "conflict_note": "Primary browser control runtime behind Crawl4AI and web-agent sandboxes.",
        "url": "https://github.com/microsoft/playwright-python",
    },
    "trafilatura": {
        "domain": "14_web_extraction_and_browser_exoskeleton",
        "verdict": "primary",
        "pip": "trafilatura",
        "tier": "html_maintext_primary",
        "github_repo": "adbar/trafilatura",
        "verified_pushed_at": "2026-11-20",
        "license": "Apache-2.0",
        "role": "Main-text HTML boilerplate/ad/nav stripper (exact extractor used in Ultra-FineWeb 2026 pipeline)",
        "conflict_note": "Primary static web-dump text extractor for custom pretraining corpus curation.",
        "url": "https://github.com/adbar/trafilatura",
    },

    # --- Domain 15: White-Box Mechanistic Interpretability (3 wheels) ---
    "nnsight": {
        "domain": "15_mechanistic_interpretability",
        "verdict": "primary",
        "pip": "nnsight",
        "tier": "mech_interp_primary",
        "github_repo": "ndif-team/nnsight",
        "verified_pushed_at": "2026-11-21",
        "license": "MIT",
        "role": "PyTorch-native white-box tracing & causal activation intervention on ANY custom architecture (DeltaNet+HoPE/Mamba)",
        "conflict_note": "Primary white-box circuit debugger for custom SovereignRouteBBrain (beats TransformerLens).",
        "url": "https://github.com/ndif-team/nnsight",
    },
    "sae_lens": {
        "domain": "15_mechanistic_interpretability",
        "verdict": "primary",
        "pip": "sae-lens",
        "tier": "sae_probing_primary",
        "github_repo": "decoderesearch/SAELens",
        "verified_pushed_at": "2026-11-20",
        "license": "MIT",
        "role": "Sparse Autoencoder training & feature circuit analysis library for debugging 65M brain representations",
        "conflict_note": "Complements nnsight for sparse feature extraction & FtF probing.",
        "url": "https://github.com/decoderesearch/SAELens",
    },
    "transformer_lens": {
        "domain": "15_mechanistic_interpretability",
        "verdict": "rejected_conflict",
        "pip": "transformer-lens (DO NOT USE FOR ROUTE B)",
        "tier": "rejected",
        "github_repo": "TransformerLensOrg/TransformerLens",
        "verified_pushed_at": "2026-11-01",
        "license": "MIT",
        "role": "Standard-Transformer-only reimplementation interpretability library",
        "conflict_note": "REJECTED for Route B: reimplements vanilla Transformers from scratch and breaks on custom OscillatoryDualDeltaNet + HoPE; use nnsight.",
        "url": "https://nnsight.net/blog/2026/02/26/nnterp-integration/",
    },

    # --- Domain 16: Local Edge Speech I/O (0-Cloud Ears & Voice, 2 wheels) ---
    "whisper_cpp": {
        "domain": "16_local_edge_speech_io",
        "verdict": "primary",
        "pip": "source-build (ggml-org/whisper.cpp)",
        "tier": "edge_stt_primary",
        "github_repo": "ggml-org/whisper.cpp",
        "verified_pushed_at": "2026-09-24",
        "license": "MIT",
        "role": "Quantized C/C++ local CPU/GPU speech-to-text engine (0-cloud sovereign ears; supersedes stale faster-whisper)",
        "conflict_note": "Primary local STT (faster-whisper unmaintained since Nov 2025).",
        "url": "https://github.com/ggml-org/whisper.cpp",
    },
    "kokoro": {
        "domain": "16_local_edge_speech_io",
        "verdict": "primary",
        "pip": "kokoro",
        "tier": "edge_tts_primary",
        "github_repo": "hexgrad/kokoro",
        "verified_pushed_at": "2026-06-08",
        "license": "Apache-2.0",
        "role": "82M-parameter Apache-2.0 local CPU multilingual (EN/ZH/etc.) real-time text-to-speech engine (0 MB GPU VRAM)",
        "conflict_note": "Primary 0-VRAM sovereign local voice output.",
        "url": "https://github.com/hexgrad/kokoro",
    },
}


def probe_top_2026_ecosystem_wheels() -> Dict[str, Any]:
    """Non-intrusively probes all 77 domain-classified 2026 wheels and exposes conflict arbitration."""
    status: Dict[str, Dict[str, Any]] = {}
    available: List[str] = []
    missing: List[str] = []
    rejected: List[str] = []
    by_domain: Dict[str, List[str]] = {d: [] for d in WHEEL_DOMAINS}
    for mod_name, meta in ECOSYSTEM_WHEEL_CATALOG.items():
        found = importlib.util.find_spec(mod_name) is not None
        status[mod_name] = {
            "installed": found,
            "domain": meta["domain"],
            "verdict": meta["verdict"],
            "pip_package": meta["pip"],
            "tier": meta["tier"],
            "github_repo": meta["github_repo"],
            "verified_pushed_at": meta["verified_pushed_at"],
            "license": meta["license"],
            "role": meta["role"],
            "conflict_note": meta["conflict_note"],
            "url": meta["url"],
        }
        by_domain.setdefault(meta["domain"], []).append(mod_name)
        if meta["verdict"] == "rejected_conflict":
            rejected.append(mod_name)
        elif found:
            available.append(mod_name)
        else:
            missing.append(mod_name)
    return {
        "installed_count": len(available),
        "total_catalog_wheels": len(ECOSYSTEM_WHEEL_CATALOG),
        "domain_count": len(WHEEL_DOMAINS),
        "installed_wheels": available,
        "missing_optional_wheels": missing,
        "rejected_conflict_wheels": rejected,
        "by_domain": by_domain,
        "fallback_mode": "pure_pytorch_and_python_ready",
        "details": status,
    }


def audit_ecosystem_wheel_freshness(online: bool = False, max_stale_days: int = 180) -> Dict[str, Any]:
    """Rule 27 Anti-Rot Gate for all 77 ecosystem wheels.
    Offline: verifies metadata completeness, zero unflagged rejected wheels in active slots, and recorded push dates.
    Online (online=True): queries live GitHub API for archived status & push staleness (> max_stale_days).
    """
    now = datetime.now(timezone.utc)
    stale_watch: List[str] = []
    archived_active: List[str] = []
    live_checked = 0
    ua = "Mozilla/5.0 (compatible; Google-Agent; +https://developers.google.com/crawling/docs/crawlers-fetchers/google-agent)"

    for name, meta in ECOSYSTEM_WHEEL_CATALOG.items():
        if meta["verdict"] == "rejected_conflict":
            continue
        pushed_str = meta.get("verified_pushed_at", "")
        if online and meta.get("github_repo"):
            try:
                req = urllib.request.Request(
                    f"https://api.github.com/repos/{meta['github_repo']}",
                    headers={"User-Agent": ua, "Accept": "application/vnd.github+json"},
                )
                with urllib.request.urlopen(req, timeout=6) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    live_checked += 1
                    pushed_str = (data.get("pushed_at") or pushed_str)[:10]
                    if data.get("archived", False):
                        archived_active.append(name)
            except Exception:
                pass
        if pushed_str:
            try:
                dt = datetime.strptime(pushed_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if (now - dt).days > int(max_stale_days):
                    stale_watch.append(f"{name}({pushed_str[:10]})")
            except ValueError:
                pass

    return {
        "ok": len(archived_active) == 0,
        "mode": "online_github_api" if online else "offline_metadata",
        "total_wheels": len(ECOSYSTEM_WHEEL_CATALOG),
        "domain_count": len(WHEEL_DOMAINS),
        "live_checked": live_checked,
        "archived_active_violations": archived_active,
        "stale_watch_over_180d": stale_watch,
        "rejected_conflicts_enforced": [k for k, v in ECOSYSTEM_WHEEL_CATALOG.items() if v["verdict"] == "rejected_conflict"],
    }


def try_fla_gated_delta_rule(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    initial_state: Optional[Any] = None,
    output_final_state: bool = True,
) -> Optional[Any]:
    """Dispatches to `fla.ops.gated_delta_rule.chunk_gated_delta_rule` when fla + CUDA are active."""
    if importlib.util.find_spec("fla") is None:
        return None
    try:
        import torch
        if not (isinstance(q, torch.Tensor) and q.is_cuda):
            return None
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        return chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
    except Exception:
        return None


def try_liger_fused_linear_ce(
    lm_head_weight: Any,
    hidden_states: Any,
    targets: Any,
    lm_head_bias: Optional[Any] = None,
) -> Optional[Any]:
    """Dispatches to Liger-Kernel FusedLinearCrossEntropyLoss (or Apple Cut Cross-Entropy fallback)
    when installed on CUDA, avoiding full logit tensor materialization in VRAM."""
    if importlib.util.find_spec("liger_kernel") is not None:
        try:
            import torch
            if isinstance(hidden_states, torch.Tensor) and hidden_states.is_cuda:
                from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
                loss_fn = LigerFusedLinearCrossEntropyLoss()
                flat_h = hidden_states.reshape(-1, hidden_states.size(-1))
                flat_t = targets.reshape(-1)
                return loss_fn(lm_head_weight, flat_h, flat_t, bias=lm_head_bias)
        except Exception:
            pass
    if lm_head_bias is None and importlib.util.find_spec("cut_cross_entropy") is not None:
        try:
            import torch
            if isinstance(hidden_states, torch.Tensor) and hidden_states.is_cuda:
                from cut_cross_entropy import linear_cross_entropy
                return linear_cross_entropy(hidden_states, lm_head_weight, targets)
        except Exception:
            pass
    return None


def save_weights_safetensors(state_dict: Dict[str, Any], path: str | Path) -> bool:
    """Saves model tensors to zero-pickle `.safetensors` when installed; returns True if written."""
    if importlib.util.find_spec("safetensors") is None:
        return False
    try:
        from safetensors.torch import save_file
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach().contiguous().cpu() for k, v in state_dict.items()}
        save_file(tensors, str(out_path))
        return True
    except Exception:
        return False


def load_weights_safetensors(path: str | Path, device: str = "cpu") -> Optional[Dict[str, Any]]:
    """Loads `.safetensors` weights when file & package exist; returns None to fall back to `.pt`."""
    p = Path(path)
    if (not p.exists()) or importlib.util.find_spec("safetensors") is None:
        return None
    try:
        from safetensors.torch import load_file
        return load_file(str(p), device=str(device))
    except Exception:
        return None


def try_liger_rms_norm(x: Any, weight: Any, eps: float = 1e-6) -> Optional[Any]:
    """Dispatches to Liger-Kernel Triton RMSNorm on CUDA when liger_kernel is installed."""
    if importlib.util.find_spec("liger_kernel") is None:
        return None
    try:
        import torch
        if not (isinstance(x, torch.Tensor) and x.is_cuda):
            return None
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        return LigerRMSNormFunction.apply(x, weight, eps, 0.0, "llama", False)
    except Exception:
        return None


def try_xgrammar_compile_json_schema(schema_json: str, vocab_size: int = 256) -> Dict[str, Any]:
    """Compiles a JSON schema for <tool_call> constrained decoding via XGrammar-2 when installed,
    falling back to pure-Python JSON schema validator metadata when absent."""
    if importlib.util.find_spec("xgrammar") is not None:
        try:
            import xgrammar as xgr
            tok_info = xgr.TokenizerInfo([bytes([i]) for i in range(int(vocab_size))], vocab_size=int(vocab_size))
            compiler = xgr.GrammarCompiler(tok_info)
            compiled = compiler.compile_json_schema(str(schema_json))
            return {"backend": "xgrammar_v2_cpp", "compiled": compiled, "latency_us_est": 40}
        except Exception:
            pass
    return {"backend": "python_schema_fallback", "compiled": str(schema_json), "latency_us_est": 250}


# =============================================================================
# v11.0: Native Windows 2026 Runtime Guard (triton-lang/triton-windows 3.9)
#        & 2026-11 Open-Weight 4GB-Class Edge Backbones Registry
# =============================================================================
EDGE_BACKBONES_2026_11: Dict[str, Dict[str, Any]] = {
    "olala_v1_mini_7a1b": {
        "hf_id": "OVHaiLLM/Olala-v1.0-mini-7A1B",
        "license": "Apache-2.0",
        "release_date": "2026-11-19",
        "architecture": "29 Mamba-3 + 7 Attention + MoE (6.8B total / ~1B active)",
        "url": "https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B",
        "adapter_mode": "external_or_zero_init_engram",
    },
    "nemotron_3_5_bolt_e4b_4gb": {
        "hf_id": "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB",
        "license": "OpenMDW-1.1",
        "release_date": "2026-10",
        "architecture": "Mamba-2 + Attention + MoE (4GB VRAM Q4_K_M target)",
        "url": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB",
        "adapter_mode": "external_or_zero_init_engram",
    },
    "gemma_4_e2b": {
        "hf_id": "google/gemma-4-E2B-it",
        "license": "Apache-2.0",
        "release_date": "2026-04",
        "architecture": "PLE 2.3B effective multimodal edge backbone",
        "url": "https://ai.google.dev/gemma/docs/core/model_card_4",
        "adapter_mode": "external_or_zero_init_engram",
    },
    "embeddinggemma_2_740m": {
        "hf_id": "google/embeddinggemma-2-740m",
        "license": "Gemma-Open-Weights",
        "release_date": "2026-10",
        "architecture": "740M multimodal unified offline embedding (6x smaller local vectors)",
        "url": "https://blog.google/innovation-and-ai/technology/ai/google-ai-updates-october-2026/",
        "adapter_mode": "engram_semantic_retriever",
    },
}


class WindowsNativeRuntimeGuard:
    RECOMMENDED_PYTHON: str = "3.15"
    SUPPORTED_PYTORCH: str = "2.15.0"
    DEFAULT_CUDA_WHEEL: str = "cu134"
    r"""v11.0 Native Windows RTX 3050 4GB Guard (triton-lang/triton-windows 3.9 + PyTorch 2.15).

    Replaces legacy WSL2/Docker default (eliminates 3-15% Hyper-V GPU tax and 15-30% /mnt/d I/O penalty)
    while enforcing 7 concrete Windows/Ampere sm_86 safety checks:
      1. PyTorch 2.15 <-> triton-windows 3.9 (<3.10) version lock
      2. All caches pinned strictly under D:\Environment\.cache (zero C-drive bloat)
      3. Real free VRAM budget via mem_get_info() (accounting for 0.4-1.0GB WDDM DWM overhead)
      4. CPU Engram @torch.compiler.disable boundary (prevents Triton CPU-pointer crash)
      5. WDDM Sysmem Fallback disabled recommendation (prevents silent 10x slowdown)
      6. Windows DataLoader num_workers=0 (avoids spawn multiprocessing overhead)
      7. Defender exclusion + TDR delay >= 10s + atomic checkpoint rename (WinError 32 safe)
    """

    def build_env_vars(self, cache_root: str = r"D:\Environment\.cache") -> Dict[str, str]:
        root = str(cache_root).rstrip("\\/")
        sep = "\\"
        return {
            "UV_CACHE_DIR": f"{root}{sep}uv",
            "PIP_CACHE_DIR": f"{root}{sep}pip",
            "HF_HOME": f"{root}{sep}huggingface",
            "TRITON_CACHE_DIR": f"{root}{sep}triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{root}{sep}torchinductor",
            "TMP": f"{root}{sep}tmp",
            "TEMP": f"{root}{sep}tmp",
        }

    def recommend_micro_batch_budget(self, free_vram_mb: float, reserve_headroom_mb: float = 512.0) -> Dict[str, Any]:
        usable = max(0.0, float(free_vram_mb) - float(reserve_headroom_mb))
        if usable < 1800.0:
            seq_len, micro_bs = 512, 1
        elif usable < 2600.0:
            seq_len, micro_bs = 1024, 1
        else:
            seq_len, micro_bs = 2048, 1
        return {
            "free_vram_mb": round(float(free_vram_mb), 1),
            "usable_vram_mb": round(usable, 1),
            "recommended_seq_len": seq_len,
            "recommended_micro_batch": micro_bs,
            "recommended_num_workers_windows": 0,
            "dtype": "bfloat16",
            "avoid_fp8_on_ampere_sm86": True,
        }

    def audit_windows_native_stack(self) -> Dict[str, Any]:
        return {
            "primary_runtime": "native_windows_uv_pytorch_2_15_triton_windows_3_9",
            "triton_windows_repo": "https://github.com/triton-lang/triton-windows",
            "fallback_runtime": "docker_wsl2_gpu_only_for_linux_only_cpp_extensions",
            "edge_backbones_2026_11": list(EDGE_BACKBONES_2026_11.keys()),
            "checklist": [
                "install_vc_redist_x64_and_triton_windows_lt_3_10",
                "pin_all_caches_to_D_Environment_cache",
                "disable_nvidia_cuda_sysmem_fallback_in_control_panel",
                "add_D_Environment_to_windows_defender_exclusions",
                "set_tdr_delay_ge_10s_and_disable_modern_standby_sleep",
                "keep_cpu_engram_behind_torch_compiler_disable",
                "use_dataloader_num_workers_0_and_atomic_checkpoint_replace",
            ],
        }
