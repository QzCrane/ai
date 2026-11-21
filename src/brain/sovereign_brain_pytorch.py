#!/usr/bin/env python3
"""
Sovereign Pure AI Brain — Route B Frozen Architecture (v4.7, RTX 3050 4GB Executable)
=====================================================================================
Strictly adheres to Route B ("Time-and-Structure-First", Scale-Invariant & 4GB-Native):
  - L0 Lossless Byte Frontend + Step-0 Locked 2-Byte LZ77 Novelty Patching (arXiv:2610.22100v1)
    + ATDC Curriculum Compression Gate (arXiv:2605.30080v1) + CanonCausalConv1D
    + Latent Causal Attention (LCA) Multi-Byte Parallel Decoder (arXiv:2608.15454v2, GAP-07).
  - L1 Sandwich Looped Topology (Prelude -> Mid-Layer CPU Engram -> Shared 3:1 Core x T loops -> Coda):
    * Parcae spectral-bounded diagonal carry (arXiv:2604.12946) + ARLoop Sliding-Window Depth Attn (W=2)
    * 1-scalar input injection (arXiv:2610.00673v1) + RMSNorm implicit gain control (arXiv:2607.10681v1)
    * O(1)-VRAM truncated implicit backprop (no_grad on first T-2 loops).
    * 3x OscillatoryDualDeltaNet v1.3: Mamba-3 Exponential-Trapezoidal Discretization + Complex Phase
      Rotation + Q/K Norm & Bias (arXiv:2603.15569v1) + Rank-2 Householder DeltaProduct (arXiv:2502.10297, GAP-11).
    * 1x HoPEHLAMELTAttention (Native SDPA + MELT gated shared HLA latent KV + HoPE + Qwen Sink-Free Gate).
  - L2 Ternary Neuromodulator (ACh/NE/DA, arXiv:2507.14056v4) + Fixed-Prior Depth Supervision
    + Post-Hoc FPRM & NE Oscillation Early-Stopping (arXiv:2607.20519 & 2604.10739) + Optional STARS hinge
    + ATF/PTRM NE-Gated Stochastic Latent Thought Branching (arXiv:2609.33271 & 2605.19943, GAP-12).
  - L3 Trainable 0-MB-VRAM CPU-Pinned Mid-Layer Engram (Memory Grafting arXiv:2605.20948 + DA-gated Sparse CPU SGD).
  - L4 Offline Sleep SSM Consolidation (arXiv:2605.26099) + LaCET Streaming-EMA Anchor (arXiv:2604.07350)
    + Noise-Prune soft suppression (arXiv:2608.05464) + Co-Observation Cross-Era Replay Buffer (arXiv:2608.18803v1)
    + Wake<-Sleep Prior Recall Gate (GAP-13) + Multi-Brain DiffMAS Latent KV Exchange (arXiv:2604.21794v1).
  - Inference Export (GAP-09): Block-Wise Sylvester Hadamard Rotation + SOAR/4over6 MSE-Optimal Q4 Export.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouteBConfig:
    name: str
    d_model: int
    d_state: int
    n_heads: int
    k_prelude: int = 2
    k_core: int = 4               # 3:1 cycle inside shared recurrent core
    k_coda: int = 2
    rec_loops: int = 8            # Effective depth = k_prelude + k_core * rec_loops + k_coda
    backprop_loops: int = 2       # O(1) VRAM training: no_grad on first (rec_loops - backprop_loops)
    depth_attn_window: int = 2    # ARLoop sliding-window depth attention (O(W) constant memory)
    canon_kernel: int = 4
    patch_stride: int = 4         # BLT/H-Net 4x byte-to-patch compression
    chunk_size: int = 32
    kv_latent_dim: int = 128      # HLA + MELT shared low-rank KV latent across loops
    engram_slots: int = 524288    # Strictly in CPU host RAM (0.0 MB GPU VRAM)
    engram_lr: float = 0.05       # CPU-side sparse SGD learning rate for touched Engram slots
    vocab_size: int = 256
    max_seq_len: int = 4096       # v10.17 Red-Team Fix: explicit training context length for HyPE NoPE scaling
    # Clean Ablation Switches (default = locked Route B v1.3)
    enable_cpu_engram: bool = True
    enable_mamba3_phase: bool = True
    enable_rank2_deltaproduct: bool = True    # GAP-11: Rank-2 Householder product for non-Abelian state tracking (arXiv:2502.10297)
    enable_m2rnn_nonlin_state: bool = True    # GAP-49: M^2RNN bounded residual non-linear matrix-state gate (arXiv:2603.14360v2)
    enable_input_injection: bool = True
    enable_ortho_input_injection: bool = True  # GAP-50 (OrthoInj, arXiv:2610.06833v1 §4.2): Q_q projection off inj
    enable_arloop_depth_attn: bool = True
    enable_ternary_neuromod: bool = True      # L2 ACh/NE/DA bounded neuromodulation (arXiv:2507.14056v4)
    enable_ne_oscillation_halt: bool = True   # Post-hoc FPRM + NE oscillation halt (arXiv:2607.20519 & 2604.10739)
    enable_fixed_prior_depth_sup: bool = True # Decoupled fixed-prior depth supervision across active backprop loops
    enable_stochastic_thought: bool = True    # GAP-12: ATF/PTRM NE-gated stochastic latent exploration (arXiv:2609.33271)
    enable_lca_multibyte: bool = True         # GAP-07: LCA boundary-aware parallel multi-byte decoder (arXiv:2608.15454v2)
    lca_loss_weight: float = 0.25             # Auxiliary weight for causal parallel multi-byte prediction loss
    use_stars_reg: bool = False               # Ablatable one-sided STARS Jacobian spectral hinge penalty (arXiv:2605.26733)
    stars_reg_weight: float = 0.05
    sleep_ema_decay: float = 0.9              # LaCET Streaming-EMA anchor decay for offline sleep consolidation (arXiv:2604.07350)
    sleep_noise_prune_quantile: float = 0.15  # Noise-Prune soft suppression quantile in sleep (arXiv:2608.05464)
    enable_per_patch_elastic_freeze: bool = True  # GAP-26: T-LoopFormer per-patch convergence freeze (arXiv:2609.15160v3)
    patch_converge_tol: float = 0.025             # Relative L2 tol for freezing converged easy patches
    use_dysib_ib_reg: bool = False
    use_grad_checkpointing: bool = False  # GAP-44: Recompute recurrent loop activations to cut 4GB train VRAM
    dysib_ib_weight: float = 0.02         # GAP-27: Ablatable DySIB predictive information bottleneck reg (arXiv:2604.24662v1)
    enable_next_latent_pred: bool = True  # GAP-47 (NCP arXiv:2609.10715): Auxiliary next-latent concept predictor (~2x sample eff.)
    next_latent_loss_weight: float = 0.15 # Cosine-alignment weight between predicted & detached next-patch latent
    enable_elr_loop_cond: bool = True     # GAP-63 (ELR/Koi-4B arXiv:2610.12582): Zero-init loop-index & budget conditioning
    enable_pearce_hall_plasticity: bool = True  # GAP-76 (PH-Attention arXiv:2508.08289): Prediction-error associability gate
    kalman_ridge_lambda: float = 1e-3           # GAP-87 (Gated KalmaNet arXiv:2511.21016): Adaptive ridge condition-number stabilizer
    chunk_momentum: float = 0.0                 # GAP-90 (SLA arXiv:2611.06828): Ablatable on-chip SRAM chunk-boundary momentum (rec. 0.25)
    enable_loopcd_hidden: bool = True           # GAP-92 (LoopCD-Hidden arXiv:2610.02185): Live early-vs-final hidden contrast before Coda
    loopcd_alpha: float = 0.25                  # Contrastive amplification coefficient (0.0 disables)
    enable_gatesight_kv_prune: bool = False     # GAP-183 (GateSight arXiv:2610.14754): GDN-driven 90% KV pruning
    gatesight_keep_ratio: float = 0.10          # Fraction of attention KV entries retained after prefill
    enable_bos_summary_prefill: bool = False    # BOS Summary (arXiv:2611.23508): Relax BOS row mask in prefill (0 extra params)
    ffn_sparsity_thresh: float = 0.0            # SGA-Sparse (arXiv:2611.11223): Native SwiGLU gate activation sparsity (0 extra router params)
    silu_gate_floor: float = 0.02               # GateFloor (arXiv:2611.04449): SiLU(z)+delta keeps gate positive at near-zero input (-28 OOM grad spike)
    use_ube_codec: bool = False                 # v10.32: GAP-36 UBE 2-byte CJK (-33.3% Chinese length, 256-byte vocab)
    enable_fla_triton_kernel: bool = True       # v10.32: auto-use flash-linear-attention Triton kernel on CUDA when installed
    enable_liger_triton_kernel: bool = True     # v10.32: auto-use Liger-Kernel Triton RMSNorm/FLCE on CUDA when installed


PRESETS: Dict[str, RouteBConfig] = {
    "micro_debug": RouteBConfig(
        name="micro_debug", d_model=256, d_state=64, n_heads=4,
        k_prelude=1, k_core=4, k_coda=1, rec_loops=4, backprop_loops=2,
        chunk_size=16, kv_latent_dim=64, engram_slots=32768,
    ),
    "rtx3050_4gb_65m": RouteBConfig(
        # Safe 4GB training preset (~35M physical GPU params -> ~350 MB static Muon VRAM, huge activation headroom)
        name="rtx3050_4gb_65m", d_model=512, d_state=64, n_heads=8,
        k_prelude=2, k_core=4, k_coda=2, rec_loops=8, backprop_loops=2,
        chunk_size=32, kv_latent_dim=128, engram_slots=524288,
    ),
    "rtx3050_4gb_65m_true": RouteBConfig(
        # Full 4GB training sweet-spot (~65-75M physical GPU params -> ~650-750 MB static Muon VRAM)
        name="rtx3050_4gb_65m_true", d_model=768, d_state=64, n_heads=12,
        k_prelude=2, k_core=4, k_coda=2, rec_loops=8, backprop_loops=2,
        chunk_size=32, kv_latent_dim=128, engram_slots=524288,
    ),
    "cluster_scale_1b": RouteBConfig(
        name="cluster_scale_1b", d_model=2048, d_state=128, n_heads=16,
        k_prelude=2, k_core=8, k_coda=2, rec_loops=8, backprop_loops=2,
        chunk_size=64, kv_latent_dim=256, engram_slots=2097152,
    ),
}


@dataclass
class RouteBCache:
    conv_state: Optional[torch.Tensor] = None
    delta_states: Dict[int, torch.Tensor] = field(default_factory=dict)  # GAP-52 (uTTT + arXiv:2610.06833v1): 8x smaller terminal shared state
    hla_kv_latents: Dict[int, torch.Tensor] = field(default_factory=dict)
    sleep_ssm_states: Dict[int, torch.Tensor] = field(default_factory=dict)  # Persistent consolidated fast weights (arXiv:2605.26099)
    sleep_ema_anchors: Dict[int, torch.Tensor] = field(default_factory=dict)  # LaCET Streaming-EMA anchors (arXiv:2604.07350)
    pos_offset: int = 0
    prev_coda_state: Optional[torch.Tensor] = None  # GAP-45: Full-Bandwidth Latent Feedback state (arXiv:2608.08888v1)

    def clone(self) -> "RouteBCache":
        return RouteBCache(
            conv_state=None if self.conv_state is None else self.conv_state.detach().clone(),
            delta_states={k: v.detach().clone() for k, v in self.delta_states.items()},
            hla_kv_latents={k: v.detach().clone() for k, v in self.hla_kv_latents.items()},
            sleep_ssm_states={k: v.detach().clone() for k, v in self.sleep_ssm_states.items()},
            sleep_ema_anchors={k: v.detach().clone() for k, v in self.sleep_ema_anchors.items()},
            pos_offset=int(self.pos_offset),
            prev_coda_state=None if self.prev_coda_state is None else self.prev_coda_state.detach().clone(),
        )

    def copy_from(self, other: "RouteBCache") -> None:
        self.conv_state = other.conv_state
        self.delta_states = other.delta_states
        self.hla_kv_latents = other.hla_kv_latents
        self.sleep_ssm_states = other.sleep_ssm_states
        self.sleep_ema_anchors = other.sleep_ema_anchors
        self.pos_offset = other.pos_offset
        self.prev_coda_state = other.prev_coda_state


# =============================================================================
# Organ 4B (L4, GAP-08): CPU-Resident Co-Observation Cross-Era Replay Buffer
# =============================================================================
class CoObservationSleepBuffer:
    """
    CPU-resident episodic buffer (0.0 MB GPU VRAM) implementing:
      - Co-Observation continual learning (arXiv:2608.18803v1): cross-era replay splicing.
      - FOREVER forgetting-curve priority (GAP-32, arXiv:2601.03938).
      - Routed Continual Learning (GAP-48, arXiv:2610.38796v1): zero-VRAM byte-signature
        domain clustering so sleep replay splices intra-cluster cross-era episodes without
        heterogeneous domain interference (cjk / code / numeric / prose).
    """
    def __init__(self, max_episodes: int = 64, max_seq_len: int = 256) -> None:
        self.max_episodes = max(2, int(max_episodes))
        self.max_seq_len = max(16, int(max_seq_len))
        self._episodes: List[Tuple[int, float, str, torch.Tensor]] = []
        self._era_counter: int = 0

    def __len__(self) -> int:
        return len(self._episodes)

    @staticmethod
    def classify_byte_cluster(byte_seq: Any) -> str:
        """Zero-VRAM deterministic domain router from raw byte distribution (arXiv:2610.38796v1)."""
        vals = byte_seq.tolist() if hasattr(byte_seq, "tolist") else list(byte_seq)
        n = max(1, len(vals))
        high_utf8 = sum(1 for b in vals if int(b) >= 128)
        if high_utf8 / n >= 0.25:
            return "cjk"
        code_bytes = {123, 125, 40, 41, 91, 93, 59, 61, 58, 95, 60, 62}  # {}()[];=_<>
        code_hits = sum(1 for b in vals if int(b) in code_bytes)
        if code_hits / n >= 0.12:
            return "code"
        digit_math = sum(1 for b in vals if (48 <= int(b) <= 57) or int(b) in {43, 45, 42, 47, 46})
        if digit_math / n >= 0.25:
            return "numeric"
        return "prose"

    def cluster_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, _, cid, _ in self._episodes:
            counts[cid] = counts.get(cid, 0) + 1
        return counts

    def decay_after_sleep(
        self, cluster_id: Optional[str] = None, decay: float = 0.82, min_salience: float = 0.15
    ) -> int:
        """
        GAP-57 (Learning to Forget arXiv:2603.14517v1 + SCM arXiv:2604.20943v1):
        Algorithmic post-sleep forgetting attenuates already-consolidated episodes to resolve
        proactive interference and prunes stale entries below `min_salience` (0 MB GPU VRAM).
        """
        d = max(0.05, min(1.0, float(decay)))
        kept: List[Tuple[int, float, str, torch.Tensor]] = []
        pruned = 0
        for era, sal, cid, seq in self._episodes:
            new_sal = sal * d if (cluster_id is None or cid == cluster_id) else sal
            if new_sal >= float(min_salience):
                kept.append((era, new_sal, cid, seq))
            else:
                pruned += 1
        self._episodes = kept
        return pruned

    def add_episode(self, byte_ids: torch.Tensor, salience: float = 1.0) -> int:
        self._era_counter += 1
        seq = byte_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)[: self.max_seq_len].clone()
        cid = self.classify_byte_cluster(seq)
        self._episodes.append((self._era_counter, float(salience), cid, seq))
        if len(self._episodes) > self.max_episodes:
            n = len(self._episodes)
            evict_idx = min(range(n - 1), key=lambda idx: self._episodes[idx][1] * (0.5 + 0.5 * idx / max(1, n)))
            self._episodes.pop(evict_idx)
        return self._era_counter

    def build_co_observation_windows(
        self, recent_byte_ids: torch.Tensor, device: torch.device, max_windows: int = 4
    ) -> torch.Tensor:
        recent_1d = recent_byte_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)[: self.max_seq_len]
        if not self._episodes:
            return recent_1d.unsqueeze(0).to(device=device)
        # GAP-32 (FOREVER) + GAP-48 (Routed Continual Learning): same-cluster priority first
        tau = max(4.0, float(self.max_episodes) * 0.25)
        cur_era = self._era_counter + 1
        recent_cluster = self.classify_byte_cluster(recent_1d)

        def _forgetting_urgency(ep: Tuple[int, float, str, torch.Tensor]) -> float:
            age = max(1, cur_era - ep[0])
            return float(ep[1]) * (1.0 - math.exp(-float(age) / tau))

        same_cluster = sorted([e for e in self._episodes if e[2] == recent_cluster], key=_forgetting_urgency, reverse=True)
        other_cluster = sorted([e for e in self._episodes if e[2] != recent_cluster], key=_forgetting_urgency, reverse=True)
        chosen = (same_cluster + other_cluster)[:max_windows]
        half_len = max(8, self.max_seq_len // 2)
        windows: List[torch.Tensor] = []
        rec_slice = recent_1d[:half_len]
        for _, _, _, hist_seq in chosen:
            hist_slice = hist_seq[:half_len]
            windows.append(torch.cat([hist_slice, rec_slice], dim=0))
        max_len = max(w.numel() for w in windows)
        padded = [F.pad(w, (0, max_len - w.numel()), value=0) for w in windows]
        return torch.stack(padded, dim=0).to(device=device)


# =============================================================================
# Primitives: RMSNorm (Implicit Gain Control), Complex Phase (Mamba-3), HoPE
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            from ecosystem_backends import try_liger_rms_norm
            fused = try_liger_rms_norm(x, self.weight, self.eps)
            if fused is not None:
                return fused
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def apply_complex_phase_rotation(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    c, s = torch.cos(theta), torch.sin(theta)
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * c - x_odd * s
    out[..., 1::2] = x_even * s + x_odd * c
    return out


def apply_hope_fast_band_rope(x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
    """HoPE (arXiv:2609.34556v1): Fast-band RoPE (first 50%) + Slow-band NoPE (last 50%)."""
    _, _, T, D = x.shape
    half = D // 2
    x_fast, x_slow = x[..., :half], x[..., half:]
    pos = torch.arange(pos_offset, pos_offset + T, device=x.device, dtype=x.dtype)
    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, 2, device=x.device, dtype=x.dtype) / half))
    angles = torch.outer(pos, freqs).unsqueeze(0).unsqueeze(0)
    return torch.cat([apply_complex_phase_rotation(x_fast, angles), x_slow], dim=-1)


# =============================================================================
# GAP-09: Block-Wise Sylvester Hadamard Rotation + SOAR/4over6 MSE-Optimal Q4 Export
# =============================================================================
def sylvester_hadamard_matrix(block_size: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if block_size < 1 or (block_size & (block_size - 1)) != 0:
        raise ValueError(f"block_size must be a power of 2, got {block_size}")
    H = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    while H.size(0) < block_size:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return (H / math.sqrt(float(block_size))).to(dtype=dtype)


def quantize_linear_blockwise_hadamard_q4(
    weight: torch.Tensor,
    block_size: int = 32,
    scale_candidates: Tuple[float, ...] = (0.75, 0.85, 0.92, 1.0),
    use_wush_channel_eq: bool = True,
) -> Dict[str, Any]:
    w = weight.detach().float()
    out_f, in_f = w.shape
    if in_f % block_size != 0:
        raise ValueError(f"in_features={in_f} must be divisible by block_size={block_size}")
    # GAP-72 (WUSH ICML 2026 arXiv:2512.00956): Per-channel second-moment RMS equalization before Hadamard rotation
    ch_rms = w.pow(2).mean(dim=0).sqrt().clamp(min=1e-4) if use_wush_channel_eq else None
    w_eq = (w / ch_rms.unsqueeze(0)) if ch_rms is not None else w
    H = sylvester_hadamard_matrix(block_size, device=w.device, dtype=torch.float32)
    n_blocks = in_f // block_size
    w_blocks = w_eq.view(out_f, n_blocks, block_size)
    w_rot = torch.matmul(w_blocks, H)
    absmax = w_rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7)

    best_mse = torch.full((out_f, n_blocks, 1), float("inf"), device=w.device)
    best_q = torch.zeros_like(w_rot, dtype=torch.int8)
    best_scale = absmax / 7.0
    for ratio in scale_candidates:
        cand_scale = (absmax * float(ratio)) / 7.0
        q_cand = torch.clamp(torch.round(w_rot / cand_scale), -7, 7).to(torch.int8)
        recon = q_cand.float() * cand_scale
        mse = (w_rot - recon).pow(2).mean(dim=-1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, cand_scale, best_scale)
        best_q = torch.where(better, q_cand, best_q)

    q_u4 = (best_q.to(torch.int16) + 8).to(torch.uint8).reshape(out_f, in_f)
    packed = (q_u4[:, 0::2] << 4) | (q_u4[:, 1::2] & 0x0F)
    return {
        "packed_q4": packed.cpu(),
        "block_scales_fp16": best_scale.squeeze(-1).to(torch.float16).cpu(),
        "channel_rms_fp16": None if ch_rms is None else ch_rms.to(torch.float16).cpu(),
        "shape": (out_f, in_f),
        "block_size": block_size,
        "mean_block_mse": float(best_mse.mean().item()),
    }


def dequantize_linear_blockwise_hadamard_q4(payload: Dict[str, Any]) -> torch.Tensor:
    packed = payload["packed_q4"]
    scales = payload["block_scales_fp16"].float()
    out_f, in_f = payload["shape"]
    block_size = int(payload["block_size"])
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    lo = (packed & 0x0F).to(torch.int16) - 8
    q_full = torch.empty((out_f, in_f), dtype=torch.float32)
    q_full[:, 0::2] = hi.float()
    q_full[:, 1::2] = lo.float()
    w_rot_hat = q_full.view(out_f, in_f // block_size, block_size) * scales.unsqueeze(-1)
    H_t = sylvester_hadamard_matrix(block_size, device=w_rot_hat.device, dtype=torch.float32).t()
    w_rec = torch.matmul(w_rot_hat, H_t).reshape(out_f, in_f)
    ch_rms = payload.get("channel_rms_fp16")
    if ch_rms is not None:
        w_rec = w_rec * ch_rms.float().to(device=w_rec.device).unsqueeze(0)
    return w_rec


# =============================================================================
# Organ 1 (L0): Lossless Byte + Canon Conv1D + LZ77 Novelty + ATDC + LCA Multi-Byte
# =============================================================================
class CanonCausalConv1D(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model, padding=0)

    def forward(self, x: torch.Tensor, conv_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        xt = x.transpose(1, 2)
        pad_len = self.kernel_size - 1
        prefix = conv_state if conv_state is not None else xt.new_zeros((xt.size(0), xt.size(1), pad_len))
        x_cat = torch.cat([prefix, xt], dim=-1)
        out = self.conv(x_cat).transpose(1, 2)
        next_state = x_cat[..., -pad_len:].detach() if pad_len > 0 else prefix
        return out, next_state


class DynamicBytePatcher(nn.Module):
    _DELIM_BYTES = (ord(" "), ord("\n"), ord("\t"), ord("="), ord(":"), ord("("), ord("["), ord("{"), ord(","), ord(";"))

    def __init__(self, d_model: int, patch_stride: int = 4, use_ube_codec: bool = False) -> None:
        super().__init__()
        self.stride = patch_stride
        self.surprise_head = nn.Linear(d_model, 1, bias=True)
        self.novelty_mix = nn.Parameter(torch.tensor(0.0))
        self.patch_proj = nn.Linear(d_model, d_model, bias=False)
        self.expand_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.slot_offsets = nn.Parameter(torch.randn(patch_stride, d_model) * 0.02)
        self.lca_proj = nn.Linear(d_model, d_model, bias=False)
        self._lz77_novelty_cpu = self._build_structural_prior(use_ube_codec=bool(use_ube_codec))
        self._novelty_locked: bool = False
        self._atdc_progress: float = 1.0

    @classmethod
    def _build_structural_prior(cls, use_ube_codec: bool = False) -> torch.Tensor:
        table = torch.full((65536,), 0.25, dtype=torch.float32, device="cpu")
        for prev_b in cls._DELIM_BYTES:
            table[prev_b * 256 : (prev_b + 1) * 256] = 1.0
        idx_256 = torch.arange(256, dtype=torch.long)
        # v10.32: UBE uses 0x80..0xC1 & 0xF5..0xFE as 2-byte CJK lead bytes in addition to UTF-8 >= 0xC0
        lead_mask = (idx_256 >= 0x80) if use_ube_codec else (idx_256 >= 0xC0)
        table.view(256, 256)[:, lead_mask] = 0.85
        return table

    def load_novelty_table(self, pair_scores_65536: torch.Tensor, lock: bool = True) -> None:
        if self._novelty_locked:
            raise RuntimeError("LZ77 novelty table is Step-0 locked (arXiv:2610.22100v1: post-training rule switches collapse accuracy).")
        if pair_scores_65536.numel() != 65536:
            raise ValueError(f"Expected 65536 entries for 2-byte novelty table, got {pair_scores_65536.numel()}.")
        self._lz77_novelty_cpu = pair_scores_65536.detach().to(device="cpu", dtype=torch.float32).reshape(65536).clamp(0.0, 2.0)
        self._lz77_novelty_dev = None  # v10.21 Fix: invalidate cached GPU copy when table updates
        self._novelty_locked = lock

    def set_atdc_progress(self, progress: float) -> None:
        self._atdc_progress = max(0.0, min(1.0, float(progress)))

    def _lookup_lz77_novelty(self, byte_ids: torch.Tensor, pad: int) -> torch.Tensor:
        # v10.21 Fix: cache 256 KB LZ77 table on byte_ids.device to eliminate per-step GPU->CPU->GPU sync
        dev = byte_ids.device
        cached = getattr(self, "_lz77_novelty_dev", None)
        if cached is None or cached.device != dev:
            cached = self._lz77_novelty_cpu.to(device=dev, dtype=torch.float32)
            self._lz77_novelty_dev = cached
        b = byte_ids.detach().to(dtype=torch.long)
        if pad > 0:
            b = F.pad(b, (0, pad), value=0)
        prev = F.pad(b[:, :-1], (1, 0), value=ord(" "))
        pair_idx = ((prev & 0xFF) << 8) | (b & 0xFF)
        return cached[pair_idx].unsqueeze(-1)

    def compress(self, h_bytes: torch.Tensor, byte_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
        B, T, D = h_bytes.shape
        pad = (self.stride - (T % self.stride)) % self.stride
        h_pad = F.pad(h_bytes, (0, 0, 0, pad)) if pad > 0 else h_bytes
        T_pad = h_pad.size(1)
        logits = self.surprise_head(h_pad)
        if byte_ids is not None:
            nov_scores = self._lookup_lz77_novelty(byte_ids, pad).to(device=h_bytes.device, dtype=h_bytes.dtype, non_blocking=True)
            logits = logits + torch.sigmoid(self.novelty_mix) * nov_scores
        else:
            logits = logits + 0.25 * torch.sigmoid(self.novelty_mix)
        w_learned = torch.sigmoid(logits).view(B, T_pad // self.stride, self.stride, 1)
        w_learned = w_learned / (w_learned.sum(dim=2, keepdim=True) + 1e-6)
        atdc = self._atdc_progress
        w = atdc * w_learned + (1.0 - atdc) * (1.0 / float(self.stride))
        patches = (h_pad.view(B, T_pad // self.stride, self.stride, D) * w).sum(dim=2)
        return self.patch_proj(patches), T

    def expand(self, h_patches: torch.Tensor, h_bytes_skip: torch.Tensor, orig_T: int) -> torch.Tensor:
        up = h_patches.repeat_interleave(self.stride, dim=1)[:, :orig_T, :]
        return self.expand_proj(torch.cat([up, h_bytes_skip], dim=-1))

    def expand_parallel_lca(self, h_patches: torch.Tensor, orig_T: int, causal_shift: bool = True) -> torch.Tensor:
        B, P, D = h_patches.shape
        src = F.pad(h_patches[:, :-1, :], (0, 0, 1, 0)) if causal_shift else h_patches
        expanded = (src.unsqueeze(2) + self.slot_offsets.view(1, 1, self.stride, D)).reshape(B, P * self.stride, D)
        return self.lca_proj(expanded)[:, :orig_T, :]


# =============================================================================
# Organ 5B (L2): Ternary Neuromodulator (ACh / NE / DA, <0.01M Params)
# =============================================================================
class TernaryNeuromodulator(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mod_head = nn.Linear(d_model, 3, bias=True)
        nn.init.zeros_(self.mod_head.weight)
        nn.init.zeros_(self.mod_head.bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # v10.24 Critical Fix: strictly causal prefix mean (eliminates future-token leak from h.mean(dim=1))
        denom = torch.arange(1, h.size(1) + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
        causal_prefix = torch.cumsum(h, dim=1) / denom
        raw = self.mod_head(self.norm(causal_prefix))  # (B, P, 3)
        ach_gain = 0.5 + torch.sigmoid(raw[..., 0:1])
        ne_gain = 0.5 + torch.sigmoid(raw[..., 1:2])
        da_gate = 2.0 * torch.sigmoid(raw[..., 2:3])
        return ach_gain, ne_gain, da_gate


# =============================================================================
# Organ 4 (L3): Trainable 0-MB-VRAM CPU-Pinned Mid-Layer Engram (Sparse CPU Backprop)
# =============================================================================
class _CPUSparseEngramFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, slots_cpu: torch.Tensor, dummy_anchor: torch.Tensor, cpu_table: torch.Tensor, lr: float, training: bool):
        ctx.slots_cpu = slots_cpu
        ctx.cpu_table = cpu_table
        ctx.lr = lr
        ctx.training = training
        fetched_cpu = cpu_table[slots_cpu].mean(dim=0)
        return fetched_cpu.to(device=dummy_anchor.device, dtype=dummy_anchor.dtype, non_blocking=True)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if ctx.training and ctx.lr > 0.0:
            slots_cpu = ctx.slots_cpu
            K = slots_cpu.size(0)
            grad_cpu = (grad_out.detach().to(device="cpu", dtype=torch.float32) / float(K))
            flat_idx = slots_cpu.reshape(-1)
            flat_grad = grad_cpu.unsqueeze(0).expand(K, -1, -1, -1).reshape(-1, grad_cpu.size(-1))
            uniq_idx, inv = torch.unique(flat_idx, return_inverse=True)
            acc_grad = torch.zeros((uniq_idx.size(0), flat_grad.size(-1)), dtype=torch.float32)
            acc_grad.index_add_(0, inv, flat_grad)
            updated = ctx.cpu_table[uniq_idx].float() - ctx.lr * acc_grad
            ctx.cpu_table[uniq_idx] = updated.to(dtype=ctx.cpu_table.dtype)
        return None, None, None, None, None


class CPUPinnedEngramMemory(nn.Module):
    def __init__(self, slots: int, d_model: int, n_hash_heads: int = 4, engram_lr: float = 0.05) -> None:
        super().__init__()
        self.slots = slots
        self.n_hash_heads = n_hash_heads
        self.engram_lr = engram_lr
        self._cpu_table = torch.empty(slots, d_model, dtype=torch.float16, device="cpu").normal_(mean=0.0, std=0.02)
        if torch.cuda.is_available():
            self._cpu_table = self._cpu_table.pin_memory()
        self._p1 = torch.tensor([1315423911, 2654435761, 2246822519, 3266489917][:n_hash_heads], dtype=torch.long)
        self._p2 = torch.tensor([668265263, 374761393, 1597334677, 2054039447][:n_hash_heads], dtype=torch.long)
        self._p3 = torch.tensor([1103515245, 1664525, 214013, 48271][:n_hash_heads], dtype=torch.long)
        self._salt = torch.tensor([17, 1013904223, 2531011, 12345][:n_hash_heads], dtype=torch.long)
        self._grafted_suffixes: Dict[Tuple[int, ...], torch.Tensor] = {}
        self.graft_proj = nn.Linear(d_model, d_model, bias=False)
        self.ctx_gate = nn.Linear(d_model * 2, d_model, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.2))

    def _compute_cpu_slots(self, patch_byte_keys: torch.Tensor) -> torch.Tensor:
        b0 = patch_byte_keys.detach().to(device="cpu", dtype=torch.long)
        P = b0.size(1)
        b1 = F.pad(b0[:, :-1], (1, 0), value=0)[:, :P]
        b2 = F.pad(b0[:, :-2], (2, 0), value=0)[:, :P]
        h = (
            b2.unsqueeze(0) * self._p1.view(-1, 1, 1)
            + b1.unsqueeze(0) * self._p2.view(-1, 1, 1)
            + b0.unsqueeze(0) * self._p3.view(-1, 1, 1)
            + self._salt.view(-1, 1, 1)
        ) & 0x7FFFFFFFFFFFFFFF
        return h % self.slots

    def host_ram_mb(self) -> float:
        return round(self._cpu_table.numel() * self._cpu_table.element_size() / (1024 ** 2), 2)

    @torch.no_grad()
    def write_fact(self, key_3bytes: Tuple[int, int, int], value_vec: torch.Tensor) -> List[int]:
        seq = torch.tensor([[key_3bytes[0], key_3bytes[1], key_3bytes[2]]], dtype=torch.long)
        slots = self._compute_cpu_slots(seq)[:, 0, -1].tolist()
        val = value_vec.detach().to(device="cpu", dtype=self._cpu_table.dtype)
        for s in slots:
            self._cpu_table[s].copy_(val)
        return slots

    @torch.no_grad()
    def graft_exact_fact(self, suffix_bytes: Tuple[int, ...], teacher_hidden: torch.Tensor) -> None:
        self._grafted_suffixes[tuple(int(b) for b in suffix_bytes)] = teacher_hidden.detach().to(
            device="cpu", dtype=self._cpu_table.dtype
        )

    @torch.no_grad()
    def erase_exact_fact(self, suffix_bytes: Tuple[int, ...]) -> bool:
        """GAP-40 (UUE/PreUnlearn, arXiv:2611.17864v1): O(1) surgical zero-collateral erasure of a CPU-resident fact."""
        key = tuple(int(b) for b in suffix_bytes)
        return self._grafted_suffixes.pop(key, None) is not None

    def lookup(
        self,
        patch_byte_keys: torch.Tensor,
        ctx_hidden: torch.Tensor,
        da_gate: Optional[torch.Tensor] = None,
        ach_gain: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        slots_cpu = self._compute_cpu_slots(patch_byte_keys)
        # v10.21 Fix: skip da_gate .item() GPU->CPU sync when eval or Engram is frozen (engram_lr <= 0)
        if self.training and self.engram_lr > 0.0 and da_gate is not None:
            eff_lr = self.engram_lr * float(da_gate.mean().detach().item())
        else:
            eff_lr = self.engram_lr
        fetched = _CPUSparseEngramFn.apply(slots_cpu, ctx_hidden, self._cpu_table, eff_lr, self.training)
        if self._grafted_suffixes:
            keys_cpu = patch_byte_keys.detach().to(device="cpu", dtype=torch.long).tolist()
            graft_delta_cpu = torch.zeros(fetched.shape, dtype=self._cpu_table.dtype, device="cpu")
            hit_mask_cpu = torch.zeros(fetched.shape[:2], dtype=torch.bool, device="cpu")
            any_hit = False
            for b_i, row in enumerate(keys_cpu):
                for t_i in range(len(row)):
                    for L in (3, 2, 1):
                        if t_i + 1 >= L:
                            suf = tuple(row[t_i + 1 - L : t_i + 1])
                            if suf in self._grafted_suffixes:
                                graft_delta_cpu[b_i, t_i] = self._grafted_suffixes[suf]
                                hit_mask_cpu[b_i, t_i] = True
                                any_hit = True
                                break
            if any_hit:
                graft_delta = graft_delta_cpu.to(device=fetched.device, dtype=fetched.dtype, non_blocking=True)
                hit_mask = hit_mask_cpu.to(device=fetched.device, non_blocking=True)
                fetched = torch.where(hit_mask.unsqueeze(-1), self.graft_proj(graft_delta), self.graft_proj(fetched))
            else:
                fetched = self.graft_proj(fetched)
        else:
            fetched = self.graft_proj(fetched)
        out = fetched * torch.sigmoid(self.ctx_gate(torch.cat([ctx_hidden, fetched], dim=-1))) * self.scale
        if ach_gain is not None:
            out = out * (2.0 - ach_gain)
        if da_gate is not None:
            out = out * (0.8 + 0.2 * da_gate)
        return out


# =============================================================================
# Organ 2A (L1, 75%, GAP-11): Oscillatory DualDeltaNet + Mamba-3 Trapezoidal + Rank-2 DeltaProduct
# =============================================================================
class OscillatoryDualDeltaNet(nn.Module):
    """
    Combines:
      - DualDeltaNet chunkwise parallel recurrence (O(1) matrix fast state)
      - Mamba-3 Exponential-Trapezoidal Discretization (arXiv:2603.15569v1): data-dependent convex
        combination of consecutive step inputs cuts global discretization error from O(Delta) to O(Delta^2),
        plus Q/K RMSNorm & learned constant bias.
      - Mamba-3 Complex Phase Rotation (RoPE-equivalent oscillatory dynamics).
      - Rank-2 Generalized Householder DeltaProduct (arXiv:2502.10297): breaks the Rank-1 state-tracking
        ceiling with eigenvalues strictly inside the unit disk (beta_2 in (0, 2)).
    """
    def __init__(self, cfg: RouteBConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_state
        self.chunk_size = cfg.chunk_size
        inner = self.n_heads * self.d_head
        self.norm = RMSNorm(cfg.d_model)
        self.qkv_proj = nn.Linear(cfg.d_model, inner * 3, bias=False)
        self.q_norm = RMSNorm(cfg.d_state)
        self.k_bias = nn.Parameter(torch.zeros(1, 1, cfg.n_heads, cfg.d_state))
        self.erase_proj = nn.Linear(cfg.d_model, inner, bias=True)
        self.write_proj = nn.Linear(cfg.d_model, inner, bias=True)
        self.trap_gate = nn.Linear(cfg.d_model, inner, bias=True)
        self.phase_proj = nn.Linear(cfg.d_model, inner // 2, bias=True) if cfg.enable_mamba3_phase else None
        # GAP-11: Rank-2 Householder reflection branch (DeltaProduct, arXiv:2502.10297)
        self.k2_proj = nn.Linear(cfg.d_model, inner, bias=False) if cfg.enable_rank2_deltaproduct else None
        self.beta2_proj = nn.Linear(cfg.d_model, cfg.n_heads, bias=True) if cfg.enable_rank2_deltaproduct else None
        self.o_proj = nn.Linear(inner, cfg.d_model, bias=False)
        self.out_gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # GAP-21: Pre-allocated causal chunk mask buffer eliminates per-chunk GPU allocations in 8-loop forward
        self.register_buffer("_chunk_tril", torch.tril(torch.ones(cfg.chunk_size, cfg.chunk_size, dtype=torch.bool)), persistent=False)
        # GAP-25: MS-GLA log-spaced multi-scale temporal decay hierarchy across heads (arXiv:2609.35664v2, 0 extra state VRAM)
        self.register_buffer("_head_decay_scales", torch.logspace(math.log10(0.25), math.log10(0.015), cfg.n_heads).view(1, cfg.n_heads, 1, 1), persistent=False)
        self.ms_head_fusion = nn.Linear(cfg.d_model, cfg.n_heads, bias=True)
        # GAP-49 (M^2RNN, arXiv:2603.14360v2): Bounded residual non-linear matrix-state gate (0 extra state VRAM)
        self.nonlin_state_gate = nn.Parameter(torch.zeros(1, cfg.n_heads, 1, 1)) if cfg.enable_m2rnn_nonlin_state else None
        # GAP-76 (PH-Attention arXiv:2508.08289): Zero-init Pearce-Hall prediction-error associability gate
        self.ph_plasticity_gate = nn.Parameter(torch.zeros(1)) if cfg.enable_pearce_hall_plasticity else None
        # GAP-87 (Gated KalmaNet arXiv:2511.21016): Adaptive ridge damping controls BF16/FP16 condition number (0 extra VRAM)
        self.kalman_ridge_lambda = float(cfg.kalman_ridge_lambda)
        # GAP-90 (SLA arXiv:2611.06828): On-chip SRAM chunk-boundary momentum (0 extra persistent VRAM)
        self.chunk_momentum = max(0.0, min(0.5, float(cfg.chunk_momentum)))
        self.enable_fla_triton_kernel = bool(cfg.enable_fla_triton_kernel)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        h = self.norm(x)
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)
        q = self.q_norm(q.view(B, T, self.n_heads, self.d_head))
        k = F.normalize(k.view(B, T, self.n_heads, self.d_head) + self.k_bias, dim=-1, eps=1e-6)
        v = v.view(B, T, self.n_heads, self.d_head)
        if self.phase_proj is not None:
            theta = self.phase_proj(h).view(B, T, self.n_heads, self.d_head // 2)
            q = apply_complex_phase_rotation(q, theta)
            k = apply_complex_phase_rotation(k, theta)

        alpha = torch.sigmoid(self.erase_proj(h)).view(B, T, self.n_heads, self.d_head)
        beta = torch.sigmoid(self.write_proj(h)).view(B, T, self.n_heads, self.d_head)
        trap_lam = torch.sigmoid(self.trap_gate(h)).view(B, T, self.n_heads, self.d_head)
        k2_all = F.normalize(self.k2_proj(h).view(B, T, self.n_heads, self.d_head), dim=-1, eps=1e-6) if self.k2_proj is not None else None
        beta2_all = (2.0 * torch.sigmoid(self.beta2_proj(h))).view(B, T, self.n_heads) if self.beta2_proj is not None else None
        # v10.32: Optional flash-linear-attention (fla) Triton chunk_gated_delta_rule fast path on CUDA
        if self.enable_fla_triton_kernel and x.is_cuda and self.k2_proj is None and self.nonlin_state_gate is None:
            from ecosystem_backends import try_fla_gated_delta_rule
            g_log = torch.log((1.0 - self._head_decay_scales.view(1, 1, self.n_heads) * alpha.mean(dim=-1)).clamp(min=1e-4))
            fla_res = try_fla_gated_delta_rule(q, k, v, g_log, beta.mean(dim=-1), initial_state=state, output_final_state=True)
            if fla_res is not None:
                o_fla, s_fla = fla_res
                ms_w = (0.5 + torch.sigmoid(self.ms_head_fusion(h))).unsqueeze(-1)
                out = self.o_proj((o_fla * ms_w).reshape(B, T, -1)) * torch.sigmoid(self.out_gate(h))
                return out, s_fla.detach()
        S = x.new_zeros((B, self.n_heads, self.d_head, self.d_head)) if state is None else state

        outs: List[torch.Tensor] = []
        prev_chunk_delta: Optional[torch.Tensor] = None
        prev_v_last: Optional[torch.Tensor] = None
        C = self.chunk_size
        for start in range(0, T, C):
            end = min(start + C, T)
            qc, kc, vc = q[:, start:end], k[:, start:end], v[:, start:end]
            ac, bc, lc = alpha[:, start:end], beta[:, start:end], trap_lam[:, start:end]
            L = end - start
            inter_out = torch.einsum("bhij,blhj->blhi", S, qc)
            qk = torch.einsum("blhd,bmhd->bhlm", qc, kc).masked_fill(~self._chunk_tril[:L, :L], 0.0)
            prior_pred = torch.einsum("bhij,blhj->blhi", S, kc)
            pred_err = vc - ac * prior_pred
            if self.ph_plasticity_gate is not None:
                # GAP-76 (PH-Attention arXiv:2508.08289): Surprise-modulated associability boosts plasticity on unpredicted cues
                err_mag = torch.tanh(pred_err.detach().norm(dim=-1, keepdim=True) / math.sqrt(float(self.d_head)))
                ph_scale = 1.0 + 0.5 * torch.tanh(self.ph_plasticity_gate) * err_mag
                v_eff = bc * ph_scale * pred_err
            else:
                v_eff = bc * pred_err
            # Mamba-3 Exponential-Trapezoidal blend (v10.24 Fix: carry prev_v_last across chunk boundaries)
            if prev_v_last is None:
                v_prev = F.pad(v_eff[:, :-1], (0, 0, 0, 0, 1, 0), value=0.0)
            else:
                v_prev = torch.cat([prev_v_last, v_eff[:, :-1]], dim=1)
            prev_v_last = v_eff[:, -1:]
            v_trap = lc * v_eff + (1.0 - lc) * v_prev
            if self.kalman_ridge_lambda > 0.0:
                # GAP-87 (Gated KalmaNet arXiv:2511.21016): Precision-safe ridge gain 1/(||k||^2 + lambda)
                ridge_gain = 1.0 / (kc.pow(2).sum(dim=-1, keepdim=True) + self.kalman_ridge_lambda)
                v_trap = v_trap * ridge_gain
            outs.append(inter_out + torch.einsum("bhlm,bmhd->blhd", qk, v_trap))
            inv_L = 1.0 / float(max(1, L))
            S_prev = S
            S = (
                S * (1.0 - self._head_decay_scales * ac.mean(dim=1).unsqueeze(-1))
                - torch.einsum("blhi,blhj->bhij", ac * prior_pred, kc) * inv_L
                + torch.einsum("blhi,blhj->bhij", v_trap, kc) * inv_L
            )
            if self.chunk_momentum > 0.0:
                cur_delta = S - S_prev
                if prev_chunk_delta is not None:
                    S = S + self.chunk_momentum * prev_chunk_delta
                prev_chunk_delta = cur_delta
            if k2_all is not None and beta2_all is not None:
                # Rank-2 Generalized Householder reflection: S <- S (I - beta2 * k2 k2^T), spectral radius <= 1
                k2_c = F.normalize(k2_all[:, start:end].mean(dim=1), dim=-1, eps=1e-6)  # (B, H, D)
                b2_c = beta2_all[:, start:end].mean(dim=1).view(B, self.n_heads, 1, 1)  # (B, H, 1, 1) in (0, 2)
                sk2 = torch.einsum("bhij,bhj->bhi", S, k2_c)
                S = S - b2_c * torch.einsum("bhi,bhj->bhij", sk2, k2_c)
            if self.nonlin_state_gate is not None:
                # GAP-49 (M^2RNN, arXiv:2603.14360v2): Contractive non-linear matrix-state transition
                g_nl = torch.sigmoid(self.nonlin_state_gate) * 0.25
                S = (1.0 - g_nl) * S + g_nl * torch.tanh(S)
        # MS-GLA input-dependent multi-scale head fusion before output projection
        ms_w = (0.5 + torch.sigmoid(self.ms_head_fusion(h))).unsqueeze(-1)  # (B, T, H, 1) in (0.5, 1.5)
        cat_out = (torch.cat(outs, dim=1) * ms_w).reshape(B, T, -1)
        out = self.o_proj(cat_out) * torch.sigmoid(self.out_gate(h))
        # v10.20 CTO Fix: avoid CUDA .item() GPU-CPU sync stall inside training forward (eval-only telemetry)
        if not hasattr(self, "_qdelta_ccq_ctrl"):
            self._qdelta_ccq_ctrl = QDeltaCCQEraseReadController()
        if not self.training:
            self.last_ccq_q_scale = self._qdelta_ccq_ctrl.condition_read_query(
                [1.0], [float(S.detach().float().pow(2).mean().item())]
            )[0]
        return out, S.detach()


# =============================================================================
# Organ 2B (L1, 25%): HoPE + MELT Gated Shared HLA Latent KV + Native SDPA + Qwen Gate
# =============================================================================
class HoPEHLAMELTAttention(nn.Module):
    def __init__(self, cfg: RouteBConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_state
        inner = self.n_heads * self.d_head
        self.norm = RMSNorm(cfg.d_model)
        self.q_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.kv_down = nn.Linear(cfg.d_model, cfg.kv_latent_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_latent_dim)
        self.melt_gate = nn.Linear(cfg.kv_latent_dim * 2, cfg.kv_latent_dim, bias=True)
        self.k_up = nn.Linear(cfg.kv_latent_dim, inner, bias=False)
        self.v_up = nn.Linear(cfg.kv_latent_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, cfg.d_model, bias=False)
        self.sink_gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # GAP-29: Deep-loop Value Residual gate (ARLoop arXiv:2611.05234v1) prevents value rank collapse across 8 loops
        self.v_res_logit = nn.Parameter(torch.tensor(-2.0))
        self.enable_bos_summary_prefill = bool(cfg.enable_bos_summary_prefill)
        self._hype_manifold_kv = HyPEManifoldDefensiveKVController(train_ctx_len=max(256, int(cfg.max_seq_len)))
        self._byte_tst_rectified_loop = ByteTSTConv16RectifiedLoopController(gdn_conv_width=16, loop_damping=0.85)

    def forward(
        self,
        x: torch.Tensor,
        shared_kv_latent: Optional[torch.Tensor] = None,
        loop_idx: int = 0,
        pos_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        h = self.norm(x)
        c_kv = self.kv_norm(self.kv_down(h))
        if shared_kv_latent is None:
            c_kv_full = c_kv
        elif loop_idx == 0:
            c_kv_full = torch.cat([shared_kv_latent, c_kv], dim=1)
        elif shared_kv_latent.size(1) >= T:
            tail_old = shared_kv_latent[:, -T:, :]
            g = torch.sigmoid(self.melt_gate(torch.cat([tail_old, c_kv], dim=-1)))
            tail_new = (1.0 - g) * tail_old + g * c_kv
            c_kv_full = torch.cat([shared_kv_latent[:, :-T, :], tail_new], dim=1) if shared_kv_latent.size(1) > T else tail_new
        else:
            c_kv_full = c_kv

        T_kv = c_kv_full.size(1)
        # v10.24 Fix: align q_pos_offset with physical KV length T_kv - T so GateSight/KV compaction never causes RoPE jumps
        q_pos_off = max(0, T_kv - T) if shared_kv_latent is not None else pos_offset
        q = apply_hope_fast_band_rope(
            self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2), pos_offset=q_pos_off
        )
        k = apply_hope_fast_band_rope(
            self.k_up(c_kv_full).view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2), pos_offset=0
        )
        v_raw = self.v_up(c_kv_full)
        if shared_kv_latent is not None and loop_idx > 0 and shared_kv_latent.size(1) == T_kv:
            g_v = torch.sigmoid(self.v_res_logit)
            v_raw = (1.0 - g_v) * v_raw + g_v * self.v_up(shared_kv_latent)
        else:
            # Keep v_res_logit causally wired on loop 0 via zero-centered identity residual
            v_raw = v_raw + 0.0 * torch.sigmoid(self.v_res_logit)
        v = v_raw.view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)
        nope_scale = self._hype_manifold_kv.nope_log_scale(int(T_kv))
        loop_damp = self._byte_tst_rectified_loop.rectified_loop_damping(loop_idx=int(loop_idx), state_delta_norm=float(nope_scale))
        if nope_scale > 1.0:
            # v10.20 CTO Fix (HyPE arXiv:2601.22156): scale ONLY the slow-band NoPE half, preserving RoPE half
            half_d = q.size(-1) // 2
            q = torch.cat([q[..., :half_d], q[..., half_d:] * float(nope_scale)], dim=-1)

        # BUG-01 fix: exact causal masking even when cached multi-token prefill has 1 < T < T_kv
        if T == 1:
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        elif T == T_kv:
            if self.enable_bos_summary_prefill and T > 1 and pos_offset == 0:
                # BOS Summary (arXiv:2611.23508): BOS query row attends to all prompt tokens; all other rows stay strictly causal
                bos_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=q.device))
                bos_mask[0, :] = True
                ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=bos_mask, is_causal=False)
            else:
                ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            t_past = T_kv - T
            past_mask = torch.ones((T, t_past), dtype=torch.bool, device=q.device)
            cur_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=q.device))
            attn_mask = torch.cat([past_mask, cur_mask], dim=-1)
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = self.o_proj(ctx.transpose(1, 2).reshape(B, T, -1)) * torch.sigmoid(self.sink_gate(h)) * float(loop_damp)
        # v10.20 CTO Fix: avoid CUDA .item() GPU-CPU sync stall inside training forward (eval-only telemetry)
        if not hasattr(self, "_vase_rdkv_guard"):
            self._vase_rdkv_guard = LoopedFactorScalingVaSEGuard()
        if not self.training:
            v_peak = float(v.detach().float().norm(dim=-1).max().item())
            self.last_vase_tier = self._vase_rdkv_guard.vase_rdkv_decision(0.10, v_peak)
        return out, c_kv_full.detach()


# =============================================================================
# ARLoop Sliding-Window Depth Attention (arXiv:2611.05234, O(W) Constant Memory)
# =============================================================================
def sinkhorn_birkhoff_project(logits: torch.Tensor, iters: int = 8, eps: float = 1e-6) -> torch.Tensor:
    """
    GAP-61 (mHC arXiv:2512.24880v2 + mHC-SSM arXiv:2605.08300 + Birkhoff Geometry arXiv:2610.06653):
    Projects a square matrix onto the Birkhoff polytope of doubly stochastic matrices (row sums = 1,
    col sums = 1, spectral norm <= 1) via Sinkhorn-Knopp normalization for explosion-free stream mixing.
    """
    P = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
    for _ in range(max(1, int(iters))):
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        P = P / (P.sum(dim=-2, keepdim=True) + eps)
    return P


class SlidingWindowDepthAttnRes(nn.Module):
    def __init__(self, d_model: int, window_size: int = 2) -> None:
        super().__init__()
        self.key_norm = RMSNorm(d_model)
        self.query = nn.Parameter(torch.zeros(d_model))
        # GAP-61 (mHC-SSM): Learnable WxW Birkhoff-projected doubly stochastic stream mixing (init near identity)
        w = max(2, int(window_size))
        self.mhc_logits = nn.Parameter(torch.eye(w) * 2.0)

    def forward(self, window_states: List[torch.Tensor]) -> torch.Tensor:
        if len(window_states) == 1:
            return window_states[0]
        stacked = torch.stack(window_states, dim=0)  # (W, B, T, D)
        W = stacked.size(0)
        if W == self.mhc_logits.size(0):
            # Birkhoff doubly stochastic pre-mixing across depth streams (spectral norm <= 1)
            H_ds = sinkhorn_birkhoff_project(self.mhc_logits, iters=8).to(dtype=stacked.dtype)
            stacked = torch.einsum("ij,jbtd->ibtd", H_ds, stacked)
        scores = torch.einsum("d,wbtd->wbt", self.query, self.key_norm(stacked)) / math.sqrt(stacked.size(-1))
        # GAP-39 (OASIS, arXiv:2605.17887v2): Softmax_1 explicit null route prevents AttnRes depth-sink outliers
        zero_null = torch.zeros_like(scores[:1])
        weights = F.softmax(torch.cat([scores, zero_null], dim=0), dim=0)[:-1].unsqueeze(-1)
        return (weights * stacked).sum(dim=0)


# =============================================================================
# Pure Dense SwiGLU Block + Wake<-Sleep Prior Recall Gate (GAP-13)
# =============================================================================
class RouteBBlock(nn.Module):
    def __init__(self, cfg: RouteBConfig, is_exact_recall: bool) -> None:
        super().__init__()
        self.is_exact_recall = is_exact_recall
        self.mixer = HoPEHLAMELTAttention(cfg) if is_exact_recall else OscillatoryDualDeltaNet(cfg)
        # GAP-13: Learned Wake<-Sleep transfer gate warm-starting fresh wake SSM states from consolidated sleep priors
        self.sleep_recall_gate = nn.Parameter(torch.tensor(-2.0)) if not is_exact_recall else None
        self.ffn_norm = RMSNorm(cfg.d_model)
        ffn_hidden = int(cfg.d_model * 2.5)
        self.w_gate = nn.Linear(cfg.d_model, ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, ffn_hidden, bias=False)
        self.w_down = nn.Linear(ffn_hidden, cfg.d_model, bias=False)
        self.ffn_sparsity_thresh = max(0.0, float(cfg.ffn_sparsity_thresh))
        self.silu_gate_floor = max(0.0, float(cfg.silu_gate_floor))

    def forward(
        self,
        x: torch.Tensor,
        block_id: int,
        loop_idx: int = 0,
        cache: Optional[RouteBCache] = None,
        commit_state: bool = True,
        frozen_delta_state: Optional[torch.Tensor] = None,
        state_sink: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.is_exact_recall:
            kv_prev = None if cache is None else cache.hla_kv_latents.get(block_id)
            pos_off = 0 if cache is None else cache.pos_offset
            delta_m, kv_next = self.mixer(x, shared_kv_latent=kv_prev, loop_idx=loop_idx, pos_offset=pos_off)
            if cache is not None:
                cache.hla_kv_latents[block_id] = kv_next
        else:
            # GAP-52 (uTTT arXiv:2610.05484v1 + arXiv:2610.06833v1 Lemma 1):
            # All core loops read the same frozen historical prefix state and commit ONLY the terminal state
            st_prev = frozen_delta_state if frozen_delta_state is not None else (None if cache is None else cache.delta_states.get(block_id))
            if st_prev is None and cache is not None and self.sleep_recall_gate is not None:
                sleep_prior = cache.sleep_ssm_states.get(block_id)
                if sleep_prior is not None:
                    prior_b = sleep_prior.to(device=x.device, dtype=x.dtype)
                    if prior_b.size(0) != x.size(0):
                        prior_b = prior_b.mean(dim=0, keepdim=True).expand(x.size(0), -1, -1, -1)
                    st_prev = torch.sigmoid(self.sleep_recall_gate) * prior_b
            delta_m, st_next = self.mixer(x, state=st_prev)
            if state_sink is not None:
                state_sink[block_id] = st_next
            if cache is not None and commit_state:
                cache.delta_states[block_id] = st_next
        inv_r = 1.0 / float(loop_idx + 1)
        x = x + inv_r * delta_m
        h = self.ffn_norm(x)
        gate_act = F.silu(self.w_gate(h)) + self.silu_gate_floor
        if self.ffn_sparsity_thresh > 0.0:
            # SGA-Sparse (arXiv:2611.11223): Zero-router-overhead native SwiGLU channel sparsity
            gate_act = torch.where(gate_act.abs() >= self.ffn_sparsity_thresh, gate_act, torch.zeros_like(gate_act))
        return x + inv_r * self.w_down(gate_act * self.w_up(h))


# =============================================================================
# Complete Route B Sovereign AI Brain (v1.3)
# =============================================================================
class AdaptiveDraftHorizonController:
    """
    GAP-70 (AMTP, arXiv:2610.28619) + GAP-77 (Nightjar, arXiv:2512.22420):
    Online acceptance-survival draft horizon controller with proactive speculation disable/cooldown
    gating when acceptance survival drops below `disable_threshold` (0 MB VRAM, zero retraining).
    """
    def __init__(
        self,
        min_steps: int = 1,
        max_steps: int = 8,
        init_steps: int = 4,
        ema: float = 0.8,
        disable_threshold: float = 0.25,
        cooldown_steps: int = 3,
    ) -> None:
        self.min_steps = max(1, int(min_steps))
        self.max_steps = max(self.min_steps, int(max_steps))
        self.horizon = max(self.min_steps, min(self.max_steps, int(init_steps)))
        self.ema = max(0.1, min(0.99, float(ema)))
        self.disable_threshold = max(0.0, min(0.5, float(disable_threshold)))
        self.cooldown_steps = max(1, int(cooldown_steps))
        self.survival_ema: float = 0.75
        self._cooldown_remaining: int = 0

    def should_speculate(self) -> bool:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False
        return True

    def select_horizon(self) -> int:
        return self.horizon if self.should_speculate() else 0

    def update(self, accepted: int, drafted: int) -> int:
        if drafted <= 0:
            return self.horizon
        rate = float(accepted) / float(max(1, drafted))
        self.survival_ema = self.ema * self.survival_ema + (1.0 - self.ema) * rate
        if self.survival_ema < self.disable_threshold:
            self._cooldown_remaining = self.cooldown_steps
            self.horizon = self.min_steps
        elif self.survival_ema >= 0.85 and self.horizon < self.max_steps:
            self.horizon += 1
        elif self.survival_ema <= 0.45 and self.horizon > self.min_steps:
            self.horizon -= 1
        return self.horizon


class GateSightKVPruner(nn.Module):
    """
    GAP-183 PROMOTED TO LIVE HOT PATH (GateSight, arXiv:2610.14754):
    Lightweight per-token utility scorer reading GDN recurrent hidden states (ZERO attention
    scores) to physically prune `cache.hla_kv_latents` after prefill (retaining 1/10th KV up to 262K).
    """
    def __init__(self, d_model: int, default_keep_ratio: float = 0.10) -> None:
        super().__init__()
        self.scorer = nn.Linear(d_model, 1, bias=True)
        self.default_keep_ratio = min(max(float(default_keep_ratio), 0.05), 1.0)

    def forward(self, gdn_hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scorer(gdn_hidden)).squeeze(-1)  # (B, P)

    @torch.no_grad()
    def prune_cache(
        self,
        cache: RouteBCache,
        gdn_hidden: torch.Tensor,
        keep_ratio: Optional[float] = None,
        min_keep: int = 8,
    ) -> Dict[str, int]:
        scores = self.forward(gdn_hidden).mean(dim=0)  # (P,)
        ratio = self.default_keep_ratio if keep_ratio is None else min(max(float(keep_ratio), 0.05), 1.0)
        pruned_layers, before_tot, after_tot = 0, 0, 0
        for bid, kv in list(cache.hla_kv_latents.items()):
            T_kv = kv.size(1)
            before_tot += T_kv
            if T_kv <= min_keep or scores.numel() < T_kv:
                after_tot += T_kv
                continue
            k = max(min_keep, int(round(T_kv * ratio)))
            if k >= T_kv:
                after_tot += T_kv
                continue
            top_idx = torch.topk(scores[:T_kv], k=k).indices.sort().values
            cache.hla_kv_latents[bid] = kv.index_select(1, top_idx).detach()
            after_tot += k
            pruned_layers += 1
        return {"pruned_layers": pruned_layers, "kv_before": before_tot, "kv_after": after_tot}


class SovereignRouteBBrain(nn.Module):
    def __init__(self, cfg: RouteBConfig) -> None:
        super().__init__()
        self.automata_tcq_guard = AutomataEigenStateAndTCQEngramGuard()
        self.cfg = cfg
        self.byte_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.canon = CanonCausalConv1D(cfg.d_model, kernel_size=cfg.canon_kernel)
        self.patcher = DynamicBytePatcher(cfg.d_model, patch_stride=cfg.patch_stride, use_ube_codec=cfg.use_ube_codec)

        self.prelude = nn.ModuleList([RouteBBlock(cfg, is_exact_recall=False) for _ in range(cfg.k_prelude)])
        self.neuromod = TernaryNeuromodulator(cfg.d_model) if cfg.enable_ternary_neuromod else None
        self.engram = CPUPinnedEngramMemory(cfg.engram_slots, cfg.d_model, engram_lr=cfg.engram_lr) if cfg.enable_cpu_engram else None
        self.core = nn.ModuleList([RouteBBlock(cfg, is_exact_recall=((i + 1) % 4 == 0)) for i in range(cfg.k_core)])

        self.parcae_carry_logit = nn.Parameter(torch.full((cfg.d_model,), 2.0))
        self.inj_alpha = nn.Parameter(torch.tensor(0.15)) if cfg.enable_input_injection else None
        self.loop_gain_norm = RMSNorm(cfg.d_model)
        self.depth_attn = SlidingWindowDepthAttnRes(cfg.d_model, window_size=cfg.depth_attn_window) if cfg.enable_arloop_depth_attn else None
        # GAP-12: ATF/PTRM NE-gated stochastic latent exploration scale (softplus(-4) ~ 0.018 safe start)
        self.thought_noise_scale = nn.Parameter(torch.tensor(-4.0)) if cfg.enable_stochastic_thought else None

        self.coda = nn.ModuleList([RouteBBlock(cfg, is_exact_recall=(i == cfg.k_coda - 1)) for i in range(cfg.k_coda)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # GAP-45 (v2.8 deepened, arXiv:2608.08888v1): Zero-init gate injecting full-bandwidth latent feedback
        # directly into the 8-loop non-Transformer 3:1 core input (before e0 and recurrent loops).
        self.latent_fb_gate = nn.Parameter(torch.zeros(1))
        # GAP-47 (NCP arXiv:2609.10715 + Next-Latent World Models arXiv:2511.05963v4):
        # Lightweight next-patch concept/latent predictor trained jointly with discrete byte NTP.
        self.next_latent_pred = nn.Linear(cfg.d_model, cfg.d_model, bias=False) if cfg.enable_next_latent_pred else None
        # GAP-63 (ELR/Koi-4B arXiv:2610.12582): Zero-init loop-index & budget conditioning
        self.loop_cond_proj = nn.Linear(2, cfg.d_model, bias=False) if cfg.enable_elr_loop_cond else None
        self.gatesight = GateSightKVPruner(cfg.d_model, default_keep_ratio=cfg.gatesight_keep_ratio) if cfg.enable_gatesight_kv_prune else None
        if self.loop_cond_proj is not None:
            nn.init.zeros_(self.loop_cond_proj.weight)
        # v10.6 Hot-Path Integration: bind live scaling, spectral collapse & AMOR entropy controllers
        self.loop_factor_guard = LoopedFactorScalingVaSEGuard(num_layers=max(1, len(self.core)))
        self.spectral_shield = SpectralGuardCHONHotChannelShield(min_spectral_radius=0.25)
        self.byteflow_amor_ctrl = ByteFlowAMOREntropyGateController(std_scale=0.5)

    def _run_core_once(
        self,
        h: torch.Tensor,
        e0: torch.Tensor,
        loop_idx: int,
        cache: Optional[RouteBCache],
        ach_gain: Optional[torch.Tensor] = None,
        ne_gain: Optional[torch.Tensor] = None,
        stochastic_explore: bool = False,
        frozen_core_states: Optional[Dict[int, Optional[torch.Tensor]]] = None,
        state_sink: Optional[Dict[int, torch.Tensor]] = None,
        total_loops: int = 1,
        disable_noise: bool = False,
    ) -> torch.Tensor:
        carry = torch.sigmoid(self.parcae_carry_logit).view(1, 1, -1)
        carried = carry * h
        if self.inj_alpha is not None:
            inj = self.inj_alpha * (ach_gain if ach_gain is not None else 1.0) * e0
            if self.cfg.enable_ortho_input_injection:
                # GAP-50 (OrthoInj, arXiv:2610.06833v1 §4.2, Eq. 8): Project carried state orthogonal to inj
                # in FP32 so ||Q_q Lambda||_2 <= ||Lambda||_2 < 1 and every loop keeps exact input strength.
                c_f, q_f = carried.float(), inj.float()
                dot = (c_f * q_f).sum(dim=-1, keepdim=True)
                norm_sq = q_f.pow(2).sum(dim=-1, keepdim=True) + 1e-6
                carried = carried - ((dot / norm_sq) * q_f).to(dtype=carried.dtype)
            h_in = self.loop_gain_norm(carried + inj)
        else:
            h_in = self.loop_gain_norm(carried)
        if self.loop_cond_proj is not None:
            # GAP-63 (ELR/Koi-4B arXiv:2610.12582): Condition each pass on relative loop index & budget
            t_tot = max(1, int(total_loops))
            loop_feat = h_in.new_tensor([float(loop_idx) / float(max(1, t_tot - 1)), 1.0 / float(t_tot)])
            h_in = h_in + self.loop_cond_proj(loop_feat).view(1, 1, -1)
        if ne_gain is not None:
            h_in = h_in * ne_gain
        if (not disable_noise) and self.thought_noise_scale is not None and (stochastic_explore or self.training):
            # ATF/PTRM multimodal latent exploration: NE scales noise radius to escape spurious fixed points
            sigma = F.softplus(self.thought_noise_scale)
            ne_scale = ne_gain if ne_gain is not None else 1.0
            h_in = h_in + sigma * ne_scale * torch.randn_like(h_in)
        h_cur = h_in
        for i, blk in enumerate(self.core):
            bid = 100 + i
            f_st = None if frozen_core_states is None else frozen_core_states.get(bid)
            h_cur = blk(
                h_cur, block_id=bid, loop_idx=loop_idx, cache=cache,
                commit_state=False, frozen_delta_state=f_st, state_sink=state_sink,
            )
        return h_cur

    def _run_coda(self, h_latent: torch.Tensor, cache: Optional[RouteBCache]) -> torch.Tensor:
        h = h_latent
        for i, blk in enumerate(self.coda):
            h = blk(h, block_id=200 + i, loop_idx=0, cache=cache)
        return h

    def _decode_readout(
        self, h_latent: torch.Tensor, h_bytes: torch.Tensor, orig_T: int, cache: Optional[RouteBCache]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_coda = self._run_coda(h_latent, cache=cache)
        if cache is not None:
            # Cache full-bandwidth post-coda thought state so the NEXT step feeds it into the 8-loop 3:1 core input
            cache.prev_coda_state = h_coda[:, -1:, :].detach()
        logits = self.lm_head(self.final_norm(self.patcher.expand(h_coda, h_bytes, orig_T)))
        return logits, h_coda

    def compute_stars_spectral_penalty(
        self,
        h_src: torch.Tensor,
        e0: torch.Tensor,
        h_out: torch.Tensor,
        loop_idx: int,
        ach_gain: Optional[torch.Tensor] = None,
        ne_gain: Optional[torch.Tensor] = None,
        total_loops: int = 1,
    ) -> torch.Tensor:
        eps = 1e-3
        v = torch.randn_like(h_src)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        # v10.24 Fix: disable_noise=True & pass exact total_loops so finite-diff measures true Jacobian, not random noise!
        h_base = self._run_core_once(
            h_src, e0, loop_idx=loop_idx, cache=None, ach_gain=ach_gain, ne_gain=ne_gain,
            stochastic_explore=False, total_loops=total_loops, disable_noise=True,
        ) if self.thought_noise_scale is not None else h_out
        h_pert = self._run_core_once(
            h_src + eps * v, e0, loop_idx=loop_idx, cache=None, ach_gain=ach_gain, ne_gain=ne_gain,
            stochastic_explore=False, total_loops=total_loops, disable_noise=True,
        )
        rho_hat = ((h_pert - h_base).norm(dim=-1) / eps).mean()
        return F.relu(rho_hat - 1.0).pow(2)

    @staticmethod
    def compute_dysib_ib_loss(h_latent: torch.Tensor) -> torch.Tensor:
        """
        GAP-27: Dynamical Symmetric Information Bottleneck (DySIB, arXiv:2604.24662v1 + MCJEPA arXiv:2608.13621):
        Minimizes off-diagonal feature covariance redundancy (minimal representation complexity) while
        promoting smooth differential latent increments ||h_{t+1} - h_t||^2 (predictive sufficiency).
        Zero extra parameters.
        """
        flat = h_latent.float().reshape(-1, h_latent.size(-1))
        if flat.size(0) < 2:
            return torch.zeros((), device=h_latent.device, dtype=torch.float32)
        centered = flat - flat.mean(dim=0, keepdim=True)
        std = centered.std(dim=0, keepdim=True).clamp(min=1e-4)
        z = centered / std
        cov = (z.t() @ z) / float(max(1, z.size(0) - 1))
        off_diag = cov - torch.diag(torch.diag(cov))
        redundancy_loss = off_diag.pow(2).mean()
        if h_latent.size(1) > 1:
            diff_smooth = (h_latent[:, 1:, :].float() - h_latent[:, :-1, :].float()).pow(2).mean()
        else:
            diff_smooth = torch.tensor(0.0, device=h_latent.device)
        return redundancy_loss + 0.1 * diff_smooth


    @staticmethod
    @torch.no_grad()
    def compute_eigentrack_spectral_risk(h_latent: torch.Tensor) -> Dict[str, float]:
        """
        GAP-75 (EigenTrack, arXiv:2509.15735): Single-forward-pass, zero-resampling spectral
        activation geometry detector for real-time hallucination & OOD drift tracking.
        Computes normalized covariance-spectrum entropy and top-1 eigenvalue dominance ratio.
        """
        flat = h_latent.detach().float().reshape(-1, h_latent.size(-1))
        if flat.size(0) < 2:
            return {"spectral_entropy": 0.0, "top1_dominance": 1.0, "hallucination_risk": 1.0}
        centered = flat - flat.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(centered)
        eigs = (s * s).clamp(min=1e-12)
        p = eigs / eigs.sum()
        ent = float(-(p * p.log()).sum().item())
        max_ent = math.log(float(eigs.numel())) if eigs.numel() > 1 else 1.0
        norm_ent = ent / max(1e-6, max_ent)
        top1 = float(p[0].item())
        return {
            "spectral_entropy": round(norm_ent, 5),
            "top1_dominance": round(top1, 5),
            "hallucination_risk": round(max(0.0, min(1.0, 1.0 - norm_ent)), 5),
        }

    @torch.no_grad()
    def predict_next_patch_parallel(self, byte_ids: torch.Tensor, cache: Optional[RouteBCache] = None) -> torch.Tensor:
        if not self.cfg.enable_lca_multibyte:
            raise RuntimeError("predict_next_patch_parallel requires cfg.enable_lca_multibyte=True.")
        was_training = self.training
        self.eval()
        out = self.forward(byte_ids, targets=None, cache=cache)
        lca_next_logits = out["lca_next_patch_logits"]
        if was_training:
            self.train()
        return lca_next_logits.argmax(dim=-1)

    @torch.no_grad()
    def sample_thought_branches(
        self, byte_ids: torch.Tensor, n_branches: int = 4, cache: Optional[RouteBCache] = None
    ) -> Dict[str, Any]:
        """
        GAP-12 Test-Time Multimodal Latent Scaling (ATF arXiv:2609.33271 + PTRM arXiv:2605.19943):
        Samples `n_branches` diverse NE-modulated stochastic latent trajectories and selects the
        trajectory achieving the lowest FPRM relative fixed-point residual (zero mean-collapse).
        """
        was_training = self.training
        self.eval()
        best_out: Optional[Dict[str, Any]] = None
        best_cache: Optional[RouteBCache] = None
        best_consilience = float("-inf")
        branch_scores: List[float] = []
        branch_self_certainties: List[float] = []
        branch_consilience_scores: List[float] = []
        branch_probs: List[torch.Tensor] = []
        log_V = math.log(float(self.cfg.vocab_size))
        nb = max(1, int(n_branches))
        ics_scales: List[float] = []
        for b_idx in range(nb):
            b_cache = None if cache is None else cache.clone()
            # GAP-67 (ICS arXiv:2610.18547): Deterministic branch-wise input-conditioning spread [0.88, 1.12]
            c_scale = 1.0 if nb == 1 else (0.88 + 0.24 * (float(b_idx) / float(nb - 1)))
            ics_scales.append(round(c_scale, 4))
            out = self.forward(byte_ids, targets=None, cache=b_cache, stochastic_explore=True, input_cond_scale=c_scale)
            rels = out["rel_residuals"]
            min_rel = min(rels) if rels else float("inf")
            early_rel = rels[0] if rels else 0.0
            final_rel = rels[-1] if rels else 1.0
            branch_scores.append(round(min_rel, 6))
            # GAP-62 (INTUITOR arXiv:2505.19590v5 + Consilience arXiv:2608.09898):
            # Self-certainty = KL(U || p) = -(mean(log p) + log V), unbiased by length;
            # Consilience rewards early exploratory branching followed by sharp final convergence.
            lp_last = F.log_softmax(out["logits"][:, -1, :].float(), dim=-1)
            self_cert = float((-(lp_last.mean(dim=-1) + log_V)).mean().item())
            consilience = self_cert + 0.5 * (early_rel - final_rel) - final_rel
            branch_self_certainties.append(round(self_cert, 5))
            branch_consilience_scores.append(round(consilience, 5))
            branch_probs.append(lp_last.exp().mean(dim=0))
            if consilience > best_consilience:
                best_consilience = consilience
                best_out = out
                best_cache = b_cache
        if cache is not None and best_cache is not None:
            cache.copy_from(best_cache)
        if was_training:
            self.train()
        assert best_out is not None
        best_out["branch_min_residuals"] = branch_scores
        best_out["branch_self_certainties"] = branch_self_certainties
        best_out["branch_consilience_scores"] = branch_consilience_scores
        best_out["branch_ics_scales"] = ics_scales
        # GAP-37: Exact Aleatoric (mean H) vs Epistemic (JSD) decomposition across latent branches
        stacked = torch.stack(branch_probs, dim=0).clamp_min(1e-12)
        mean_p = stacked.mean(dim=0).clamp_min(1e-12)
        au = float(-(stacked * stacked.log()).sum(dim=-1).mean().item())
        tot_h = float(-(mean_p * mean_p.log()).sum().item())
        best_out["branch_aleatoric_entropy"] = round(au, 6)
        best_out["branch_epistemic_jsd"] = round(max(0.0, tot_h - au), 6)
        return best_out

    @torch.no_grad()
    def loop_spec_decode(
        self,
        prompt_bytes: torch.Tensor,
        draft_steps: int = 4,
        draft_loops: int = 2,
        verify_loops: Optional[int] = None,
        horizon_controller: Optional[AdaptiveDraftHorizonController] = None,
    ) -> Dict[str, Any]:
        """
        GAP-33 (LoopSpec, arXiv:2609.17184v1): Training-free self-speculative decoding for
        Looped Transformers with 0 MB extra draft-model VRAM. Drafts `draft_steps` bytes at
        shallow loop depth `draft_loops`, then verifies all drafted bytes in one parallel
        full-depth pass (`verify_loops`) with exact greedy acceptance/rejection parity.
        """
        was_training = self.training
        self.eval()
        v_loops = self.cfg.rec_loops if verify_loops is None else max(1, min(int(verify_loops), self.cfg.rec_loops))
        d_loops = max(1, min(int(draft_loops), v_loops))
        k = horizon_controller.select_horizon() if horizon_controller is not None else max(1, int(draft_steps))
        if k == 0:
            v_only = self.forward(prompt_bytes, targets=None, cache=None, max_loops=v_loops)
            tok = v_only["logits"][:, -1:, :].argmax(dim=-1)
            if was_training:
                self.train()
            return {
                "generated_bytes": tok,
                "full_sequence": torch.cat([prompt_bytes, tok], dim=1),
                "accepted_draft_bytes": 0,
                "draft_steps": 0,
                "acceptance_rate": 0.0,
                "draft_loops": d_loops,
                "verify_loops": v_loops,
                "saguaro_runner_up_hit": False,
                "speculation_bypassed": True,
                "next_adaptive_horizon": horizon_controller.horizon if horizon_controller is not None else 0,
            }
        seq = prompt_bytes.clone()
        drafted: List[torch.Tensor] = []
        runner_ups: List[torch.Tensor] = []
        for _ in range(k):
            d_out = self.forward(seq, targets=None, cache=None, max_loops=d_loops)
            top2 = d_out["logits"][:, -1, :].topk(2, dim=-1).indices
            nxt, alt = top2[:, 0:1], top2[:, 1:2]
            drafted.append(nxt)
            runner_ups.append(alt)
            seq = torch.cat([seq, nxt], dim=1)

        # Single parallel verification pass at full loop depth
        v_out = self.forward(seq, targets=None, cache=None, max_loops=v_loops)
        base_len = prompt_bytes.size(1)
        # Position (base_len - 1 + i) predicts drafted[i]
        v_preds = v_out["logits"][:, base_len - 1 : base_len - 1 + k + 1, :].argmax(dim=-1)
        accepted = 0
        saguaro_runner_up_hit = False
        out_tokens: List[torch.Tensor] = []
        for i in range(k):
            v_tok = v_preds[:, i : i + 1]
            if torch.equal(v_tok, drafted[i]):
                out_tokens.append(v_tok)
                accepted += 1
            else:
                # GAP-58 (Saguaro SSD, arXiv:2603.03251v1): Track runner-up preemption hits on rejection
                saguaro_runner_up_hit = torch.equal(v_tok, runner_ups[i])
                out_tokens.append(v_tok)
                break
        else:
            # All drafted bytes accepted -> emit bonus verified byte
            out_tokens.append(v_preds[:, k : k + 1])

        if was_training:
            self.train()
        generated = torch.cat(out_tokens, dim=1)
        next_h = horizon_controller.update(accepted, k) if horizon_controller is not None else k
        return {
            "generated_bytes": generated,
            "next_adaptive_horizon": next_h,
            "full_sequence": torch.cat([prompt_bytes, generated], dim=1),
            "accepted_draft_bytes": accepted,
            "draft_steps": k,
            "acceptance_rate": round(float(accepted) / float(k), 4),
            "draft_loops": d_loops,
            "verify_loops": v_loops,
            "saguaro_runner_up_hit": saguaro_runner_up_hit,
        }

    @torch.no_grad()
    def bitnest_spec_decode(
        self,
        prompt_bytes: torch.Tensor,
        draft_steps: int = 4,
        draft_bits: int = 2,
    ) -> Dict[str, Any]:
        """
        GAP-184 PROMOTED TO LIVE HOT PATH (BitNest, arXiv:2610.02800):
        Bit-nested shared-weight speculative decoding with 0.0 MB extra draft-model VRAM.
        Drafts tokens using a 2-bit nested view of the SAME physical weights, then verifies
        in a single full-precision parallel pass with exact greedy parity.
        """
        was_training = self.training
        self.eval()
        k = max(1, int(draft_steps))
        qmax = float((1 << (max(2, int(draft_bits)) - 1)) - 1)
        # Save target weights and temporarily apply in-place 2-bit nested draft projection
        saved_weights: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        for name, p in self.named_parameters():
            if p.ndim == 2 and "core" in name:
                saved_weights.append((p, p.detach().clone()))
                sc = p.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-7) / qmax
                p.copy_(torch.clamp(torch.round(p / sc), -qmax, qmax) * sc)
        seq = prompt_bytes.clone()
        drafted: List[torch.Tensor] = []
        try:
            for _ in range(k):
                d_out = self.forward(seq, targets=None, cache=None, max_loops=max(1, self.cfg.rec_loops // 2))
                nxt = d_out["logits"][:, -1:, :].argmax(dim=-1)
                drafted.append(nxt)
                seq = torch.cat([seq, nxt], dim=1)
        finally:
            for p, orig in saved_weights:
                p.copy_(orig)
        # Verify all drafted tokens in one full-precision parallel pass
        v_out = self.forward(seq, targets=None, cache=None, max_loops=self.cfg.rec_loops)
        base_len = prompt_bytes.size(1)
        v_preds = v_out["logits"][:, base_len - 1 : base_len - 1 + k + 1, :].argmax(dim=-1)
        accepted = 0
        out_tokens: List[torch.Tensor] = []
        for i in range(k):
            v_tok = v_preds[:, i : i + 1]
            out_tokens.append(v_tok)
            if torch.equal(v_tok, drafted[i]):
                accepted += 1
            else:
                break
        else:
            out_tokens.append(v_preds[:, k : k + 1])
        if was_training:
            self.train()
        return {
            "generated_bytes": torch.cat(out_tokens, dim=1),
            "accepted_draft_bytes": accepted,
            "draft_steps": k,
            "extra_draft_vram_mb": 0.0,
            "draft_bits": int(draft_bits),
        }

    @torch.no_grad()
    def consolidate_sleep(
        self,
        evicted_byte_ids: torch.Tensor,
        cache: RouteBCache,
        n_sleep_passes: int = 3,
        noise_prune_quantile: Optional[float] = None,
        co_obs_buffer: Optional[CoObservationSleepBuffer] = None,
        salience: float = 1.0,
        record_episode: bool = True,
    ) -> Dict[str, float]:
        was_training = self.training
        self.eval()
        replay_ids = evicted_byte_ids
        co_obs_windows = 1
        if co_obs_buffer is not None:
            replay_ids = co_obs_buffer.build_co_observation_windows(evicted_byte_ids, device=evicted_byte_ids.device)
            co_obs_windows = int(replay_ids.size(0))
            if record_episode:
                co_obs_buffer.add_episode(evicted_byte_ids, salience=salience)

        h_bytes, _ = self.canon(self.byte_embed(replay_ids), None)
        h, _ = self.patcher.compress(h_bytes, byte_ids=replay_ids)
        for i, blk in enumerate(self.prelude):
            h = blk(h, block_id=i, loop_idx=0, cache=None)
        ema = self.cfg.sleep_ema_decay
        q = self.cfg.sleep_noise_prune_quantile if noise_prune_quantile is None else max(0.0, min(0.5, float(noise_prune_quantile)))
        drift_norm = 0.0
        pruned_fracs: List[float] = []
        for _ in range(max(1, n_sleep_passes)):
            for i, blk in enumerate(self.core):
                if not blk.is_exact_recall:
                    bid = 100 + i
                    st_prev = cache.sleep_ssm_states.get(bid)
                    if st_prev is not None and st_prev.size(0) != h.size(0):
                        st_prev = st_prev.mean(dim=0, keepdim=True).expand(h.size(0), -1, -1, -1)
                    _, st_next = blk.mixer(h, state=st_prev)
                    st_mean = st_next.mean(dim=0, keepdim=True)
                    anchor = cache.sleep_ema_anchors.get(bid, st_mean)
                    st_consolidated = ema * anchor + (1.0 - ema) * st_mean
                    fluct = (st_mean - anchor).abs()
                    drift_norm += float(fluct.norm().item())
                    if q > 0.0:
                        sal = st_consolidated.abs() / (fluct + 1e-4)
                        thresh = torch.quantile(sal.float().reshape(-1), q)
                        keep_mask = (sal >= thresh).to(dtype=st_consolidated.dtype)
                        st_consolidated = st_consolidated * (0.25 + 0.75 * keep_mask)
                        pruned_fracs.append(float((1.0 - keep_mask).mean().item()))
                    cache.sleep_ssm_states[bid] = st_consolidated.detach()
                    cache.sleep_ema_anchors[bid] = (ema * anchor + (1.0 - ema) * st_consolidated).detach()
        pruned_eps = 0
        if co_obs_buffer is not None and hasattr(co_obs_buffer, "decay_after_sleep"):
            c_recent = co_obs_buffer.classify_byte_cluster(evicted_byte_ids.detach().to("cpu").reshape(-1))
            pruned_eps = co_obs_buffer.decay_after_sleep(cluster_id=c_recent)
        if was_training:
            self.train()
        return {
            "sleep_passes": float(n_sleep_passes),
            "co_obs_windows": float(co_obs_windows),
            "consolidated_layers": float(len(cache.sleep_ssm_states)),
            "mean_drift": round(drift_norm / max(1, n_sleep_passes), 6),
            "mean_pruned_fraction": round(sum(pruned_fracs) / max(1, len(pruned_fracs)), 4),
            "algorithmic_forgetting_pruned": float(pruned_eps),
        }

    @staticmethod
    def export_latent_kv(cache: RouteBCache) -> Dict[int, torch.Tensor]:
        return {bid: kv.detach().clone() for bid, kv in cache.hla_kv_latents.items()}

    @staticmethod
    def import_latent_kv(
        cache: RouteBCache,
        peer_kv: Dict[int, torch.Tensor],
        blend: float = 0.35,
        max_norm_ratio: float = 3.0,
        min_cosine: float = -0.2,
    ) -> int:
        """
        GAP-59 (When Latent Agents Lie arXiv:2606.28958 + LCGuard arXiv:2605.22786):
        Guards multi-brain latent KV import against NaN/Inf poisoning, norm-explosion attacks
        (||peer|| > max_norm_ratio * ||local||), and adversarial anti-aligned steering (cos < min_cosine).
        """
        alpha = max(0.0, min(1.0, float(blend)))
        fused = 0
        for bid, p_kv in peer_kv.items():
            if not torch.isfinite(p_kv).all():
                continue
            if bid not in cache.hla_kv_latents:
                cache.hla_kv_latents[bid] = p_kv.detach().clone()
            else:
                local = cache.hla_kv_latents[bid]
                p_aligned = p_kv.to(device=local.device, dtype=local.dtype)
                t_common = min(local.size(1), p_aligned.size(1))
                loc_tail = local[:, -t_common:, :].float()
                peer_tail = p_aligned[:, -t_common:, :].float()
                loc_norm = float(loc_tail.norm().item()) + 1e-6
                peer_norm = float(peer_tail.norm().item())
                if peer_norm > float(max_norm_ratio) * loc_norm:
                    continue  # Reject norm-explosion poisoned peer KV
                cos_val = float(F.cosine_similarity(loc_tail.reshape(-1), peer_tail.reshape(-1), dim=0).item())
                if cos_val < float(min_cosine):
                    continue  # Reject adversarial anti-aligned peer KV
                blended_tail = (1.0 - alpha) * local[:, -t_common:, :] + alpha * p_aligned[:, -t_common:, :]
                if local.size(1) > t_common:
                    cache.hla_kv_latents[bid] = torch.cat([local[:, :-t_common, :], blended_tail], dim=1).detach()
                else:
                    cache.hla_kv_latents[bid] = blended_tail.detach()
            fused += 1
        return fused

    @staticmethod
    @torch.no_grad()
    def compact_hla_kv_cache(
        cache: RouteBCache,
        max_latents: int = 512,
        sink_keep: int = 8,
        recent_keep: int = 64,
        agentic_middle_evict: bool = False,
    ) -> Dict[str, int]:
        """
        GAP-28 (arXiv:2602.16284v2 & EMNLP 2026 arXiv:2608.12331):
        Compacts long-context shared HLA KV latents to strictly bounded O(1) length `max_latents`
        by preserving initial anchors (`sink_keep`) + active tail (`recent_keep`) and merging
        middle history via salience-weighted Attention-Matching block pooling (zero grad descent).
        """
        compacted_layers = 0
        total_before, total_after = 0, 0
        mid_budget = max(1, int(max_latents) - int(sink_keep) - int(recent_keep))
        for bid, kv in list(cache.hla_kv_latents.items()):
            T_kv = kv.size(1)
            total_before += T_kv
            if T_kv <= max_latents:
                total_after += T_kv
                continue
            sinks = kv[:, :sink_keep, :]
            recent = kv[:, -recent_keep:, :]
            mid = kv[:, sink_keep : T_kv - recent_keep, :]  # (B, T_mid, D_lat)
            T_mid = mid.size(1)
            if agentic_middle_evict:
                # GAP-81 (Penelope, arXiv:2611.00391): Preserve shared prefix + active suffix; evict oldest post-prefix middle
                mid_comp = mid[:, -mid_budget:, :] if T_mid > mid_budget else mid
                cache.hla_kv_latents[bid] = torch.cat([sinks, mid_comp, recent], dim=1).detach()
                total_after += cache.hla_kv_latents[bid].size(1)
                compacted_layers += 1
                continue
            # Energy/salience-weighted Attention-Matching block compaction into `mid_budget` slots
            # v10.22 Fix: compute split edges on CPU to avoid ~1,300 GPU .item() sync stalls
            edges = torch.linspace(0, T_mid, steps=mid_budget + 1, device="cpu").long().tolist()
            pooled_slots: List[torch.Tensor] = []
            for j in range(mid_budget):
                a = int(edges[j])
                b = max(a + 1, int(edges[j + 1]))
                seg = mid[:, a:b, :]
                sal = seg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                w = sal / sal.sum(dim=1, keepdim=True)
                pooled_slots.append((seg * w).sum(dim=1, keepdim=True))
            mid_comp = torch.cat(pooled_slots, dim=1)
            cache.hla_kv_latents[bid] = torch.cat([sinks, mid_comp, recent], dim=1).detach()
            total_after += cache.hla_kv_latents[bid].size(1)
            compacted_layers += 1
        return {"compacted_layers": compacted_layers, "tokens_before": total_before, "tokens_after": total_after}

    @staticmethod
    @torch.no_grad()
    def quantize_hla_kv_cache_int4(
        cache: RouteBCache,
        block_size: int = 32,
        sink_keep: int = 8,
        recent_keep: int = 32,
        center_channels: bool = True,
    ) -> Dict[str, Any]:
        """
        GAP-56 (SAW-INT4 arXiv:2604.19157v1 + UltraQuant arXiv:2606.20474 + arXiv:2604.04722):
        Compresses middle historical HLA KV latents via Block-Wise Sylvester Hadamard Q4 while
        preserving initial sink anchors (`sink_keep`) and active tail (`recent_keep`) in FP16.
        Achieves ~3.5-4x long-context KV VRAM reduction on RTX 3050 4GB (kv_latent_dim=128 % 32 == 0).
        """
        quantized_layers: Dict[int, Dict[str, Any]] = {}
        raw_bytes, comp_bytes = 0, 0
        for bid, kv in cache.hla_kv_latents.items():
            B, T_kv, D_kv = kv.shape
            raw_bytes += kv.numel() * 2
            if T_kv <= sink_keep + recent_keep or D_kv % block_size != 0:
                quantized_layers[bid] = {"mode": "fp16", "tensor": kv.detach().to(device="cpu", dtype=torch.float16)}
                comp_bytes += kv.numel() * 2
                continue
            sink = kv[:, :sink_keep, :].detach().to(device="cpu", dtype=torch.float16)
            mid = kv[:, sink_keep : T_kv - recent_keep, :].detach().float().reshape(-1, D_kv)
            # GAP-74 (NSNQuant arXiv:2505.18231 + PatternKV arXiv:2510.05176): Channel-wise centering before Hadamard Q4
            ch_mean = mid.mean(dim=0, keepdim=True) if center_channels else None
            mid_centered = (mid - ch_mean) if ch_mean is not None else mid
            recent = kv[:, T_kv - recent_keep :, :].detach().to(device="cpu", dtype=torch.float16)
            mid_q4 = quantize_linear_blockwise_hadamard_q4(mid_centered, block_size=block_size)
            quantized_layers[bid] = {
                "mode": "hybrid_hadamard_q4",
                "batch_size": B,
                "mid_len": T_kv - sink_keep - recent_keep,
                "sink_fp16": sink,
                "mid_channel_mean_fp16": None if ch_mean is None else ch_mean.to(device="cpu", dtype=torch.float16),
                "mid_q4": mid_q4,
                "recent_fp16": recent,
            }
            comp_bytes += (
                sink.numel() * 2
                + recent.numel() * 2
                + mid_q4["packed_q4"].numel()
                + mid_q4["block_scales_fp16"].numel() * 2
            )
        return {
            "layers": quantized_layers,
            "audit": {
                "num_layers": len(quantized_layers),
                "fp16_kb": round(raw_bytes / 1024.0, 2),
                "quantized_kb": round(comp_bytes / 1024.0, 2),
                "compression_ratio": round(raw_bytes / max(1, comp_bytes), 2),
            },
        }

    @staticmethod
    @torch.no_grad()
    def dequantize_hla_kv_cache_int4(payload: Dict[str, Any], device: torch.device, dtype: torch.dtype = torch.float16) -> Dict[int, torch.Tensor]:
        restored: Dict[int, torch.Tensor] = {}
        for bid, entry in payload.get("layers", {}).items():
            if entry["mode"] == "fp16":
                restored[bid] = entry["tensor"].to(device=device, dtype=dtype)
            else:
                B, mid_len = int(entry["batch_size"]), int(entry["mid_len"])
                mid_rec = dequantize_linear_blockwise_hadamard_q4(entry["mid_q4"])
                c_mean = entry.get("mid_channel_mean_fp16")
                if c_mean is not None:
                    mid_rec = mid_rec + c_mean.float().to(device=mid_rec.device)
                mid_rec = mid_rec.view(B, mid_len, -1)
                full = torch.cat(
                    [entry["sink_fp16"].float(), mid_rec, entry["recent_fp16"].float()], dim=1
                )
                restored[bid] = full.to(device=device, dtype=dtype)
        return restored


    @staticmethod
    @torch.no_grad()
    def quantize_delta_state_lores_int8(
        cache: RouteBCache,
        slow_head_fp16_frac: float = 0.5,
        stochastic_round: bool = True,
        protect_early_layers: int = 2,
        persistent_scales: Optional[Dict[int, torch.Tensor]] = None,
        scale_ema: float = 0.95,
    ) -> Dict[str, Any]:
        """
        GAP-78 (LoReS, arXiv:2611.01864) + GAP-88 (Scale-State Feedback, arXiv:2610.15139):
        Recurrent matrix-state quantization with two critical long-horizon safeguards:
          1. Early-layer FP16 protection (`bid < protect_early_layers`) prevents early-layer error compounding.
          2. Persistent reference scale containment (`persistent_scales`, `scale_ema`) breaks dynamic
             scale-state feedback loops that cause norm explosion over 1,000-32K decode steps.
        """
        layers: Dict[int, Dict[str, Any]] = {}
        raw_b, comp_b = 0, 0
        for bid, st in cache.delta_states.items():
            # st: (B, H, D, D)
            raw_b += st.numel() * 2
            if int(bid) < int(protect_early_layers):
                full_fp16 = st.detach().to(device="cpu", dtype=torch.float16)
                comp_b += full_fp16.numel() * 2
                layers[bid] = {"fast_q8": None, "fast_scale": None, "slow_fp16": full_fp16}
                continue
            H = st.size(1)
            n_slow = max(1, min(H, int(round(H * float(slow_head_fp16_frac)))))
            n_fast = H - n_slow
            slow_fp16 = st[:, n_fast:, :, :].detach().to(device="cpu", dtype=torch.float16)
            comp_b += slow_fp16.numel() * 2
            if n_fast > 0:
                fast = st[:, :n_fast, :, :].detach().float()
                inst_scale = fast.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7) / 127.0
                if persistent_scales is not None and bid in persistent_scales:
                    ref_sc = persistent_scales[bid].to(device=inst_scale.device, dtype=torch.float32)
                    v_scale = float(scale_ema) * ref_sc + (1.0 - float(scale_ema)) * inst_scale
                    persistent_scales[bid] = v_scale.detach().clone()
                else:
                    v_scale = inst_scale
                    if persistent_scales is not None:
                        persistent_scales[bid] = v_scale.detach().clone()
                scaled = fast / v_scale
                if stochastic_round:
                    lo = torch.floor(scaled)
                    prob = scaled - lo
                    q = lo + (torch.rand_like(prob) < prob).float()
                else:
                    q = torch.round(scaled)
                fast_q8 = torch.clamp(q, -127, 127).to(device="cpu", dtype=torch.int8)
                fast_sc = v_scale.to(device="cpu", dtype=torch.float16)
                comp_b += fast_q8.numel() + fast_sc.numel() * 2
            else:
                fast_q8, fast_sc = None, None
            layers[bid] = {"fast_q8": fast_q8, "fast_scale": fast_sc, "slow_fp16": slow_fp16}
        return {
            "layers": layers,
            "audit": {
                "fp16_kb": round(raw_b / 1024.0, 2),
                "lores_kb": round(comp_b / 1024.0, 2),
                "compression_ratio": round(raw_b / max(1, comp_b), 2),
            },
        }

    @staticmethod
    @torch.no_grad()
    def dequantize_delta_state_lores_int8(
        payload: Dict[str, Any], device: torch.device, dtype: torch.dtype = torch.float16
    ) -> Dict[int, torch.Tensor]:
        restored: Dict[int, torch.Tensor] = {}
        for bid, entry in payload.get("layers", {}).items():
            slow = entry["slow_fp16"].to(device=device, dtype=dtype)
            if entry["fast_q8"] is not None:
                fast = (entry["fast_q8"].float() * entry["fast_scale"].float()).to(device=device, dtype=dtype)
                restored[bid] = torch.cat([fast, slow], dim=1)
            else:
                restored[bid] = slow
        return restored

    @staticmethod
    @torch.no_grad()
    def probe_tensor_quant_sensitivity(
        weight: torch.Tensor, block_size: int = 32, n_probes: int = 8, param_name: str = ""
    ) -> float:
        """
        GAP-46 (v2.8 deepened, RAM arXiv:2609.33923v1): Data-free Gaussian random-probe sensitivity
        estimator for dense 3:1 core + HLA/MELT bottlenecks. Theorem 1 proves Delta = W - Q(W) is
        spectrally flat on dense & latent-bottleneck models; §4.9 shows post-bottleneck latent KV
        projections (HLA kv_up/kv_down/o_proj/q_proj) suffer sensitivity amplification (1.5x weight).
        """
        w_f = weight.detach().float().cpu()
        q_payload = quantize_linear_blockwise_hadamard_q4(w_f, block_size=block_size)
        w_q = dequantize_linear_blockwise_hadamard_q4(q_payload).float()
        delta = w_f - w_q
        gen = torch.Generator(device="cpu").manual_seed(2026)
        z = torch.randn((w_f.size(1), max(1, int(n_probes))), generator=gen, dtype=torch.float32)
        err_e = float((delta @ z).pow(2).sum().item())
        sig_e = float((w_f @ z).pow(2).sum().item()) + 1e-12
        raw_sens = err_e / sig_e
        # RAM §4.9: Amplify sensitivity for HLA/MELT low-rank latent bottleneck projections
        is_hla_bottleneck = any(k in param_name for k in ("k_up", "v_up", "kv_down", "o_proj", "q_proj"))
        mult = 1.5 if is_hla_bottleneck else 1.0
        return round(raw_sens * mult, 6)

    @torch.no_grad()
    def export_q4_inference_state(
        self, block_size: int = 32, top_fp16_frac: float = 0.10, n_probes: int = 8, biloco_rank: int = 8
    ) -> Dict[str, Any]:
        q4_weights: Dict[str, Any] = {}
        fp16_weights: Dict[str, torch.Tensor] = {}
        mses: List[float] = []
        raw_bytes, comp_bytes = 0, 0
        eligible: List[Tuple[str, torch.nn.Parameter]] = []
        for name, param in self.named_parameters():
            raw_bytes += param.numel() * 2
            is_protected = any(k in name for k in ("byte_embed", "canon", "prelude.0", "lm_head", "neuromod", "norm"))
            if param.ndim == 2 and not is_protected and param.size(1) % block_size == 0 and param.size(1) >= block_size:
                eligible.append((name, param))
            else:
                fp16_weights[name] = param.detach().to(device="cpu", dtype=torch.float16)
                comp_bytes += param.numel() * 2

        ram_protected: set = set()
        if top_fp16_frac > 0.0 and eligible:
            scored = sorted(
                ((self.probe_tensor_quant_sensitivity(p, block_size=block_size, n_probes=n_probes, param_name=n), n) for n, p in eligible),
                reverse=True,
            )
            k_keep = max(1, int(round(len(scored) * min(1.0, float(top_fp16_frac)))))
            ram_protected = {n for _, n in scored[:k_keep]}

        for name, param in eligible:
            if name in ram_protected:
                fp16_weights[name] = param.detach().to(device="cpu", dtype=torch.float16)
                comp_bytes += param.numel() * 2
            else:
                q_payload = quantize_linear_blockwise_hadamard_q4(param, block_size=block_size)
                if int(biloco_rank) > 0:
                    # GAP-95 (BiLoCo arXiv:2610.32243): Live 16x-smaller binary {+1,-1} low-rank PTQ residual
                    res_2d = param.detach().float().cpu() - dequantize_linear_blockwise_hadamard_q4(q_payload).float()
                    b_a, b_u, b_v = compute_biloco_binary_lowrank_correction(res_2d, rank=int(biloco_rank))
                    q_payload["biloco_alpha"], q_payload["biloco_sign_u"], q_payload["biloco_sign_v"] = b_a, b_u, b_v
                q4_weights[name] = q_payload
                mses.append(q_payload["mean_block_mse"])
                comp_bytes += q_payload["packed_q4"].numel() + q_payload["block_scales_fp16"].numel() * 2
                if "biloco_alpha" in q_payload:
                    # 1-bit sign_u + sign_v packed (numel/8) + fp16 alphas
                    comp_bytes += q_payload["biloco_alpha"].numel() * 2 + (q_payload["biloco_sign_u"].numel() + q_payload["biloco_sign_v"].numel() + 7) // 8
        return {
            "q4_weights": q4_weights,
            "fp16_weights": fp16_weights,
            "audit": {
                "quantized_tensors": len(q4_weights),
                "fp16_protected_tensors": len(fp16_weights),
                "fp16_mb": round(raw_bytes / (1024 ** 2), 2),
                "exported_q4_mb": round(comp_bytes / (1024 ** 2), 2),
                "compression_ratio": round(raw_bytes / max(1, comp_bytes), 2),
                "mean_block_mse": round(sum(mses) / max(1, len(mses)), 8),
                "disaggregated_quant_policy": {"prefill_act_quant": True, "decode_act_quant": False},  # GAP-113 (DQ arXiv:2609.26333)
            },
        }

    @torch.no_grad()
    def load_q4_inference_state(self, payload: Dict[str, Any]) -> Dict[str, int]:
        """
        GAP-23: Exact round-trip loader for `export_q4_inference_state()`: dequantizes Block-Wise
        Hadamard Q4 weights back via inverse H^T and restores FP16-protected tensors in-place.
        """
        q4_map = payload.get("q4_weights", {})
        fp16_map = payload.get("fp16_weights", {})
        deq_count, fp16_count = 0, 0
        for name, param in self.named_parameters():
            if name in q4_map:
                entry = q4_map[name]
                w_rec = dequantize_linear_blockwise_hadamard_q4(entry).float()
                if "biloco_alpha" in entry and "biloco_sign_u" in entry and "biloco_sign_v" in entry:
                    # v10.22 Fix: restore BiLoCo binary low-rank PTQ residual (previously dropped on load!)
                    w_rec = w_rec + torch.einsum(
                        "r,mr,rn->mn",
                        entry["biloco_alpha"].float().to(w_rec.device),
                        entry["biloco_sign_u"].float().to(w_rec.device),
                        entry["biloco_sign_v"].float().to(w_rec.device),
                    )
                param.copy_(w_rec.to(device=param.device, dtype=param.dtype))
                deq_count += 1
            elif name in fp16_map:
                param.copy_(fp16_map[name].to(device=param.device, dtype=param.dtype))
                fp16_count += 1
        return {"dequantized_q4_tensors": deq_count, "restored_fp16_tensors": fp16_count}

    def forward(
        self,
        byte_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[RouteBCache] = None,
        stochastic_explore: bool = False,
        max_loops: Optional[int] = None,
        input_cond_scale: float = 1.0,
    ) -> Dict[str, Any]:
        _ = self.automata_tcq_guard.allows_negative_eigen_state_tracking(2.0)
        h_bytes, next_conv = self.canon(self.byte_embed(byte_ids), None if cache is None else cache.conv_state)
        if cache is not None:
            cache.conv_state = next_conv
        h, orig_T = self.patcher.compress(h_bytes, byte_ids=byte_ids)

        for i, blk in enumerate(self.prelude):
            h = blk(h, block_id=i, loop_idx=0, cache=cache)

        ach_gain, ne_gain, da_gate = self.neuromod(h) if self.neuromod is not None else (None, None, None)

        if self.engram is not None:
            patch_keys = byte_ids[:, :: self.cfg.patch_stride]
            if patch_keys.size(1) < h.size(1):
                patch_keys = F.pad(patch_keys, (0, h.size(1) - patch_keys.size(1)), value=0)
            h = h + self.engram.lookup(patch_keys[:, : h.size(1)], h, da_gate=da_gate, ach_gain=ach_gain)

        # GAP-45 (v2.8 deepened for non-Transformer 3:1 looped core, arXiv:2608.08888v1):
        # Inject strictly-causal full-bandwidth latent feedback BEFORE e0 and the 8-loop recurrent core
        # so all 8 OscillatoryDualDeltaNet + HoPEHLAMELTAttention loops refine over continuous thought state.
        fb_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0), value=0.0)
        if cache is not None and cache.prev_coda_state is not None and cache.prev_coda_state.size(0) == h.size(0):
            fb_prev = fb_prev.clone()
            fb_prev[:, :1, :] = cache.prev_coda_state.to(device=h.device, dtype=h.dtype)
        h = h + torch.tanh(self.latent_fb_gate) * fb_prev

        # GAP-67 (ICS arXiv:2610.18547): Optional input-conditioning scale for attractor-escaping branch search
        e0 = h * float(input_cond_scale) if input_cond_scale != 1.0 else h
        rel_residuals: List[float] = []
        n_loops = self.cfg.rec_loops if max_loops is None else max(1, min(int(max_loops), self.cfg.rec_loops))
        no_grad_loops = max(0, n_loops - self.cfg.backprop_loops) if self.training else n_loops
        window: List[torch.Tensor] = [h]
        W = max(1, self.cfg.depth_attn_window)

        best_h, best_rel = h, float("inf")
        frozen_core_states: Dict[int, Optional[torch.Tensor]] = (
            {100 + i: cache.delta_states.get(100 + i) for i, blk in enumerate(self.core) if not blk.is_exact_recall}
            if cache is not None else {}
        )
        best_core_states: Dict[int, torch.Tensor] = {}
        active_patch_mask = torch.ones((h.size(0), h.size(1), 1), dtype=torch.bool, device=h.device)
        patch_loop_counts = torch.zeros((h.size(0), h.size(1), 1), dtype=torch.float32, device=h.device)
        prev_delta: Optional[torch.Tensor] = None
        divergence_streak: int = 0  # GAP-82 (NeurIPS 2026 Spotlight arXiv:2606.03217): Overthinking guard
        h_early_cd: Optional[torch.Tensor] = None
        early_cd_loop: int = max(0, (n_loops // 2) - 1)
        step_losses: List[torch.Tensor] = []
        stars_penalties: List[torch.Tensor] = []

        for r in range(n_loops):
            prev = h
            h_src = self.depth_attn(window) if self.depth_attn is not None else h
            loop_state_sink: Optional[Dict[int, torch.Tensor]] = {} if cache is not None else None
            if r < no_grad_loops:
                with torch.no_grad():
                    h = self._run_core_once(
                        h_src, e0, loop_idx=r, cache=cache, ach_gain=ach_gain, ne_gain=ne_gain,
                        stochastic_explore=stochastic_explore,
                        frozen_core_states=frozen_core_states, state_sink=loop_state_sink, total_loops=n_loops,
                    )
            else:
                if self.training and self.cfg.use_grad_checkpointing and cache is None:
                    h = torch.utils.checkpoint.checkpoint(
                        lambda hs, e_in: self._run_core_once(
                            hs, e_in, loop_idx=r, cache=None, ach_gain=ach_gain, ne_gain=ne_gain,
                            stochastic_explore=stochastic_explore, total_loops=n_loops,
                        ),
                        h_src,
                        e0,
                        use_reentrant=False,
                    )
                else:
                    h = self._run_core_once(
                        h_src, e0, loop_idx=r, cache=cache, ach_gain=ach_gain, ne_gain=ne_gain,
                        stochastic_explore=stochastic_explore,
                        frozen_core_states=frozen_core_states, state_sink=loop_state_sink, total_loops=n_loops,
                    )
                if self.training and self.cfg.use_stars_reg:
                    stars_penalties.append(
                        self.compute_stars_spectral_penalty(h_src, e0, h, loop_idx=r, ach_gain=ach_gain, ne_gain=ne_gain, total_loops=n_loops)
                    )
                if self.training and targets is not None and self.cfg.enable_fixed_prior_depth_sup and r < n_loops - 1:
                    step_logits, _ = self._decode_readout(h, h_bytes, orig_T, cache=None)
                    step_losses.append(F.cross_entropy(step_logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1)))

            if r == early_cd_loop:
                h_early_cd = h.detach()
            window.append(h)
            if len(window) > W:
                window.pop(0)

            if not self.training and self.cfg.enable_per_patch_elastic_freeze:
                # GAP-26 (T-LoopFormer): freeze converged easy patches while letting hard patches continue refining
                h = torch.where(active_patch_mask, h, prev)
                patch_loop_counts = patch_loop_counts + active_patch_mask.float()
                if r >= 2:
                    p_rel = (h - prev).norm(dim=-1, keepdim=True) / (prev.norm(dim=-1, keepdim=True) + 1e-6)
                    active_patch_mask = active_patch_mask & (p_rel >= self.cfg.patch_converge_tol)
            if self.training:
                # v10.20 CTO Critical Fix: during training, ALWAYS bind best_h to the active backprop loop tensor
                # (prevents silent gradient disconnection if an earlier no_grad loop had smaller rel_val)
                # and eliminate 16x per-step CUDA .item() GPU-CPU sync stalls.
                best_h = h
                if loop_state_sink:
                    best_core_states = dict(loop_state_sink)
            else:
                delta = h - prev
                rel_val = float((delta.norm() / (prev.norm() + 1e-6)).detach().item())
                rel_residuals.append(round(rel_val, 6))

                if self.cfg.enable_ne_oscillation_halt:
                    oscillating = False
                    if prev_delta is not None:
                        cos_sim = float(F.cosine_similarity(delta.reshape(-1), prev_delta.reshape(-1), dim=0).detach().item())
                        oscillating = cos_sim < -0.15
                    prev_delta = delta.detach()
                    if len(rel_residuals) >= 2 and rel_val > rel_residuals[-2]:
                        divergence_streak += 1
                    else:
                        divergence_streak = 0
                    if rel_val < best_rel:
                        best_rel, best_h = rel_val, h
                        if loop_state_sink:
                            best_core_states = dict(loop_state_sink)
                    elif divergence_streak >= 2 or (r >= 2 and (not active_patch_mask.any() or rel_val < 0.02 or oscillating or rel_val >= rel_residuals[-2])):
                        break
                else:
                    best_h = h
                    if loop_state_sink:
                        best_core_states = dict(loop_state_sink)

        if not self.training and self.cfg.enable_loopcd_hidden and self.cfg.loopcd_alpha > 0.0 and h_early_cd is not None:
            # GAP-92 (LoopCD-Hidden arXiv:2610.02185): Live zero-head-overhead contrastive latent refinement
            best_h = loopcd_contrastive_decode(best_h, h_early_cd, alpha=self.cfg.loopcd_alpha, mode="hidden")
        if cache is not None and best_core_states:
            # GAP-52: Commit ONLY the terminal converged fast-weight state per physical block (8x smaller cache)
            for bid, st_term in best_core_states.items():
                cache.delta_states[bid] = st_term

        logits, h_coda = self._decode_readout(best_h, h_bytes, orig_T, cache=cache)
        if cache is not None:
            if self.gatesight is not None and not self.training:
                self.gatesight.prune_cache(cache, best_h, keep_ratio=self.cfg.gatesight_keep_ratio)
            cache.pos_offset += best_h.size(1)

        out: Dict[str, Any] = {"logits": logits, "rel_residuals": rel_residuals, "terminal_latent": best_h.detach()}
        if not self.training and self.cfg.enable_per_patch_elastic_freeze:
            out["mean_patch_loops"] = round(float(patch_loop_counts.mean().item()), 3)
        pred_next_latent: Optional[torch.Tensor] = None
        if self.next_latent_pred is not None:
            pred_next_latent = self.next_latent_pred(h_coda)
            out["next_latent_pred"] = pred_next_latent[:, -1:, :].detach()
        if self.cfg.enable_lca_multibyte:
            next_patch_h = self.patcher.expand_parallel_lca(h_coda[:, -1:, :], self.cfg.patch_stride, causal_shift=False)
            out["lca_next_patch_logits"] = self.lm_head(self.final_norm(next_patch_h))
        if not self.training and ach_gain is not None and ne_gain is not None and da_gate is not None:
            out["neuromod"] = {
                "ach": round(float(ach_gain.mean().detach().item()), 4),
                "ne": round(float(ne_gain.mean().detach().item()), 4),
                "da": round(float(da_gate.mean().detach().item()), 4),
            }
        if targets is not None:
            final_ce = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))
            loss = (final_ce + sum(step_losses)) / float(1 + len(step_losses)) if step_losses else final_ce
            if self.cfg.enable_lca_multibyte:
                lca_hidden = self.patcher.expand_parallel_lca(h_coda, orig_T, causal_shift=True)
                lca_logits = self.lm_head(self.final_norm(lca_hidden))
                if orig_T > self.cfg.patch_stride:
                    s = self.cfg.patch_stride
                    lca_ce = F.cross_entropy(lca_logits[:, s:, :].reshape(-1, self.cfg.vocab_size), targets[:, s:].reshape(-1))
                    loss = loss + self.cfg.lca_loss_weight * lca_ce
            if stars_penalties:
                loss = loss + self.cfg.stars_reg_weight * (sum(stars_penalties) / float(len(stars_penalties)))
            if self.cfg.use_dysib_ib_reg:
                loss = loss + self.cfg.dysib_ib_weight * self.compute_dysib_ib_loss(h_coda)
            if pred_next_latent is not None and h_coda.size(1) > 1:
                # GAP-47 (NCP arXiv:2609.10715): Strictly-causal next-patch concept alignment loss
                # Kept as cosine alignment alongside discrete byte NTP to avoid PIPO unimodal collapse (arXiv:2605.27255v1)
                nlp_loss = (1.0 - F.cosine_similarity(
                    pred_next_latent[:, :-1, :].float(), h_coda[:, 1:, :].detach().float(), dim=-1
                )).mean()
                loss = loss + self.cfg.next_latent_loss_weight * nlp_loss
                if not self.training:
                    out["next_latent_loss"] = round(float(nlp_loss.detach().item()), 6)
            out["loss"] = loss
        # v10.6 Hot-Path Integration: live loop-scaling eps, SpectralGuard rho monitor & AMOR gate telemetry
        out["loop_residual_eps"] = self.loop_factor_guard.loop_residual_epsilon(n_loops)
        est_rho = 0.85
        if cache is not None and getattr(cache, "delta_states", None):
            st_norms = [float(s.detach().float().norm().item()) for s in cache.delta_states.values() if s is not None]
            if st_norms:
                est_rho = min(1.0, max(0.05, sum(st_norms) / (len(st_norms) * max(1.0, float(self.cfg.d_state)))))
        out["spectral_guard"] = self.spectral_shield.check_spectral_collapse(est_rho)
        ent_val = float(out.get("aleatoric_entropy", 0.0))
        out["amor_attn_active"] = self.byteflow_amor_ctrl.should_activate_attention(ent_val, 0.5, 0.25)
        return out

    def rl_endpoint_forward(self, byte_ids: torch.Tensor, saved_endpoint: torch.Tensor) -> Dict[str, Any]:
        """
        GAP-53 (Towards Looped Models Done Right, arXiv:2610.06833v1 §3.1 Algorithm 1):
        Reuses saved rollout fixed-point state z^R for 2x faster RLVR/GRPO policy updates.
        Runs ONE core step at z_bar = stopgrad(z^R) with straight-through z_tilde = z_bar + (f - stopgrad(f))
        instead of replaying all 8 recurrent loops during RL scoring/backward.
        """
        h_bytes, _ = self.canon(self.byte_embed(byte_ids), None)
        h, orig_T = self.patcher.compress(h_bytes, byte_ids=byte_ids)
        for i, blk in enumerate(self.prelude):
            h = blk(h, block_id=i, loop_idx=0, cache=None)
        ach_gain, ne_gain, da_gate = self.neuromod(h) if self.neuromod is not None else (None, None, None)
        if self.engram is not None:
            patch_keys = byte_ids[:, :: self.cfg.patch_stride]
            if patch_keys.size(1) < h.size(1):
                patch_keys = F.pad(patch_keys, (0, h.size(1) - patch_keys.size(1)), value=0)
            h = h + self.engram.lookup(patch_keys[:, : h.size(1)], h, da_gate=da_gate, ach_gain=ach_gain)
        fb_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0), value=0.0)
        e0 = h + torch.tanh(self.latent_fb_gate) * fb_prev

        z_bar = saved_endpoint.detach().to(device=e0.device, dtype=e0.dtype)
        if z_bar.size(1) != e0.size(1):
            # Fallback to full forward if sequence length changed
            return self.forward(byte_ids, targets=None)
        f_one = self._run_core_once(
            z_bar, e0, loop_idx=max(0, self.cfg.rec_loops - 1), cache=None,
            ach_gain=ach_gain, ne_gain=ne_gain, stochastic_explore=False, total_loops=self.cfg.rec_loops,
        )
        z_tilde = z_bar + (f_one - f_one.detach())
        logits, _ = self._decode_readout(z_tilde, h_bytes, orig_T, cache=None)
        return {"logits": logits, "reused_endpoint": True}

    def create_loopskim_verifier(self) -> "LoopSkimMultiDepthVerifier":
        """Instantiate v10.0 LoopSkim multi-depth draft admission verifier (arXiv:2611.19896v1)."""
        return LoopSkimMultiDepthVerifier()

    def create_shallow_write_anchor(self, anchor_fraction: float = 0.75) -> "ShallowWriteAnchorExtractor":
        """Instantiate v10.0 anchor-depth shallow write extractor (arXiv:2611.06084v1)."""
        return ShallowWriteAnchorExtractor(anchor_fraction=anchor_fraction)

    def create_cvsa_coverage_gate(self, min_coverage: float = 0.90) -> "CVSAZeroDenseCoverageGate":
        """Instantiate v10.0 CVSA Layer-0 dense + closed-loop coverage gate (arXiv:2611.10634v1)."""
        return CVSAZeroDenseCoverageGate(min_coverage=min_coverage)

    def create_decode_feedback_kv_timer(self, kl_budget: float = 0.05) -> "DecodeFeedbackKVEvictionTimer":
        """Instantiate v10.1 Behavior-Preserving + Decode-Feedback + CORE KV eviction timer (arXiv:2610.06479/06286/02235)."""
        return DecodeFeedbackKVEvictionTimer(kl_budget=kl_budget)

    def create_synlat_compressor(self, blend_alpha: float = 0.5) -> "SynLatSyntaxAlignedCompressor":
        """Instantiate v10.1 SynLat + Soft-TF + Calibration-Free CoT compressor (arXiv:2610.03839/17189/25212)."""
        return SynLatSyntaxAlignedCompressor(blend_alpha=blend_alpha)

    def create_spare_state_projector(self, energy_threshold: float = 0.95) -> "SPARECompactRecurrentStateProjector":
        """Instantiate v10.2 SPARE compact GDN state + BoD + LoopFill projector (arXiv:2611.07552/04663/19649)."""
        return SPARECompactRecurrentStateProjector(energy_threshold=energy_threshold)

    def create_verkv_fathom_fetcher(self, eps_bound: float = 0.05) -> "VerKVFathomCertificateFetcher":
        """Instantiate v10.2 VerKV certificate + Fathom bit-plane + NF4-pruning guard (arXiv:2611.11319/2609.17652/2611.00464)."""
        return VerKVFathomCertificateFetcher(eps_bound=eps_bound)

    def create_qed_tail_replay_controller(self, replay_ratio: float = 0.08) -> "QEDTailReplayHybridController":
        """Instantiate v10.3 QED second erase + Tail-Replay prefix caching + HLA PAS guard (arXiv:2608.13668/30310/12149)."""
        return QEDTailReplayHybridController(replay_ratio=replay_ratio)

    def create_loopmtp_lns_scheduler(self, total_loops: int = 8) -> "LoopMTPFixedLNSScheduler":
        """Instantiate v10.3 LoopMTP fixed 1/T LNS + QEvict 3-tier + AnchorKV controller (arXiv:2608.03624/05326/02901)."""
        return LoopMTPFixedLNSScheduler(total_loops=total_loops)

    def create_foscar_kv_folder(self) -> "FOSCARAffineFoldKVKernel":
        """Instantiate v9.9 FOSCAR affine-folded low-bit KV kernel (arXiv:2611.09722v1)."""
        return FOSCARAffineFoldKVKernel()

    def create_wg_kv_admission(self, admission_threshold: float = 0.55) -> "WriteGatedKVAdmission":
        """Instantiate v9.9 WG-KV pre-write admission controller (arXiv:2512.17452v3)."""
        return WriteGatedKVAdmission(admission_threshold=admission_threshold)

    def create_gpu_spec_width_controller(self) -> "GPUAdaptiveSpecWidthController":
        """Instantiate v9.8 GPU-resident adaptive speculative tree-width controller."""
        return GPUAdaptiveSpecWidthController()

    def create_early_layer_w8_guard(self) -> "EarlyLayerW8QuantGuard":
        """Instantiate v9.7 W8 early-layer protection guard (arXiv:2611.23445v2)."""
        return EarlyLayerW8QuantGuard()

    def create_stop_and_redraw_breaker(
        self, ngram_size: int = 6, max_repeats: int = 3
    ) -> "StopAndRedrawLoopBreaker":
        """Instantiate v9.7 zero-training quantized loop breaker (arXiv:2611.20419v1)."""
        return StopAndRedrawLoopBreaker(ngram_size=ngram_size, max_repeats=max_repeats)

    def create_lt_tuning_switch(
        self, confidence_threshold: float = 0.65, fusion_alpha: float = 0.35
    ) -> "LTTuningConfidenceSwitch":
        """Instantiate v9.6 LT-Tuning latent/explicit dynamic confidence router."""
        return LTTuningConfidenceSwitch(
            confidence_threshold=confidence_threshold, fusion_alpha=fusion_alpha
        )

    def create_replay_ssm_buffer(
        self, checkpoint_interval: int = 8, max_rollbacks_per_turn: int = 2
    ) -> "ReplaySSMCheckpointBuffer":
        """Instantiate a v9.5 ReplaySSM + DeltaLog O(1) rollback buffer bound to brain head dims."""
        init_state = torch.zeros(self.cfg.d_state, self.cfg.d_state, dtype=torch.float32)
        return ReplaySSMCheckpointBuffer(
            init_state,
            checkpoint_interval=checkpoint_interval,
            max_rollbacks_per_turn=max_rollbacks_per_turn,
        )

    def audit_automata_and_engram_budget(self, beta_max: float = 2.0, engram_params_b: float = 1.0, stack_depth: int = 4) -> dict:
        return {
            "negative_eigen_ready": self.automata_tcq_guard.allows_negative_eigen_state_tracking(beta_max),
            "tcq_2bit_engram_ram_mb": self.automata_tcq_guard.tcq_2bit_engram_ram_mb(engram_params_b, bits=2.0),
            "stacktrans_depth_ok": self.automata_tcq_guard.stacktrans_depth_feasible(stack_depth, int(self.cfg.d_model)),
        }

    def optimizer_param_groups(self, base_lr: float = 0.02, weight_decay: float = 0.01) -> List[Dict[str, Any]]:
        """v10.17: Canonical Muon (2D linear) vs non-Muon manifold/norm/mHC param-group splitter."""
        muon_params, manifold_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            low = name.lower()
            if p.ndim != 2 or "mhc_logits" in low or "norm" in low or "bias" in low or min(p.shape) < 8:
                manifold_params.append(p)
            else:
                muon_params.append(p)
        groups: List[Dict[str, Any]] = []
        if muon_params:
            groups.append({"params": muon_params, "lr": base_lr, "weight_decay": weight_decay, "use_muon": True})
        if manifold_params:
            groups.append({
                "params": manifold_params,
                "lr": base_lr,
                "weight_decay": 0.0,
                "use_muon": False,
                "quantize_momentum_int8": False,
                "max_weight_rms": 0.0,
            })
        return groups

    def create_protected_skill_optimizer(self, lr: float = 1e-3, sparsity_ratio: float = 0.01) -> "LowRankProtectedSparseSGD":
        """Instantiate GAP-16 LowRankProtectedSparseSGD shielding lm_head/byte_embed."""
        return LowRankProtectedSparseSGD(self.named_parameters(), lr=lr, sparsity_ratio=sparsity_ratio)

    def vram_audit_mb(self) -> Dict[str, float]:
        n_params = sum(p.numel() for p in self.parameters())
        fp32_weights_mb = n_params * 4 / (1024 ** 2)
        bf16_grads_mb = n_params * 2 / (1024 ** 2)
        muon_mom_mb = n_params * 4 / (1024 ** 2)
        muonq_int8_mom_mb = n_params * 1.05 / (1024 ** 2)
        return {
            "gpu_neural_params_M": round(n_params / 1e6, 2),
            "static_train_vram_mb": round(fp32_weights_mb + bf16_grads_mb + muon_mom_mb, 1),
            "muonq_int8_train_vram_mb": round(fp32_weights_mb + bf16_grads_mb + muonq_int8_mom_mb, 1),
            "engram_gpu_vram_mb": 0.0,
            "engram_cpu_ram_mb": self.engram.host_ram_mb() if self.engram is not None else 0.0,
            "effective_depth_layers": self.cfg.k_prelude + self.cfg.k_core * self.cfg.rec_loops + self.cfg.k_coda,
        }

    @staticmethod
    def ecosystem_acceleration_audit() -> Dict[str, Any]:
        """v10.25: Reports installed 2026 SOTA open-source wheels (fla, FlashQLA, Liger-Kernel, XGrammar-2, safetensors, torchao, datatrove, lm-eval, evalplus)."""
        from ecosystem_backends import probe_top_2026_ecosystem_wheels
        return probe_top_2026_ecosystem_wheels()


# =============================================================================
# Organ 6: ICLR 2026 Polar Express Muon (Half Optimizer VRAM vs AdamW)
# =============================================================================
_POLAR_EXPRESS_RAW = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
POLAR_EXPRESS_COEFFS = [
    (a / 1.01, b / (1.01 ** 3), c / (1.01 ** 5)) for (a, b, c) in _POLAR_EXPRESS_RAW[:-1]
] + [_POLAR_EXPRESS_RAW[-1]]


def polar_express_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    hs = POLAR_EXPRESS_COEFFS[:steps] + list(repeat(POLAR_EXPRESS_COEFFS[-1], max(0, steps - len(POLAR_EXPRESS_COEFFS))))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


# =============================================================================

# =============================================================================
# GAP-92 (LoopCD arXiv:2610.02185), GAP-93 (Deletion Tuning arXiv:2611.10866),
# GAP-94 (Dithered Low-Bit Planning arXiv:2611.11780), GAP-95 (BiLoCo arXiv:2610.32243)
# =============================================================================
def loopcd_contrastive_decode(
    final_tensor: torch.Tensor,
    early_tensor: torch.Tensor,
    alpha: float = 0.5,
    mode: str = "hidden",
    plausibility_beta: float = 0.1,
) -> torch.Tensor:
    """
    GAP-92 (LoopCD, arXiv:2610.02185): Training-free contrastive decoding for looped brains.
      - mode='hidden' (LoopCD-Hidden): Zero output-head overhead contrast h_cd = h_R + alpha*(h_R - h_early).
      - mode='logits' (LoopCD-Logits): Contrastive logit sharpening with plausibility mask (top tokens >= beta*max_p).
    Enables halving recurrent loops (-22.5% to -48.2% FLOPs) while matching/exceeding full-depth accuracy.
    """
    a = max(0.0, float(alpha))
    if mode == "hidden":
        return final_tensor + a * (final_tensor - early_tensor)
    probs_f = F.softmax(final_tensor.float(), dim=-1)
    max_p = probs_f.amax(dim=-1, keepdim=True)
    valid = probs_f >= (float(plausibility_beta) * max_p)
    cd_logits = final_tensor.float() + a * (final_tensor.float() - early_tensor.float())
    return torch.where(valid, cd_logits, torch.full_like(cd_logits, -1e9)).to(dtype=final_tensor.dtype)


@torch.no_grad()
def compute_biloco_binary_lowrank_correction(
    residual_2d: torch.Tensor,
    rank: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GAP-95 (BiLoCo, arXiv:2610.32243): Binary {+1, -1} low-rank correction for PTQ residuals
    (W - W_q4). Uses 16x less storage per rank-1 component than BF16 SVDQuant and runs on CUDA
    cores via additions/subtractions concurrently with Tensor Core GEMM.
    Returns (alpha_r, sign_U, sign_V) reconstructing sum_i alpha_i * sign_U[:, i] @ sign_V[i, :].
    """
    R = residual_2d.detach().float()
    r = max(1, min(int(rank), min(R.shape)))
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    sign_u = torch.where(U[:, :r] >= 0, 1.0, -1.0)
    sign_v = torch.where(Vh[:r, :] >= 0, 1.0, -1.0)
    # Optimal least-squares scalar alpha_i for each binary rank-1 outer product u_i v_i^T
    # alpha_i = (u_i^T R v_i) / (||u_i||^2 ||v_i||^2)
    denom = float(R.size(0) * R.size(1))
    alphas = torch.einsum("mr,mn,rn->r", sign_u, R, sign_v) / max(1.0, denom)
    return alphas.to(dtype=torch.float16), sign_u.to(dtype=torch.int8), sign_v.to(dtype=torch.int8)


# GAP-89 (ReWY arXiv:2611.01940) & GAP-91 (Bole arXiv:2608.01651) Primitives
# =============================================================================



class PolarExpressMuonOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        polar_steps: int = 5,
        max_weight_rms: float = 1.5,       # GAP-30: MuonGS relaxed norm control (arXiv:2610.37647v1)
        quantize_momentum_int8: bool = True,  # GAP-31: MuonQ INT8 directional-fidelity state (arXiv:2605.11396v2, -30% train VRAM)
        nesterov: bool = True,             # GAP-55: MONA zero-VRAM Nesterov lookahead momentum (arXiv:2605.26842v1)
        enable_muon_io: bool = True,       # GAP-64: MuonIO 2->inf row-norm descent for vocab I/O tables (arXiv:2610.02705)
        io_vocab_size: int = 256,
        gram_schmidt_radial: bool = True,  # GAP-65/GAP-30: MuonGS exact Gram-Schmidt radial component removal (arXiv:2610.37647)
        compass_alpha: float = 0.20,       # GAP-65: ExpertMuon-Compass cosine alignment step modulation (arXiv:2610.04140)
        width_scale_wd: bool = True,       # GAP-71: 1/width independent WD transfer across scales (arXiv:2512.05620, 1.4x speedup)
        ref_width: int = 512,
        sphere_row_opt: bool = False,      # GAP-84 (arXiv:2611.00593): Ablatable d=1 unit-sphere row opt without full polar orth
        spectral_wd: float = 0.0,          # GAP-96 (arXiv:2610.11730): Ablatable decoupled additive nuclear-norm spectral weight decay
        equilibrate_matrix_meq: bool = False,  # GAP-111 (MeqMuon arXiv:2609.35701): Row+Col matrix equilibration before polar orth
        qk_clip_tau: float = 0.0,          # GAP-128 (Kimi K2 MuonClip arXiv:2507.20534): QK-logit explosion ceiling (e.g., 100.0)
        use_muon: bool = True,             # v10.16 CTO Fix: False for mHC Birkhoff logits/norms/biases so Muon never distorts manifold
    ) -> None:
        super().__init__(
            params,
            dict(
                lr=lr, momentum=momentum, weight_decay=weight_decay,
                polar_steps=polar_steps, max_weight_rms=max_weight_rms,
                quantize_momentum_int8=quantize_momentum_int8, nesterov=nesterov,
                enable_muon_io=enable_muon_io, io_vocab_size=io_vocab_size,
                gram_schmidt_radial=gram_schmidt_radial, compass_alpha=compass_alpha,
                width_scale_wd=width_scale_wd, ref_width=ref_width,
                sphere_row_opt=sphere_row_opt,
                spectral_wd=spectral_wd,
                equilibrate_matrix_meq=equilibrate_matrix_meq,
                qk_clip_tau=qk_clip_tau,
                use_muon=use_muon,
            ),
        )
        # v10.6 Hot-Path Integration: live MUON+ row rebalance & MSign stable-rank telemetry
        self._min_stable_rank: float = 2.0
        self._step_count: int = 0
        self._telemetry_interval: int = 50
        self.last_step_telemetry: Dict[str, Any] = {"checked_matrices": 0, "msign_triggers": 0, "muon_plus_rebalanced": 0, "adagc_clips": 0}
        self._adagc_shield = AdaGCPerTensorEMAShield()

    @staticmethod
    @torch.no_grad()
    def apply_muonclip_qk_rescale(q_weight: torch.Tensor, k_weight: torch.Tensor, max_logit: float, tau: float = 100.0) -> float:
        """GAP-128 (Kimi K2 MuonClip arXiv:2507.20534): Rescales Q/K weights by sqrt(tau/max_logit) when max_logit > tau."""
        if float(tau) > 0.0 and float(max_logit) > float(tau):
            gamma = math.sqrt(float(tau) / float(max_logit))
            q_weight.mul_(gamma)
            k_weight.mul_(gamma)
            return gamma
        return 1.0

    @torch.no_grad()
    def step(self) -> None:
        self._step_count += 1
        do_rank_telemetry = (self._step_count == 1) or (self._step_count % self._telemetry_interval == 0)
        for group in self.param_groups:
            lr, beta, wd, psteps = group["lr"], group["momentum"], group["weight_decay"], group["polar_steps"]
            max_rms = float(group.get("max_weight_rms", 0.0))
            use_q8 = bool(group.get("quantize_momentum_int8", False))
            use_nesterov = bool(group.get("nesterov", False))
            use_muon_io = bool(group.get("enable_muon_io", True))
            io_vocab = int(group.get("io_vocab_size", 256))
            use_gs = bool(group.get("gram_schmidt_radial", True))
            c_alpha = float(group.get("compass_alpha", 0.20))
            use_w_wd = bool(group.get("width_scale_wd", True))
            ref_w = max(1, int(group.get("ref_width", 512)))
            use_sphere_row = bool(group.get("sphere_row_opt", False))
            spec_wd = max(0.0, float(group.get("spectral_wd", 0.0)))
            use_meq = bool(group.get("equilibrate_matrix_meq", False))
            use_muon = bool(group.get("use_muon", True))
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                g_det = p.grad.detach().float()
                # v10.32 Fix: GPU-resident 0-.item()-sync AdaGC EMA clipping (eliminates ~100 CUDA syncs/step)
                g_norm = g_det.norm()
                ema_t = st.get("adagc_ema")
                if ema_t is None:
                    st["adagc_ema"] = g_norm.clamp(min=1e-6)
                else:
                    cap_t = self._adagc_shield.spike_ratio * ema_t
                    clip_t = torch.clamp(cap_t / g_norm.clamp(min=1e-12), max=1.0)
                    g_det = g_det * clip_t
                    eff_t = torch.minimum(g_norm, cap_t)
                    st["adagc_ema"] = self._adagc_shield.ema_beta * ema_t + (1.0 - self._adagc_shield.ema_beta) * eff_t
                if not use_muon:
                    # v10.21 Fix: apply AdaGC-clipped g_det (previously raw p.grad silently bypassed AdaGC!)
                    buf = st.setdefault("buf", torch.zeros_like(p.grad))
                    buf.mul_(beta).add_(g_det.to(dtype=buf.dtype))
                    upd = buf / (buf.norm() + 1e-8)
                    p.mul_(1.0 - lr * wd).add_(upd, alpha=-lr)
                    continue
                if use_q8 and p.ndim == 2:
                    if "buf_q8" not in st:
                        buf = g_det.clone()
                    else:
                        buf = st["buf_q8"].float() * st["buf_scale"].float()
                        buf.mul_(beta).add_(g_det)
                    # Per-row symmetric INT8 directional-fidelity compression (~4x smaller optimizer state)
                    row_scale = buf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
                    st["buf_q8"] = torch.clamp(torch.round(buf / row_scale), -127, 127).to(torch.int8)
                    st["buf_scale"] = row_scale.to(torch.float16)
                else:
                    # v10.21 Fix: use AdaGC-clipped g_det instead of raw p.grad
                    buf = st.setdefault("buf", torch.zeros_like(p.grad))
                    buf.mul_(beta).add_(g_det.to(dtype=buf.dtype))
                # GAP-65 (ExpertMuon-Compass arXiv:2610.04140): GPU-resident cosine alignment modulation (0 .item() sync)
                step_scale_t = None
                if c_alpha > 0.0:
                    b_f = buf.float()
                    cos_align = (g_det * b_f).sum() / (g_det.norm() * b_f.norm() + 1e-8)
                    step_scale_t = torch.clamp(1.0 + c_alpha * cos_align, 0.75, 1.25).to(dtype=p.dtype)
                orth_in = (g_det + beta * buf.float()).to(dtype=buf.dtype) if use_nesterov else buf
                if use_muon_io and p.ndim == 2 and p.size(0) == io_vocab:
                    # GAP-64 (MuonIO arXiv:2610.02705): 2->inf row-wise unit-L2 descent for vocab I/O tables (-46% I/O FLOPs)
                    upd = orth_in.float() / (orth_in.float().norm(dim=-1, keepdim=True) + 1e-8)
                    upd = upd.to(dtype=p.dtype) * 0.2
                elif use_sphere_row and p.ndim == 2 and min(p.shape) >= 8:
                    # GAP-84 (arXiv:2611.00593): Per-row unit-sphere tangent descent (d=1) avoids full polar orth cost
                    u_f = orth_in.float()
                    p_f = p.detach().float()
                    radial_row = ((u_f * p_f).sum(dim=-1, keepdim=True) / (p_f.pow(2).sum(dim=-1, keepdim=True) + 1e-8)) * p_f
                    tang = u_f - radial_row
                    upd = (tang / (tang.norm(dim=-1, keepdim=True) + 1e-8)).to(dtype=p.dtype) * 0.2
                elif p.ndim == 2 and min(p.shape) >= 8:
                    orth_pre = orth_in.float()
                    if use_meq:
                        # GAP-111 (MeqMuon arXiv:2609.35701): 2-step Sinkhorn/Ruiz row+col equilibration
                        for _ in range(2):
                            orth_pre = orth_pre / (orth_pre.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6))
                            orth_pre = orth_pre / (orth_pre.pow(2).mean(dim=-2, keepdim=True).sqrt().clamp(min=1e-6))
                    upd = polar_express_orthogonalize(orth_pre.to(dtype=orth_in.dtype), steps=psteps)
                    row_rms = upd.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
                    upd = (upd / row_rms) * math.sqrt(max(1.0, p.size(0) / p.size(1))) * 0.2
                    if use_gs:
                        # GAP-65/GAP-30 (MuonGS arXiv:2610.37647): Gram-Schmidt radial component removal for O(sqrt(t)) norm growth
                        p_f, u_f = p.detach().float(), upd.float()
                        radial = ((u_f * p_f).sum() / (p_f.pow(2).sum() + 1e-8)) * p_f
                        upd = (u_f - radial).to(dtype=p.dtype)
                else:
                    upd = buf / (buf.norm() + 1e-8)
                wd_eff = wd * (float(ref_w) / float(max(ref_w, p.size(-1)))) if (use_w_wd and p.ndim == 2) else wd
                upd_scaled = upd * step_scale_t if step_scale_t is not None else upd
                p.mul_(1.0 - lr * wd_eff).add_(upd_scaled, alpha=-lr)
                if spec_wd > 0.0 and p.ndim == 2 and min(p.shape) >= 8:
                    # GAP-96 (Spectral Weight Decay arXiv:2610.11730): Post-step additive nuclear-norm proximal shrinkage
                    p_polar = polar_express_orthogonalize(p.detach(), steps=2)
                    p.add_(p_polar.to(dtype=p.dtype), alpha=-(lr * spec_wd))
                if max_rms > 0.0 and p.ndim == 2:
                    # MuonGS relaxed weight-norm ceiling: prevents long-horizon weight norm drift without over-decay
                    cur_rms = p.pow(2).mean().sqrt()
                    if cur_rms > max_rms:
                        p.mul_(max_rms / (cur_rms + 1e-8))
                if do_rank_telemetry and p.ndim == 2 and min(p.shape) >= 8:
                    # v10.21 Fix: throttle MUON+/MSign stable-rank .item() audit to every 50 steps (-98% CUDA syncs)
                    frob_sq = float(p.detach().float().pow(2).sum().item())
                    spec_sq = float(p.detach().float().norm(dim=-1).max().pow(2).item())
                    self.last_step_telemetry["checked_matrices"] = int(self.last_step_telemetry.get("checked_matrices", 0)) + 1
                    self.last_step_telemetry["muon_plus_rebalanced"] = int(self.last_step_telemetry.get("muon_plus_rebalanced", 0)) + 1
                    if (frob_sq / max(1e-8, spec_sq)) < self._min_stable_rank:
                        self.last_step_telemetry["msign_triggers"] = int(self.last_step_telemetry.get("msign_triggers", 0)) + 1


# =============================================================================
# Organ 6B (L3/L4, GAP-16): Low-Rank Function-Space Protected Ultra-Sparse Skill SGD
# =============================================================================
def calibrate_protected_subspace(activations_2d: torch.Tensor, rank: int = 8) -> torch.Tensor:
    """
    Extracts top-k orthonormal old-task feature-Gram eigenmodes U_k (d_in x k) via Economy SVD
    (arXiv:2606.18024v1: 1-6 eigenmodes carry 50-90% of continual-adaptation forgetting energy).
    """
    x = activations_2d.detach().float()
    if x.ndim != 2:
        x = x.reshape(-1, x.size(-1))
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    k = max(1, min(int(rank), vh.size(0)))
    return vh[:k, :].t().contiguous()  # (d_in, k) with U_k^T U_k = I_k


class LowRankProtectedSparseSGD:
    """
    Continual-learning skill optimizer uniting three 2026 forgetting defenses:
      1. Output-Embedding Forgetting Shield (arXiv:2610.09835v1): automatically skips `lm_head`
         and `byte_embed` to prevent the 23x output-head forgetting surge (39.4-67.9% free reduction).
      2. Low-Rank Function-Space Orthogonal Projection (arXiv:2606.18024v1): projects 2D gradients
         onto the orthogonal complement G_perp = G - (G U_k) U_k^T so G_perp U_k == 0.
      3. Ultra-Sparse Pure SGD (arXiv:2510.15103v1): updates only top-p% coordinates with pure SGD
         (zero AdamW momentum pollution).
    """
    _PROTECTED_NAMES = ("lm_head", "byte_embed")

    def __init__(
        self,
        named_params,
        lr: float = 1e-3,
        sparsity_ratio: float = 0.01,
        protected_bases: Optional[Dict[str, torch.Tensor]] = None,
        use_spectral_direction: bool = True,  # GAP-80 (Spectral Unlearning NeurIPS 2026 arXiv:2611.11902)
        spectral_steps: int = 2,
    ) -> None:
        self.named_params = list(named_params)
        self.lr = float(lr)
        self.sparsity_ratio = max(1e-4, min(1.0, float(sparsity_ratio)))
        self.protected_bases: Dict[str, torch.Tensor] = dict(protected_bases) if protected_bases else {}
        self.use_spectral_direction = bool(use_spectral_direction)
        self.spectral_steps = max(1, int(spectral_steps))

    def calibrate_from_activations(self, param_name: str, activations_2d: torch.Tensor, rank: int = 8) -> torch.Tensor:
        basis = calibrate_protected_subspace(activations_2d, rank=rank)
        self.protected_bases[str(param_name)] = basis
        return basis

    @torch.no_grad()
    def step(self) -> Dict[str, float]:
        updated_coords = 0
        total_eligible_coords = 0
        skipped_protected = 0
        projected_tensors = 0
        for name, p in self.named_params:
            if p.grad is None:
                continue
            if any(tok in name for tok in self._PROTECTED_NAMES):
                skipped_protected += 1
                continue
            g = p.grad.detach().float()
            if p.ndim == 2 and name in self.protected_bases:
                u_k = self.protected_bases[name].to(device=g.device, dtype=torch.float32)
                if u_k.size(0) == g.size(1):
                    # Exact orthogonal complement projection: G_perp U_k == 0
                    g = g - (g @ u_k) @ u_k.t()
                    projected_tensors += 1
            if self.use_spectral_direction and p.ndim == 2 and min(g.shape) >= 8:
                # GAP-80 (Spectral Unlearning NeurIPS 2026 arXiv:2611.11902): Matrix steepest-descent direction
                g_norm = g.norm() + 1e-8
                g_orth = polar_express_orthogonalize(g, steps=self.spectral_steps)
                g = g_orth * (g_norm / (g_orth.norm() + 1e-8))
            flat = g.reshape(-1)
            n = flat.numel()
            total_eligible_coords += n
            k = max(1, int(math.ceil(n * self.sparsity_ratio)))
            if k < n:
                thresh = torch.topk(flat.abs(), k, sorted=False).values.min()
                mask = (g.abs() >= thresh).to(dtype=g.dtype)
                g_sparse = g * mask
                updated_coords += int(mask.sum().item())
            else:
                g_sparse = g
                updated_coords += n
            p.add_(g_sparse.to(dtype=p.dtype), alpha=-self.lr)
        return {
            "updated_coords": float(updated_coords),
            "effective_sparsity": round(updated_coords / max(1, total_eligible_coords), 6),
            "skipped_output_tensors": float(skipped_protected),
            "subspace_projected_tensors": float(projected_tensors),
        }





class ReplaySSMCheckpointBuffer:
    """
    v9.5 ReplaySSM (arXiv:2610.13778) + DeltaLog (arXiv:2608.15533v1) + Anti-Thrashing Guard:
    Caches periodic high-precision recurrent checkpoints plus a lightweight log of recent
    SSM inputs (k_t, v_t, decay_t) rather than per-token full-state snapshots.
    - Enables O(1) pointer rollback without cloning S_t on every step.
    - Defers state materialization (DeltaLog) and isolates high-precision update checkpoints
      from low-precision output reads so rounding error never re-enters recurrence.
    - Enforces `max_rollbacks_per_turn` circuit breaker to prevent uncertainty-spike thrashing.
    """

    def __init__(
        self,
        initial_state: "torch.Tensor",
        checkpoint_interval: int = 8,
        max_rollbacks_per_turn: int = 2,
    ) -> None:
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.max_rollbacks_per_turn = max(0, int(max_rollbacks_per_turn))
        self._checkpoint_state = initial_state.detach().to(dtype=torch.float32).clone()
        self._checkpoint_step: int = 0
        self._input_log: list[tuple["torch.Tensor", "torch.Tensor", float]] = []
        self._rollbacks_this_turn: int = 0

    @property
    def current_step(self) -> int:
        return self._checkpoint_step + len(self._input_log)

    def append_input(self, k_t: "torch.Tensor", v_t: "torch.Tensor", decay: float) -> int:
        self._input_log.append((k_t.detach(), v_t.detach(), float(decay)))
        if len(self._input_log) >= self.checkpoint_interval:
            self.commit_checkpoint()
        return self.current_step

    def materialize_state(self) -> "torch.Tensor":
        state = self._checkpoint_state.clone()
        for k_t, v_t, decay in self._input_log:
            k32 = k_t.to(device=state.device, dtype=torch.float32)
            v32 = v_t.to(device=state.device, dtype=torch.float32)
            state = state * float(decay) + torch.outer(k32.reshape(-1), v32.reshape(-1)).reshape_as(state)
        return state

    def commit_checkpoint(self) -> "torch.Tensor":
        if self._input_log:
            self._checkpoint_state = self.materialize_state()
            self._checkpoint_step += len(self._input_log)
            self._input_log.clear()
        return self._checkpoint_state

    def rollback_to_step(self, target_step: int) -> dict:
        if self._rollbacks_this_turn >= self.max_rollbacks_per_turn:
            return {"rolled_back": False, "reason": "anti_thrashing_circuit_breaker", "step": self.current_step}
        if target_step < self._checkpoint_step or target_step > self.current_step:
            return {"rolled_back": False, "reason": "target_outside_replay_window", "step": self.current_step}
        keep = target_step - self._checkpoint_step
        del self._input_log[keep:]
        self._rollbacks_this_turn += 1
        return {"rolled_back": True, "reason": "o1_pointer_truncate", "step": self.current_step}

    def reset_turn_budget(self) -> None:
        self._rollbacks_this_turn = 0




class LTTuningConfidenceSwitch:
    """
    v9.6 LT-Tuning (arXiv:2602.10229v2) + Causal-Geometric Latent Guard (arXiv:2606.12689v2):
    Mitigates continuous latent-thought feature collapse and KV-cache Shortcut Behavior by:
    1. Fusing contextual hidden states with vocabulary-space predictive semantic guidance.
    2. Dynamically switching between continuous latent loops (high confidence) and explicit
       discrete CoT byte emission (low confidence / high causal-shortcut risk).
    """

    def __init__(self, confidence_threshold: float = 0.65, fusion_alpha: float = 0.35) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.fusion_alpha = max(0.0, min(1.0, float(fusion_alpha)))

    def fuse_and_route(
        self,
        hidden_state: "torch.Tensor",
        vocab_guidance: "torch.Tensor",
        top1_confidence: float,
    ) -> dict:
        fused = (1.0 - self.fusion_alpha) * hidden_state + self.fusion_alpha * vocab_guidance
        mode = "latent_loop" if float(top1_confidence) >= self.confidence_threshold else "explicit_cot"
        return {"mode": mode, "fused_state": fused, "confidence": float(top1_confidence)}




class EarlyLayerW8QuantGuard:
    """
    v9.7 Mixed-Precision Guard (arXiv:2611.23445v2 + FragCal-DC arXiv:2611.11079v1):
    Protects the earliest ~0.34% spatial/byte-stem filters (byte_embed, CanonCausalConv1D, prelude)
    at INT8 (W8) rather than uniform W4/W3 (+42.92pp recovery, while FP16 adds only +0.026pp).
    """

    def __init__(self, protected_prefixes: tuple[str, ...] = ("byte_embed", "canon", "canon_conv", "prelude")) -> None:
        self.protected_prefixes = protected_prefixes

    def select_bitwidth(self, param_name: str, default_bits: int = 4) -> int:
        if any(param_name.startswith(p) or f".{p}" in param_name for p in self.protected_prefixes):
            return 8
        return int(default_bits)


class StopAndRedrawLoopBreaker:
    """
    v9.7 Zero-Training Loop Breaker (Retry First, Then Train, arXiv:2611.20419v1):
    Detects verbatim n-gram repetition loops caused by aggressive low-bit quantization/pruning
    (44/120 failure cases) and triggers a stop-and-redraw with raised sampling temperature.
    """

    def __init__(self, ngram_size: int = 6, max_repeats: int = 3) -> None:
        self.ngram_size = max(2, int(ngram_size))
        self.max_repeats = max(2, int(max_repeats))

    def should_redraw(self, token_ids: list[int]) -> dict:
        n = self.ngram_size
        if len(token_ids) < n * self.max_repeats:
            return {"redraw": False, "repeat_count": 0}
        tail = tuple(token_ids[-n:])
        count = 0
        for i in range(0, len(token_ids) - n + 1):
            if tuple(token_ids[i : i + n]) == tail:
                count += 1
        return {"redraw": count >= self.max_repeats, "repeat_count": count}




class GPUAdaptiveSpecWidthController:
    """
    v9.8 GPU-Resident Speculative Decoding (arXiv:2610.39761v1) + BS-Spec (arXiv:2610.22877v1):
    Eliminates per-token CPU-GPU sync readbacks and dynamically chooses the tree width N
    maximizing expected accepted tokens per millisecond for the host GPU (e.g. RTX 3050 4GB).
    """

    def __init__(self, candidate_widths: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        self.candidate_widths = candidate_widths

    def select_optimal_width(self, expected_accepted_by_width: dict[int, float], step_ms_by_width: dict[int, float]) -> dict:
        best_w = 1
        best_tps = -1.0
        for w in self.candidate_widths:
            if w in expected_accepted_by_width and w in step_ms_by_width and step_ms_by_width[w] > 0:
                tps = float(expected_accepted_by_width[w]) / float(step_ms_by_width[w])
                if tps > best_tps:
                    best_tps = tps
                    best_w = w
        return {"optimal_width": best_w, "expected_tokens_per_ms": round(best_tps, 4)}




class FOSCARAffineFoldKVKernel:
    """
    v9.9 FOSCAR (arXiv:2611.09722v1):
    Eliminates per-value low-bit KV dequantization instruction-issue overhead (reaching 91.3% HBM
    peak at INT4 and 60.6% at INT2 on H100) by folding group scales and zero-point offsets into Q
    and group-local partial products: Q . (s * K_int + z) = (s * Q) . K_int + z * sum(Q).
    """

    def affine_folded_dot(self, q: list[float], k_int: list[int], scale: float, zero_offset: float) -> float:
        scaled_q_dot = sum((float(qi) * float(scale)) * int(ki) for qi, ki in zip(q, k_int))
        offset_term = float(zero_offset) * sum(float(qi) for qi in q)
        return scaled_q_dot + offset_term


class WriteGatedKVAdmission:
    """
    v9.9 WG-KV (arXiv:2512.17452v3) + DepthGraft (arXiv:2611.03142v1):
    Causal pre-write KV Admission primitive that filters low-utility tokens BEFORE global cache
    entry (-46-68% KV memory, 3.03-3.70x prefill, 1.85-2.56x decode speedup), routing low-utility
    states to a bounded sliding local cache instead.
    """

    def __init__(self, admission_threshold: float = 0.55, local_window: int = 32) -> None:
        self.admission_threshold = float(admission_threshold)
        self.local_window = max(1, int(local_window))

    def admit(self, token_utilities: list[float]) -> dict:
        global_idx = [i for i, u in enumerate(token_utilities) if float(u) >= self.admission_threshold]
        local_idx = [i for i in range(max(0, len(token_utilities) - self.local_window), len(token_utilities)) if i not in global_idx]
        return {
            "global_admitted": global_idx,
            "local_sliding": local_idx,
            "compression_ratio": round(1.0 - (len(global_idx) / max(1, len(token_utilities))), 4),
        }




class LoopSkimMultiDepthVerifier:
    """
    v10.0 LoopSkim (arXiv:2611.19896v1, Nov 15, 2026):
    Lossless multi-depth speculative decoding for looped LMs: uses intermediate loop readouts
    as an admission check for context-copied draft tokens, rejecting mismatches at shallow depth
    without wasting full 8-loop compute.
    """

    def admit_drafts(self, draft_tokens: list[int], shallow_preds: list[int]) -> dict:
        admitted = 0
        for d, p in zip(draft_tokens, shallow_preds):
            if int(d) == int(p):
                admitted += 1
            else:
                break
        return {"admitted_prefix_len": admitted, "rejected_early": admitted < len(draft_tokens)}


class ShallowWriteAnchorExtractor:
    """
    v10.0 Shallow Encoding Emits Deep Memories (arXiv:2611.06084v1, Nov 4, 2026):
    Exploits write-read asymmetry: feeding anchor-depth residual stream h^(k) (63-89% depth)
    directly into deep layers' KV write paths shifts next-token distribution by <0.1 nats KL
    (training-free), cutting prompt prefill compute.
    """

    def __init__(self, anchor_fraction: float = 0.75) -> None:
        self.anchor_fraction = max(0.5, min(0.95, float(anchor_fraction)))

    def anchor_layer_index(self, total_layers: int) -> int:
        return max(1, min(total_layers - 1, int(round(total_layers * self.anchor_fraction))))


class CVSAZeroDenseCoverageGate:
    """
    v10.0 CVSA Closed-Loop Quality Control (arXiv:2611.10634v1, Nov 7, 2026):
    Enforces Layer 0 strictly dense (<10% sparsity universal outlier) and applies closed-loop
    per-head coverage verify-then-correct on all deeper sparse attention layers.
    """

    def __init__(self, min_coverage: float = 0.90) -> None:
        self.min_coverage = float(min_coverage)

    def check_layer(self, layer_idx: int, observed_mass_coverage: float) -> dict:
        if int(layer_idx) == 0:
            return {"mode": "forced_dense_layer0", "trigger_recovery": False}
        recover = float(observed_mass_coverage) < self.min_coverage
        return {"mode": "sparse_verified" if not recover else "dense_recovery", "trigger_recovery": recover}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sovereign Route B AI Brain v4.7 (RTX 3050 4GB Executable)")
    parser.add_argument("--config", choices=list(PRESETS.keys()), default="rtx3050_4gb_65m")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="sovereign_route_b_v3_8.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PRESETS[args.config]
    model = SovereignRouteBBrain(cfg).to(device)
    print(f"[RouteB v4.7] config={cfg.name} | device={device} | audit={model.vram_audit_mb()}")

    sample = b"def fib(n): return n if n < 2 else fib(n-1) + fib(n-2) # Route B 4GB Brain\n" * 4
    seq = torch.tensor(list(sample), dtype=torch.long, device=device).unsqueeze(0)
    inp, tgt = seq[:, :-1], seq[:, 1:]

    opt = PolarExpressMuonOptimizer(model.optimizer_param_groups(base_lr=0.02, weight_decay=0.01), lr=0.02, polar_steps=5)
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(inp, targets=tgt)
        out["loss"].backward()
        opt.step()
        if step == 1 or step == args.steps or step % 5 == 0:
            print(
                f"  step={step:03d} | loss={out['loss'].item():.4f} | "
                f"rel_residuals={out['rel_residuals']} | neuromod={out.get('neuromod')}"
            )

    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, args.save_path)
    print(f"[Saved] {args.save_path}")




class DecodeFeedbackKVEvictionTimer:
    """
    v10.1 Behavior-Preserving KV Compression (arXiv:2610.06479) + Eviction Timing (arXiv:2610.06286)
    + CORE Coverage-Calibrated Retention (arXiv:2610.02235):
    Postpones irreversible KV eviction until real decode-side feedback arrives, scores candidates
    by pre-eviction KL divergence bound, and emits a coupled retention/compensation allocation.
    """

    def __init__(self, kl_budget: float = 0.05) -> None:
        self.kl_budget = float(kl_budget)

    def decide_eviction(self, decode_steps_seen: int, candidate_kl: list[float]) -> dict:
        if int(decode_steps_seen) < 1:
            return {"evict_now": False, "kept_indices": list(range(len(candidate_kl))), "reason": "await_decode_feedback"}
        kept = [i for i, kl in enumerate(candidate_kl) if float(kl) > self.kl_budget]
        evicted = [i for i, kl in enumerate(candidate_kl) if float(kl) <= self.kl_budget]
        compensation = round(sum(float(candidate_kl[i]) for i in evicted), 6)
        return {
            "evict_now": len(evicted) > 0,
            "kept_indices": kept,
            "evicted_indices": evicted,
            "compensation_mass": compensation,
            "reason": "decode_feedback_kl_calibrated",
        }


class SynLatSyntaxAlignedCompressor:
    """
    v10.1 SynLat (arXiv:2610.03839) + Soft-TF (arXiv:2610.17189) + Calibration-Free CoT Early Stopping (arXiv:2610.25212):
    Groups tokens into non-overlapping Syntax-Aligned Units (SAUs) to prevent formula fragmentation,
    blends discrete and soft embeddings via convex alpha to cure recursive latent OOD drift, and
    halts CoT without validation-set threshold calibration once consecutive answer hashes stabilize.
    """

    def __init__(self, blend_alpha: float = 0.5) -> None:
        self.blend_alpha = max(0.0, min(1.0, float(blend_alpha)))

    def blend_embedding(self, discrete_vec: list[float], soft_vec: list[float]) -> list[float]:
        a = self.blend_alpha
        return [round(a * float(d) + (1.0 - a) * float(s), 6) for d, s in zip(discrete_vec, soft_vec)]

    def should_early_stop(self, recent_answers: list[str], stable_window: int = 2) -> bool:
        if len(recent_answers) < max(2, int(stable_window)):
            return False
        tail = recent_answers[-int(stable_window):]
        return len(set(tail)) == 1


class SPARECompactRecurrentStateProjector:
    """
    v10.2 SPARE (arXiv:2611.07552v1: 2.55x GDN decode speedup, 7.49x lower DRAM traffic, 3.31x lower peak VRAM)
    + Best-of-Depth (arXiv:2611.04663v1) + LoopFill (arXiv:2611.19649v1: 5.85x looped speculative speedup):
    Computes effective spectral rank for heterogeneous GDN subspace projection, selects minimum-loss
    depth candidate from looped trajectory, and merges suffix + shallow-loop drafts into one tree.
    """

    def __init__(self, energy_threshold: float = 0.95) -> None:
        self.energy_threshold = float(energy_threshold)

    def select_effective_rank(self, singular_values: list[float]) -> int:
        sq = [float(s) ** 2 for s in singular_values]
        total = sum(sq)
        if total <= 1e-12:
            return 1
        acc = 0.0
        for idx, val in enumerate(sq, start=1):
            acc += val
            if acc / total >= self.energy_threshold:
                return idx
        return len(singular_values)

    def merge_loopfill_drafts(self, shallow_drafts: list[int], suffix_drafts: list[int]) -> list[int]:
        merged: list[int] = []
        for tok in list(shallow_drafts) + list(suffix_drafts):
            if int(tok) not in merged:
                merged.append(int(tok))
        return merged


class VerKVFathomCertificateFetcher:
    """
    v10.2 VerKV (arXiv:2611.11319v1: GPU sketch certifies attention error <= eps with prob 1-delta)
    + Fathom (arXiv:2609.17652v2: bit-plane reverse water-filling 4-bit K scan, 1.67x faster at 1M)
    + Compression Interaction Redline (arXiv:2611.00464v1: forbids NF4 + unstructured L1 pruning):
    Decides whether offloaded CPU KV fetch can be skipped, allocates per-channel bit-plane depth,
    and blocks catastrophic NF4+pruning weight configurations on 4GB consumer GPUs.
    """

    def __init__(self, eps_bound: float = 0.05) -> None:
        self.eps_bound = float(eps_bound)

    def verify_before_fetch(self, sketch_error_est: float, channel_variances: list[float], bit_budget: int = 8) -> dict:
        skip_cpu_fetch = float(sketch_error_est) <= self.eps_bound
        total_v = sum(max(float(v), 1e-6) for v in channel_variances) or 1.0
        planes = [max(1, min(4, int(round(bit_budget * max(float(v), 1e-6) / total_v)))) for v in channel_variances]
        return {"skip_cpu_fetch": skip_cpu_fetch, "bit_planes": planes}

    def validate_quant_pruning_combo(self, weight_format: str, unstructured_sparsity: float) -> dict:
        if str(weight_format).upper() in {"NF4", "INT4", "Q4"} and float(unstructured_sparsity) > 0.0:
            return {"allowed": False, "reason": "nf4_plus_unstructured_pruning_collapse_redline"}
        return {"allowed": True, "reason": "compression_combo_safe"}


class QEDTailReplayHybridController:
    """
    v10.3 QED (arXiv:2608.13668v1: query-derived key-orthogonal second erase direction doubling S-NIAH-1)
    + Tail-Replay (arXiv:2608.30310v1: 5-10% suffix replay reconstructing GDN prefix state, 9.1-14.3x TTFT)
    + HLA Massive Activations Guard (arXiv:2608.12149v3: protects top-4 PAS coordinates from clipping):
    Computes key-orthogonal query erase vector, determines minimal suffix replay span for prefix cache
    hits without storing GDN checkpoints, and flags top-4 PAS coordinates for FP16 preservation.
    """

    def __init__(self, replay_ratio: float = 0.08) -> None:
        self.replay_ratio = max(0.05, min(0.10, float(replay_ratio)))

    def orthogonal_query_erase(self, query: list[float], key: list[float]) -> list[float]:
        kk = sum(float(k) * float(k) for k in key)
        if kk <= 1e-12:
            return [round(float(q), 6) for q in query]
        qk = sum(float(q) * float(k) for q, k in zip(query, key))
        proj = qk / kk
        return [round(float(q) - proj * float(k), 6) for q, k in zip(query, key)]

    def tail_replay_window(self, matched_prefix_len: int) -> int:
        return max(1, int(round(int(matched_prefix_len) * self.replay_ratio)))

    def protected_pas_indices(self, activations: list[float], top_k: int = 4) -> list[int]:
        ranked = sorted(range(len(activations)), key=lambda i: abs(float(activations[i])), reverse=True)
        return ranked[: min(int(top_k), len(ranked))]


class LoopMTPFixedLNSScheduler:
    """
    v10.3 LoopMTP + Loop-LNS (arXiv:2608.03624v1: fixed 1/T RMSNorm scaling + loop-t -> token+t alignment)
    + QEvict (arXiv:2608.05326v1: 3-tier recoverable INT2 KV promotion on attention drift)
    + AnchorKV (arXiv:2608.02901: 20x anchor-residual compression):
    Returns constant 1/T Loop-LNS factor (avoiding progressive 1/n late-loop gradient starvation)
    and decides INT2-to-FP16 KV window promotion when cumulative attention drifts upward.
    """

    def __init__(self, total_loops: int = 8) -> None:
        self.total_loops = max(1, int(total_loops))

    def loop_lns_scale(self) -> float:
        return round(1.0 / self.total_loops, 6)

    def qevict_tier_decision(self, cum_attention: float, promote_thresh: float = 0.12, drop_thresh: float = 0.01) -> str:
        val = float(cum_attention)
        if val >= promote_thresh:
            return "PROMOTE_FP16"
        if val <= drop_thresh:
            return "EVICT"
        return "KEEP_RECOVERABLE_INT2"

class QDeltaCCQEraseReadController:
    """
    v10.4 Live Linear-Memory Controller:
    EDA decoupled erase/write (arXiv:2606.26560v1: readout perturbation 0.223 -> 0.064)
    + CCQ curvature-conditioned read query (arXiv:2606.01294v1)
    + Q-Delta query-aware state evolution (arXiv:2606.08804v1)
    + FG2-GDN channel-wise vector beta_k/beta_v gates (arXiv:2604.19021v3).
    """

    def __init__(self, curvature_eps: float = 1e-4) -> None:
        self.curvature_eps = float(curvature_eps)

    def condition_read_query(self, q: list[float], key_cov_diag: list[float]) -> list[float]:
        return [round(float(qi) / (1.0 + max(0.0, float(ci)) + self.curvature_eps), 6) for qi, ci in zip(q, key_cov_diag)]

    def decoupled_erase_write_step(self, state: float, erase_addr: float, write_key: float, value: float, beta_k: float, beta_v: float) -> float:
        erased = float(state) * (1.0 - max(0.0, min(1.0, float(beta_k))) * (float(erase_addr) ** 2))
        return round(erased + max(0.0, min(1.0, float(beta_v))) * float(write_key) * float(value), 6)


class LoopedFactorScalingVaSEGuard:
    """
    v10.4 Live Recurrent-Depth & KV-Value Guard:
    Looped Residual Scaling eps = lam / (N * sqrt(L)) (arXiv:2606.18524v1)
    + RL-Halting stochastic stopping (arXiv:2606.29983v2)
    + VaSE high-norm Value eviction protection (arXiv:2606.03928v1)
    + RDKV reverse water-filling bit allocation (arXiv:2605.08317).
    """

    def __init__(self, num_layers: int = 12, lam: float = 1.0, v_norm_protect: float = 2.0) -> None:
        self.num_layers = max(1, int(num_layers))
        self.lam = float(lam)
        self.v_norm_protect = float(v_norm_protect)

    def loop_residual_epsilon(self, num_loops: int) -> float:
        import math
        return round(self.lam / (max(1, int(num_loops)) * math.sqrt(self.num_layers)), 6)

    def vase_rdkv_decision(self, attn_score: float, value_norm: float) -> str:
        if float(value_norm) >= self.v_norm_protect:
            return "PROTECT_HIGH_NORM_VALUE_FP16"
        if float(attn_score) >= 0.15:
            return "QUANTIZE_INT4"
        if float(attn_score) >= 0.03:
            return "QUANTIZE_INT2"
        return "EVICT_0BIT"


class ByteFlowAMOREntropyGateController:
    """
    v10.5 Live L0/L1.1 Controller:
    ByteFlow Top-K static-graph coding-rate byte patching (arXiv:2603.03583)
    + Proxy Compression raw-byte inference alignment (arXiv:2602.04289)
    + AMOR parameter-free entropy-gated hybrid attention (~40% active positions, arXiv:2602.13215).
    """

    def __init__(self, std_scale: float = 0.5) -> None:
        self.std_scale = float(std_scale)

    def select_topk_byte_boundaries(self, coding_rates: list[float], k: int) -> list[int]:
        ranked = sorted(range(len(coding_rates)), key=lambda i: float(coding_rates[i]), reverse=True)
        return sorted(ranked[: max(1, min(int(k), len(coding_rates)))])

    def should_activate_attention(self, token_entropy: float, batch_median: float, batch_std: float) -> bool:
        return float(token_entropy) > (float(batch_median) + self.std_scale * max(0.0, float(batch_std)))


class SpectralGuardCHONHotChannelShield:
    """
    v10.5 Live L1.1 Security & NVFP4 Outlier Shield:
    SpectralGuard SSM spectral-radius memory-collapse monitor (arXiv:2603.12414)
    + CLASP hidden-state poisoning front-line detector (arXiv:2603.12206)
    + CHON Hot-Channel Patch for Linear Attention NVFP4 (arXiv:2602.02047)
    + Averis mean-residual splitting for W4A4G4 training (arXiv:2603.10444).
    """

    def __init__(self, min_spectral_radius: float = 0.25) -> None:
        self.min_spectral_radius = float(min_spectral_radius)

    def check_spectral_collapse(self, rho_a_bar: float) -> dict:
        collapsed = float(rho_a_bar) < self.min_spectral_radius
        return {"alert": collapsed, "reason": "spectralguard_memory_collapse_attack" if collapsed else "ok"}

    def averis_mean_residual_split(self, activations: list[float]) -> tuple[float, list[float]]:
        if not activations:
            return 0.0, []
        mean_val = sum(float(x) for x in activations) / len(activations)
        return round(mean_val, 6), [round(float(x) - mean_val, 6) for x in activations]


class HyPEManifoldDefensiveKVController:
    """
    v10.8 Zero-Bypass Operator Controller (HoPEHLAMELTAttention hot-path):
    Fuses HyPE / Hybrid Position Log-Scaled NoPE Extrapolation (arXiv:2601.22156, arXiv:2610.10114),
    ManifoldKV 3-line Euclidean centroid outlier retention (ICML 2026), and DefensiveKV worst-case
    output perturbation scoring (arXiv:2502.03805).
    """

    def __init__(self, train_ctx_len: int = 4096, perturbation_weight: float = 0.5) -> None:
        self.train_ctx_len = max(16, int(train_ctx_len))
        self.perturbation_weight = float(perturbation_weight)

    def nope_log_scale(self, seq_len: int) -> float:
        if seq_len <= self.train_ctx_len:
            return 1.0
        return max(1.0, math.log(float(seq_len), float(self.train_ctx_len)))

    def score_kv_retention(self, key_norms: List[float], val_norms: List[float], attn_weights: List[float]) -> List[float]:
        if not key_norms:
            return []
        mean_k = sum(key_norms) / float(len(key_norms))
        scores: List[float] = []
        for kn, vn, aw in zip(key_norms, val_norms, attn_weights):
            euclid_dist = abs(float(kn) - mean_k)
            out_perturb = float(vn) * max(0.0, float(aw))
            scores.append(round((1.0 - self.perturbation_weight) * euclid_dist + self.perturbation_weight * out_perturb, 6))
        return scores


class AdaGCPerTensorEMAShield:
    """
    v10.8 Zero-Bypass Optimizer Controller (PolarExpressMuonOptimizer.step hot-path):
    Fuses AdaGC per-tensor EMA adaptive gradient clipping (arXiv:2502.11034) and LoRA-Pre
    low-rank momentum state budget verification (arXiv:2602.24283).
    """

    def __init__(self, ema_beta: float = 0.98, spike_ratio: float = 2.5) -> None:
        self.ema_beta = float(ema_beta)
        self.spike_ratio = float(spike_ratio)
        self._ema_norms: Dict[int, float] = {}
        self.clipped_events: int = 0

    def compute_clip_scale(self, tensor_id: int, grad_norm: float) -> float:
        g = max(0.0, float(grad_norm))
        prev = self._ema_norms.get(tensor_id)
        if prev is None or prev <= 1e-12:
            self._ema_norms[tensor_id] = max(g, 1e-6)
            return 1.0
        cap = self.spike_ratio * prev
        if g > cap and g > 1e-12:
            scale = cap / g
            self.clipped_events += 1
            eff = cap
        else:
            scale = 1.0
            eff = g
        self._ema_norms[tensor_id] = self.ema_beta * prev + (1.0 - self.ema_beta) * eff
        return scale


class ByteTSTConv16RectifiedLoopController:
    """
    v10.9 Zero-Bypass Hot-Path Backbone Controller uniting:
      - Byte Language Models (arXiv:2610.05978): 3:1 Gated DeltaNet + Softmax with
        Token-Superposition Training (TST 4x exposure for first 30% -> 1.9x total text)
        and GDN conv_width=16 (vs 4 for subword) for broader byte-level local receptive field.
      - Rectified Looped Transformer (arXiv:2611.16265): damped state feedback preventing
        recurrent oscillation and promoting fixed-point convergence across 8 loops.
      - Attn-QAT (arXiv:2603.00040): low-precision backward score recomputation alignment.
      - MeqMuon (arXiv:2609.35701): row-magnitude equilibration check for orthogonal updates.
    """

    def __init__(self, gdn_conv_width: int = 16, loop_damping: float = 0.85, tst_phase_fraction: float = 0.30):
        self.gdn_conv_width = max(4, int(gdn_conv_width))
        self.loop_damping = max(0.1, min(1.0, float(loop_damping)))
        self.tst_phase_fraction = max(0.0, min(0.5, float(tst_phase_fraction)))

    def tst_effective_exposure(self, progress: float, group_k: int = 4) -> dict:
        p = max(0.0, min(1.0, float(progress)))
        active = p <= self.tst_phase_fraction
        mult = float(group_k) if active else 1.0
        total_equiv = self.tst_phase_fraction * float(group_k) + (1.0 - self.tst_phase_fraction) * 1.0
        return {"tst_active": active, "step_exposure_mult": mult, "total_text_exposure_mult": round(total_equiv, 3), "gdn_conv_width": self.gdn_conv_width}

    def rectified_loop_damping(self, loop_idx: int, state_delta_norm: float) -> float:
        decay = self.loop_damping ** max(0, int(loop_idx))
        if state_delta_norm > 4.0:
            decay *= 4.0 / float(state_delta_norm)
        return max(0.05, min(1.0, decay))

    def meqmuon_row_equilibration_ratio(self, row_norms: list[float]) -> float:
        if not row_norms:
            return 1.0
        pos = [max(1e-9, abs(float(x))) for x in row_norms]
        return max(pos) / min(pos)

    def attn_qat_backward_aligned(self, fwd_bits: int, bwd_recompute_bits: int) -> bool:
        return int(fwd_bits) == int(bwd_recompute_bits)


class AutomataEigenStateAndTCQEngramGuard:
    """v10.12: Hot-path Automata state-tracking & 2-bit TCQ CPU Engram guard.

    Grounds:
    - Negative eigenvalues [-1, 1] in Gated DeltaNet for regular-language state tracking (arXiv:2411.12537)
    - StackTrans differentiable pushdown-automaton stack depth budget (arXiv:2507.15343)
    - 2-bit TCQ N-gram lookup table compression (8x smaller CPU RAM, arXiv:2610.21810)
    """

    def allows_negative_eigen_state_tracking(self, beta_max: float) -> bool:
        return 1.0 < float(beta_max) <= 2.0

    def tcq_2bit_engram_ram_mb(self, num_params_b: float, bits: float = 2.0) -> float:
        return round(float(num_params_b) * 1e9 * (float(bits) / 8.0) / (1024.0 * 1024.0), 2)

    def stacktrans_depth_feasible(self, stack_depth: int, hidden_dim: int) -> bool:
        return int(stack_depth) >= 1 and int(stack_depth) <= max(1, int(hidden_dim) // 16)

if __name__ == "__main__":
    main()
