#!/usr/bin/env python3
"""
Sovereign Pure AI Brain — Route B Frozen Architecture (v1.0, RTX 3050 4GB Executable)
=====================================================================================
Strictly adheres to Route B ("Time-and-Structure-First", Scale-Invariant & 4GB-Native):
  - Sandwich Looped Topology (Prelude -> Mid-Layer CPU Engram -> Shared 3:1 Core x T loops -> Coda)
    * Parcae spectral-bounded diagonal carry (arXiv:2604.12946): h_{r+1} = sigmoid(a) * h_r + delta
    * ARLoop Sliding-Window Depth Attention (arXiv:2611.05234, window W=2): O(1) memory selective
      cross-loop residual access without full-history overhead.
    * 1-scalar input injection (arXiv:2610.00673v1) + RMSNorm implicit gain control (arXiv:2607.10681v1)
    * O(1)-VRAM truncated implicit backprop (no_grad on first T-2 loops).
  - Trainable 0-MB-VRAM CPU-Pinned Mid-Layer Engram (DeepSeek Engram + SlackTier arXiv:2611.13413v1
    + TF-Engram offline load hook arXiv:2607.07388v1):
    * Custom autograd Function routes sparse gradients ONLY for touched slots back to CPU RAM,
      updating `_cpu_table` in-place on CPU (0.0 MB GPU VRAM, 100% trainable!).
  - 3:1 Hybrid Recurrent Core:
    * 3x OscillatoryDualDeltaNet (DualDeltaNet + Mamba-3 complex phase rotation)
    * 1x HoPEHLAMELTAttention (PyTorch native F.scaled_dot_product_attention + MELT learnable gated
      shared HLA latent KV cache across loops arXiv:2605.07721v3 & 2610.07940 + HoPE + Qwen Sink-Free Gate)
  - Pure Dense SwiGLU FFN (NO MoE on 4GB) + Native BF16 AMP + ICLR 2026 Polar Express Muon.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from itertools import repeat
from typing import Dict, List, Optional, Tuple

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
    # Clean Ablation Switches (default = locked Route B v1.0)
    enable_cpu_engram: bool = True
    enable_mamba3_phase: bool = True
    enable_input_injection: bool = True
    enable_arloop_depth_attn: bool = True
    enable_erm_halt: bool = True


PRESETS: Dict[str, RouteBConfig] = {
    "micro_debug": RouteBConfig(
        name="micro_debug", d_model=256, d_state=64, n_heads=4,
        k_prelude=1, k_core=4, k_coda=1, rec_loops=4, backprop_loops=2,
        chunk_size=16, kv_latent_dim=64, engram_slots=32768,
    ),
    "rtx3050_4gb_65m": RouteBConfig(
        name="rtx3050_4gb_65m", d_model=512, d_state=64, n_heads=8,
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
    delta_states: Dict[Tuple[int, int], torch.Tensor] = field(default_factory=dict)
    hla_kv_latents: Dict[int, torch.Tensor] = field(default_factory=dict)
    pos_offset: int = 0


# =============================================================================
# Primitives: RMSNorm (Implicit Gain Control), Complex Phase (Mamba-3), HoPE
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
# Organ 1: Lossless Byte + Streaming Canon Conv1D + 4x DynamicBytePatcher
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
    def __init__(self, d_model: int, patch_stride: int = 4) -> None:
        super().__init__()
        self.stride = patch_stride
        self.surprise_head = nn.Linear(d_model, 1, bias=True)
        self.patch_proj = nn.Linear(d_model, d_model, bias=False)
        self.expand_proj = nn.Linear(d_model * 2, d_model, bias=False)

    def compress(self, h_bytes: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, T, D = h_bytes.shape
        pad = (self.stride - (T % self.stride)) % self.stride
        h_pad = F.pad(h_bytes, (0, 0, 0, pad)) if pad > 0 else h_bytes
        T_pad = h_pad.size(1)
        w = torch.sigmoid(self.surprise_head(h_pad)).view(B, T_pad // self.stride, self.stride, 1)
        w = w / (w.sum(dim=2, keepdim=True) + 1e-6)
        patches = (h_pad.view(B, T_pad // self.stride, self.stride, D) * w).sum(dim=2)
        return self.patch_proj(patches), T

    def expand(self, h_patches: torch.Tensor, h_bytes_skip: torch.Tensor, orig_T: int) -> torch.Tensor:
        up = h_patches.repeat_interleave(self.stride, dim=1)[:, :orig_T, :]
        return self.expand_proj(torch.cat([up, h_bytes_skip], dim=-1))


# =============================================================================
# Organ 4: Trainable 0-MB-VRAM CPU-Pinned Mid-Layer Engram (Sparse CPU Backprop)
# =============================================================================
class _CPUSparseEngramFn(torch.autograd.Function):
    """
    Forward: gathers K-head rows from CPU-pinned table -> non_blocking transfer to GPU.
    Backward: transfers ONLY touched slot gradients back to CPU and applies sparse in-place SGD
    on CPU RAM (strictly 0.0 MB GPU VRAM for table & optimizer state!).
    """
    @staticmethod
    def forward(ctx, slots_cpu: torch.Tensor, dummy_anchor: torch.Tensor, cpu_table: torch.Tensor, lr: float, training: bool):
        ctx.slots_cpu = slots_cpu
        ctx.cpu_table = cpu_table
        ctx.lr = lr
        ctx.training = training
        fetched_cpu = cpu_table[slots_cpu].mean(dim=0)  # (B, T, D) on CPU
        return fetched_cpu.to(device=dummy_anchor.device, dtype=dummy_anchor.dtype, non_blocking=True)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if ctx.training and ctx.lr > 0.0:
            slots_cpu = ctx.slots_cpu  # (K, B, T)
            K = slots_cpu.size(0)
            grad_cpu = (grad_out.detach().to(device="cpu", dtype=torch.float32) / float(K))  # (B, T, D)
            flat_idx = slots_cpu.reshape(-1)
            flat_grad = grad_cpu.unsqueeze(0).expand(K, -1, -1, -1).reshape(-1, grad_cpu.size(-1))
            uniq_idx, inv = torch.unique(flat_idx, return_inverse=True)
            acc_grad = torch.zeros((uniq_idx.size(0), flat_grad.size(-1)), dtype=torch.float32)
            acc_grad.index_add_(0, inv, flat_grad)
            # In-place sparse update on CPU table
            updated = ctx.cpu_table[uniq_idx].float() - ctx.lr * acc_grad
            ctx.cpu_table[uniq_idx] = updated.to(dtype=ctx.cpu_table.dtype)
        return None, None, None, None, None


class CPUPinnedEngramMemory(nn.Module):
    """
    Trainable 0-MB-VRAM CPU-Pinned Mid-Layer Engram (DeepSeek Engram + SlackTier arXiv:2611.13413v1).
    Supports BOTH online sparse CPU-RAM training AND offline TF-Engram (arXiv:2607.07388v1) bulk load.
    """
    def __init__(self, slots: int, d_model: int, n_hash_heads: int = 4, engram_lr: float = 0.05) -> None:
        super().__init__()
        self.slots = slots
        self.n_hash_heads = n_hash_heads
        self.engram_lr = engram_lr
        # Plain CPU attribute -> guaranteed 0.0 MB GPU VRAM even after model.to("cuda")
        self._cpu_table = torch.empty(slots, d_model, dtype=torch.float16, device="cpu").normal_(mean=0.0, std=0.02)
        if torch.cuda.is_available():
            self._cpu_table = self._cpu_table.pin_memory()
        self._p1 = torch.tensor([1315423911, 2654435761, 2246822519, 3266489917][:n_hash_heads], dtype=torch.long)
        self._p2 = torch.tensor([668265263, 374761393, 1597334677, 2054039447][:n_hash_heads], dtype=torch.long)
        self._p3 = torch.tensor([1103515245, 1664525, 214013, 48271][:n_hash_heads], dtype=torch.long)
        self._salt = torch.tensor([17, 1013904223, 2531011, 12345][:n_hash_heads], dtype=torch.long)
        self.ctx_gate = nn.Linear(d_model * 2, d_model, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.2))

    def _compute_cpu_slots(self, patch_byte_keys: torch.Tensor) -> torch.Tensor:
        b0 = patch_byte_keys.detach().to(device="cpu", dtype=torch.long)
        b1 = F.pad(b0[:, :-1], (1, 0), value=0)
        b2 = F.pad(b0[:, :-2], (2, 0), value=0)
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

    def lookup(self, patch_byte_keys: torch.Tensor, ctx_hidden: torch.Tensor) -> torch.Tensor:
        slots_cpu = self._compute_cpu_slots(patch_byte_keys)
        fetched = _CPUSparseEngramFn.apply(slots_cpu, ctx_hidden, self._cpu_table, self.engram_lr, self.training)
        return fetched * torch.sigmoid(self.ctx_gate(torch.cat([ctx_hidden, fetched], dim=-1))) * self.scale


# =============================================================================
# Organ 2A (75%): Oscillatory DualDeltaNet
# =============================================================================
class OscillatoryDualDeltaNet(nn.Module):
    def __init__(self, cfg: RouteBConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_state
        self.chunk_size = cfg.chunk_size
        inner = self.n_heads * self.d_head
        self.norm = RMSNorm(cfg.d_model)
        self.qkv_proj = nn.Linear(cfg.d_model, inner * 3, bias=False)
        self.erase_proj = nn.Linear(cfg.d_model, inner, bias=True)
        self.write_proj = nn.Linear(cfg.d_model, inner, bias=True)
        self.phase_proj = nn.Linear(cfg.d_model, inner // 2, bias=True) if cfg.enable_mamba3_phase else None
        self.o_proj = nn.Linear(inner, cfg.d_model, bias=False)
        self.out_gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        h = self.norm(x)
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head)
        k = F.normalize(k.view(B, T, self.n_heads, self.d_head), dim=-1, eps=1e-6)
        v = v.view(B, T, self.n_heads, self.d_head)
        if self.phase_proj is not None:
            theta = self.phase_proj(h).view(B, T, self.n_heads, self.d_head // 2)
            q = apply_complex_phase_rotation(q, theta)
            k = apply_complex_phase_rotation(k, theta)

        alpha = torch.sigmoid(self.erase_proj(h)).view(B, T, self.n_heads, self.d_head)
        beta = torch.sigmoid(self.write_proj(h)).view(B, T, self.n_heads, self.d_head)
        S = x.new_zeros((B, self.n_heads, self.d_head, self.d_head)) if state is None else state

        outs: List[torch.Tensor] = []
        C = self.chunk_size
        for start in range(0, T, C):
            end = min(start + C, T)
            qc, kc, vc = q[:, start:end], k[:, start:end], v[:, start:end]
            ac, bc = alpha[:, start:end], beta[:, start:end]
            L = end - start
            inter_out = torch.einsum("bhij,blhj->blhi", S, qc)
            qk = torch.einsum("blhd,bmhd->bhlm", qc, kc) * torch.tril(torch.ones(L, L, device=x.device, dtype=x.dtype))
            prior_pred = torch.einsum("bhij,blhj->blhi", S, kc)
            v_eff = bc * (vc - ac * prior_pred)
            outs.append(inter_out + torch.einsum("bhlm,bmhd->blhd", qk, v_eff))
            S = (
                S * (1.0 - 0.1 * ac.mean(dim=1).unsqueeze(-1))
                - torch.einsum("blhi,blhj->bhij", ac * prior_pred, kc) / max(1, L)
                + torch.einsum("blhi,blhj->bhij", v_eff, kc)
            )
        out = self.o_proj(torch.cat(outs, dim=1).reshape(B, T, -1)) * torch.sigmoid(self.out_gate(h))
        return out, S.detach()


# =============================================================================
# Organ 2B (25%): HoPE + MELT Gated Shared HLA Latent KV + Native SDPA + Qwen Gate
# =============================================================================
class HoPEHLAMELTAttention(nn.Module):
    """
    Combines:
      - PyTorch native F.scaled_dot_product_attention (Ampere Flash/MemEff O(T) VRAM)
      - HLA shared low-rank KV latent cache (arXiv:2610.07940)
      - MELT learnable gated cross-loop cache refinement (arXiv:2605.07721v3)
      - HoPE (50% fast RoPE + 50% slow NoPE) + Qwen 1-line Sink-Free Gate
    """
    def __init__(self, cfg: RouteBConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_state
        inner = self.n_heads * self.d_head
        self.norm = RMSNorm(cfg.d_model)
        self.q_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.kv_down = nn.Linear(cfg.d_model, cfg.kv_latent_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_latent_dim)
        # MELT learnable gate blending current loop's refined latent into shared KV cache
        self.melt_gate = nn.Linear(cfg.kv_latent_dim * 2, cfg.kv_latent_dim, bias=True)
        self.k_up = nn.Linear(cfg.kv_latent_dim, inner, bias=False)
        self.v_up = nn.Linear(cfg.kv_latent_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, cfg.d_model, bias=False)
        self.sink_gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

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
        else:
            # MELT gated in-place refinement of the active tail slice across loops (constant cache size!)
            tail_old = shared_kv_latent[:, -T:, :]
            g = torch.sigmoid(self.melt_gate(torch.cat([tail_old, c_kv], dim=-1)))
            tail_new = (1.0 - g) * tail_old + g * c_kv
            c_kv_full = torch.cat([shared_kv_latent[:, :-T, :], tail_new], dim=1) if shared_kv_latent.size(1) > T else tail_new

        T_kv = c_kv_full.size(1)
        q = apply_hope_fast_band_rope(
            self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2), pos_offset=pos_offset
        )
        k = apply_hope_fast_band_rope(
            self.k_up(c_kv_full).view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2), pos_offset=0
        )
        v = self.v_up(c_kv_full).view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)

        is_causal = (T == T_kv)
        ctx = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = self.o_proj(ctx.transpose(1, 2).reshape(B, T, -1)) * torch.sigmoid(self.sink_gate(h))
        return out, c_kv_full.detach()


# =============================================================================
# ARLoop Sliding-Window Depth Attention (arXiv:2611.05234, O(W) Constant Memory)
# =============================================================================
class SlidingWindowDepthAttnRes(nn.Module):
    """
    ARLoop (arXiv:2611.05234): Selectively aggregates only the last W loop states via
    learned depth attention, matching/beating full-history attention at constant O(W) VRAM.
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.key_norm = RMSNorm(d_model)
        self.query = nn.Parameter(torch.zeros(d_model))

    def forward(self, window_states: List[torch.Tensor]) -> torch.Tensor:
        if len(window_states) == 1:
            return window_states[0]
        stacked = torch.stack(window_states, dim=0)  # (W, B, T, D)
        scores = torch.einsum("d,wbtd->wbt", self.query, self.key_norm(stacked)) / math.sqrt(stacked.size(-1))
        weights = F.softmax(scores, dim=0).unsqueeze(-1)
        return (weights * stacked).sum(dim=0)


# =============================================================================
# Pure Dense SwiGLU Block (Zero MoE Overhead on 4GB Single GPU)
# =============================================================================
class RouteBBlock(nn.Module):
    def __init__(self, cfg: RouteBConfig, is_exact_recall: bool) -> None:
        super().__init__()
        self.is_exact_recall = is_exact_recall
        self.mixer = HoPEHLAMELTAttention(cfg) if is_exact_recall else OscillatoryDualDeltaNet(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        ffn_hidden = int(cfg.d_model * 2.5)
        self.w_gate = nn.Linear(cfg.d_model, ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, ffn_hidden, bias=False)
        self.w_down = nn.Linear(ffn_hidden, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        block_id: int,
        loop_idx: int = 0,
        cache: Optional[RouteBCache] = None,
    ) -> torch.Tensor:
        if self.is_exact_recall:
            kv_prev = None if cache is None else cache.hla_kv_latents.get(block_id)
            pos_off = 0 if cache is None else cache.pos_offset
            delta_m, kv_next = self.mixer(x, shared_kv_latent=kv_prev, loop_idx=loop_idx, pos_offset=pos_off)
            if cache is not None:
                cache.hla_kv_latents[block_id] = kv_next
        else:
            st_prev = None if cache is None else cache.delta_states.get((block_id, loop_idx))
            delta_m, st_next = self.mixer(x, state=st_prev)
            if cache is not None:
                cache.delta_states[(block_id, loop_idx)] = st_next
        # Scaling Laws for Looped Transformers (2026): 1/r residual scaling prevents correlated loop inflation
        inv_r = 1.0 / float(loop_idx + 1)
        x = x + inv_r * delta_m
        h = self.ffn_norm(x)
        return x + inv_r * self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


# =============================================================================
# Complete Route B Sovereign AI Brain (v1.0)
# =============================================================================
class SovereignRouteBBrain(nn.Module):
    def __init__(self, cfg: RouteBConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.byte_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.canon = CanonCausalConv1D(cfg.d_model, kernel_size=cfg.canon_kernel)
        self.patcher = DynamicBytePatcher(cfg.d_model, patch_stride=cfg.patch_stride)

        self.prelude = nn.ModuleList([RouteBBlock(cfg, is_exact_recall=False) for _ in range(cfg.k_prelude)])
        self.engram = CPUPinnedEngramMemory(cfg.engram_slots, cfg.d_model, engram_lr=cfg.engram_lr) if cfg.enable_cpu_engram else None
        self.core = nn.ModuleList([RouteBBlock(cfg, is_exact_recall=((i + 1) % 4 == 0)) for i in range(cfg.k_core)])

        # Parcae bounded diagonal carry (arXiv:2604.12946) + 1-scalar input injection + ARLoop window
        self.parcae_carry_logit = nn.Parameter(torch.full((cfg.d_model,), 2.0))
        self.inj_alpha = nn.Parameter(torch.tensor(0.15)) if cfg.enable_input_injection else None
        self.loop_gain_norm = RMSNorm(cfg.d_model)
        self.depth_attn = SlidingWindowDepthAttnRes(cfg.d_model) if cfg.enable_arloop_depth_attn else None
        self.energy_head = nn.Linear(cfg.d_model, 1, bias=False) if cfg.enable_erm_halt else None

        self.coda = nn.ModuleList([RouteBBlock(cfg, is_exact_recall=(i == cfg.k_coda - 1)) for i in range(cfg.k_coda)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def _run_core_once(self, h: torch.Tensor, e0: torch.Tensor, loop_idx: int, cache: Optional[RouteBCache]) -> torch.Tensor:
        carry = torch.sigmoid(self.parcae_carry_logit).view(1, 1, -1)
        h_in = self.loop_gain_norm(carry * h + (self.inj_alpha * e0 if self.inj_alpha is not None else 0.0))
        h_cur = h_in
        for i, blk in enumerate(self.core):
            h_cur = blk(h_cur, block_id=100 + i, loop_idx=loop_idx, cache=cache)
        return h_cur

    def forward(
        self,
        byte_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[RouteBCache] = None,
    ) -> Dict[str, torch.Tensor | List[float]]:
        h_bytes, next_conv = self.canon(self.byte_embed(byte_ids), None if cache is None else cache.conv_state)
        if cache is not None:
            cache.conv_state = next_conv
        h, orig_T = self.patcher.compress(h_bytes)

        for i, blk in enumerate(self.prelude):
            h = blk(h, block_id=i, loop_idx=0, cache=cache)

        if self.engram is not None:
            patch_keys = byte_ids[:, :: self.cfg.patch_stride]
            if patch_keys.size(1) < h.size(1):
                patch_keys = F.pad(patch_keys, (0, h.size(1) - patch_keys.size(1)), value=0)
            h = h + self.engram.lookup(patch_keys[:, : h.size(1)], h)

        e0 = h
        energies: List[float] = []
        n_loops = self.cfg.rec_loops
        no_grad_loops = max(0, n_loops - self.cfg.backprop_loops) if self.training else n_loops
        window: List[torch.Tensor] = [h]
        W = max(1, self.cfg.depth_attn_window)

        best_h, best_e = h, float("inf")
        for r in range(n_loops):
            prev = h
            h_src = self.depth_attn(window) if self.depth_attn is not None else h
            if r < no_grad_loops:
                with torch.no_grad():
                    h = self._run_core_once(h_src, e0, loop_idx=r, cache=cache)
            else:
                h = self._run_core_once(h_src, e0, loop_idx=r, cache=cache)

            window.append(h)
            if len(window) > W:
                window.pop(0)

            if self.energy_head is not None:
                # FPRM (arXiv:2606.18206) relative fixed-point residual + ERM scalar energy
                rel_diff = (h - prev).norm() / (prev.norm() + 1e-6)
                e_val = float((rel_diff + 0.1 * self.energy_head(h).abs().mean()).detach().item())
                energies.append(round(e_val, 6))
                if e_val < best_e:
                    best_e, best_h = e_val, h
                elif not self.training and r >= 2 and (float(rel_diff.item()) < 0.02 or e_val >= energies[-2]):
                    break
            else:
                best_h = h

        h = best_h
        for i, blk in enumerate(self.coda):
            h = blk(h, block_id=200 + i, loop_idx=0, cache=cache)
        if cache is not None:
            cache.pos_offset += h.size(1)

        logits = self.lm_head(self.final_norm(self.patcher.expand(h, h_bytes, orig_T)))
        out: Dict[str, torch.Tensor | List[float]] = {"logits": logits, "energies": energies}
        if targets is not None:
            out["loss"] = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))
        return out

    def vram_audit_mb(self) -> Dict[str, float]:
        n_params = sum(p.numel() for p in self.parameters())
        fp32_weights_mb = n_params * 4 / (1024 ** 2)
        bf16_grads_mb = n_params * 2 / (1024 ** 2)
        muon_mom_mb = n_params * 4 / (1024 ** 2)
        return {
            "gpu_neural_params_M": round(n_params / 1e6, 2),
            "static_train_vram_mb": round(fp32_weights_mb + bf16_grads_mb + muon_mom_mb, 1),
            "engram_gpu_vram_mb": 0.0,
            "engram_cpu_ram_mb": self.engram.host_ram_mb() if self.engram is not None else 0.0,
            "effective_depth_layers": self.cfg.k_prelude + self.cfg.k_core * self.cfg.rec_loops + self.cfg.k_coda,
        }


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


class PolarExpressMuonOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.9, weight_decay: float = 0.01, polar_steps: int = 5) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, polar_steps=polar_steps))

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr, beta, wd, psteps = group["lr"], group["momentum"], group["weight_decay"], group["polar_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("buf", torch.zeros_like(p.grad))
                buf.mul_(beta).add_(p.grad)
                if p.ndim == 2 and min(p.shape) >= 8:
                    upd = polar_express_orthogonalize(buf, steps=psteps)
                    row_rms = upd.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
                    upd = (upd / row_rms) * math.sqrt(max(1.0, p.size(0) / p.size(1))) * 0.2
                else:
                    upd = buf / (buf.norm() + 1e-8)
                p.mul_(1.0 - lr * wd).add_(upd, alpha=-lr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sovereign Route B AI Brain v1.0 (RTX 3050 4GB Executable)")
    parser.add_argument("--config", choices=list(PRESETS.keys()), default="rtx3050_4gb_65m")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="sovereign_route_b_v1.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PRESETS[args.config]
    model = SovereignRouteBBrain(cfg).to(device)
    print(f"[RouteB v1.0] config={cfg.name} | device={device} | audit={model.vram_audit_mb()}")

    sample = b"def fib(n): return n if n < 2 else fib(n-1) + fib(n-2) # Route B 4GB Brain\n" * 4
    seq = torch.tensor(list(sample), dtype=torch.long, device=device).unsqueeze(0)
    inp, tgt = seq[:, :-1], seq[:, 1:]

    opt = PolarExpressMuonOptimizer(model.parameters(), lr=0.02, polar_steps=5)
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(inp, targets=tgt)
        out["loss"].backward()
        opt.step()
        if step == 1 or step == args.steps or step % 5 == 0:
            print(f"  step={step:03d} | loss={out['loss'].item():.4f} | erm_energies={out['energies']}")

    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, args.save_path)
    print(f"[Saved] {args.save_path}")


if __name__ == "__main__":
    main()
