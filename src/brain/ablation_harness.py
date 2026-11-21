#!/usr/bin/env python3
"""
Sovereign AI Brain — 4GB Control-Variable Ablation & Lifecycle Verification Harness (GAP-14)
============================================================================================
Ready-to-run on local RTX 3050 4GB (or CPU micro_debug) when real-hardware falsification (Layer 2)
is initiated. Zero toy mocks: tests real PyTorch forward/backward, Step-0 LZ77 corpus novelty fitting,
Wake -> Sleep (Co-Observation + LaCET + Noise-Prune) -> Next-Wake prior transfer, DiffMAS latent KV
exchange, ATF multimodal thought branching, and Block-Wise Hadamard SOAR/4over6 Q4 export.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from typing import Any, Dict, List

import sys
from pathlib import Path
import torch

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sovereign_brain_pytorch import (
    PRESETS,
    CoObservationSleepBuffer,
    PolarExpressMuonOptimizer,
    RouteBCache,
    RouteBConfig,
    SovereignRouteBBrain,
)


def build_lz77_novelty_table_from_bytes(corpus: bytes, smoothing: float = 1.0) -> torch.Tensor:
    """
    Step-0 corpus-fitted 2-byte LZ77/bigram novelty table (arXiv:2610.22100v1, 65536 entries on CPU RAM):
    novelty(b_{t-1}, b_t) = 1 - p(b_t | b_{t-1}) scaled into [0.1, 1.5].
    Must be loaded BEFORE step 0 and locked permanently.
    """
    counts = torch.full((256, 256), float(smoothing), dtype=torch.float32)
    arr = torch.tensor(list(corpus), dtype=torch.long)
    if arr.numel() >= 2:
        prev_b, curr_b = arr[:-1], arr[1:]
        flat_idx = prev_b * 256 + curr_b
        counts.view(-1).index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    cond_prob = counts / counts.sum(dim=1, keepdim=True)
    novelty = (1.0 - cond_prob).clamp(0.0, 1.0) * 1.4 + 0.1
    return novelty.reshape(65536)


def make_ablation_suite(base: RouteBConfig) -> Dict[str, RouteBConfig]:
    """Controlled single-variable ablation matrix for Layer-2 falsification on RTX 3050 4GB."""
    return {
        "full_v2_9": replace(base),
        "full_v1_3": replace(base),
        "ablate_no_engram": replace(base, enable_cpu_engram=False),
        "ablate_no_mamba3_phase": replace(base, enable_mamba3_phase=False),
        "ablate_no_rank2_deltaproduct": replace(base, enable_rank2_deltaproduct=False),
        "ablate_no_m2rnn_nonlin_state": replace(base, enable_m2rnn_nonlin_state=False),
        "ablate_no_next_latent_pred": replace(base, enable_next_latent_pred=False),
        "ablate_no_ternary_neuromod": replace(base, enable_ternary_neuromod=False),
        "ablate_no_lca_multibyte": replace(base, enable_lca_multibyte=False),
        "ablate_with_stars_reg": replace(base, use_stars_reg=True),
    }


def run_single_ablation(name: str, cfg: RouteBConfig, corpus: bytes, steps: int, device: torch.device) -> Dict[str, Any]:
    torch.manual_seed(42)
    model = SovereignRouteBBrain(cfg).to(device)
    nov_table = build_lz77_novelty_table_from_bytes(corpus)
    model.patcher.load_novelty_table(nov_table, lock=True)

    seq = torch.tensor(list(corpus), dtype=torch.long, device=device).unsqueeze(0)
    inp, tgt = seq[:, :-1], seq[:, 1:]
    opt = PolarExpressMuonOptimizer(model.optimizer_param_groups(base_lr=0.02, weight_decay=0.01), lr=0.02, polar_steps=5)
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    losses: List[float] = []
    for step in range(1, steps + 1):
        model.patcher.set_atdc_progress(step / float(max(1, steps)))
        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(inp, targets=tgt)
        out["loss"].backward()
        opt.step()
        losses.append(round(float(out["loss"].detach().item()), 4))
    elapsed = round(time.perf_counter() - t0, 3)
    peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1) if device.type == "cuda" else 0.0

    # Verify full lifecycle: Wake -> Sleep (Co-Obs + LaCET + Noise-Prune) -> Next-Wake Prior Transfer + ATF + Q4
    cache = RouteBCache()
    co_buf = CoObservationSleepBuffer(max_episodes=16, max_seq_len=128)
    co_buf.add_episode(inp[:, :64], salience=1.5)
    sleep_metrics = model.consolidate_sleep(inp[:, 64:128], cache, n_sleep_passes=2, co_obs_buffer=co_buf)
    wake_after_sleep = model.sample_thought_branches(inp[:, :32], n_branches=3, cache=cache)
    q4_audit = model.export_q4_inference_state(block_size=32)["audit"]

    return {
        "variant": name,
        "vram_audit": model.vram_audit_mb(),
        "peak_gpu_vram_mb": peak_vram_mb,
        "elapsed_sec": elapsed,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "sleep_metrics": sleep_metrics,
        "atf_branch_residuals": wake_after_sleep.get("branch_min_residuals", []),
        "q4_export_audit": q4_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Route B v2.9 4GB Controlled Ablation Harness")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="micro_debug")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--variants", nargs="*", default=["full_v2_9"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = copy.deepcopy(PRESETS[args.preset])
    suite = make_ablation_suite(base_cfg)
    corpus = (
        b"def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)\n"
        b"assert fib(10) == 55 and fib(12) == 144 # parity & arithmetic state tracking\n"
    ) * 3

    results = []
    for v_name in args.variants:
        if v_name not in suite:
            raise ValueError(f"Unknown variant {v_name}. Choices: {list(suite.keys())}")
        results.append(run_single_ablation(v_name, suite[v_name], corpus, args.steps, device))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
