#!/usr/bin/env python3
"""
Real-corpus controlled ablation on RTX 3050 4GB.

Axes: pure-baseline vs full stack, selective-token-loss on/off, rec_loops 4/8, BF16/FP16.
Metric of record: held-out bits-per-byte (BPB) = CE_nats / ln(2) / bytes,
valid for a byte-level vocab (vocab_size == 256 => 1 token == 1 byte).

Windows are drawn with a fixed RNG seed shared across variants, so every variant
sees exactly the same byte ranges. Held-out slice is never trained on.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "brain"))

from sovereign_brain_pytorch import (  # noqa: E402
    PRESETS,
    PolarExpressMuonOptimizer,
    SovereignRouteBBrain,
)
from ablation_harness import build_lz77_novelty_table_from_bytes  # noqa: E402
from training_engine import compute_selective_token_loss  # noqa: E402

_MB = 1024.0 * 1024.0
LN2 = math.log(2.0)

# Mechanisms that a "pure baseline" switches off (keeps only the core looped backbone).
PURE_OFF = dict(
    enable_cpu_engram=False,
    enable_mamba3_phase=False,
    enable_rank2_deltaproduct=False,
    enable_m2rnn_nonlin_state=False,
    enable_input_injection=False,
    enable_ortho_input_injection=False,
    enable_arloop_depth_attn=False,
    enable_ternary_neuromod=False,
    enable_ne_oscillation_halt=False,
    enable_fixed_prior_depth_sup=False,
    enable_stochastic_thought=False,
    enable_lca_multibyte=False,
    enable_per_patch_elastic_freeze=False,
    enable_next_latent_pred=False,
    enable_elr_loop_cond=False,
    enable_pearce_hall_plasticity=False,
    enable_loopcd_hidden=False,
)


def make_cfg(preset: str, pure: bool, rec_loops: int):
    cfg = PRESETS[preset]
    if pure:
        cfg = replace(cfg, **PURE_OFF)
    cfg = replace(cfg, rec_loops=rec_loops, backprop_loops=min(2, rec_loops))
    return cfg


def grad_norm(model: torch.nn.Module) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(max(sq, 0.0))


@torch.no_grad()
def eval_bpb(model, data: np.memmap, seq_len: int, n_batches: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    model.eval()
    tot_nats, tot_bytes = 0.0, 0
    lo, hi = 0, len(data) - seq_len - 1
    for _ in range(n_batches):
        off = int(rng.integers(lo, hi))
        arr = np.asarray(data[off: off + seq_len + 1]).astype(np.int64)
        w = torch.from_numpy(arr).unsqueeze(0).cuda()
        out = model(w[:, :-1], targets=w[:, 1:])
        tot_nats += float(out["loss"].detach()) * seq_len
        tot_bytes += seq_len
    model.train()
    return tot_nats / LN2 / max(tot_bytes, 1)


def run_variant(name, cfg, train_data, eval_data, args, sel_loss: bool, dtype: str, win_seed: int = 1234):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = SovereignRouteBBrain(cfg).cuda()
    nov = build_lz77_novelty_table_from_bytes(bytes(train_data[: 1 << 21]))
    model.patcher.load_novelty_table(nov, lock=True)
    n_params = sum(p.numel() for p in model.parameters())

    opt = PolarExpressMuonOptimizer(
        model.optimizer_param_groups(base_lr=args.lr, weight_decay=0.01),
        lr=args.lr, polar_steps=5,
    )
    torch.cuda.synchronize()
    static_mb = torch.cuda.memory_allocated() / _MB

    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "fp16"))
    model.train()

    rng = np.random.default_rng(win_seed)  # identical windows for variants sharing a seed
    hi = len(train_data) - args.seq_len - 1
    n_inf = 0
    losses = []
    step_times = []
    t0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        _step_t0 = time.perf_counter()
        off = int(rng.integers(0, hi))
        arr = np.asarray(train_data[off: off + args.seq_len + 1]).astype(np.int64)
        w = torch.from_numpy(arr).unsqueeze(0).cuda()
        inp, tgt = w[:, :-1], w[:, 1:]

        model.patcher.set_atdc_progress(step / float(args.steps))
        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp, enabled=True):
            out = model(inp, targets=tgt)
            loss = out["loss"]
            if sel_loss:
                sel_ce, _ = compute_selective_token_loss(out["logits"], tgt)
                eff_bp = min(cfg.rec_loops, cfg.backprop_loops)
                w_final = 1.0 / float(max(1, eff_bp)) if cfg.enable_fixed_prior_depth_sup else 1.0
                raw_ce = torch.nn.functional.cross_entropy(
                    out["logits"].reshape(-1, out["logits"].size(-1)), tgt.reshape(-1)
                )
                loss = loss + w_final * (sel_ce - raw_ce)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()
        if not math.isfinite(grad_norm(model)):
            n_inf += 1
        if scaler.is_enabled():
            scaler.step(opt); scaler.update()
        else:
            opt.step()

        lv = float(loss.detach())
        if not math.isfinite(lv):
            n_inf += 1
        losses.append(round(lv, 5))
        step_times.append(time.perf_counter() - _step_t0)

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / _MB
    val = eval_bpb(model, eval_data, args.seq_len, args.eval_batches, seed=7)
    train_bpb = (sum(losses[-5:]) / 5) / LN2
    steady = sorted(step_times[5:]) if len(step_times) > 6 else sorted(step_times)
    steady_s = steady[len(steady) // 2] if steady else float("nan")

    return {
        "variant": name,
        "params_M": round(n_params / 1e6, 2),
        "rec_loops": cfg.rec_loops,
        "selective_token_loss": sel_loss,
        "dtype": dtype,
        "static_vram_MB": round(static_mb, 1),
        "peak_vram_MB": round(peak_mb, 1),
        "wall_sec": round(wall, 2),
        "ms_per_step": round(1000 * wall / args.steps, 1),
        "steady_ms_per_step": round(1000 * steady_s, 1),
        "steady_bytes_per_sec": round(args.seq_len / steady_s, 1) if steady_s else 0.0,
        "bytes_per_sec": round(args.steps * args.seq_len / wall, 1),
        "train_bpb_tail5": round(train_bpb, 4),
        "heldout_bpb": round(val, 4),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "nan_inf_events": n_inf,
    }


def prewarm(preset: str, seq_len: int, dtypes, steps: int = 2) -> None:
    """Populate the Triton/inductor kernel cache at the exact shape and dtype used by the
    measured variants, so variant wall time is not polluted by first-call compilation."""
    for dt in dtypes:
        torch.manual_seed(0)
        cfg = replace(PRESETS[preset], rec_loops=4, backprop_loops=2)
        m = SovereignRouteBBrain(cfg).cuda()
        m.train()
        amp = {"bf16": torch.bfloat16, "fp16": torch.float16}[dt]
        for _ in range(steps):
            w = torch.randint(0, 256, (1, seq_len + 1), device="cuda", dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=amp, enabled=True):
                out = m(w[:, :-1], targets=w[:, 1:])
                loss = out["loss"]
            loss.backward()
            m.zero_grad(set_to_none=True)
        del m
        torch.cuda.empty_cache()
    print(f"[prewarm] kernel cache primed for dtypes={list(dtypes)} seq_len={seq_len}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--preset", default="rtx3050_4gb_65m")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--variants", nargs="*", default=["all"])
    ap.add_argument("--prewarm", type=int, default=2, help="throwaway steps per dtype to prime the kernel cache; 0 disables")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = np.memmap(args.corpus, dtype=np.uint8, mode="r")
    split = int(len(data) * 0.98)
    train_data, eval_data = data[:split], data[split:]
    print(f"corpus={len(data)/_MB:.1f}MB train={len(train_data)/_MB:.1f}MB heldout={len(eval_data)/_MB:.1f}MB")

    matrix = {
        # name:                        pure,  rec, dtype,  sel,   window_seed
        "A_pure_base_r4_bf16":      (True,  4, "bf16", False, 1234),
        "A2_pure_base_r4_bf16_rep": (True,  4, "bf16", False, 999),   # noise floor (same cfg, different windows)
        "B_pure_seloss_r4_bf16":    (True,  4, "bf16", True,  1234),
        "C_pure_base_r8_bf16":      (True,  8, "bf16", False, 1234),
        "D_pure_base_r4_fp16":      (True,  4, "fp16", False, 1234),
        "E_full_stack_r8_bf16":     (False, 8, "bf16", True,  1234),
    }
    selected = list(matrix) if args.variants == ["all"] else args.variants
    dtypes = tuple(sorted({matrix[v][2] for v in selected}))
    if args.prewarm:
        prewarm(args.preset, args.seq_len, dtypes, steps=args.prewarm)

    results = []
    for v in selected:
        pure, rec, dtype, sel, wseed = matrix[v]
        cfg = make_cfg(args.preset, pure, rec)
        r = run_variant(v, cfg, train_data, eval_data, args, sel, dtype, wseed)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)
        torch.cuda.empty_cache()

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== SUMMARY (held-out BPB, lower is better) ===")
    base = next((r["heldout_bpb"] for r in results if r["variant"] == "A_pure_base_r4_bf16"), None)
    rep = next((r["heldout_bpb"] for r in results if r["variant"].startswith("A2_")), None)
    noise = None if (base is None or rep is None) else abs(rep - base)
    if noise is not None:
        print(f"noise floor (A vs A2, same cfg different windows): {noise:.4f} BPB")
    for r in results:
        d = "" if base is None else f"  Δ vs A = {r['heldout_bpb'] - base:+.4f}"
        sig = ""
        if noise and not r["variant"].startswith("A_"):
            sig = "  [significant]" if abs(r["heldout_bpb"] - base) > noise else "  [within noise]"
        print(f"{r['variant']:26} BPB={r['heldout_bpb']:.4f}{d}{sig}  "
              f"{r.get('steady_bytes_per_sec', r['bytes_per_sec']):.0f} B/s  peak={r['peak_vram_MB']:.0f}MB")


if __name__ == "__main__":
    main()
