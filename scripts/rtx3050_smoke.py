#!/usr/bin/env python3
"""
RTX 3050 4GB — real-card 20-step smoke telemetry driver.

Reports, per step: loss, pre-clip global grad-norm, step wall time, bytes/s.
Reports once: param count, static VRAM (post-init), peak allocated/reserved VRAM,
dtype/AMP mode, FLA + Liger-Kernel real dispatch status, NaN/Inf counters.

This driver lives in scripts/ (outside the AST dead-code audit scope of src/brain/).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "brain"))

from sovereign_brain_pytorch import (  # noqa: E402
    PRESETS,
    PolarExpressMuonOptimizer,
    SovereignRouteBBrain,
)
from ablation_harness import build_lz77_novelty_table_from_bytes  # noqa: E402
from ecosystem_backends import (  # noqa: E402
    probe_top_2026_ecosystem_wheels,
    try_fla_gated_delta_rule,
    try_liger_fused_linear_ce,
)

_MB = 1024.0 * 1024.0


def synth_corpus(nbytes: int, seed: int = 1234) -> bytes:
    """Deterministic, compressible byte stream with real code/math/CJK mix (no network)."""
    chunks = [
        b"def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)\n",
        b"assert fib(10) == 55 and fib(12) == 144\n",
        b"The entropy of an isolated system never decreases; dS >= 0.\n",
        "孤立系统的总熵永不减少，这是热力学第二定律的核心表述。\n".encode("utf-8"),
        b"for i in range(1000): acc += i * i - (i >> 1)\n",
        b"lemma: sum_{k=1}^{n} k = n(n+1)/2 forall n in N\n",
    ]
    g = torch.Generator().manual_seed(seed)
    order = torch.randint(0, len(chunks), (nbytes // 40 + 8,), generator=g).tolist()
    out = bytearray()
    for idx in order:
        out.extend(chunks[idx])
        if len(out) >= nbytes:
            break
    return bytes(out[:nbytes])


def grad_norm(model: torch.nn.Module) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(max(sq, 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="rtx3050_4gb_65m", choices=list(PRESETS.keys()))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--corpus", default="", help="path to .bin; synthetic if omitted")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — aborting real-card smoke test")

    dev = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    # ---- corpus ----
    need = max(args.steps * args.batch_size * (args.seq_len + 1) + 4096, 1 << 20)
    if args.corpus:
        raw = Path(args.corpus).read_bytes()
        if len(raw) < need:
            raw = (raw * (need // len(raw) + 1))[:need]
            src = f"file:{args.corpus} (tiled to {need} bytes)"
        else:
            src = f"file:{args.corpus}"
    else:
        raw = synth_corpus(need)
        src = "synthetic-mix"

    data = torch.tensor(list(raw), dtype=torch.long, device=dev)

    # ---- model ----
    cfg = PRESETS[args.preset]
    model = SovereignRouteBBrain(cfg).to(dev)
    nov = build_lz77_novelty_table_from_bytes(raw[: min(len(raw), 2_000_000)])
    model.patcher.load_novelty_table(nov, lock=True)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = PolarExpressMuonOptimizer(
        model.optimizer_param_groups(base_lr=args.lr, weight_decay=0.01),
        lr=args.lr, polar_steps=5,
    )
    torch.cuda.synchronize()
    static_alloc = torch.cuda.memory_allocated(dev) / _MB
    static_reserved = torch.cuda.memory_reserved(dev) / _MB

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    use_amp = args.dtype != "fp32" and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "fp16"))

    # ---- kernel dispatch probes (real tensors, not find_spec) ----
    B, H, T, Dh = 1, 4, 64, 32
    q, k, v = (torch.randn(B, H, T, Dh, device=dev, dtype=torch.bfloat16) for _ in range(3))
    g = torch.randn(B, H, T, Dh, device=dev, dtype=torch.bfloat16)
    beta = torch.randn(B, H, T, device=dev, dtype=torch.bfloat16)
    fla_res = try_fla_gated_delta_rule(q, k, v, g, beta)
    del q, k, v, g, beta
    torch.cuda.empty_cache()

    W = torch.randn(256, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
    Hs = torch.randn(8, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
    Tg = torch.randint(0, 256, (8,), device=dev)
    lig_res = try_liger_fused_linear_ce(W, Hs, Tg)
    liger_ok = lig_res is not None and torch.is_tensor(lig_res) and torch.isfinite(lig_res).all()
    del W, Hs, Tg, lig_res
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(dev)
    model.train()

    losses, gnorm, times = [], [], []
    nan_or_inf = 0
    total_bytes = 0
    t_all0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        off = (step - 1) * args.batch_size * args.seq_len
        win = data[off: off + args.batch_size * (args.seq_len + 1)].unsqueeze(0)
        if win.numel() < args.batch_size * (args.seq_len + 1):
            off = 0
            win = data[: args.batch_size * (args.seq_len + 1)].unsqueeze(0)
        inp, tgt = win[:, :-1, ], win[:, 1:]

        model.patcher.set_atdc_progress(step / float(args.steps))
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(inp, targets=tgt)
            loss = out["loss"]

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()

        gn = grad_norm(model)
        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        lv = float(loss.detach().item())
        if not math.isfinite(lv):
            nan_or_inf += 1
        if not math.isfinite(gn):
            nan_or_inf += 1
        losses.append(round(lv, 5))
        gnorm.append(round(gn, 4))
        times.append(round(dt, 4))
        total_bytes += args.batch_size * args.seq_len

    wall = time.perf_counter() - t_all0
    peak_alloc = torch.cuda.max_memory_allocated(dev) / _MB
    peak_res = torch.cuda.max_memory_reserved(dev) / _MB
    free_b, total_b = torch.cuda.mem_get_info(dev)

    probe = probe_top_2026_ecosystem_wheels()
    report = {
        "tag": args.tag or args.preset,
        "preset": args.preset,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "amp_mode": args.dtype,
        "corpus_source": src,
        "params_total": n_params,
        "params_trainable": n_train,
        "params_total_M": round(n_params / 1e6, 2),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "static_allocated_MB": round(static_alloc, 1),
        "static_reserved_MB": round(static_reserved, 1),
        "peak_allocated_MB": round(peak_alloc, 1),
        "peak_reserved_MB": round(peak_res, 1),
        "gpu_total_MB": round(total_b / _MB, 1),
        "gpu_free_MB_at_end": round(free_b / _MB, 1),
        "wall_sec": round(wall, 3),
        "ms_per_step": round(1000.0 * wall / args.steps, 2),
        "bytes_per_sec": round(total_bytes / wall, 1),
        "tokens_per_sec": round(total_bytes / wall, 1),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_delta": round(losses[-1] - losses[0], 5),
        "loss_curve": losses,
        "grad_norm_curve": gnorm,
        "step_sec_curve": times,
        "nan_or_inf_events": nan_or_inf,
        "fla_dispatch_real": fla_res is not None,
        "liger_dispatch_real": bool(liger_ok),
        "fla_installed": probe["details"].get("fla", {}).get("installed"),
        "liger_installed": probe["details"].get("liger_kernel", {}).get("installed"),
        "xgrammar_installed": probe["details"].get("xgrammar", {}).get("installed"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
