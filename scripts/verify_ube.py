#!/usr/bin/env python3
"""
UBE (Universal Byte Codec) verification on real Chinese records.

1. Lossless round-trip: decode(encode(text)) == text for every sampled record.
2. Per-record compression ratio on CJK-dominant text vs UTF-8.
3. Whole-file size comparison of the two built corpora (.bin), with the caveat
   that the two builds are independent streams (dedup decisions differ), so the
   file ratio is indicative, not an exact per-record measure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "brain"))

from exoskeleton_bridge import UniversalByteCodec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--utf8-bin", default="")
    ap.add_argument("--ube-bin", default="")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(
        "openbmb/Ultra-FineWeb-L3",
        "Ultra-FineWeb-L3-zh-Multi-Style-Synthetic",
        split="train",
        streaming=True,
    )

    n = 0
    ok = 0
    fail = 0
    fail_samples = []
    utf8_bytes = 0
    ube_bytes = 0
    cjk_chars = 0

    for row in ds:
        text = row.get("content") or ""
        if not text:
            continue
        n += 1
        enc = UniversalByteCodec.encode(text)
        try:
            dec = UniversalByteCodec.decode(enc)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            if len(fail_samples) < 3:
                fail_samples.append(f"decode error: {exc}")
            continue
        if dec == text:
            ok += 1
        else:
            fail += 1
            if len(fail_samples) < 3:
                fail_samples.append(f"mismatch at len {len(text)}")
        utf8_bytes += len(text.encode("utf-8"))
        ube_bytes += len(enc)
        cjk_chars += sum(1 for ch in text if ord(ch) >= 0x4E00)

        if n >= args.samples:
            break

    ratio = (ube_bytes / utf8_bytes) if utf8_bytes else float("nan")
    out = {
        "records_sampled": n,
        "roundtrip_ok": ok,
        "roundtrip_fail": fail,
        "lossless": fail == 0 and n > 0,
        "utf8_bytes": utf8_bytes,
        "ube_bytes": ube_bytes,
        "ube_over_utf8_ratio": round(ratio, 4),
        "byte_saving_pct": round((1 - ratio) * 100, 2),
        "cjk_chars": cjk_chars,
        "bytes_per_cjk_char_utf8": round(utf8_bytes / max(1, cjk_chars), 3),
        "bytes_per_cjk_char_ube": round(ube_bytes / max(1, cjk_chars), 3),
        "fail_examples": fail_samples,
    }

    for p, key in ((args.utf8_bin, "utf8_file"), (args.ube_bin, "ube_file")):
        if p:
            fp = Path(p)
            out[key] = {"path": p, "bytes": fp.stat().st_size if fp.exists() else None}

    if args.utf8_bin and args.ube_bin:
        a, b = Path(args.utf8_bin), Path(args.ube_bin)
        if a.exists() and b.exists() and a.stat().st_size:
            out["file_size_ratio_ube_over_utf8"] = round(b.stat().st_size / a.stat().st_size, 4)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
