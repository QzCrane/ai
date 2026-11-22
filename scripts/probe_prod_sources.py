#!/usr/bin/env python3
"""Bounded reachability probe for every source of a corpus profile.

For each DatasetSource: resolve the HF dataset, load the config in streaming mode,
and pull the first record. Reports per-source OK/FAIL plus whether the declared
text_field is actually present. Written so stage-3 (30GB) cannot stall for hours
on a silently broken source.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "brain"))

from prepare_seed_corpus import CORPUS_PROFILES  # noqa: E402


def probe_one(src) -> dict:
    from datasets import load_dataset

    t0 = time.time()
    rec = {"name": src.name, "hf_id": src.hf_id, "subset": src.subset, "split": src.split,
           "text_field": src.text_field, "weight": src.weight, "status": "?"}
    try:
        ds = load_dataset(src.hf_id, src.subset, split=src.split, streaming=True)
        row = next(iter(ds))
        keys = list(row.keys())
        rec["keys"] = keys[:10]
        rec["text_field_present"] = src.text_field in row
        if src.text_field in row:
            val = row[src.text_field]
            rec["sample_chars"] = len(val) if isinstance(val, str) else None
            rec["sample_head"] = (val[:80] if isinstance(val, str) else str(type(val)))
        rec["status"] = "OK" if src.text_field in row else "FIELD_MISSING"
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "FAIL"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    rec["sec"] = round(time.time() - t0, 1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="rtx3050_4gb_prod_30gb")
    ap.add_argument("--out", default="D:/Environment/logs/prod_sources.json")
    args = ap.parse_args()

    prof = CORPUS_PROFILES[args.profile]
    out = []
    for src in prof.sources:
        r = probe_one(src)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        out.append(r)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    bad = [r["name"] for r in out if r["status"] != "OK"]
    print(f"\n[probe] {len(out) - len(bad)}/{len(out)} OK; problems: {bad}", flush=True)


if __name__ == "__main__":
    main()
