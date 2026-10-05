#!/usr/bin/env python3
"""Equal-width sliding-window layer-reversion sweep (Concern 3, size-controlled).

Slides a fixed-width window across all transformer layers; for each window position
the window's layers are reverted to the base model (alpha=0) while every other layer
keeps its Dense-CPT weights, then held-in Belebele is evaluated. Because the window
width is constant, the resulting accuracy-vs-depth curve is not confounded by the
first/middle/last groups having different sizes (19 vs 9 vs 6) -- the exact objection
XbTP raised. Doubles as a downstream boundary-sensitivity analysis for Concern 4.

Loads the base and CPT models once and rebuilds each window in place (fast), reusing
the interpolation/eval helpers from middle_layer_alpha_sweep.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regularization import get_num_layers  # noqa: E402
from scripts.middle_layer_alpha_sweep import (  # noqa: E402
    is_target_param, load_text_model, run_lm_eval,
)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-3-4b-pt")
    ap.add_argument("--cpt_model", required=True, help="Dense CPT checkpoint dir")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--window", type=int, default=6, help="window width (default 6 = smallest group)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start_from", type=int, default=0, help="first window start (for GPU sharding)")
    ap.add_argument("--start_to", type=int, default=-1, help="last window start; -1 = auto (num_layers-window)")
    ap.add_argument("--tasks", required=True, help="comma-separated lm-eval tasks (held-in Belebele)")
    ap.add_argument("--batch_size", default="8")
    ap.add_argument("--include_path", default="")
    ap.add_argument("--log_samples", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", default="", help="lm-eval --limit for smoke tests; '' = full")
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.jsonl"

    print(f"[load] CPT: {args.cpt_model}", flush=True)
    cpt = load_text_model(args.cpt_model, args.device)
    print(f"[load] base: {args.base_model}", flush=True)
    base = load_text_model(args.base_model, args.device)
    num_layers = get_num_layers(cpt)

    # Cache both parameter sets on CPU; rebuild the working model per window.
    base_sd = {n: p.detach().to("cpu").clone() for n, p in base.named_parameters()}
    cpt_sd = {n: p.detach().to("cpu").clone() for n, p in cpt.named_parameters()}
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.cpt_model, trust_remote_code=True)

    W = args.window
    hi = (num_layers - W) if args.start_to < 0 else min(args.start_to, num_layers - W)
    starts = list(range(args.start_from, hi + 1, args.stride))
    print(f"[plan] {len(starts)} windows of width {W} over {num_layers} layers "
          f"(starts {args.start_from}..{hi})", flush=True)

    for start in starts:
        end = start + W
        for n, p in cpt.named_parameters():
            src = base_sd if is_target_param(n, start, end) else cpt_sd
            p.data.copy_(src[n].to(p.device))
        ckpt_dir = out_root / f"window_{start:02d}_{end:02d}"
        cpt.save_pretrained(ckpt_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(ckpt_dir)

        eval_rec = run_lm_eval(
            checkpoint=ckpt_dir,
            tasks=args.tasks,
            output_dir=out_root / f"eval_{start:02d}",
            cache_dir=out_root / f"cache_{start:02d}",
            batch_size=args.batch_size,
            limit=args.limit or None,
            include_path=args.include_path or None,
            log_samples=args.log_samples,
        )
        accs = [m["acc,none"] for m in eval_rec["metrics"].values() if "acc,none" in m]
        rec = {
            "window_start": start,
            "window_end": end,
            "window_center": (start + end - 1) / 2.0,
            "n_tasks": len(accs),
            "belebele_acc": (sum(accs) / len(accs)) if accs else None,
            "metrics": eval_rec["metrics"],
        }
        with summary_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[window {start:02d}-{end - 1:02d}] belebele_acc={rec['belebele_acc']}", flush=True)
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    print(f"[done] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
