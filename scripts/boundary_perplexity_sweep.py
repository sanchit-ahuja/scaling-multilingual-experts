#!/usr/bin/env python3
"""Validation-perplexity sweep over candidate layer-split boundaries (Concern 4, corroboration).

For each candidate (first_n, last_n), hard-revert the Dense-CPT model's middle band
[first_n, num_layers-last_n) to the base weights (alpha=0) and measure held-in validation
perplexity via train.py's eval-only path. Shows how much acquisition (perplexity) is
sacrificed as a function of where the boundary is drawn; paired with E4's downstream
sensitivity this corroborates the 9/6 choice. NOTE: this is a post-hoc reversion-based
sweep, NOT the training-time selection (see rebuttal_notes.md). Consistency check: the
(9,6) point must reproduce the existing dense_25b_v2_reverted perplexity.

Loads base+CPT once; rebuilds each boundary in place; evals via a train.py subprocess.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regularization import get_num_layers  # noqa: E402
from scripts.middle_layer_alpha_sweep import is_target_param, load_text_model  # noqa: E402


def parse_boundaries(raw: str):
    """'9,6;6,6;12,6' -> [(9,6),(6,6),(12,6)]."""
    out = []
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        f, l = pair.split(",")
        out.append((int(f), int(l)))
    return out


def run_perplexity(ckpt_dir: Path, eval_dir: Path, *, data_prefix: str, families: str,
                   model_name: str, data_fraction: float) -> list:
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--data.data_prefix", data_prefix,
        "--checkpoint.serialization_dir", str(eval_dir),
        "--checkpoint.checkpoint_path", str(ckpt_dir),
        "--data.families", f"[{families}]",
        "--model.model_name", model_name,
        "--training.valid_bsz", "8",
        "--eval.eval_only", "true",
        "--eval.per_language_eval", "true",
        "--data.data_fraction", str(data_fraction),
        "--data.max_length", "2048",
        "--data.packing", "true",
    ]
    print("[ppl] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    res = eval_dir / "per_language_eval_results.json"
    return json.loads(res.read_text()) if res.exists() else []


def summarize(entries: list) -> dict:
    """Per-family and overall mean perplexity from per-language eval entries."""
    from collections import defaultdict
    by_fam = defaultdict(list)
    allv = []
    for e in entries:
        p = e.get("perplexity")
        if p is None or (isinstance(p, float) and p != p):  # skip None/NaN
            continue
        by_fam[e.get("family", "?")].append(p)
        allv.append(p)
    fam_mean = {f: sum(v) / len(v) for f, v in by_fam.items()}
    return {"per_family_ppl": fam_mean,
            "overall_ppl": (sum(allv) / len(allv)) if allv else None,
            "n_langs": len(allv)}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-3-4b-pt")
    ap.add_argument("--cpt_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--boundaries", default="9,6;6,6;12,6;9,4;9,8;6,4;12,8;15,6;9,10",
                    help="semicolon-separated first,last pairs (middle = [first, N-last))")
    ap.add_argument("--data_prefix", required=True)
    ap.add_argument("--families", default="slavic,germanic,indic,romance,austronesian")
    ap.add_argument("--data_fraction", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.jsonl"

    print(f"[load] CPT: {args.cpt_model}", flush=True)
    cpt = load_text_model(args.cpt_model, args.device)
    print(f"[load] base: {args.base_model}", flush=True)
    base = load_text_model(args.base_model, args.device)
    num_layers = get_num_layers(cpt)
    base_sd = {n: p.detach().to("cpu").clone() for n, p in base.named_parameters()}
    cpt_sd = {n: p.detach().to("cpu").clone() for n, p in cpt.named_parameters()}
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.cpt_model, trust_remote_code=True)

    for first_n, last_n in parse_boundaries(args.boundaries):
        start, end = first_n, num_layers - last_n  # middle band reverted to base
        for n, p in cpt.named_parameters():
            src = base_sd if is_target_param(n, start, end) else cpt_sd
            p.data.copy_(src[n].to(p.device))
        tag = f"first{first_n}_last{last_n}"
        ckpt_dir = out_root / f"revert_{tag}"
        cpt.save_pretrained(ckpt_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(ckpt_dir)

        entries = run_perplexity(
            ckpt_dir, out_root / f"eval_{tag}",
            data_prefix=args.data_prefix, families=args.families,
            model_name=args.base_model, data_fraction=args.data_fraction)
        rec = {"first_n": first_n, "last_n": last_n,
               "middle_start": start, "middle_end": end, "middle_width": end - start,
               **summarize(entries)}
        with summary_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{tag}] middle=[{start},{end}) width={end - start} "
              f"overall_ppl={rec['overall_ppl']}", flush=True)
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    print(f"[done] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
