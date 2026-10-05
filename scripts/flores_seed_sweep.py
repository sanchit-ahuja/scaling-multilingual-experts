#!/usr/bin/env python3
"""FLORES demonstration-seed robustness (Concern 1, generation benchmark).

FLORES uses the default (random) 2-shot sampler with fewshot_split=dev, so the paper's
numbers are one random-demo draw at the default seed (1234). This re-runs the held-in
FLORES generation for a checkpoint under additional fewshot seeds and records corpus ChrF
per task. Combined with the existing seed-1234 results (read offline in the aggregator),
this gives a mean+/-std over seeds so we can show the FLORES conclusions (D-Rev ~ Dense;
Expert < Dense) are stable to demonstration sampling.

Model + TaskManager built once; seeds looped (each re-samples only the in-context demos).
Unlike the MC study, no `_rs` tasks are needed -- the released FLORES tasks are already random.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--tasks", required=True, help="comma-separated flores task names")
    ap.add_argument("--seeds", default="1,2", help="NEW fewshot seeds (baseline 1234 read offline)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", default="32")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    tasks = [t for t in args.tasks.split(",") if t]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[flores] loading {args.pretrained} (bs={args.batch_size}, dtype={args.dtype})", flush=True)
    t0 = time.time()
    lm = HFLM(pretrained=args.pretrained, batch_size=args.batch_size, dtype=args.dtype,
              trust_remote_code=True)
    print(f"[flores] model ready in {time.time()-t0:.0f}s; {len(tasks)} tasks x {len(seeds)} seeds",
          flush=True)

    for seed in seeds:
        ts = time.time()
        res = simple_evaluate(
            model=lm, tasks=tasks, num_fewshot=2, fewshot_random_seed=seed,
            batch_size=args.batch_size, limit=args.limit, log_samples=False, bootstrap_iters=0,
        )
        rows = []
        for task, m in res["results"].items():
            if not task.startswith("flores_"):
                continue
            chrf = m.get("chrf,none", m.get("chrf"))
            if not isinstance(chrf, (int, float)):
                continue
            rows.append({"pretrained": args.pretrained, "task": task, "seed": seed,
                         "chrf": chrf, "bleu": m.get("bleu,none", m.get("bleu"))})
        with out.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        chrfs = [r["chrf"] for r in rows]
        macro = sum(chrfs) / len(chrfs) if chrfs else float("nan")
        print(f"[flores] seed={seed} done in {time.time()-ts:.0f}s macro_chrf={macro:.3f} "
              f"({len(rows)} tasks)", flush=True)

    print(f"[flores] all seeds done -> {out}", flush=True)


if __name__ == "__main__":
    main()
