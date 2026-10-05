#!/usr/bin/env python3
"""E3: few-shot demonstration-seed robustness for the MC benchmarks (Concern 1).

For one checkpoint and one benchmark, evaluate the random-sampler ("_rs") held-in tasks
under several fewshot seeds. The released tasks use `sampler: first_n` (deterministic), so
the paper's headline numbers are untouched; here we vary only the *fewshot* seed (4th slot of
lm-eval's seed tuple) with `sampler: default` to measure demonstration-sampling variance.

The 4B model and the TaskManager are built once and reused across all seeds (each seed only
re-samples the in-context demonstrations), so cost ≈ one model load + N_seeds eval passes.
Writes one JSON line per (seed, task) with acc/acc_norm to --output.
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
    ap.add_argument("--pretrained", required=True, help="checkpoint dir or HF id")
    ap.add_argument("--benchmark", required=True, choices=["belebele", "global_piqa"])
    ap.add_argument("--tasks", required=True, help="comma-separated _rs task names")
    ap.add_argument("--include_path", required=True, help="dir with the _rs task yamls")
    ap.add_argument("--seeds", default="1,2,3,4,5", help="comma-separated fewshot seeds")
    ap.add_argument("--output", required=True, help="output jsonl (appended)")
    ap.add_argument("--batch_size", default="32")
    ap.add_argument("--limit", type=int, default=None, help="per-task doc cap for smoke tests")
    ap.add_argument("--dtype", default="auto")  # match paper's baseline eval (no forced dtype)
    args = ap.parse_args()

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    tasks = [t for t in args.tasks.split(",") if t]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[e3] loading {args.pretrained} (bs={args.batch_size}, dtype={args.dtype})", flush=True)
    t0 = time.time()
    lm = HFLM(pretrained=args.pretrained, batch_size=args.batch_size, dtype=args.dtype,
              trust_remote_code=True)
    tm = TaskManager(include_path=args.include_path)
    print(f"[e3] model+taskmgr ready in {time.time()-t0:.0f}s; {len(tasks)} tasks x {len(seeds)} seeds",
          flush=True)

    for seed in seeds:
        ts = time.time()
        res = simple_evaluate(
            model=lm,
            tasks=tasks,
            task_manager=tm,
            num_fewshot=2,
            fewshot_random_seed=seed,
            batch_size=args.batch_size,
            limit=args.limit,
            log_samples=False,
            bootstrap_iters=0,
        )
        rows = []
        for task, m in res["results"].items():
            if "acc,none" not in m:
                continue
            rows.append({
                "pretrained": args.pretrained,
                "benchmark": args.benchmark,
                "task": task,
                "seed": seed,
                "acc": m.get("acc,none"),
                "acc_norm": m.get("acc_norm,none"),
                "n": res.get("n-samples", {}).get(task, {}).get("effective"),
            })
        with out.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        accs = [r["acc"] for r in rows if r["acc"] is not None]
        macro = sum(accs) / len(accs) if accs else float("nan")
        print(f"[e3] seed={seed} done in {time.time()-ts:.0f}s  macro_acc={macro:.4f} "
              f"({len(rows)} tasks)", flush=True)

    print(f"[e3] all seeds done -> {out}", flush=True)


if __name__ == "__main__":
    main()
