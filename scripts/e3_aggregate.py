#!/usr/bin/env python3
"""Aggregate E3 seed-sweep results into per-(family, strategy) cells.

Reads the per-seed JSONL from e3_seed_sweep.py, groups tasks into family cells (same
grouping as the paper tables), and reports, per cell:
  - random-sampler mean +/- std across fewshot seeds (macro over the family's langs), and
  - a *matched* first_n baseline computed over the identical task set from the released
    (first_n) result JSONs (scripts/e3_matched_firstn.py),
so the delta isolates the demonstration-sampler effect (no language-set/grouping mismatch).
Both macros are taken over the intersection of tasks present in the rs run and the first_n
run for that exact checkpoint, so they are strictly apples-to-apples.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e3_jobs import heldin_by_family, BEL_GRP, PIQA_GRP  # noqa: E402
from scripts.e3_matched_firstn import matched_firstn  # noqa: E402

OUT_ROOT = Path("/work/nvme/bfzp/e3_seed_sweep")
TABLES = ROOT / "rebuttal" / "tables"

TAG2COL = {"base": "Base", "dense": "Dense", "expert": "Expert", "dense-reverted": "D.-Rev.",
           "expert-reverted": "E.-Rev.", "freeze": "Freeze", "layer-reg": "L.-Reg", "soup": "Soup"}
FAMILIES = ["Slavic", "Germanic", "Indic", "Austronesian", "Romance"]


def task_family_map(bench):
    grp = BEL_GRP if bench == "belebele" else PIQA_GRP
    rex = r"-\s*(belebele_\S+)" if bench == "belebele" else r"task:\s*(global_piqa_completions_\S+)"
    fam_tasks = heldin_by_family(grp, rex)
    m = {}
    for fam, tasks in fam_tasks.items():
        for t in tasks:
            m[t] = fam                 # base task name (no _rs) -> family
    return m


def strat_of(tag: str) -> str:
    stem = tag
    for b in ("_global_piqa", "_belebele"):
        if stem.endswith(b):
            stem = stem[: -len(b)]
            break
    return stem.rsplit("_", 1)[0]


def main():
    fam_maps = {b: task_family_map(b) for b in ("belebele", "global_piqa")}
    firstn_idx = {b: matched_firstn(b) for b in ("belebele", "global_piqa")}

    # cell[(bench, col, fam)] = {"pretrained":..., "seed_task_acc": {seed: {base_task: acc}}}
    cell = defaultdict(lambda: {"pretrained": None, "seed": defaultdict(dict)})
    for fp in sorted(OUT_ROOT.glob("*.jsonl")):
        col = TAG2COL.get(strat_of(fp.stem))
        if col is None:
            continue
        for line in fp.open():
            r = json.loads(line)
            bench = r["benchmark"]
            base_task = r["task"][:-3] if r["task"].endswith("_rs") else r["task"]
            fam = fam_maps[bench].get(base_task)
            if fam is None or r["acc"] is None:
                continue
            c = cell[(bench, col, fam)]
            c["pretrained"] = r["pretrained"]
            c["seed"][r["seed"]][base_task] = r["acc"]

    for bench in ("belebele", "global_piqa"):
        print(f"\n===== {bench}: matched first_n vs random-sampler mean+/-std "
              f"(macro over family langs, identical task set) =====")
        print(f"{'family':13s} {'strategy':9s} {'first_n':>8s} {'rs_mean':>8s} {'rs_std':>7s} "
              f"{'delta':>7s} {'#lang':>5s} {'#seed':>5s}")
        out_csv = TABLES / f"e3_{bench}_seed_robustness.csv"
        with out_csv.open("w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["benchmark", "family", "strategy", "first_n_matched",
                        "rs_mean", "rs_std", "delta", "n_langs", "n_seeds"])
            for fam in FAMILIES:
                for strat, col in TAG2COL.items():
                    c = cell.get((bench, col, fam))
                    if not c or not c["seed"]:
                        continue
                    fn_accs = firstn_idx[bench].get(c["pretrained"], {})
                    # tasks present across ALL seeds and in the first_n run (strict intersection)
                    seed_sets = [set(td) for td in c["seed"].values()]
                    common = set.intersection(*seed_sets) & set(fn_accs)
                    if not common:
                        continue
                    per_seed_macro = [statistics.mean(c["seed"][s][t] for t in common)
                                      for s in c["seed"]]
                    fn_macro = statistics.mean(fn_accs[t] for t in common)
                    mean = statistics.mean(per_seed_macro)
                    sd = statistics.pstdev(per_seed_macro) if len(per_seed_macro) > 1 else 0.0
                    delta = mean - fn_macro
                    print(f"{fam:13s} {col:9s} {fn_macro:8.4f} {mean:8.4f} {sd:7.4f} "
                          f"{delta:+7.4f} {len(common):5d} {len(per_seed_macro):5d}")
                    w.writerow([bench, fam, col, round(fn_macro, 4), round(mean, 4),
                                round(sd, 4), round(delta, 4), len(common), len(per_seed_macro)])
        print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
