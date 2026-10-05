#!/usr/bin/env python3
"""Aggregate FLORES seed-sweep into per-(family, strategy) cells.

Combines the newly generated seeds (from flores_seed_sweep.py) with the existing seed-1234
baseline (read from the released FLORES result JSONs), and reports per cell the ChrF
mean+/-std over the 3 seeds, macro over the family's held-in languages x 2 directions.
This shows the FLORES conclusions are stable to demonstration sampling.
"""
from __future__ import annotations

import csv
import glob
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.flores_jobs as fj  # noqa: E402
from scripts.flores_postprocess import truncate_translation as _trunc  # noqa: E402

OUT_ROOT = Path("/work/nvme/bfzp/flores_seed_sweep")
TABLES = ROOT / "rebuttal" / "tables"
BASELINE_SEED = 1234

# base task name -> family
TASK_FAMILY = {}
for fam, langs in fj.FAM_LANGS.items():
    for t in fj.tasks_for(langs):
        TASK_FAMILY[t] = fam

TAG2COL = {"dense": "Dense", "dense-reverted": "D.-Rev.", "expert": "Expert",
           "freeze": "Freeze", "layer-reg": "L.-Reg"}
FAMILIES = ["Slavic", "Germanic", "Indic", "Austronesian", "Romance"]


def _baseline_dir_index():
    """{pretrained: result_dir} for the existing (seed-1234) FLORES runs; the dir holds the
    per-task samples_flores_*.jsonl with raw generations. Prefer most tasks, then newest."""
    idx = {}
    for jf in glob.glob("/work/nvme/bfzp/**/results_*.json", recursive=True):
        if "flores" not in jf.lower():
            continue
        try:
            d = json.loads(Path(jf).read_text())
        except Exception:
            continue
        chrf = {k for k, v in d.get("results", {}).items()
                if k.startswith("flores_") and isinstance(v.get("chrf,none", v.get("chrf")), (int, float))}
        if not chrf:
            continue
        ma = d.get("config", {}).get("model_args", "")
        pre = ma.get("pretrained", "") if isinstance(ma, dict) else \
            (re.search(r"pretrained=([^,]+)", ma).group(1) if isinstance(ma, str) and "pretrained=" in ma else "")
        key = (len(chrf), Path(jf).stat().st_mtime)
        if pre and (pre not in idx or key > idx[pre][0]):
            idx[pre] = (key, str(Path(jf).parent))
    return {p: d for p, (k, d) in idx.items()}


def _truncated_corpus_chrf(samples_file: str):
    """Recompute paper-protocol (truncated) corpus ChrF from a FLORES samples jsonl."""
    refs, hyps = [], []
    for line in open(samples_file):
        s = json.loads(line)
        resp = s.get("filtered_resps") or s.get("resps")
        while isinstance(resp, list):
            resp = resp[0]
        tgt = s.get("target")
        refs.append(tgt if isinstance(tgt, str) else tgt[0])
        hyps.append(_trunc(resp))
    if not refs:
        return None
    return sacrebleu.corpus_chrf(hyps, [refs]).score


class BaselineScorer:
    """Lazily recompute paper-protocol (truncated) seed-1234 ChrF for needed (checkpoint, task)."""

    def __init__(self):
        self.dir_idx = _baseline_dir_index()
        self.cache = {}  # (pretrained, task) -> chrf

    def chrf(self, pretrained: str, task: str):
        key = (pretrained, task)
        if key in self.cache:
            return self.cache[key]
        dirp = self.dir_idx.get(pretrained)
        val = None
        if dirp:
            sfs = glob.glob(f"{dirp}/samples_{task}_*.jsonl")
            if sfs:
                val = _truncated_corpus_chrf(sorted(sfs)[-1])
        self.cache[key] = val
        return val


def strat_of(tag: str) -> str:
    return tag.rsplit("_", 1)[0]  # dense / dense-reverted / expert


def main():
    scorer = BaselineScorer()
    # cell[(col, fam)] = {"pretrained":.., "seed": {seed: {task: chrf}}}
    cell = defaultdict(lambda: {"pretrained": None, "seed": defaultdict(dict)})
    for fp in sorted(OUT_ROOT.glob("*.jsonl")):
        col = TAG2COL.get(strat_of(fp.stem))
        if col is None:
            continue
        for line in fp.open():
            r = json.loads(line)
            fam = TASK_FAMILY.get(r["task"])
            if fam is None or r["chrf"] is None:
                continue
            c = cell[(col, fam)]
            c["pretrained"] = r["pretrained"]
            c["seed"][r["seed"]][r["task"]] = r["chrf"]

    # attach seed-1234 baseline for each cell: re-truncate the raw generations (paper protocol),
    # restricted to that family's held-in tasks that the new-seed run also covered.
    for (col, fam), c in cell.items():
        new_seed_tasks = set().union(*[set(td) for td in c["seed"].values()]) if c["seed"] else set()
        base = {}
        for t in new_seed_tasks:
            v = scorer.chrf(c["pretrained"], t)
            if v is not None:
                base[t] = v
        if base:
            c["seed"][BASELINE_SEED] = base

    print("===== FLORES: ChrF mean+/-std over seeds (macro over family langs x 2 dirs) =====")
    print(f"{'family':13s} {'strategy':9s} {'seed1234':>9s} {'mean':>8s} {'std':>7s} "
          f"{'range':>7s} {'#task':>5s} {'#seed':>5s}")
    out_csv = TABLES / "flores_seed_robustness.csv"
    with out_csv.open("w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["family", "strategy", "seed1234", "mean_chrf", "std_chrf",
                    "range_chrf", "n_tasks", "n_seeds"])
        for fam in FAMILIES:
            for strat, col in TAG2COL.items():
                c = cell.get((col, fam))
                if not c or not c["seed"]:
                    continue
                seed_sets = [set(td) for td in c["seed"].values()]
                common = set.intersection(*seed_sets)
                if not common:
                    continue
                per_seed = {s: statistics.mean(c["seed"][s][t] for t in common) for s in c["seed"]}
                vals = list(per_seed.values())
                mean = statistics.mean(vals)
                sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                base = per_seed.get(BASELINE_SEED, float("nan"))
                rng = max(vals) - min(vals)
                print(f"{fam:13s} {col:9s} {base:9.3f} {mean:8.3f} {sd:7.3f} {rng:7.3f} "
                      f"{len(common):5d} {len(vals):5d}")
                w.writerow([fam, col, round(base, 3), round(mean, 3), round(sd, 3),
                            round(rng, 3), len(common), len(vals)])
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
