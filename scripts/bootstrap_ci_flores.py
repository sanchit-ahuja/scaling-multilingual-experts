#!/usr/bin/env python3
"""Bootstrap confidence intervals for the main FLORES-200 ChrF table.

FLORES generation is not re-run: we read the saved per-sample generations, apply
the paper's post-truncation protocol (identical to x-elm-v2/retruncate_flores.py),
and bootstrap **corpus** ChrF by resampling sentences. Corpus ChrF cannot be
resampled as a scalar, so we precompute each sentence's chrF statistics once
(sacrebleu's `_extract_corpus_statistics`) and, per replicate, resample sentence
counts, sum the stat vectors, and recompute the score.

A family cell = macro-mean over the family's training languages x {en->xx, xx->en}.
Reproduces tables/flores_main.tex, then attaches a 95% percentile CI + std.
Outputs to rebuttal/tables/ (paper tree untouched).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from aggregate_downstream_results import LANG_CODE_MAP  # noqa: E402
from configs.constants import LANGS  # noqa: E402

RESULTS_BASE = Path(os.environ.get("RESULTS_BASE", os.environ.get("DATA_ROOT", "/work/nvme/bfzp")))
OUT_DIR = ROOT / "rebuttal" / "tables"
FAMILIES = ["Slavic", "Germanic", "Indic", "Austronesian", "Romance"]
STRATEGY_COLS = ["base", "dense", "expert", "dense-reverted",
                 "expert-reverted", "freeze", "layer-reg", "expert-soup"]
COL_LABELS = {"base": "Base", "dense": "Dense", "expert": "Expert", "dense-reverted": "D.-Rev.",
              "expert-reverted": "E.-Rev.", "freeze": "Freeze", "layer-reg": "L.-Reg", "expert-soup": "Soup"}
SHARED = {"base", "dense", "dense-reverted", "expert-soup"}

# Canonical FLORES result-dir keys per (strategy, ckpt_family); ckpt_family None = shared.
# (Identical to x-elm-v2/retruncate_flores.py, which produced flores_truncated_rescore.csv.)
DIRS = {
    ("base", None): ["google_gemma-3-4b-pt", "Base_google_gemma-3-4b-pt"],
    ("dense", None): ["Dense_gemma_4b_dense_25b_v2_final"],
    ("dense-reverted", None): ["Dense_gemma_4b_dense_25b_v2_reverted"],
    ("expert-soup", None): ["All_checkpoints_gemma_4b_expert_soup"],
}
for fam, tag in [("Austronesian", "austronesian"), ("Germanic", "germanic"),
                 ("Romance", "romance"), ("Slavic", "slavic")]:
    DIRS[("expert", fam)] = [f"{fam}_{tag}_gemma_4b_expert_final"]
    DIRS[("expert-reverted", fam)] = [f"{fam}_{tag}_gemma_4b_expert_reverted"]
    DIRS[("freeze", fam)] = [f"{fam}_gemma_4b_{tag}_freeze_final"]
    DIRS[("layer-reg", fam)] = [f"{fam}_gemma_4b_{tag}_layer_reg_final"]
DIRS[("expert", "Indic")] = ["Indic_Indic_gemma_4b_expert_checkpoint-7000"]
DIRS[("expert-reverted", "Indic")] = ["Indic_Indic_gemma_4b_expert_reverted-7000"]
DIRS[("expert-reverted", "Austronesian")] = ["Austronesian_austronesian_gemma_4b_expert_reverted-9000"]
DIRS[("freeze", "Indic")] = ["Indic_gemma_4b_indic_freeze_final"]
DIRS[("layer-reg", "Indic")] = ["Indic_gemma_4b_indic_layer_reg_final"]

PAPER_FLORES = {  # tables/flores_main.tex, column order = STRATEGY_COLS
    "Slavic": [33.6, 52.7, 47.4, 53.6, 48.6, 45.7, 44.1, 46.2],
    "Germanic": [34.8, 59.0, 57.0, 59.5, 57.1, 58.0, 49.4, 50.4],
    "Indic": [29.9, 43.5, 36.1, 44.2, 40.7, 40.1, 36.2, 42.6],
    "Austronesian": [33.8, 55.7, 25.0, 53.6, 27.8, 35.2, 42.4, 48.1],
    "Romance": [35.0, 58.2, 55.0, 59.0, 48.2, 44.3, 53.1, 49.6],
}

_CHRF = CHRF()  # defaults == sacrebleu.corpus_chrf


def truncate(g: str) -> str:
    """Paper protocol: lstrip, cut at first literal '\\n' or real newline, strip."""
    if not g:
        return g
    g = g.lstrip()
    cuts = [i for i in (g.find("\\n"), g.find("\n")) if i >= 0]
    if cuts:
        g = g[: min(cuts)]
    return g.strip().lstrip(" ").strip()


def parse_pair(fname: str):
    """(direction, non_english_lang_code) from a samples filename, or None."""
    m = re.match(r"samples_flores_([a-z]+_[A-Z][a-z]+)-([a-z]+_[A-Z][a-z]+)_", os.path.basename(fname))
    if not m:
        return None
    src, tgt = m.group(1), m.group(2)
    if src.startswith("eng_"):
        return "en_xx", tgt
    if tgt.startswith("eng_"):
        return "xx_en", src
    return None


def find_task_files(dir_keys):
    """(direction, lang_code) -> jsonl path, deduped across matching dirs."""
    out = {}
    for key in dir_keys:
        for d in glob.glob(str(RESULTS_BASE / f"baseline_lm_eval_{key}_flores")):
            for fp in glob.glob(f"{d}/*/samples_flores_*.jsonl"):
                parsed = parse_pair(fp)
                if parsed:
                    out.setdefault(parsed, fp)
    return out


def task_stats(jsonl_path: str) -> np.ndarray:
    """Per-sentence chrF statistic vectors (n_sent x 18) after truncation."""
    refs, preds = [], []
    for line in open(jsonl_path):
        s = json.loads(line)
        resps = s.get("resps") or s.get("filtered_resps")
        if not resps:
            continue
        g = resps[0][0] if isinstance(resps[0], list) else resps[0]
        refs.append(s.get("target", "") or "")
        preds.append(truncate(g))
    if not refs:
        return np.empty((0, 0))
    return np.asarray(_CHRF._extract_corpus_statistics(preds, [refs]), dtype=float)


def chrf_from_stats(summed) -> float:
    return _CHRF._compute_score_from_stats(list(summed)).score


def bootstrap_task(stats: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """B corpus-ChrF replicates for one task (resample sentences via multinomial counts)."""
    n = stats.shape[0]
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)  # (B, n)
    summed = counts @ stats  # (B, 18)
    return np.array([chrf_from_stats(summed[b]) for b in range(n_boot)])


def cell_langs(family: str, lang_codes) -> list:
    train = set(LANGS[family.lower()])
    return sorted(c for c in lang_codes if LANG_CODE_MAP.get(c.split("_")[0], c.split("_")[0]) in train)


def build_table(n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    # cache task stats by path to avoid recomputation across cells
    stats_cache: dict = {}
    rows, boot_by_col = {}, defaultdict(list)
    for fam in FAMILIES:
        rows[fam] = {}
        for strat in STRATEGY_COLS:
            ckpt_fam = None if strat in SHARED else fam
            task_files = find_task_files(DIRS[(strat, ckpt_fam)])
            # keep this family's training languages, both directions
            tasks = [(d, lg, fp) for (d, lg), fp in task_files.items()
                     if lg in cell_langs(fam, [lg2 for (_dd, lg2) in task_files])]
            if not tasks:
                rows[fam][strat] = None
                continue
            task_points, task_reps = [], []
            for _d, _lg, fp in tasks:
                if fp not in stats_cache:
                    stats_cache[fp] = task_stats(fp)
                st = stats_cache[fp]
                if st.size == 0:
                    continue
                task_points.append(chrf_from_stats(st.sum(axis=0)))
                task_reps.append(bootstrap_task(st, n_boot, rng))
            point = float(np.mean(task_points))
            reps = np.mean(np.vstack(task_reps), axis=0)  # macro over tasks per replicate
            lo, hi = np.percentile(reps, [2.5, 97.5])
            rows[fam][strat] = {"point": point, "lo": float(lo), "hi": float(hi),
                                "std": float(reps.std()), "n_tasks": len(task_points)}
            boot_by_col[strat].append(reps)
    avg = {}
    for strat in STRATEGY_COLS:
        arrs, pts = boot_by_col[strat], [rows[f][strat]["point"] for f in FAMILIES if rows[f][strat]]
        if len(arrs) == len(FAMILIES):
            reps = np.mean(np.vstack(arrs), axis=0)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            avg[strat] = {"point": float(np.mean(pts)), "lo": float(lo), "hi": float(hi), "std": float(reps.std())}
        else:
            avg[strat] = {"point": float(np.mean(pts)), "lo": None, "hi": None, "std": None} if pts else None
    return rows, avg


def validate(rows):
    print("\n=== VALIDATION: FLORES held-in point estimates vs paper ===")
    maxd = 0.0
    for fam in FAMILIES:
        cells = []
        for j, s in enumerate(STRATEGY_COLS):
            c = rows[fam][s]
            if c is None:
                cells.append(f"{COL_LABELS[s]}:MISS"); continue
            d = abs(c["point"] - PAPER_FLORES[fam][j]); maxd = max(maxd, d)
            cells.append(f"{c['point']:.1f}" + ("" if d <= 0.1 else f"!{PAPER_FLORES[fam][j]}"))
        print(f"  {fam:<13}" + "  ".join(cells))
    print(f"  --> max |mine - paper| = {maxd:.2f} ChrF (want <= 0.1)")


def fmt(c):
    if c is None:
        return "--"
    return f"{c['point']:.1f}" + ("" if c["lo"] is None else f" [{c['lo']:.1f},{c['hi']:.1f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    rows, avg = build_table(args.n_boot, args.seed)
    if args.validate:
        validate(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "flores_held-in_ci.csv"
    with open(csv_path, "w") as f:
        f.write("family," + ",".join(f"{COL_LABELS[s]}_point,{COL_LABELS[s]}_lo,{COL_LABELS[s]}_hi,{COL_LABELS[s]}_std"
                                     for s in STRATEGY_COLS) + "\n")
        for fam in FAMILIES + ["Average"]:
            src = rows[fam] if fam != "Average" else avg
            parts = [fam]
            for s in STRATEGY_COLS:
                c = src[s]
                parts += ["", "", "", ""] if c is None else [
                    f"{c['point']:.2f}", "" if c["lo"] is None else f"{c['lo']:.2f}",
                    "" if c["hi"] is None else f"{c['hi']:.2f}", "" if c["std"] is None else f"{c['std']:.3f}"]
            f.write(",".join(parts) + "\n")
    print(f"\nWrote {csv_path}")
    print("\n=== FLORES (held-in) ChrF point [95% CI] ===")
    print("Family".ljust(13) + "".join(COL_LABELS[s].ljust(20) for s in STRATEGY_COLS))
    for fam in FAMILIES:
        print(fam.ljust(13) + "".join(fmt(rows[fam][s]).ljust(20) for s in STRATEGY_COLS))
    print("Average".ljust(13) + "".join(fmt(avg[s]).ljust(20) for s in STRATEGY_COLS))


if __name__ == "__main__":
    main()
