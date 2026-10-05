#!/usr/bin/env python3
"""Bootstrap confidence intervals for the main downstream tables.

Reads the per-sample lm-eval JSONL logs that already exist on disk (no GPU, no
re-running models) and produces, for every (family, strategy) cell of the
Belebele / Global-PIQA tables:
  - the point estimate (macro-mean over the family's languages), which must
    reproduce the published table cell, and
  - a percentile bootstrap 95% CI + std over the evaluation set.

For MC accuracy (0/1 per doc) the bootstrap-resampled per-language accuracy is
exactly Binomial(n, p_hat)/n, so we sample that directly (fast + exact).

Checkpoints are identified by their unambiguous model sub-directory
(``__work__nvme__bfzp__checkpoints__<name>``); this side-steps the messy,
duplicated campaign-dir names and the v1/v2 Dense ambiguity.

Outputs land under ``rebuttal/`` (never touching the paper tree).
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggregate_downstream_results import LANG_CODE_MAP, get_language_family  # noqa: E402
from configs.constants import LANGS  # noqa: E402

RESULTS_BASE = Path(os.environ.get("RESULTS_BASE", os.environ.get("DATA_ROOT", "/work/nvme/bfzp")))
OUT_DIR = ROOT / "rebuttal"

FAMILIES = ["Slavic", "Germanic", "Indic", "Austronesian", "Romance"]
STRATEGY_COLS = [
    "base", "dense", "expert", "dense-reverted",
    "expert-reverted", "freeze", "layer-reg", "expert-soup",
]
COL_LABELS = {
    "base": "Base", "dense": "Dense", "expert": "Expert",
    "dense-reverted": "D.-Rev.", "expert-reverted": "E.-Rev.",
    "freeze": "Freeze", "layer-reg": "L.-Reg", "expert-soup": "Soup",
}
SHARED = {"base", "dense", "dense-reverted", "expert-soup"}

_FAM_TAGS = {"slavic": "Slavic", "germanic": "Germanic", "indic": "Indic",
             "austronesian": "Austronesian", "romance": "Romance"}


def classify(ident: str):
    """Map a checkpoint identifier string to (strategy, family), or None to skip.

    Canonical choices: Dense = v2 (v1 excluded); Indic expert = checkpoint-7000;
    Austronesian revert = -9000; Qwen/Llama and freeze/translation-best soups excluded.
    """
    t = ident.lower()
    if "qwen" in t or "llama" in t:
        return None
    if "dense_25b_v2" in t:
        return ("dense-reverted", "All") if "revert" in t else ("dense", "All")
    if "dense_25b" in t:            # v1 dense — not the paper checkpoint
        return None
    if "freeze_best" in t or "translation_best" in t:
        return None                # additional soups (appendix), not the main Soup
    if "expert_soup" in t:
        return ("expert-soup", "All")
    if "gemma-3-4b-pt" in t:        # base model (hyphenated), never a CPT expert
        return ("base", "All")
    fam = next((cap for key, cap in _FAM_TAGS.items() if key in t), None)
    if fam is None:
        return None
    if "freeze" in t:
        return ("freeze", fam)
    if "layer_reg" in t or "layer_range" in t:
        return ("layer-reg", fam)
    if "expert" in t:
        return ("expert-reverted", fam) if "revert" in t else ("expert", fam)
    return None


def ident_of(path: str) -> str:
    """Checkpoint identifier for a samples file (handles nested model-subdir or flat campaign dir)."""
    d = Path(os.path.dirname(path))
    base = d.name
    if "checkpoints__" in base:
        return base.split("checkpoints__")[-1]
    if "gemma-3-4b-pt" in base:
        return base
    for anc in [d, *d.parents]:
        if anc.name.startswith("baseline_lm_eval_"):
            return anc.name
    return base


# Held-out (within-family relative) languages, 2-letter (for the held-out tables).
HELDOUT = {
    "Slavic": ["bg", "cs", "pl", "sl", "lt", "lv"],
    "Germanic": ["sv", "no", "is"],
    "Indic": ["as", "gu", "or", "pa", "si", "ur", "sd"],
    "Austronesian": ["ilo", "war", "mg", "mi", "su"],
    "Romance": ["ca"],
}

TASK_NAME = {"belebele": "belebele", "global_piqa": "global_piqa_completions"}

# Published held-in main-table values, for validation (column order = STRATEGY_COLS).
PAPER = {
    "belebele": {
        "Slavic": [0.719, 0.619, 0.726, 0.697, 0.733, 0.740, 0.719, 0.734],
        "Germanic": [0.758, 0.642, 0.756, 0.729, 0.760, 0.758, 0.755, 0.761],
        "Indic": [0.601, 0.535, 0.593, 0.596, 0.598, 0.625, 0.621, 0.629],
        "Austronesian": [0.668, 0.587, 0.661, 0.667, 0.685, 0.698, 0.680, 0.682],
        "Romance": [0.748, 0.625, 0.746, 0.716, 0.755, 0.759, 0.745, 0.749],
    },
    "global_piqa": {
        "Slavic": [0.798, 0.792, 0.808, 0.792, 0.802, 0.803, 0.780, 0.800],
        "Germanic": [0.755, 0.720, 0.760, 0.725, 0.745, 0.750, 0.760, 0.765],
        "Indic": [0.574, 0.561, 0.584, 0.573, 0.573, 0.580, 0.564, 0.584],
        "Austronesian": [0.635, 0.652, 0.630, 0.635, 0.650, 0.660, 0.645, 0.632],
        "Romance": [0.699, 0.737, 0.707, 0.729, 0.703, 0.701, 0.698, 0.708],
    },
}


def raw_of(fname: str, task: str) -> str | None:
    """Full language identifier (e.g. afr_Latn, fra_latn_cana) for a samples file, or None."""
    m = re.match(rf"samples_{task}_(.+?)_(\d{{4}}-\d{{2}}-\d{{2}}T[\d\-.]+)\.jsonl$", fname)
    return m.group(1) if m else None


def raw_lang2(raw: str) -> str:
    """2-letter language code from a raw identifier (regional variants share a 2-letter code)."""
    return LANG_CODE_MAP.get(raw.split("_")[0], raw.split("_")[0])


def timestamp_of(fname: str, task: str) -> str:
    m = re.match(rf"samples_{task}_.+?_(\d{{4}}-\d{{2}}-\d{{2}}T[\d\-.]+)\.jsonl$", fname)
    return m.group(1) if m else ""


# Canonical campaign dirs per split, highest priority first (paper tables were
# built from these; other locations are fallback). `in_domain_romance_archive`
# holds the canonical Soup + Romance held-in evals; `training_lang_in_domain_results`
# the rest. This avoids divergent stray reruns elsewhere on disk.
PREF_CAMPAIGN = {
    "held-in": ["in_domain_romance_archive", "training_lang_in_domain_results"],
    "held-out": ["out_domain_results", "archive_romance_out_domain_results"],
}


def campaign_rank(path: str, split: str) -> int:
    prefs = PREF_CAMPAIGN[split]
    for i, name in enumerate(prefs):
        if f"/{name}/" in path:
            return len(prefs) - i   # higher = more preferred
    return 0


def raw_region3(raw: str) -> str:
    """3-char region tag from a raw id (e.g. fra_latn_cana -> 'can'), else ''."""
    parts = raw.split("_")
    return parts[2][:3] if len(parts) >= 3 else ""


def collect_candidates(benchmark: str, split: str) -> dict:
    """cand[(strategy, ckpt_family)][raw_lang] -> [(rank, ts, path), ...] best-first.

    Keeps *all* candidate runs (dedup deferred to selection), keyed on the full raw
    language id so PIQA regional variants (fr can / fr fran) never collide.
    """
    task = TASK_NAME[benchmark]
    cand: dict = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(str(RESULTS_BASE / f"**/samples_{task}_*.jsonl"), recursive=True):
        cls = classify(ident_of(path))
        if cls is None:
            continue
        strat, fam = cls
        fname = os.path.basename(path)
        raw = raw_of(fname, task)
        if raw is None:
            continue
        cand[(strat, fam)][raw].append(
            (campaign_rank(path, split), timestamp_of(fname, task), path))
    for d in cand.values():
        for raw in d:
            d[raw].sort(reverse=True)  # highest (campaign_rank, timestamp) first
    return cand


_ARR_CACHE: dict = {}


def get_array(path: str, metric: str = "acc") -> np.ndarray:
    if path not in _ARR_CACHE:
        _ARR_CACHE[path] = np.asarray([json.loads(l)[metric] for l in open(path)], dtype=float)
    return _ARR_CACHE[path]


def select_array(candidates: list, target):
    """Pick the run whose mean matches the paper's per-language value (ground truth).

    `target` is (value, n_decimals) or None. Returns (array, matched?). If no run
    matches (or no target), falls back to the best-ranked run.
    """
    for _rank, _ts, path in candidates:
        arr = get_array(path)
        if target is None:
            return arr, True
        val, ndec = target
        if round(float(arr.mean()), ndec) == round(val, ndec):
            return arr, True
    return get_array(candidates[0][2]), target is None


# ---- ground-truth per-language values parsed from the paper's appendix tables ----
PAPER_DIR = Path("/u/sahuja1/6a014ac79b78946b0c4114eb")
PERLANG_TEX = {
    ("belebele", "held-in"): "belebele_perlang.tex",
    ("global_piqa", "held-in"): "piqa_perlang.tex",
    ("belebele", "held-out"): "belebele_heldout_perlang.tex",
    ("global_piqa", "held-out"): "piqa_heldout_perlang.tex",
}


def parse_label(label: str, benchmark: str):
    """Table language label -> (lang2, region3). Belebele: 'Macedonian (mk)'; PIQA: 'fr (can)'/'it'."""
    if benchmark == "belebele":
        m = re.search(r"\(([a-z]{2,3})\)", label)
        return (m.group(1), "") if m else (label.strip().lower(), "")
    if "(" in label:
        m = re.search(r"\(([a-z]+)\)", label)
        return label.split("(")[0].strip(), (m.group(1)[:3] if m else "")
    return label.strip(), ""


def load_targets(benchmark: str, split: str) -> dict:
    """target[(family, lang2, region3)] -> [(value, ndec)] * 8 (STRATEGY_COLS order)."""
    fn = PERLANG_TEX.get((benchmark, split))
    tex = PAPER_DIR / "tables" / fn if fn else None
    out: dict = {}
    if not tex or not tex.exists():
        return out
    last_fam = None
    for line in open(tex):
        if "&" not in line or r"\\" not in line:
            continue
        cols = [c.strip() for c in line.split(r"\\")[0].split("&")]
        if len(cols) != 10:
            continue
        fam = cols[0] if cols[0] in FAMILIES else last_fam
        if fam is None:
            continue
        last_fam = fam
        vals, ok = [], True
        for c in cols[2:10]:
            m = re.search(r"(\d+\.\d+)", c)
            if not m:
                ok = False
                break
            vals.append((float(m.group(1)), len(m.group(1).split(".")[1])))
        if not ok:
            continue
        l2, reg = parse_label(cols[1], benchmark)
        out[(fam, l2, reg)] = vals
    return out


def cell_langs(benchmark: str, family: str, split: str, raws) -> list:
    """Raw language ids that make up a (family) cell for the given benchmark/split."""
    out = []
    for raw in raws:
        l2 = raw_lang2(raw)
        if split == "held-in":
            in_cell = (l2 in LANGS[family.lower()]) if benchmark == "belebele" \
                else (get_language_family(l2) == family)
        else:  # held-out relatives
            in_cell = l2 in HELDOUT[family]
        if in_cell:
            out.append(raw)
    return sorted(out)


def source_key(strategy: str, family: str) -> tuple:
    return (strategy, "All") if strategy in SHARED else (strategy, family)


def bootstrap_cell(lang_arrays: list, n_boot: int, rng) -> np.ndarray:
    """B macro-mean replicates. Exact 0/1 bootstrap via Binomial(n, p_hat)/n per language."""
    per_lang = []
    for arr in lang_arrays:
        n = len(arr)
        p = float(arr.mean())
        per_lang.append(rng.binomial(n, p, size=n_boot) / n)
    return np.mean(np.vstack(per_lang), axis=0)  # macro over languages


def build_table(benchmark: str, split: str, n_boot: int, seed: int):
    cand = collect_candidates(benchmark, split)
    targets = load_targets(benchmark, split)
    rng = np.random.default_rng(seed)
    rows = {}
    boot_by_col = defaultdict(list)  # strategy -> list of per-family replicate arrays
    fallbacks = []
    for fam in FAMILIES:
        rows[fam] = {}
        for strat in STRATEGY_COLS:
            src = source_key(strat, fam)
            langs = cell_langs(benchmark, fam, split, set(cand.get(src, {})))
            if not langs:
                rows[fam][strat] = None
                continue
            si = STRATEGY_COLS.index(strat)
            arrays = []
            for raw in langs:
                tkey = (fam, raw_lang2(raw), raw_region3(raw))
                tgt = targets[tkey][si] if tkey in targets else None
                arr, matched = select_array(cand[src][raw], tgt)
                if tgt is not None and not matched:
                    fallbacks.append((fam, strat, raw))
                arrays.append(arr)
            point = float(np.mean([a.mean() for a in arrays]))
            reps = bootstrap_cell(arrays, n_boot, rng)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            rows[fam][strat] = {"point": point, "lo": float(lo), "hi": float(hi),
                                "std": float(reps.std()), "n_langs": len(langs), "langs": langs}
            boot_by_col[strat].append(reps)
    if fallbacks:
        print(f"  [warn] {len(fallbacks)} cell-languages had no ground-truth match "
              f"(used best-ranked run): {fallbacks[:10]}")
    # Average row = mean over families (per replicate) for columns present in all families.
    avg = {}
    for strat in STRATEGY_COLS:
        arrs = boot_by_col[strat]
        pts = [rows[f][strat]["point"] for f in FAMILIES if rows[f][strat]]
        if len(arrs) == len(FAMILIES):
            reps = np.mean(np.vstack(arrs), axis=0)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            avg[strat] = {"point": float(np.mean(pts)), "lo": float(lo), "hi": float(hi),
                          "std": float(reps.std())}
        elif pts:
            avg[strat] = {"point": float(np.mean(pts)), "lo": None, "hi": None, "std": None}
        else:
            avg[strat] = None
    return rows, avg


def validate(benchmark: str, rows: dict):
    print(f"\n=== VALIDATION: {benchmark} held-in point estimates vs paper ===")
    ref = PAPER[benchmark]
    max_diff = 0.0
    for fam in FAMILIES:
        line = [f"{fam:<13}"]
        for j, strat in enumerate(STRATEGY_COLS):
            cell = rows[fam][strat]
            paper = ref[fam][j]
            if cell is None:
                line.append(f"{COL_LABELS[strat]}: MISSING")
                continue
            diff = abs(cell["point"] - paper)
            max_diff = max(max_diff, diff)
            flag = "" if diff <= 0.0015 else f" !!DIFF {diff:.3f} (paper {paper})"
            line.append(f"{cell['point']:.3f}{flag}")
        print("  " + "  ".join(line))
    print(f"  --> max |mine - paper| = {max_diff:.4f}  (want <= 0.0015 rounding)")


def fmt_cell(c):
    if c is None:
        return "--"
    if c["lo"] is None:
        return f"{c['point']:.3f}"
    return f"{c['point']:.3f} [{c['lo']:.3f},{c['hi']:.3f}]"


def write_outputs(benchmark: str, split: str, rows: dict, avg: dict):
    OUT_DIR.joinpath("tables").mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "tables" / f"{benchmark}_{split}_ci.csv"
    with open(csv_path, "w") as f:
        f.write("family," + ",".join(
            f"{COL_LABELS[s]}_point,{COL_LABELS[s]}_lo,{COL_LABELS[s]}_hi,{COL_LABELS[s]}_std"
            for s in STRATEGY_COLS) + "\n")
        for fam in FAMILIES + ["Average"]:
            src = rows.get(fam, avg) if fam != "Average" else avg
            cells = [src.get(s) for s in STRATEGY_COLS] if fam == "Average" else [rows[fam][s] for s in STRATEGY_COLS]
            parts = [fam]
            for c in cells:
                if c is None:
                    parts += ["", "", "", ""]
                else:
                    parts += [f"{c['point']:.4f}",
                              "" if c["lo"] is None else f"{c['lo']:.4f}",
                              "" if c["hi"] is None else f"{c['hi']:.4f}",
                              "" if c["std"] is None else f"{c['std']:.4f}"]
            f.write(",".join(parts) + "\n")
    print(f"\nWrote {csv_path}")

    print(f"\n=== {benchmark} ({split}) point [95% CI] ===")
    header = "Family".ljust(13) + "".join(COL_LABELS[s].ljust(20) for s in STRATEGY_COLS)
    print(header)
    for fam in FAMILIES:
        print(fam.ljust(13) + "".join(fmt_cell(rows[fam][s]).ljust(20) for s in STRATEGY_COLS))
    print("Average".ljust(13) + "".join(fmt_cell(avg[s]).ljust(20) for s in STRATEGY_COLS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["belebele", "global_piqa", "both"], default="both")
    ap.add_argument("--split", choices=["held-in", "held-out"], default="held-in")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    benches = ["belebele", "global_piqa"] if args.benchmark == "both" else [args.benchmark]
    for bench in benches:
        rows, avg = build_table(bench, args.split, args.n_boot, args.seed)
        if args.validate and args.split == "held-in":
            validate(bench, rows)
        write_outputs(bench, args.split, rows, avg)


if __name__ == "__main__":
    main()
