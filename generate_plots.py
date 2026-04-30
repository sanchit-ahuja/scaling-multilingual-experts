#!/usr/bin/env python3
"""
Plotting script for multilingual expert results on Gemma 4B.
Generates:
  1. Heatmaps (delta from base) per benchmark per expert family
  2. Perplexity vs. downstream tradeoff scatter plots (with Pareto frontier)
  3. Radar charts per family expert

Paper-ready styling: clean fonts, muted academic palette, Pareto curves.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from math import pi
import warnings
warnings.filterwarnings("ignore")

# ─── Matplotlib global style ────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.linewidth": 0.8,
})

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.environ.get("PLOTS_OUTPUT_DIR", os.path.join(BASE_DIR, "plots"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAMILIES = ["Austronesian", "Germanic", "Indic", "Romance", "Slavic"]
FAMILY_KEYS = ["austronesian", "germanic", "indic", "romance", "slavic"]

STRATEGIES_ORDER = [
    "Dense", "Dense_Reverted", "Expert", "Expert_Reverted", "Expert_Soup", "Freeze", "Layer_Reg",
]

# Academic-friendly palette (colorblind-safe, print-friendly)
STRATEGY_DISPLAY = {
    "Dense":           {"label": "Dense",           "color": "#d62728", "marker": "s"},
    "Dense_Reverted":  {"label": "Dense Reverted",  "color": "#ff7f0e", "marker": "D"},
    "Expert":          {"label": "Expert",          "color": "#2ca02c", "marker": "o"},
    "Expert_Reverted": {"label": "Expert Reverted", "color": "#17becf", "marker": "^"},
    "Expert_Soup":     {"label": "Expert Soup",     "color": "#8c564b", "marker": "v"},
    "Freeze":          {"label": "Freeze",          "color": "#1f77b4", "marker": "P"},
    "Layer_Reg":       {"label": "Layer Reg",       "color": "#9467bd", "marker": "X"},
}

STRATEGY_LABELS_HEATMAP = {
    "Dense": "Dense",
    "Dense_Reverted": "Dense\nReverted",
    "Expert": "Expert",
    "Expert_Reverted": "Expert\nReverted",
    "Expert_Soup": "Expert\nSoup",
    "Freeze": "Freeze",
    "Layer_Reg": "Layer\nReg",
}

FAMILY_COLORS = {
    "Austronesian": "#d62728",
    "Germanic":     "#1f77b4",
    "Indic":        "#2ca02c",
    "Romance":      "#ff7f0e",
    "Slavic":       "#9467bd",
}

LANG_NAMES = {
    "ceb": "Cebuano", "fil": "Filipino", "id": "Indonesian", "jv": "Javanese",
    "ms": "Malay", "sm": "Samoan", "af": "Afrikaans", "da": "Danish",
    "en": "English", "fy": "Frisian", "lb": "Luxembourgish", "nl": "Dutch",
    "bn": "Bengali", "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam",
    "mr": "Marathi", "ne": "Nepali", "ta": "Tamil", "te": "Telugu",
    "es": "Spanish", "fr": "French", "ga": "Irish", "it": "Italian",
    "pt": "Portuguese", "ro": "Romanian", "hr": "Croatian", "mk": "Macedonian",
    "ru": "Russian", "sk": "Slovak", "sr": "Serbian", "uk": "Ukrainian",
    "de": "German",
}

# Languages each expert was trained on (superset; not all appear in every benchmark)
EXPERT_LANGS = {
    "austronesian": {"sm", "jv", "ceb", "fil", "id", "ms"},
    "germanic":     {"af", "fy", "lb", "da", "nl", "de", "en"},
    "indic":        {"bn", "hi", "kn", "ml", "mr", "ne", "ta", "te"},
    "romance":      {"es", "pt", "fr", "ga", "it", "ro"},
    "slavic":       {"mk", "hr", "ru", "sk", "sr", "uk"},
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sc(s):
    """Shorthand to get strategy display props."""
    return STRATEGY_DISPLAY[s]


def find_pivoted(benchmark_dir, prefix, family_key, prefer_date="20260315"):
    pattern = os.path.join(BASE_DIR, benchmark_dir, f"{prefix}_{family_key}_pivoted_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    for f in reversed(files):
        if prefer_date in f:
            return f
    return files[-1]


def load_pivoted(fp):
    return pd.read_csv(fp) if fp else None


def get_strategy_cols(df, suffix):
    out = {}
    for col in df.columns:
        if col.startswith("Base"):
            continue
        if col.endswith(suffix):
            out[col[: -len(suffix)]] = col
    return out


def get_base_col(df, suffix):
    for col in df.columns:
        if col.startswith("Base") and col.endswith(suffix):
            return col
    return None


def pareto_frontier_indices(points):
    """Return indices of Pareto-optimal points (minimise x, maximise y)."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i: lower-or-equal PPL AND higher-or-equal downstream, strict in ≥1
            if pts[j, 0] <= pts[i, 0] and pts[j, 1] >= pts[i, 1]:
                if pts[j, 0] < pts[i, 0] or pts[j, 1] > pts[i, 1]:
                    is_pareto[i] = False
                    break
    return np.where(is_pareto)[0]


# ─── Load data ───────────────────────────────────────────────────────────────

print("Loading data …")
data = {}
data["belebele"] = {fk: load_pivoted(find_pivoted("belebele_gemma_2", "belebele", fk)) for fk in FAMILY_KEYS}
data["piqa"] = {fk: load_pivoted(find_pivoted("piqa_gemma_2", "global_piqa_completions", fk)) for fk in FAMILY_KEYS}
data["perplexity"] = {fk: load_pivoted(find_pivoted("perplexity_gemma_2", "perplexity", fk)) for fk in FAMILY_KEYS}
data["flores"] = {fk: load_pivoted(find_pivoted("flores_gemma_2", "flores", fk)) for fk in FAMILY_KEYS}
print("Data loaded.\n")


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — HEATMAPS
# ═════════════════════════════════════════════════════════════════════════════

def _build_delta_frame(df, suffix, base_col):
    strat_cols = get_strategy_cols(df, suffix)
    ordered = [s for s in STRATEGIES_ORDER if s in strat_cols]
    out = df[["Language_Family", "Lang"]].copy()
    for s in ordered:
        out[s] = df[strat_cols[s]] - df[base_col]
    fam_ord = {f: i for i, f in enumerate(FAMILIES)}
    out["_fo"] = out["Language_Family"].map(fam_ord)
    out = out.sort_values(["_fo", "Lang"]).reset_index(drop=True)
    return out, ordered


def plot_heatmap(benchmark, metric_suffix, metric_label, is_ppl=False):
    fig, axes = plt.subplots(1, 5, figsize=(26, 13), gridspec_kw={"wspace": 0.08})
    fig.suptitle(
        f"{benchmark.upper()} — Δ from Base ({metric_label})",
        fontsize=16, fontweight="bold", y=0.99,
    )

    last_im = None

    for idx, (fk, fam) in enumerate(zip(FAMILY_KEYS, FAMILIES)):
        ax = axes[idx]
        df = data[benchmark][fk]
        if df is None:
            ax.axis("off"); continue

        base_col = get_base_col(df, metric_suffix)
        if base_col is None:
            ax.axis("off"); continue

        delta, ordered = _build_delta_frame(df, metric_suffix, base_col)
        matrix = delta[ordered].values

        if is_ppl:
            vmax = min(np.nanmax(np.abs(matrix)), 6.0)
            cmap = "RdYlGn_r"
        else:
            vmax = min(np.nanmax(np.abs(matrix)), 0.15)
            cmap = "RdYlGn"

        last_im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels(
            [STRATEGY_LABELS_HEATMAP.get(s, s) for s in ordered],
            fontsize=8, rotation=45, ha="right",
        )

        labels = [f"{LANG_NAMES.get(r['Lang'], r['Lang'])} ({r['Lang']})"
                  for _, r in delta.iterrows()]
        expert_langs = EXPERT_LANGS.get(fk, set())
        lang_codes = delta["Lang"].values
        ax.set_yticks(range(len(labels)))
        if idx == 0:
            ax.set_yticklabels(labels, fontsize=7.5)
        else:
            ax.set_yticklabels([], fontsize=7.5)

        # Dim non-expert rows with semi-transparent white overlay
        n_cols = len(ordered)
        for i, lc in enumerate(lang_codes):
            if lc not in expert_langs:
                ax.add_patch(plt.Rectangle(
                    (-0.5, i - 0.5), n_cols, 1.0,
                    facecolor="white", edgecolor="none",
                    alpha=0.35, zorder=2, linewidth=0,
                ))

        ax.set_title(f"{fam} Expert", fontsize=12, fontweight="bold")

        # Family separators
        fams = delta["Language_Family"].values
        for i in range(1, len(fams)):
            if fams[i] != fams[i - 1]:
                ax.axhline(y=i - 0.5, color="black", linewidth=1.2, alpha=0.7)

        # Cell text
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:+.1f}" if is_ppl else f"{v:+.3f}"
                c = "white" if abs(v) > vmax * 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5, color=c)

    if last_im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.68])
        arrow = "↓ better" if is_ppl else "↑ better"
        fig.colorbar(last_im, cax=cbar_ax, label=f"Δ {metric_label}  ({arrow})")

    fig.text(
        0.02, 0.005,
        "Saturated rows = expert training languages; dimmed rows = non-training languages",
        fontsize=8, fontstyle="italic", color="#555555",
    )

    plt.tight_layout(rect=[0.01, 0.02, 0.92, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{benchmark}.{ext}"), bbox_inches="tight")
    print(f"  heatmap_{benchmark}")
    plt.close(fig)


def plot_heatmap_flores(direction="xx_en"):
    sfx = f"_{direction}_ChrF"
    label = f"ChrF ({direction.replace('_', '→')})"

    fig, axes = plt.subplots(1, 5, figsize=(26, 13), gridspec_kw={"wspace": 0.08})
    fig.suptitle(f"FLORES — Δ from Base ({label})", fontsize=16, fontweight="bold", y=0.99)

    last_im = None
    for idx, (fk, fam) in enumerate(zip(FAMILY_KEYS, FAMILIES)):
        ax = axes[idx]
        df = data["flores"][fk]
        if df is None:
            ax.axis("off"); continue
        base_col = f"Base_{direction}_ChrF"
        if base_col not in df.columns:
            ax.axis("off"); continue

        strat_cols = {s: f"{s}_{direction}_ChrF" for s in STRATEGIES_ORDER if f"{s}_{direction}_ChrF" in df.columns}
        ordered = [s for s in STRATEGIES_ORDER if s in strat_cols]

        delta = df[["Language_Family", "Lang"]].copy()
        for s in ordered:
            delta[s] = df[strat_cols[s]] - df[base_col]
        fo = {f: i for i, f in enumerate(FAMILIES)}
        delta["_fo"] = delta["Language_Family"].map(fo)
        delta = delta.sort_values(["_fo", "Lang"]).reset_index(drop=True)

        matrix = delta[ordered].values
        vmax = min(np.nanmax(np.abs(matrix)), 30)

        last_im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels([STRATEGY_LABELS_HEATMAP.get(s, s) for s in ordered], fontsize=8, rotation=45, ha="right")

        labels = [f"{LANG_NAMES.get(r['Lang'], r['Lang'])} ({r['Lang']})" for _, r in delta.iterrows()]
        expert_langs = EXPERT_LANGS.get(fk, set())
        lang_codes = delta["Lang"].values
        ax.set_yticks(range(len(labels)))
        if idx == 0:
            ax.set_yticklabels(labels, fontsize=7.5)
        else:
            ax.set_yticklabels([], fontsize=7.5)
        ax.set_title(f"{fam} Expert", fontsize=12, fontweight="bold")

        # Dim non-expert rows with semi-transparent white overlay
        n_cols = len(ordered)
        for i, lc in enumerate(lang_codes):
            if lc not in expert_langs:
                ax.add_patch(plt.Rectangle(
                    (-0.5, i - 0.5), n_cols, 1.0,
                    facecolor="white", edgecolor="none",
                    alpha=0.35, zorder=2, linewidth=0,
                ))

        fams = delta["Language_Family"].values
        for i in range(1, len(fams)):
            if fams[i] != fams[i - 1]:
                ax.axhline(y=i - 0.5, color="black", linewidth=1.2, alpha=0.7)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if np.isnan(v):
                    continue
                c = "white" if abs(v) > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=5, color=c)

    if last_im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.68])
        fig.colorbar(last_im, cax=cbar_ax, label=f"Δ {label}  (↑ better)")

    fig.text(
        0.02, 0.005,
        "Saturated rows = expert training languages; dimmed rows = non-training languages",
        fontsize=8, fontstyle="italic", color="#555555",
    )

    plt.tight_layout(rect=[0.01, 0.02, 0.92, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"heatmap_flores_{direction}.{ext}"), bbox_inches="tight")
    print(f"  heatmap_flores_{direction}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — TRADEOFF with Pareto frontier
# ═════════════════════════════════════════════════════════════════════════════

def _collect_points(bench_df, bench_suffix, ppl_df, flores_direction=None, family_key=None):
    """
    Return dict {strategy_or_base: (ppl, downstream)}.
    For FLORES, bench_suffix is ignored; flores_direction specifies xx_en or en_xx.
    If family_key is given, averages are restricted to that family's expert languages.
    """
    if bench_df is None or ppl_df is None:
        return {}
    common = set(bench_df["Lang"]) & set(ppl_df["Lang"])
    if family_key is not None and family_key in EXPERT_LANGS:
        common &= EXPERT_LANGS[family_key]
    b = bench_df[bench_df["Lang"].isin(common)]
    p = ppl_df[ppl_df["Lang"].isin(common)]
    if len(b) == 0:
        return {}

    base_p = get_base_col(p, "_PPL")
    results = {}

    if flores_direction is not None:
        base_b = f"Base_{flores_direction}_ChrF"
        if base_b in b.columns and base_p:
            results["Base"] = (p[base_p].mean(), b[base_b].mean())
        p_strats = get_strategy_cols(p, "_PPL")
        for s in STRATEGIES_ORDER:
            bc = f"{s}_{flores_direction}_ChrF"
            if bc in b.columns and s in p_strats:
                results[s] = (p[p_strats[s]].mean(), b[bc].mean())
    else:
        base_b = get_base_col(b, bench_suffix)
        if base_b and base_p:
            results["Base"] = (p[base_p].mean(), b[base_b].mean())
        b_strats = get_strategy_cols(b, bench_suffix)
        p_strats = get_strategy_cols(p, "_PPL")
        for s in STRATEGIES_ORDER:
            if s in b_strats and s in p_strats:
                results[s] = (p[p_strats[s]].mean(), b[b_strats[s]].mean())
    return results


def _draw_pareto(ax, points_dict):
    """Draw Pareto frontier curve on ax. points_dict = {name: (x, y)}."""
    if len(points_dict) < 2:
        return
    names = list(points_dict.keys())
    pts = np.array([points_dict[n] for n in names])
    idx = pareto_frontier_indices(pts)
    if len(idx) < 2:
        return
    pareto_pts = pts[idx]
    # Sort by x (PPL) for a smooth curve
    order = np.argsort(pareto_pts[:, 0])
    pareto_pts = pareto_pts[order]
    ax.plot(
        pareto_pts[:, 0], pareto_pts[:, 1],
        color="#555555", linewidth=1.5, linestyle="--", alpha=0.5, zorder=1,
    )
    # Shade the dominated region lightly
    # Extend to axes limits for a proper fill
    xs = list(pareto_pts[:, 0])
    ys = list(pareto_pts[:, 1])
    ax.fill_between(
        xs, ys, y2=ax.get_ylim()[0] if ax.get_ylim()[0] < min(ys) else min(ys) - 1,
        alpha=0.04, color="#555555", zorder=0,
    )


def plot_tradeoff_single(bench, ylabel, title, filename,
                         suffix=None, flores_direction=None):
    """
    1 row × 5 cols (one per family) tradeoff plot for a single benchmark.

    For Belebele/PIQA: set bench + suffix (e.g. "belebele", "_Acc").
    For FLORES:        set bench="flores" + flores_direction (e.g. "xx_en").
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), gridspec_kw={"wspace": 0.3})
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.04)

    ppl = data["perplexity"]

    for col, (fk, fam) in enumerate(zip(FAMILY_KEYS, FAMILIES)):
        ax = axes[col]
        pts = _collect_points(
            data[bench][fk], suffix, ppl[fk],
            flores_direction=flores_direction, family_key=fk,
        )

        for strat, (x, y) in pts.items():
            if strat == "Base":
                ax.scatter(x, y, marker="*", s=200, c="black", zorder=10,
                           edgecolors="black", linewidths=0.8)
            else:
                d = _sc(strat)
                ax.scatter(x, y, marker=d["marker"], s=90,
                           c=d["color"], edgecolors=d["color"],
                           linewidths=1.2, zorder=5)

        _draw_pareto(ax, pts)

        ax.set_title(fam, fontsize=11, fontweight="bold")
        ax.set_xlabel("Avg Perplexity ↓", fontsize=9)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    elems = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
               markersize=11, label="Base", markeredgewidth=0.8),
    ]
    for s in STRATEGIES_ORDER:
        d = _sc(s)
        elems.append(Line2D(
            [0], [0], marker=d["marker"], color="w",
            markerfacecolor=d["color"], markeredgecolor=d["color"],
            markersize=8, label=d["label"], markeredgewidth=1.0,
        ))
    elems.append(Line2D([0], [0], color="#555555", linewidth=1.5, linestyle="--",
                        alpha=0.6, label="Pareto frontier"))
    fig.legend(handles=elems, loc="lower center", ncol=8, fontsize=9,
               bbox_to_anchor=(0.5, -0.18), frameon=True, fancybox=True,
               edgecolor="#cccccc")

    plt.tight_layout(rect=[0, 0.0, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"{filename}.{ext}"), bbox_inches="tight")
    print(f"  {filename}")
    plt.close(fig)


def plot_tradeoff():
    """5 rows × 4 cols, all benchmarks — appendix figure."""
    fig, axes = plt.subplots(5, 4, figsize=(24, 28))
    fig.suptitle(
        "Perplexity vs. Downstream Performance Tradeoff (All Benchmarks)",
        fontsize=17, fontweight="bold", y=1.005,
    )

    col_specs = [
        ("belebele", "_Acc",  None,     "Belebele Acc"),
        ("piqa",     "_Acc",  None,     "PIQA Acc"),
        (None,       None,    "xx_en",  "FLORES ChrF (xx→en)"),
        (None,       None,    "en_xx",  "FLORES ChrF (en→xx)"),
    ]

    ppl = data["perplexity"]

    for row, (fk, fam) in enumerate(zip(FAMILY_KEYS, FAMILIES)):
        for col, (bench, suffix, fdir, title) in enumerate(col_specs):
            ax = axes[row, col]

            if bench is not None:
                pts = _collect_points(data[bench][fk], suffix, ppl[fk], family_key=fk)
            else:
                pts = _collect_points(data["flores"][fk], None, ppl[fk], flores_direction=fdir, family_key=fk)

            # Plot each point
            for strat, (x, y) in pts.items():
                if strat == "Base":
                    ax.scatter(x, y, marker="*", s=250, c="black", zorder=10,
                               edgecolors="black", linewidths=0.8)
                else:
                    d = _sc(strat)
                    ax.scatter(x, y, marker=d["marker"], s=100,
                               c=d["color"], edgecolors=d["color"],
                               linewidths=1.2, zorder=5)

            # Pareto frontier (include Base)
            _draw_pareto(ax, pts)

            # Axes labels
            if row == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{fam}\nDownstream ↑", fontsize=10, fontweight="bold")
            if row == 4:
                ax.set_xlabel("Avg Perplexity ↓", fontsize=10)

            ax.grid(True, alpha=0.15, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Legend
    elems = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
               markersize=13, label="Base", markeredgewidth=0.8),
    ]
    for s in STRATEGIES_ORDER:
        d = _sc(s)
        elems.append(Line2D(
            [0], [0], marker=d["marker"], color="w",
            markerfacecolor=d["color"], markeredgecolor=d["color"],
            markersize=9, label=d["label"], markeredgewidth=1.0,
        ))
    elems.append(Line2D([0], [0], color="#555555", linewidth=1.5, linestyle="--",
                         alpha=0.6, label="Pareto frontier"))
    fig.legend(handles=elems, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.015), frameon=True, fancybox=True,
               edgecolor="#cccccc")

    plt.tight_layout(rect=[0, 0.035, 1, 0.975])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"tradeoff_ppl_vs_downstream.{ext}"), bbox_inches="tight")
    print("  tradeoff_ppl_vs_downstream")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — RADAR CHARTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_radar(benchmark, metric_suffix, bench_label):
    N = len(FAMILIES)
    angles = [n / N * 2 * pi for n in range(N)] + [0]

    fig, axes = plt.subplots(1, 5, figsize=(28, 6.5), subplot_kw=dict(polar=True))
    fig.suptitle(
        f"{bench_label}  (Δ from Base, avg per target family)",
        fontsize=15, fontweight="bold", y=1.06,
    )

    for idx, (fk, fam) in enumerate(zip(FAMILY_KEYS, FAMILIES)):
        ax = axes[idx]
        df = data[benchmark][fk]
        if df is None:
            ax.set_title(f"{fam}\n(no data)", fontsize=11); continue

        base_col = get_base_col(df, metric_suffix)
        strat_cols = get_strategy_cols(df, metric_suffix)

        for s in STRATEGIES_ORDER:
            if s not in strat_cols:
                continue
            d = _sc(s)
            vals = []
            for tf in FAMILIES:
                sub = df[df["Language_Family"] == tf]
                vals.append((sub[strat_cols[s]] - sub[base_col]).mean() if len(sub) > 0 else 0)
            vals.append(vals[0])
            ax.plot(angles, vals, linewidth=2, color=d["color"],
                    marker=d["marker"], markersize=4.5, label=d["label"])
            ax.fill(angles, vals, alpha=0.035, color=d["color"])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(FAMILIES, fontsize=8)
        ax.set_title(f"{fam} Expert", fontsize=11, fontweight="bold", pad=18)
        ax.plot(angles, [0] * len(angles), "k-", linewidth=0.6, alpha=0.4)

    elems = [Line2D([0], [0], color=_sc(s)["color"], marker=_sc(s)["marker"],
                     markersize=7, label=_sc(s)["label"], linewidth=2)
             for s in STRATEGIES_ORDER]
    fig.legend(handles=elems, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=True, fancybox=True,
               edgecolor="#cccccc")

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"radar_{benchmark}.{ext}"), bbox_inches="tight")
    print(f"  radar_{benchmark}")
    plt.close(fig)


def plot_radar_flores(direction="xx_en"):
    N = len(FAMILIES)
    angles = [n / N * 2 * pi for n in range(N)] + [0]
    label = f"FLORES ChrF ({direction.replace('_', '→')})"

    fig, axes = plt.subplots(1, 5, figsize=(28, 6.5), subplot_kw=dict(polar=True))
    fig.suptitle(f"{label}  (Δ from Base, avg per target family)",
                 fontsize=15, fontweight="bold", y=1.06)

    for idx, (fk, fam) in enumerate(zip(FAMILY_KEYS, FAMILIES)):
        ax = axes[idx]
        df = data["flores"][fk]
        if df is None:
            ax.set_title(f"{fam}\n(no data)", fontsize=11); continue

        base_col = f"Base_{direction}_ChrF"
        if base_col not in df.columns:
            continue

        for s in STRATEGIES_ORDER:
            c = f"{s}_{direction}_ChrF"
            if c not in df.columns:
                continue
            d = _sc(s)
            vals = []
            for tf in FAMILIES:
                sub = df[df["Language_Family"] == tf]
                vals.append((sub[c] - sub[base_col]).mean() if len(sub) > 0 else 0)
            vals.append(vals[0])
            ax.plot(angles, vals, linewidth=2, color=d["color"],
                    marker=d["marker"], markersize=4.5, label=d["label"])
            ax.fill(angles, vals, alpha=0.035, color=d["color"])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(FAMILIES, fontsize=8)
        ax.set_title(f"{fam} Expert", fontsize=11, fontweight="bold", pad=18)
        ax.plot(angles, [0] * len(angles), "k-", linewidth=0.6, alpha=0.4)

    elems = [Line2D([0], [0], color=_sc(s)["color"], marker=_sc(s)["marker"],
                     markersize=7, label=_sc(s)["label"], linewidth=2)
             for s in STRATEGIES_ORDER]
    fig.legend(handles=elems, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=True, fancybox=True,
               edgecolor="#cccccc")

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"radar_flores_{direction}.{ext}"), bbox_inches="tight")
    print(f"  radar_flores_{direction}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Heatmaps ===")
    plot_heatmap("belebele",   "_Acc", "Accuracy",   is_ppl=False)
    plot_heatmap("piqa",       "_Acc", "Accuracy",   is_ppl=False)
    plot_heatmap("perplexity", "_PPL", "Perplexity", is_ppl=True)
    plot_heatmap_flores("xx_en")
    plot_heatmap_flores("en_xx")

    print("\n=== Tradeoff ===")
    plot_tradeoff_single("belebele", "Belebele Accuracy ↑",
                         "Perplexity vs. Belebele Accuracy Tradeoff",
                         "tradeoff_belebele", suffix="_Acc")
    plot_tradeoff_single("piqa", "PIQA Accuracy ↑",
                         "Perplexity vs. PIQA Accuracy Tradeoff",
                         "tradeoff_piqa", suffix="_Acc")
    plot_tradeoff_single("flores", "FLORES ChrF ↑",
                         "Perplexity vs. FLORES ChrF (xx→en) Tradeoff",
                         "tradeoff_flores_xx_en", flores_direction="xx_en")
    plot_tradeoff_single("flores", "FLORES ChrF ↑",
                         "Perplexity vs. FLORES ChrF (en→xx) Tradeoff",
                         "tradeoff_flores_en_xx", flores_direction="en_xx")
    plot_tradeoff()

    print("\n=== Radar ===")
    plot_radar("belebele",   "_Acc", "Belebele Accuracy")
    plot_radar("piqa",       "_Acc", "PIQA Accuracy")
    plot_radar("perplexity", "_PPL", "Perplexity")
    plot_radar_flores("xx_en")
    plot_radar_flores("en_xx")

    print(f"\n✓ All plots in {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")