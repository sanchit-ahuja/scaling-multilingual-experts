#!/usr/bin/env python3
"""Plot post-truncation FLORES alpha-sweep ChrF from rescored task CSVs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_flores_layer_group_sweeps import (
    DIRECTION_LABELS,
    DIRECTION_ORDER,
    FAMILY_ORDER,
    TARGET_ORDER,
    save_plot,
)


def aggregate(task_df: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    return (
        task_df.groupby(keys, as_index=False)
        .agg(
            chrf=(metric, "mean"),
            raw_chrf=("raw_chrf", "mean"),
            fixed_chrf=("fixed_chrf", "mean"),
            pct_truncated=("pct_truncated", "mean"),
            n_tasks=("task", "nunique"),
            n_languages=("language", "nunique"),
        )
        .sort_values(keys)
    )


def plot_overall(overall_df: pd.DataFrame, output_prefix: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sns.lineplot(
        data=overall_df,
        x="alpha",
        y="chrf",
        hue="target",
        hue_order=TARGET_ORDER,
        style="target",
        markers=True,
        dashes=False,
        linewidth=2.4,
        markersize=7,
        ax=ax,
    )
    ax.set_xlabel("Layer-group interpolation alpha")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.03, 1.03)
    ax.legend(title="Interpolated layers", frameon=False)
    save_plot(fig, output_prefix)


def plot_family(family_df: pd.DataFrame, output_prefix: Path, ylabel: str) -> None:
    plot_df = family_df.copy()
    plot_df["family"] = pd.Categorical(
        plot_df["family"], categories=FAMILY_ORDER, ordered=True
    )

    grid = sns.relplot(
        data=plot_df,
        x="alpha",
        y="chrf",
        hue="target",
        hue_order=TARGET_ORDER,
        style="target",
        markers=True,
        dashes=False,
        col="family",
        col_wrap=3,
        kind="line",
        linewidth=2.0,
        height=2.35,
        aspect=1.18,
        facet_kws={"sharey": False},
    )
    grid.set_axis_labels("", ylabel)
    grid.set_titles("{col_name}")
    for idx, ax in enumerate(grid.axes.flat):
        if idx < 3:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
    sns.move_legend(
        grid,
        "upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        title="Interpolated layers",
        frameon=False,
    )
    grid.figure.text(0.5, 0.02, "Alpha", ha="center", va="center")
    grid.figure.subplots_adjust(top=0.86, bottom=0.12, hspace=0.32, wspace=0.22)
    save_plot(grid.figure, output_prefix)


def plot_direction(direction_df: pd.DataFrame, output_prefix: Path, ylabel: str) -> None:
    plot_df = direction_df.copy()
    plot_df["direction"] = pd.Categorical(
        plot_df["direction"], categories=DIRECTION_ORDER, ordered=True
    )
    plot_df["Direction"] = plot_df["direction"].map(DIRECTION_LABELS)

    grid = sns.relplot(
        data=plot_df,
        x="alpha",
        y="chrf",
        hue="target",
        hue_order=TARGET_ORDER,
        style="target",
        markers=True,
        dashes=False,
        col="Direction",
        col_order=[DIRECTION_LABELS[d] for d in DIRECTION_ORDER],
        kind="line",
        linewidth=2.2,
        height=3.1,
        aspect=1.05,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("Alpha", ylabel)
    grid.set_titles("{col_name}")
    sns.move_legend(
        grid,
        "upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        title="Interpolated layers",
        frameon=False,
    )
    grid.figure.subplots_adjust(top=0.78, bottom=0.16, wspace=0.18)
    save_plot(grid.figure, output_prefix)


def plot_family_direction(
    family_direction_df: pd.DataFrame, output_prefix: Path, ylabel: str
) -> None:
    plot_df = family_direction_df.copy()
    plot_df["family"] = pd.Categorical(
        plot_df["family"], categories=FAMILY_ORDER, ordered=True
    )
    plot_df["direction"] = pd.Categorical(
        plot_df["direction"], categories=DIRECTION_ORDER, ordered=True
    )
    plot_df["Direction"] = plot_df["direction"].map(DIRECTION_LABELS)

    grid = sns.relplot(
        data=plot_df,
        x="alpha",
        y="chrf",
        hue="target",
        hue_order=TARGET_ORDER,
        style="target",
        markers=True,
        dashes=False,
        row="Direction",
        col="family",
        col_order=FAMILY_ORDER,
        row_order=[DIRECTION_LABELS[d] for d in DIRECTION_ORDER],
        kind="line",
        linewidth=1.8,
        height=2.15,
        aspect=1.1,
        facet_kws={"sharey": False},
    )
    grid.set_axis_labels("", "")
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    sns.move_legend(
        grid,
        "upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        title="Interpolated layers",
        frameon=False,
    )
    grid.figure.supxlabel("Alpha", y=0.04)
    grid.figure.supylabel(ylabel, x=0.01)
    grid.figure.subplots_adjust(
        left=0.07, right=0.99, top=0.82, bottom=0.13, hspace=0.32, wspace=0.28
    )
    save_plot(grid.figure, output_prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-csv",
        type=Path,
        default=Path("results/flores_alpha_sweep_fixed_task.csv"),
    )
    parser.add_argument("--csv-dir", type=Path, default=Path("results"))
    parser.add_argument("--plot-dir", type=Path, default=Path("plots"))
    parser.add_argument(
        "--metric",
        choices=["fixed_chrf", "raw_chrf"],
        default="fixed_chrf",
        help="ChrF column to plot. Use fixed_chrf for paper-facing analysis.",
    )
    args = parser.parse_args()

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.15,
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )

    task_df = pd.read_csv(args.task_csv)
    if task_df.empty:
        raise SystemExit(f"No rows found in {args.task_csv}")

    overall_df = aggregate(task_df, ["target", "alpha"], args.metric)
    family_df = aggregate(task_df, ["target", "alpha", "family"], args.metric)
    direction_df = aggregate(task_df, ["target", "alpha", "direction"], args.metric)
    family_direction_df = aggregate(
        task_df, ["target", "alpha", "family", "direction"], args.metric
    )

    args.csv_dir.mkdir(parents=True, exist_ok=True)
    family_direction_csv = args.csv_dir / "flores_alpha_sweep_fixed_family_direction.csv"
    family_direction_df.to_csv(family_direction_csv, index=False)

    stem = "fixed" if args.metric == "fixed_chrf" else "raw_from_samples"
    ylabel = "Held-in FLORES ChrF"

    overall_prefix = args.plot_dir / f"flores_alpha_sweep_{stem}_overall"
    family_prefix = args.plot_dir / f"flores_alpha_sweep_{stem}_family"
    direction_prefix = args.plot_dir / f"flores_alpha_sweep_{stem}_direction"
    family_direction_prefix = args.plot_dir / f"flores_alpha_sweep_{stem}_family_direction"

    plot_overall(overall_df, overall_prefix, ylabel)
    plot_family(family_df, family_prefix, ylabel)
    plot_direction(direction_df, direction_prefix, ylabel)
    plot_family_direction(family_direction_df, family_direction_prefix, ylabel)

    for prefix in (
        overall_prefix,
        family_prefix,
        direction_prefix,
        family_direction_prefix,
    ):
        print(f"Wrote {prefix.with_suffix('.pdf')}")
        print(f"Wrote {prefix.with_suffix('.png')}")
    print(f"Wrote {family_direction_csv}")


if __name__ == "__main__":
    main()
