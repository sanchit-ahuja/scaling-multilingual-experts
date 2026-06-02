#!/usr/bin/env python3
"""Plot held-in Belebele accuracy for expert-soup alpha sweeps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from plot_belebele_layer_group_sweeps import FAMILY_ORDER, FLORES_CODE, load_training_languages


def load_rows(summary_paths: dict[str, Path], training_languages: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for target, summary_path in summary_paths.items():
        for line in summary_path.read_text().splitlines():
            record = json.loads(line)
            metrics = record["eval"]["metrics"]
            drift = record["alpha_scaled_mean_target_drift_to_base"]
            original_drift = record["mean_original_target_drift_to_base"]
            for family, languages in training_languages.items():
                for language in languages:
                    flores_code = FLORES_CODE.get(language)
                    if not flores_code:
                        continue
                    task = f"belebele_{flores_code}"
                    if task not in metrics:
                        continue
                    rows.append(
                        {
                            "target": target,
                            "alpha": float(record["alpha"]),
                            "target_drift": float(drift),
                            "original_target_drift": float(original_drift),
                            "family": family,
                            "language": language,
                            "task": task,
                            "accuracy": float(metrics[task]["acc,none"]),
                        }
                    )
    return pd.DataFrame(rows)


def save_plot(fig, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overall(overall_df: pd.DataFrame, output_prefix: Path) -> None:
    plot_df = overall_df.copy()
    plot_df["accuracy_pct"] = 100 * plot_df["accuracy"]
    plot_df["target"] = plot_df["target"].map({"middle": "Middle layers", "all": "Full model"})

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sns.lineplot(
        data=plot_df,
        x="alpha",
        y="accuracy_pct",
        hue="target",
        marker="o",
        linewidth=2.2,
        ax=ax,
    )
    ax.set_xlabel("Alpha: expert drift retained")
    ax.set_ylabel("Held-in Belebele accuracy (%)")
    ax.set_title("Expert-soup interpolation is flat across alpha")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="")
    save_plot(fig, output_prefix)


def plot_family(family_df: pd.DataFrame, output_prefix: Path) -> None:
    plot_df = family_df.copy()
    plot_df["accuracy_pct"] = 100 * plot_df["accuracy"]
    plot_df["target"] = plot_df["target"].map({"middle": "Middle", "all": "Full"})
    plot_df["family"] = pd.Categorical(
        plot_df["family"], categories=FAMILY_ORDER, ordered=True
    )

    grid = sns.relplot(
        data=plot_df,
        x="alpha",
        y="accuracy_pct",
        hue="target",
        col="family",
        col_wrap=3,
        kind="line",
        marker="o",
        linewidth=2.0,
        height=2.6,
        aspect=1.05,
        facet_kws={"sharey": False},
    )
    grid.set_axis_labels("Alpha", "Accuracy (%)")
    grid.set_titles("{col_name}")
    grid.legend.set_title("")
    for ax in grid.axes.flat:
        ax.grid(True, axis="y", alpha=0.25)
    save_plot(grid.fig, output_prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--middle_summary",
        type=Path,
        default=Path(
            "data/drift_alpha_sweep/expert_soup_middle_belebele_all/summary.jsonl"
        ),
    )
    parser.add_argument(
        "--all_summary",
        type=Path,
        default=Path(
            "data/drift_alpha_sweep/expert_soup_all_belebele_all/summary.jsonl"
        ),
    )
    parser.add_argument(
        "--data_stats",
        type=Path,
        help="Optional training data_stats.json; defaults to configs.constants.LANGS.",
    )
    parser.add_argument("--plot_dir", type=Path, default=Path("plots"))
    parser.add_argument("--csv_dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    training_languages = load_training_languages(args.data_stats)
    rows = load_rows(
        {"middle": args.middle_summary, "all": args.all_summary},
        training_languages,
    )
    overall_df = (
        rows.groupby(["target", "alpha", "target_drift", "original_target_drift"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), n=("accuracy", "size"))
        .sort_values(["target", "alpha"])
    )
    family_df = (
        rows.groupby(
            ["target", "alpha", "target_drift", "original_target_drift", "family"],
            as_index=False,
        )
        .agg(accuracy=("accuracy", "mean"), n=("accuracy", "size"))
        .sort_values(["target", "family", "alpha"])
    )

    args.csv_dir.mkdir(parents=True, exist_ok=True)
    language_csv = args.csv_dir / "expert_soup_alpha_sweep_heldin_language.csv"
    family_csv = args.csv_dir / "expert_soup_alpha_sweep_heldin_family.csv"
    overall_csv = args.csv_dir / "expert_soup_alpha_sweep_heldin_overall.csv"
    rows.sort_values(["target", "family", "language", "alpha"]).to_csv(language_csv, index=False)
    family_df.to_csv(family_csv, index=False)
    overall_df.to_csv(overall_csv, index=False)

    overall_prefix = args.plot_dir / "expert_soup_alpha_sweep_heldin_overall"
    family_prefix = args.plot_dir / "expert_soup_alpha_sweep_heldin_family"
    sns.set_theme(style="whitegrid", context="talk")
    plot_overall(overall_df, overall_prefix)
    plot_family(family_df, family_prefix)

    print(overall_df)
    print(f"Wrote {overall_prefix.with_suffix('.pdf')}")
    print(f"Wrote {overall_prefix.with_suffix('.png')}")
    print(f"Wrote {family_prefix.with_suffix('.pdf')}")
    print(f"Wrote {family_prefix.with_suffix('.png')}")
    print(f"Wrote {overall_csv}")
    print(f"Wrote {family_csv}")
    print(f"Wrote {language_csv}")


if __name__ == "__main__":
    main()
