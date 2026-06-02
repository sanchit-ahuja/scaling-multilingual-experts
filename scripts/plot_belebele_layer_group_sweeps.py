#!/usr/bin/env python3
"""Plot held-in Belebele accuracy for first/middle/last layer-group sweeps."""

from __future__ import annotations

import argparse
import json
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

from configs.constants import LANGS  # noqa: E402


FLORES_CODE = {
    "es": "spa_Latn",
    "pt": "por_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "ro": "ron_Latn",
    "gl": "glg_Latn",
    "ga": "gle_Latn",
    "mk": "mkd_Cyrl",
    "hr": "hrv_Latn",
    "ru": "rus_Cyrl",
    "sk": "slk_Latn",
    "sr": "srp_Cyrl",
    "uk": "ukr_Cyrl",
    "af": "afr_Latn",
    "fy": "fry_Latn",
    "lb": "ltz_Latn",
    "da": "dan_Latn",
    "nl": "nld_Latn",
    "en": "eng_Latn",
    "bn": "ben_Beng",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "sm": "smo_Latn",
    "jv": "jav_Latn",
    "ceb": "ceb_Latn",
    "fil": "tgl_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
}

TARGET_ORDER = ["first", "middle", "last"]
FAMILY_ORDER = ["slavic", "germanic", "indic", "austronesian", "romance"]


def load_training_languages(data_stats_path: Path | None) -> dict[str, list[str]]:
    if data_stats_path is None:
        return {family: sorted(languages) for family, languages in LANGS.items()}
    with data_stats_path.open() as f:
        data_stats = json.load(f)
    return {
        family: sorted(family_stats["languages"])
        for family, family_stats in data_stats["families"].items()
    }


def load_rows(summary_paths: dict[str, Path], training_languages: dict[str, list[str]]):
    rows = []
    missing = {}
    for target, summary_path in summary_paths.items():
        target_missing = set()
        for line in summary_path.read_text().splitlines():
            record = json.loads(line)
            metrics = record["eval"]["metrics"]
            drift = record.get(
                "alpha_scaled_normalized_target_drift",
                record.get("alpha_scaled_normalized_middle_drift"),
            )
            original_drift = record.get(
                "original_normalized_target_drift",
                record.get("original_normalized_middle_drift"),
            )
            for family, languages in training_languages.items():
                for language in languages:
                    task = f"belebele_{FLORES_CODE.get(language, '')}"
                    if task not in metrics:
                        target_missing.add(f"{family}/{language}:{task}")
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
        missing[target] = sorted(target_missing)
    return pd.DataFrame(rows), missing


def save_plot(fig, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overall(overall_df: pd.DataFrame, output_prefix: Path) -> None:
    plot_df = overall_df.copy()
    plot_df["accuracy_pct"] = 100 * plot_df["accuracy"]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sns.lineplot(
        data=plot_df,
        x="alpha",
        y="accuracy_pct",
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
    ax.set_ylabel("Held-in Belebele accuracy (%)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(58, 68)
    ax.legend(title="Interpolated layers", frameon=False)
    save_plot(fig, output_prefix)


def plot_family(family_df: pd.DataFrame, output_prefix: Path) -> None:
    plot_df = family_df.copy()
    plot_df["accuracy_pct"] = 100 * plot_df["accuracy"]
    plot_df["family"] = pd.Categorical(
        plot_df["family"], categories=FAMILY_ORDER, ordered=True
    )

    grid = sns.relplot(
        data=plot_df,
        x="alpha",
        y="accuracy_pct",
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
    grid.set_axis_labels("", "Accuracy (%)")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--first-summary",
        type=Path,
        default=Path("data/drift_alpha_sweep/dense_first_belebele_all/summary.jsonl"),
    )
    parser.add_argument(
        "--middle-summary",
        type=Path,
        default=Path("data/drift_alpha_sweep/dense_middle_belebele_all/summary.jsonl"),
    )
    parser.add_argument(
        "--last-summary",
        type=Path,
        default=Path("data/drift_alpha_sweep/dense_last_belebele_all/summary.jsonl"),
    )
    parser.add_argument(
        "--data-stats",
        type=Path,
        help="Optional training data_stats.json; defaults to configs.constants.LANGS.",
    )
    parser.add_argument("--csv-dir", type=Path, default=Path("results"))
    parser.add_argument("--plot-dir", type=Path, default=Path("plots"))
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

    summary_paths = {
        "first": args.first_summary,
        "middle": args.middle_summary,
        "last": args.last_summary,
    }
    training_languages = load_training_languages(args.data_stats)
    language_df, missing = load_rows(summary_paths, training_languages)
    if language_df.empty:
        raise SystemExit("No held-in Belebele rows found.")

    family_df = (
        language_df.groupby(
            ["target", "alpha", "target_drift", "original_target_drift", "family"],
            as_index=False,
        )
        .agg(accuracy=("accuracy", "mean"), n_languages=("language", "nunique"))
        .sort_values(["target", "family", "alpha"])
    )
    overall_df = (
        language_df.groupby(
            ["target", "alpha", "target_drift", "original_target_drift"],
            as_index=False,
        )
        .agg(accuracy=("accuracy", "mean"), n_languages=("language", "nunique"))
        .sort_values(["target", "alpha"])
    )

    args.csv_dir.mkdir(parents=True, exist_ok=True)
    language_csv = args.csv_dir / "belebele_alpha_sweep_layer_groups_language.csv"
    family_csv = args.csv_dir / "belebele_alpha_sweep_layer_groups_family.csv"
    overall_csv = args.csv_dir / "belebele_alpha_sweep_layer_groups_overall.csv"
    language_df.sort_values(["target", "family", "language", "alpha"]).to_csv(
        language_csv, index=False
    )
    family_df.to_csv(family_csv, index=False)
    overall_df.to_csv(overall_csv, index=False)

    overall_prefix = args.plot_dir / "belebele_alpha_sweep_layer_groups_overall"
    family_prefix = args.plot_dir / "belebele_alpha_sweep_layer_groups_family"
    plot_overall(overall_df, overall_prefix)
    plot_family(family_df, family_prefix)

    print(f"Wrote {overall_prefix.with_suffix('.pdf')}")
    print(f"Wrote {overall_prefix.with_suffix('.png')}")
    print(f"Wrote {family_prefix.with_suffix('.pdf')}")
    print(f"Wrote {family_prefix.with_suffix('.png')}")
    print(f"Wrote {language_csv}")
    print(f"Wrote {family_csv}")
    print(f"Wrote {overall_csv}")
    print(f"Held-in languages evaluated: {language_df['language'].nunique()}")
    for target in TARGET_ORDER:
        if missing[target]:
            print(f"Missing held-in Belebele tasks for {target}:")
            for item in missing[target]:
                print(f"  {item}")


if __name__ == "__main__":
    main()
