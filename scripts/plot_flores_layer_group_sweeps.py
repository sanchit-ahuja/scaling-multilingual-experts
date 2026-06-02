#!/usr/bin/env python3
"""Plot held-in FLORES ChrF for first/middle/last layer-group sweeps."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.flores_heldin_tasks import DEFAULT_FLORES_DIR, heldin_tasks


TARGET_ORDER = ["first", "middle", "last"]
FAMILY_ORDER = ["slavic", "germanic", "indic", "austronesian", "romance"]
DIRECTION_ORDER = ["en_xx", "xx_en"]
DIRECTION_LABELS = {"en_xx": "en->xx", "xx_en": "xx->en"}
CODE_TO_FAMILY_LANGUAGE = {
    "mkd_Cyrl": ("slavic", "mk"),
    "hrv_Latn": ("slavic", "hr"),
    "rus_Cyrl": ("slavic", "ru"),
    "slk_Latn": ("slavic", "sk"),
    "srp_Cyrl": ("slavic", "sr"),
    "ukr_Cyrl": ("slavic", "uk"),
    "afr_Latn": ("germanic", "af"),
    "ltz_Latn": ("germanic", "lb"),
    "dan_Latn": ("germanic", "da"),
    "nld_Latn": ("germanic", "nl"),
    "ben_Beng": ("indic", "bn"),
    "hin_Deva": ("indic", "hi"),
    "kan_Knda": ("indic", "kn"),
    "mal_Mlym": ("indic", "ml"),
    "mar_Deva": ("indic", "mr"),
    "npi_Deva": ("indic", "ne"),
    "tam_Taml": ("indic", "ta"),
    "tel_Telu": ("indic", "te"),
    "smo_Latn": ("austronesian", "sm"),
    "jav_Latn": ("austronesian", "jv"),
    "ceb_Latn": ("austronesian", "ceb"),
    "tgl_Latn": ("austronesian", "fil"),
    "ind_Latn": ("austronesian", "id"),
    "zsm_Latn": ("austronesian", "ms"),
    "spa_Latn": ("romance", "es"),
    "por_Latn": ("romance", "pt"),
    "fra_Latn": ("romance", "fr"),
    "glg_Latn": ("romance", "gl"),
    "ita_Latn": ("romance", "it"),
    "ron_Latn": ("romance", "ro"),
}


def parse_task(task: str) -> tuple[str, str, str]:
    match = re.match(r"flores_(.+)-(.+)$", task)
    if not match:
        raise ValueError(f"Unexpected FLORES task name: {task}")
    src, tgt = match.groups()
    if src == "eng_Latn":
        direction = "en_xx"
        code = tgt
    elif tgt == "eng_Latn":
        direction = "xx_en"
        code = src
    else:
        raise ValueError(f"Task is not English-centric: {task}")
    family, language = CODE_TO_FAMILY_LANGUAGE[code]
    return direction, family, language


def load_rows(
    summary_paths: dict[str, Path], tasks: list[str]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
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
            for task in tasks:
                if task not in metrics or "chrf,none" not in metrics[task]:
                    target_missing.add(task)
                    continue
                direction, family, language = parse_task(task)
                rows.append(
                    {
                        "target": target,
                        "alpha": float(record["alpha"]),
                        "target_drift": float(drift),
                        "original_target_drift": float(original_drift),
                        "family": family,
                        "language": language,
                        "direction": direction,
                        "task": task,
                        "chrf": float(metrics[task]["chrf,none"]),
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
    ax.set_ylabel("Held-in FLORES ChrF")
    ax.set_xlim(-0.03, 1.03)
    ax.legend(title="Interpolated layers", frameon=False)
    save_plot(fig, output_prefix)


def plot_family(family_df: pd.DataFrame, output_prefix: Path) -> None:
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
    grid.set_axis_labels("", "ChrF")
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


def plot_direction(direction_df: pd.DataFrame, output_prefix: Path) -> None:
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
    grid.set_axis_labels("Alpha", "ChrF")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--first-summary",
        type=Path,
        default=Path(
            "data/drift_alpha_sweep/dense_first_flores_heldin/summary.jsonl"
        ),
    )
    parser.add_argument(
        "--middle-summary",
        type=Path,
        default=Path(
            "data/drift_alpha_sweep/dense_middle_flores_heldin/summary.jsonl"
        ),
    )
    parser.add_argument(
        "--last-summary",
        type=Path,
        default=Path(
            "data/drift_alpha_sweep/dense_last_flores_heldin/summary.jsonl"
        ),
    )
    parser.add_argument("--flores-dir", type=Path, default=DEFAULT_FLORES_DIR)
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

    tasks = heldin_tasks(args.flores_dir)
    summary_paths = {
        "first": args.first_summary,
        "middle": args.middle_summary,
        "last": args.last_summary,
    }
    language_df, missing = load_rows(summary_paths, tasks)
    if language_df.empty:
        raise SystemExit("No held-in FLORES rows found.")

    family_df = (
        language_df.groupby(
            [
                "target",
                "alpha",
                "target_drift",
                "original_target_drift",
                "family",
                "direction",
            ],
            as_index=False,
        )
        .agg(chrf=("chrf", "mean"), n_languages=("language", "nunique"))
        .sort_values(["target", "family", "direction", "alpha"])
    )
    overall_df = (
        language_df.groupby(
            ["target", "alpha", "target_drift", "original_target_drift"],
            as_index=False,
        )
        .agg(chrf=("chrf", "mean"), n_languages=("language", "nunique"))
        .sort_values(["target", "alpha"])
    )
    direction_df = (
        language_df.groupby(
            ["target", "alpha", "target_drift", "original_target_drift", "direction"],
            as_index=False,
        )
        .agg(chrf=("chrf", "mean"), n_languages=("language", "nunique"))
        .sort_values(["target", "direction", "alpha"])
    )

    args.csv_dir.mkdir(parents=True, exist_ok=True)
    language_csv = args.csv_dir / "flores_alpha_sweep_layer_groups_language.csv"
    family_csv = args.csv_dir / "flores_alpha_sweep_layer_groups_family.csv"
    direction_csv = args.csv_dir / "flores_alpha_sweep_layer_groups_direction.csv"
    overall_csv = args.csv_dir / "flores_alpha_sweep_layer_groups_overall.csv"
    language_df.sort_values(
        ["target", "family", "language", "direction", "alpha"]
    ).to_csv(language_csv, index=False)
    family_df.to_csv(family_csv, index=False)
    direction_df.to_csv(direction_csv, index=False)
    overall_df.to_csv(overall_csv, index=False)

    overall_prefix = args.plot_dir / "flores_alpha_sweep_layer_groups_overall"
    family_prefix = args.plot_dir / "flores_alpha_sweep_layer_groups_family"
    direction_prefix = args.plot_dir / "flores_alpha_sweep_layer_groups_direction"
    plot_overall(overall_df, overall_prefix)
    plot_family(family_df, family_prefix)
    plot_direction(direction_df, direction_prefix)

    print(f"Wrote {overall_prefix.with_suffix('.pdf')}")
    print(f"Wrote {overall_prefix.with_suffix('.png')}")
    print(f"Wrote {family_prefix.with_suffix('.pdf')}")
    print(f"Wrote {family_prefix.with_suffix('.png')}")
    print(f"Wrote {direction_prefix.with_suffix('.pdf')}")
    print(f"Wrote {direction_prefix.with_suffix('.png')}")
    print(f"Wrote {language_csv}")
    print(f"Wrote {family_csv}")
    print(f"Wrote {direction_csv}")
    print(f"Wrote {overall_csv}")
    print(f"Held-in FLORES languages evaluated: {language_df['language'].nunique()}")
    for target in TARGET_ORDER:
        if missing[target]:
            print(f"Missing held-in FLORES tasks for {target}:")
            for item in missing[target]:
                print(f"  {item}")


if __name__ == "__main__":
    main()
