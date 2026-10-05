#!/usr/bin/env python3
"""Report paper-protocol FLORES ChrF by direction for Expert vs E.-Rev.

The paper's FLORES table first truncates each saved generation to its first
line. This script applies that same protocol to the existing saved samples and
reports held-in family macros separately for xx->en and en->xx.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bootstrap_ci_flores import (  # noqa: E402
    DIRS,
    FAMILIES,
    cell_langs,
    chrf_from_stats,
    find_task_files,
    task_stats,
)


def direction_scores(family: str, strategy: str) -> dict[str, float]:
    task_files = find_task_files(DIRS[(strategy, family)])
    held = set(cell_langs(family, [lang for _direction, lang in task_files]))
    by_direction: dict[str, list[float]] = defaultdict(list)
    for (direction, language), sample_path in sorted(task_files.items()):
        if language not in held:
            continue
        stats = task_stats(sample_path)
        if stats.size:
            by_direction[direction].append(chrf_from_stats(stats.sum(axis=0)))
    return {direction: sum(scores) / len(scores) for direction, scores in by_direction.items()}


def main() -> None:
    print("| Family | Expert xx→en | E.-Rev. xx→en | Δ | Expert en→xx | E.-Rev. en→xx | Δ |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for family in FAMILIES:
        expert = direction_scores(family, "expert")
        reverted = direction_scores(family, "expert-reverted")
        print(
            f"| {family} | {expert['xx_en']:.2f} | {reverted['xx_en']:.2f} | "
            f"{reverted['xx_en'] - expert['xx_en']:+.2f} | {expert['en_xx']:.2f} | "
            f"{reverted['en_xx']:.2f} | {reverted['en_xx'] - expert['en_xx']:+.2f} |"
        )


if __name__ == "__main__":
    main()
