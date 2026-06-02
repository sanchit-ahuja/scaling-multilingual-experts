#!/usr/bin/env python3
"""Extract held-in FLORES task names from the local lm-eval task YAMLs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_FLORES_DIR = Path(
    os.environ.get(
        "FLORES_DIR",
        Path.home() / "lm-evaluation-harness" / "lm_eval" / "tasks" / "flores",
    )
)
GROUP_FILES = ("flores_en_xx.yaml", "flores_xx_en.yaml")


def heldin_tasks(flores_dir: Path) -> list[str]:
    tasks: list[str] = []
    for name in GROUP_FILES:
        path = flores_dir / name
        in_heldin_block = False
        for raw_line in path.read_text().splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("#") and "Held-in languages" in stripped:
                in_heldin_block = True
                continue
            if in_heldin_block and (
                stripped.startswith("aggregate_metric_list:")
                or stripped.startswith("metadata:")
            ):
                break
            if not in_heldin_block or not stripped.startswith("- "):
                continue
            task = stripped[2:].strip()
            if task.startswith("flores_"):
                tasks.append(task)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flores-dir", type=Path, default=DEFAULT_FLORES_DIR)
    parser.add_argument("--separator", default=",")
    args = parser.parse_args()

    tasks = heldin_tasks(args.flores_dir)
    if not tasks:
        raise SystemExit(f"No held-in FLORES tasks found under {args.flores_dir}")
    print(args.separator.join(tasks))


if __name__ == "__main__":
    main()
