#!/usr/bin/env python3
"""Re-score FLORES alpha-sweep samples with the paper truncation rule.

The sample-writing reruns reused the original eval directories, so summary JSONLs
and results JSONs may be duplicated. This script ignores summaries entirely and
dedupes sample files by selecting the latest timestamped sample JSONL for each
(target layer group, alpha, task).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.flores_postprocess import parse_sample_pair, truncate_translation
from scripts.plot_flores_layer_group_sweeps import CODE_TO_FAMILY_LANGUAGE


DEFAULT_ROOT = Path("data/drift_alpha_sweep")
TARGETS = ("first", "middle", "last")
ALPHA_RE = re.compile(r"eval_alpha_(.+)$")
SAMPLE_RE = re.compile(r"samples_(flores_.+?)_(\d{4}-\d{2}-\d{2}T.+)\.jsonl$")


def alpha_from_dir(path: Path) -> float:
    match = ALPHA_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected eval dir name: {path}")
    return float(match.group(1).replace("p", "."))


def task_and_timestamp(path: Path) -> tuple[str, str]:
    match = SAMPLE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected sample filename: {path.name}")
    return match.group(1), match.group(2)


def collect_latest_samples(root: Path) -> dict[tuple[str, float, str], Path]:
    latest: dict[tuple[str, float, str], tuple[str, Path]] = {}
    for target in TARGETS:
        target_root = root / f"dense_{target}_flores_heldin"
        for eval_dir in sorted(target_root.glob("eval_alpha_*")):
            alpha = alpha_from_dir(eval_dir)
            for sample_path in eval_dir.glob("*/samples_flores_*.jsonl"):
                task, timestamp = task_and_timestamp(sample_path)
                key = (target, alpha, task)
                if key not in latest or timestamp > latest[key][0]:
                    latest[key] = (timestamp, sample_path)
    return {key: path for key, (_, path) in latest.items()}


def family_language(lang_code: str) -> tuple[str, str]:
    if lang_code not in CODE_TO_FAMILY_LANGUAGE:
        raise KeyError(f"No family/language mapping for {lang_code}")
    return CODE_TO_FAMILY_LANGUAGE[lang_code]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def score_task_chrf(jsonl_path: Path) -> dict | None:
    refs, raws, fixed = [], [], []
    n_truncated = 0
    with jsonl_path.open() as f:
        for line in f:
            sample = json.loads(line)
            target = sample.get("target", "") or ""
            resps = sample.get("resps") or sample.get("filtered_resps")
            if not resps:
                continue
            response = resps[0]
            raw = response[0] if isinstance(response, list) else response
            truncated = truncate_translation(raw)
            refs.append(target)
            raws.append(raw)
            fixed.append(truncated)
            if truncated != raw.strip():
                n_truncated += 1
    if not refs:
        return None
    return {
        "n": len(refs),
        "raw_chrf": sacrebleu.corpus_chrf(raws, [refs]).score,
        "fixed_chrf": sacrebleu.corpus_chrf(fixed, [refs]).score,
        "pct_truncated": 100 * n_truncated / len(refs),
    }


def aggregate(rows: list[dict], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)

    out = []
    for group_key, group_rows in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item.update(
            {
                "raw_chrf": mean([r["raw_chrf"] for r in group_rows]),
                "fixed_chrf": mean([r["fixed_chrf"] for r in group_rows]),
                "pct_truncated": mean([r["pct_truncated"] for r in group_rows]),
                "n_tasks": len(group_rows),
                "n_languages": len({r["language"] for r in group_rows}),
            }
        )
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    latest = collect_latest_samples(args.root)
    rows = []
    items = sorted(latest.items())
    for idx, ((target, alpha, task), sample_path) in enumerate(items, start=1):
        parsed = parse_sample_pair(sample_path)
        if not parsed:
            continue
        _src, _tgt, direction, lang_code = parsed
        family, language = family_language(lang_code)
        stats = score_task_chrf(sample_path)
        if not stats:
            continue
        rows.append(
            {
                "target": target,
                "alpha": alpha,
                "family": family,
                "language": language,
                "lang_code": lang_code,
                "direction": direction,
                "task": task,
                "sample_path": str(sample_path),
                "n": stats["n"],
                "raw_chrf": stats["raw_chrf"],
                "fixed_chrf": stats["fixed_chrf"],
                "pct_truncated": stats["pct_truncated"],
            }
        )
        if idx % 60 == 0:
            print(f"Scored {idx}/{len(items)} sample files", flush=True)

    if not rows:
        raise SystemExit("No FLORES alpha-sweep sample rows found.")

    out_dir = args.output_dir
    write_csv(out_dir / "flores_alpha_sweep_fixed_task.csv", rows)
    write_csv(
        out_dir / "flores_alpha_sweep_fixed_overall.csv",
        aggregate(rows, ["target", "alpha"]),
    )
    write_csv(
        out_dir / "flores_alpha_sweep_fixed_direction.csv",
        aggregate(rows, ["target", "alpha", "direction"]),
    )
    write_csv(
        out_dir / "flores_alpha_sweep_fixed_family.csv",
        aggregate(rows, ["target", "alpha", "family"]),
    )
    write_csv(
        out_dir / "flores_alpha_sweep_fixed_family_direction.csv",
        aggregate(rows, ["target", "alpha", "family", "direction"]),
    )

    print(f"Latest sample files: {len(latest)}")
    print(f"Scored task rows: {len(rows)}")
    print(f"Wrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
