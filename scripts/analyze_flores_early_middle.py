#!/usr/bin/env python3
"""Paired, first-line FLORES analysis for the early/middle follow-up plan.

This deliberately consumes logged lm-eval samples rather than harness metrics.
It is safe to smoke-test on a login node with ``--n-bootstrap 10 --max-tasks 1``;
the registered 10,000-replicate run belongs in the supplied CPU Slurm job.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.flores_postprocess import truncate_translation


PAPER_TARGETS = {"first": "[0,9)", "middle": "[9,28)"}
WINDOWS = {0: "[0,6)", 5: "[5,11)", 10: "[10,16)", 15: "[15,21)", 20: "[20,26)"}
EUROPEAN = {"slavic", "germanic", "romance"}


def response(sample: dict[str, Any]) -> str:
    value: Any = sample.get("filtered_resps") or sample.get("resps") or ""
    while isinstance(value, list):
        value = value[0] if value else ""
    return value if isinstance(value, str) else str(value)


def reference(sample: dict[str, Any]) -> str:
    value = sample.get("target")
    if isinstance(value, list):
        value = value[0] if value else ""
    if not isinstance(value, str):
        raise ValueError("Sample has no string target")
    return value


def signature(sample: dict[str, Any]) -> str:
    """Fields which must agree before a response pair can be treated as paired."""
    arguments = sample.get("arguments", {})
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def read_samples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            sample = json.loads(line)
            rows.append({"doc_id": sample.get("doc_id"), "reference": reference(sample),
                         "response": response(sample), "signature": signature(sample)})
    if not rows:
        raise ValueError(f"No samples in {path}")
    if any(row["doc_id"] is None for row in rows):
        raise ValueError(f"Missing doc_id in {path}")
    return rows


def pair_samples(dense_path: Path, intervention_path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    dense, intervention = read_samples(dense_path), read_samples(intervention_path)
    dense_by_id = {row["doc_id"]: row for row in dense}
    intervention_by_id = {row["doc_id"]: row for row in intervention}
    if len(dense_by_id) != len(dense) or len(intervention_by_id) != len(intervention):
        raise ValueError("Duplicate doc_id prevents paired analysis")
    if dense_by_id.keys() != intervention_by_id.keys():
        raise ValueError(f"doc_id mismatch: {dense_path} vs {intervention_path}")
    pairs = [(dense_by_id[key], intervention_by_id[key]) for key in sorted(dense_by_id)]
    for left, right in pairs:
        if left["reference"] != right["reference"]:
            raise ValueError(f"Reference mismatch at doc_id={left['doc_id']}")
        if left["signature"] != right["signature"]:
            raise ValueError(f"Prompt/generation configuration mismatch at doc_id={left['doc_id']}")
    return pairs


def deterministic_sample(pairs: list[tuple[dict[str, Any], dict[str, Any]]], size: int,
                         seed: int, task: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Select a reproducible diagnostic subset without breaking response pairs."""
    if size <= 0 or size >= len(pairs):
        return pairs
    task_seed = int.from_bytes(hashlib.sha256(f"{seed}:{task}".encode()).digest()[:8], "big")
    indices = sorted(random.Random(task_seed).sample(range(len(pairs)), size))
    return [pairs[index] for index in indices]


def chrf(outputs: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_chrf(outputs, [refs]).score


def scores(pairs: list[tuple[dict[str, Any], dict[str, Any]]], indices: Iterable[int] | None = None) -> tuple[float, float]:
    chosen = range(len(pairs)) if indices is None else indices
    dense, intervention, refs = [], [], []
    for index in chosen:
        left, right = pairs[index]
        dense.append(truncate_translation(left["response"]))
        intervention.append(truncate_translation(right["response"]))
        refs.append(left["reference"])
    return chrf(dense, refs), chrf(intervention, refs)


def diagnostics(rows: list[dict[str, Any]]) -> dict[str, float]:
    raws = [row["response"] for row in rows]
    first = [truncate_translation(item) for item in raws]
    lengths = sorted(len(item) for item in first)
    def quantile(q: float) -> float:
        return lengths[round((len(lengths) - 1) * q)] if lengths else 0.0
    return {
        "empty_output_rate": sum(not item for item in first) / len(first),
        "mean_first_line_length": mean(map(len, first)), "p50_first_line_length": quantile(.50),
        "p90_first_line_length": quantile(.90), "p95_first_line_length": quantile(.95),
        "p99_first_line_length": quantile(.99),
        "long_first_line_rate_512_chars": sum(len(item) >= 512 for item in first) / len(first),
        "first_line_truncation_rate": sum(item.lstrip() != truncate_translation(item) for item in raws) / len(raws),
    }


def _bootstrap_chunk(task_pairs: list[list[tuple[dict[str, Any], dict[str, Any]]]], n: int,
                     seed: int) -> list[float]:
    """One deterministic bootstrap shard, suitable for a worker process."""
    rng = random.Random(seed)
    deltas = []
    for _ in range(n):
        per_task = []
        for pairs in task_pairs:
            selected = [rng.randrange(len(pairs)) for _ in pairs]
            dense, intervention = scores(pairs, selected)
            per_task.append(intervention - dense)
        deltas.append(mean(per_task))
    return deltas


def bootstrap(task_pairs: list[list[tuple[dict[str, Any], dict[str, Any]]]], n: int, seed: int,
              workers: int) -> tuple[float, float]:
    """Paired sentence bootstrap, retaining the paper's equal-weight task macro."""
    workers = min(workers, n)
    counts = [n // workers + (index < n % workers) for index in range(workers)]
    seeds = [seed + index for index in range(workers)]
    if workers == 1:
        deltas = _bootstrap_chunk(task_pairs, counts[0], seeds[0])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            shards = pool.map(_bootstrap_chunk, [task_pairs] * workers, counts, seeds)
            deltas = [value for shard in shards for value in shard]
    deltas.sort()
    return deltas[round(.025 * (n - 1))], deltas[round(.975 * (n - 1))]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def paper_conditions(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    needed = {"target", "alpha", "family", "language", "direction", "task", "sample_path"}
    if not rows or not needed <= rows[0].keys():
        raise ValueError("Paper task CSV lacks required sample metadata columns")
    return [row for row in rows if row["target"] in PAPER_TARGETS]


def sliding_conditions(root: Path) -> list[dict[str, str]]:
    rows = []
    for start, label in WINDOWS.items():
        result = root / f"screen_start_{start}" / f"result_dense-revert-{start:02d}-{start + 6:02d}.json"
        payload = json.loads(result.read_text())
        for task in payload["tasks"]:
            sample_files = sorted(Path(payload["eval_dir"]).rglob(f"samples_{task['task']}_*.jsonl"))
            if not sample_files:
                raise FileNotFoundError(f"No samples for {task['task']} under {payload['eval_dir']}")
            language = task["task"].split("_")[1].split("-")[0] if task["direction"] == "xx_en" else task["task"].split("-")[1]
            rows.append({"condition": f"sliding {label}", "target": "sliding", "alpha": "0",
                         "family": task["family"].lower(), "language": language,
                         "direction": task["direction"], "task": task["task"],
                         "sample_path": str(sample_files[-1])})
    return rows


def phase_b_conditions(summary_path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    """Build paired conditions from a completed matched-control Phase-B summary."""
    payload = json.loads(summary_path.read_text())
    records = {record["tag"]: record for record in payload["records"]}
    control = records.get("dense-control")
    intervention = records.get("dense-revert-05-11")
    if not control or not intervention:
        raise ValueError("Phase-B summary needs dense-control and dense-revert-05-11 records")
    dense, conditions = {}, []
    for row in intervention["tasks"]:
        task = row["task"]
        control_files = sorted(Path(control["eval_dir"]).rglob(f"samples_{task}_*.jsonl"))
        intervention_files = sorted(Path(intervention["eval_dir"]).rglob(f"samples_{task}_*.jsonl"))
        if not control_files or not intervention_files:
            raise FileNotFoundError(f"Missing matched samples for {task}")
        family, direction = row["family"].lower(), row["direction"]
        language = task.split("_")[1].split("-")[0] if direction == "xx_en" else task.split("-")[1]
        dense[task] = {"task": task, "sample_path": str(control_files[-1])}
        conditions.append({"condition": "phase-b [5,11) alpha=0", "target": "phase-b", "alpha": "0",
                           "family": family, "language": language, "direction": direction,
                           "task": task, "sample_path": str(intervention_files[-1])})
    conditions.extend({**row, "family": "European"} for row in conditions
                      if row["family"] in EUROPEAN and row["direction"] == "xx_en")
    return conditions, dense


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        # LOO rows intentionally add ``dropped_language`` and ``macro_delta``
        # to the interval table. Preserve every column when row schemas differ.
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def endpoint_check(paper: list[dict[str, str]]) -> None:
    values: dict[str, list[float]] = defaultdict(list)
    for row in paper:
        if float(row["alpha"]) == 1.0:
            values[row["target"]].append(float(row.get("fixed_chrf", "nan")))
    means = {target: mean(value) for target, value in values.items() if value}
    if len(means) != 3 or max(means.values()) - min(means.values()) > .1:
        raise ValueError(f"Paper alpha=1 endpoint disagreement exceeds 0.1 macro ChrF: {means}")


def analyze(conditions: list[dict[str, str]], dense_by_task: dict[str, dict[str, str]], n_bootstrap: int,
            seed: int, workers: int, sample_size: int, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cell_rows, language_rows, interval_rows, diagnostic_rows = [], [], [], []
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in conditions:
        condition = row.get("condition") or f"paper {PAPER_TARGETS[row['target']]} alpha={row['alpha']}"
        grouped[(condition, row["family"], row["direction"], row["alpha"])].append(row)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    total_cells = len(grouped)
    completed_cells = completed_tasks = completed_samples = 0
    progress_path = output_dir / "progress.json"
    write_json_atomic(progress_path, {"status": "running", "total_cells": total_cells,
        "completed_cells": 0, "completed_task_condition_pairs": 0,
        "completed_paired_samples": 0, "n_bootstrap": n_bootstrap,
        "sample_size_per_task": sample_size})
    for (condition, family, direction, alpha), rows in sorted(grouped.items()):
        identity = json.dumps([condition, family, direction, alpha], separators=(",", ":"))
        partial_path = partial_dir / f"{hashlib.sha256(identity.encode()).hexdigest()}.json"
        if partial_path.exists():
            bundle = json.loads(partial_path.read_text())
            cell_rows.append(bundle["cell"]); language_rows.extend(bundle["languages"])
            interval_rows.append(bundle["interval"]); diagnostic_rows.extend(bundle["diagnostics"])
            completed_cells += 1; completed_tasks += bundle["n_task_condition_pairs"]
            completed_samples += bundle["n_paired_samples"]
            write_json_atomic(progress_path, {"status": "running", "total_cells": total_cells,
                "completed_cells": completed_cells, "completed_task_condition_pairs": completed_tasks,
                "completed_paired_samples": completed_samples, "n_bootstrap": n_bootstrap,
                "sample_size_per_task": sample_size, "last_cell": identity, "resumed": True})
            continue
        items = []
        for row in rows:
            dense = dense_by_task.get(row["task"])
            if not dense:
                raise ValueError(f"No canonical Dense sample for {row['task']}")
            pairs = deterministic_sample(pair_samples(Path(dense["sample_path"]), Path(row["sample_path"])),
                                         sample_size, seed, row["task"])
            items.append((row, pairs))
        task_deltas = []
        cell_languages, cell_diagnostics = [], []
        for row, pairs in items:
            dense_score, intervention_score = scores(pairs)
            delta = intervention_score - dense_score
            task_deltas.append(delta)
            cell_languages.append({"condition": condition, "family": family, "direction": direction,
                                  "language": row["language"], "task": row["task"], "alpha": alpha,
                                  "dense_chrf": dense_score, "intervention_chrf": intervention_score, "delta": delta})
            for label, collection in (("dense", [left for left, _ in pairs]), ("intervention", [right for _, right in pairs])):
                cell_diagnostics.append({"condition": condition, "family": family, "direction": direction,
                                        "language": row["language"], "task": row["task"], "state": label,
                                        **diagnostics(collection)})
        cell = {"condition": condition, "family": family, "direction": direction, "alpha": alpha,
                          "n_tasks": len(items), "macro_delta": mean(task_deltas),
                          "dense_macro_chrf": mean([scores(p)[0] for _, p in items]),
                          "intervention_macro_chrf": mean([scores(p)[1] for _, p in items])}
        lo, hi = bootstrap([pairs for _, pairs in items], n_bootstrap, seed, workers)
        interval = {"condition": condition, "family": family, "direction": direction, "alpha": alpha,
                    "n_tasks": len(items), "delta_ci95_low": lo, "delta_ci95_high": hi}
        bundle = {"cell": cell, "languages": cell_languages, "interval": interval,
                  "diagnostics": cell_diagnostics, "n_task_condition_pairs": len(items),
                  "n_paired_samples": sum(len(pairs) for _, pairs in items)}
        write_json_atomic(partial_path, bundle)
        cell_rows.append(cell); language_rows.extend(cell_languages); interval_rows.append(interval)
        diagnostic_rows.extend(cell_diagnostics)
        completed_cells += 1; completed_tasks += len(items)
        completed_samples += bundle["n_paired_samples"]
        write_json_atomic(progress_path, {"status": "running", "total_cells": total_cells,
            "completed_cells": completed_cells, "completed_task_condition_pairs": completed_tasks,
            "completed_paired_samples": completed_samples, "n_bootstrap": n_bootstrap,
            "sample_size_per_task": sample_size, "last_cell": identity, "resumed": False})
    write_json_atomic(progress_path, {"status": "completed", "total_cells": total_cells,
        "completed_cells": completed_cells, "completed_task_condition_pairs": completed_tasks,
        "completed_paired_samples": completed_samples, "n_bootstrap": n_bootstrap,
        "sample_size_per_task": sample_size})
    return cell_rows, language_rows, interval_rows, diagnostic_rows


def add_robustness(language_rows: list[dict[str, Any]], interval_rows: list[dict[str, Any]], n_bootstrap: int, seed: int) -> None:
    early = [row for row in language_rows if row["condition"] == "sliding [5,11)" and row["direction"] == "xx_en"]
    pooled = [row for row in early if row["family"] in EUROPEAN]
    if pooled:
        # Language is one task in this screen; task-macro LOO remains explicit and transparent.
        for dropped in pooled:
            kept = [row for row in pooled if row is not dropped]
            interval_rows.append({"condition": "sliding [5,11) European LOO", "family": "European",
                                  "direction": "xx_en", "alpha": "0", "dropped_language": dropped["language"],
                                  "n_tasks": len(kept), "macro_delta": mean(row["delta"] for row in kept),
                                  "delta_ci95_low": "", "delta_ci95_high": ""})


def binomial_interval(successes: int, total: int, level: float = .95) -> tuple[float, float]:
    """Two-sided exact (Clopper-Pearson) interval by inverting binomial tails."""
    if not 0 <= successes <= total or total == 0:
        raise ValueError("Invalid binomial count")
    tail = (1.0 - level) / 2.0
    def cdf(k: int, probability: float) -> float:
        return sum(math.comb(total, i) * probability ** i * (1 - probability) ** (total - i)
                   for i in range(k + 1))
    def solve(predicate: Any) -> float:
        low, high = 0.0, 1.0
        for _ in range(70):
            mid = (low + high) / 2
            if predicate(mid): high = mid
            else: low = mid
        return (low + high) / 2
    lower = 0.0 if successes == 0 else solve(lambda p: 1 - cdf(successes - 1, p) >= tail)
    upper = 1.0 if successes == total else solve(lambda p: cdf(successes, p) <= tail)
    return lower, upper


def sign_robustness(language_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in language_rows:
        groups[(row["condition"], row["family"], row["direction"], row["alpha"])].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        positive = sum(row["delta"] > 0 for row in rows)
        low, high = binomial_interval(positive, len(rows))
        output.append({"condition": key[0], "family": key[1], "direction": key[2], "alpha": key[3],
                       "n_languages": len(rows), "n_positive": positive,
                       "exact_binomial_ci95_low": low, "exact_binomial_ci95_high": high})
    return output


def cross_experiment_signs(language_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # The paper CSV uses display-language labels (for example ``af``), whereas
    # localization rows carry FLORES codes (``afr_Latn``). Task names are the
    # shared stable identifier for this replication comparison.
    first = {(r["family"], r["direction"], r["task"]): r for r in language_rows
             if r["condition"] == "paper [0,9) alpha=0.0"}
    sliding = {(r["family"], r["direction"], r["task"]): r for r in language_rows
               if r["condition"] == "sliding [5,11)"}
    return [{"family": key[0], "direction": key[1], "language": row["language"], "task": key[2],
             "paper_delta": row["delta"], "sliding_delta": sliding[key]["delta"],
             "same_sign": (row["delta"] > 0) == (sliding[key]["delta"] > 0)}
            for key, row in sorted(first.items()) if key in sliding]


def self_test() -> None:
    assert truncate_translation("  hello\\nmore ") == "hello"
    assert alpha_to_beta(0.0) == 1.0 and alpha_to_beta(1.0) == 0.0
    pairs = [({"response": "a", "reference": "b"}, {"response": "b", "reference": "b"})]
    dense, intervention = scores(pairs)
    assert intervention > dense
    assert bootstrap([pairs], 10, 7, 1)[0] == bootstrap([pairs], 10, 7, 1)[1]
    with tempfile.TemporaryDirectory() as directory:
        left, right = Path(directory) / "left.jsonl", Path(directory) / "right.jsonl"
        base = {"doc_id": 7, "target": "ref", "arguments": {"gen_args_0": {"arg_1": {"max_gen_toks": 512}}}}
        left.write_text(json.dumps({**base, "resps": [["dense"]]}) + "\n")
        right.write_text(json.dumps({**base, "resps": [["intervention"]]}) + "\n")
        assert len(pair_samples(left, right)) == 1


def alpha_to_beta(alpha: float) -> float:
    """Paper alpha is Dense weight; the localization runner uses Base-donor beta."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return 1.0 - alpha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-task-csv", type=Path)
    parser.add_argument("--sliding-root", type=Path)
    parser.add_argument("--phase-b-summary", type=Path,
                        help="Completed matched-control Phase-B summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                        help="Bootstrap worker processes (default: Slurm CPU allocation or 1)")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Deterministic paired sentences per task; 0 uses every sentence")
    parser.add_argument("--max-tasks", type=int, default=0, help="Smoke-test cap per input collection")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); print("self-test passed"); return
    if args.n_bootstrap < 2 or args.workers < 1 or args.sample_size < 0:
        raise ValueError("--n-bootstrap >= 2, --workers >= 1, and --sample-size >= 0 are required")
    if bool(args.phase_b_summary) == bool(args.paper_task_csv or args.sliding_root):
        raise ValueError("Use --phase-b-summary, or both --paper-task-csv and --sliding-root")
    if args.phase_b_summary:
        conditions, canonical = phase_b_conditions(args.phase_b_summary)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cell, languages, intervals, audits = analyze(
            conditions, canonical, args.n_bootstrap, args.seed, args.workers, args.sample_size, args.output_dir,
        )
        write_csv(args.output_dir / "phase_b_family_direction.csv", cell)
        write_csv(args.output_dir / "phase_b_per_language.csv", languages)
        write_csv(args.output_dir / "paired_bootstrap_intervals.csv", intervals)
        write_csv(args.output_dir / "output_diagnostics.csv", audits)
        (args.output_dir / "offline_findings.md").write_text("# Phase B paired bootstrap\n\nResults in CSV files.\n")
        return
    if not args.paper_task_csv or not args.sliding_root:
        raise ValueError("--paper-task-csv and --sliding-root are required outside Phase B")
    all_paper_rows = read_csv(args.paper_task_csv)
    endpoint_check(all_paper_rows)
    paper = paper_conditions(all_paper_rows)
    canonical = {row["task"]: row for row in paper if row["target"] == "first" and float(row["alpha"]) == 1.0}
    if not canonical:
        raise ValueError("No target=first, alpha=1 canonical Dense samples")
    conditions = []
    for row in paper:
        row = dict(row); row["condition"] = f"paper {PAPER_TARGETS[row['target']]} alpha={row['alpha']}"; conditions.append(row)
    conditions.extend(sliding_conditions(args.sliding_root))
    # The registered pooled endpoint is a task-level macro over the three
    # European xx->en family cells, not a sentence-pooled corpus score.
    conditions.extend({**row, "family": "European"} for row in conditions
                      if row["family"] in EUROPEAN and row["direction"] == "xx_en")
    if args.max_tasks:
        conditions = conditions[:args.max_tasks]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cell, languages, intervals, audits = analyze(
        conditions, canonical, args.n_bootstrap, args.seed, args.workers, args.sample_size, args.output_dir,
    )
    add_robustness(languages, intervals, args.n_bootstrap, args.seed)
    write_csv(args.output_dir / "paper_family_direction_alpha.csv", [row for row in cell if row["condition"].startswith("paper")])
    write_csv(args.output_dir / "sliding_family_direction.csv", [row for row in cell if row["condition"].startswith("sliding")])
    write_csv(args.output_dir / "early_per_language.csv", [row for row in languages if row["condition"] == "sliding [5,11)"])
    write_csv(args.output_dir / "paired_bootstrap_intervals.csv", intervals)
    write_csv(args.output_dir / "output_diagnostics.csv", audits)
    write_csv(args.output_dir / "sign_robustness.csv", sign_robustness(languages))
    write_csv(args.output_dir / "cross_experiment_signs.csv", cross_experiment_signs(languages))
    report = ["# FLORES early/middle offline findings", "", f"Bootstrap replicates: {args.n_bootstrap}; seed: {args.seed}.", "", "## Family × direction cells", ""]
    report += [f"- {row['condition']} | {row['family']} {row['direction']}: Δ={row['macro_delta']:+.3f}" for row in cell]
    (args.output_dir / "offline_findings.md").write_text("\n".join(report) + "\n")
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps({"seed": args.seed, "n_bootstrap": args.n_bootstrap, "workers": args.workers, "sample_size_per_task": args.sample_size, "alpha_beta_mapping": "beta = 1 - alpha", "canonical_dense": "paper first alpha=1"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
