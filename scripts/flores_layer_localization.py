#!/usr/bin/env python3
"""Compute-bounded FLORES layer localization and Dense-to-Expert transfer.

Every FLORES decision is scored from logged samples using ``flores_first_line``;
the harness's untruncated metric is deliberately never consumed here.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.flores_jobs import FAM_LANGS  # noqa: E402
from scripts.flores_postprocess import truncate_translation  # noqa: E402

SCREEN_LANGS = {
    "Slavic": ["mkd_Cyrl", "rus_Cyrl"],
    "Germanic": ["afr_Latn", "nld_Latn"],
    "Indic": ["hin_Deva", "tam_Taml"],
    "Austronesian": ["smo_Latn", "ind_Latn"],
    "Romance": ["spa_Latn", "ron_Latn"],
}
SCREEN_STARTS = (0, 5, 10, 15, 20, 24, 25, 26, 27, 28)
LENGTH_QUANTILES = (0.50, 0.90, 0.95, 0.99)
LONG_FIRST_LINE_CHARS = 512


def _response(sample: dict[str, Any]) -> str:
    value: Any = sample.get("filtered_resps") or sample.get("resps") or ""
    while isinstance(value, list):
        value = value[0] if value else ""
    return value if isinstance(value, str) else str(value)


def alpha_to_beta(alpha: float) -> float:
    """Map paper Dense-weight alpha to this runner's donor-movement beta."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return 1.0 - alpha


def beta_to_alpha(beta: float) -> float:
    """Map this runner's donor-movement beta to paper Dense-weight alpha."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    return 1.0 - beta


def select_families(raw: str) -> dict[str, list[str]]:
    """Resolve explicit, case-insensitive family names without changing defaults."""
    lookup = {name.lower(): name for name in FAM_LANGS}
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not requested:
        raise ValueError("--families must name at least one family")
    unknown = sorted(set(requested) - lookup.keys())
    if unknown:
        raise ValueError(f"Unknown FLORES families: {', '.join(unknown)}; choices: {', '.join(FAM_LANGS)}")
    return {lookup[name]: FAM_LANGS[lookup[name]] for name in requested}


def _quantile(values: list[int], quantile: float) -> int:
    """Return the nearest-rank-like quantile without another runtime dependency."""
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _output_diagnostics(responses: list[str], first_lines: list[str]) -> dict[str, float | int]:
    """Summarize pre-filter and scored-completion lengths in characters.

    ``flores_first_line`` deliberately drops text following a line boundary.
    Reporting both lengths exposes generations that score after filtering but
    continue producing text, while the long-first-line rate flags completions
    whose scored first line is itself unusually long.
    """
    response_lengths = [len(response.strip()) for response in responses]
    first_line_lengths = [len(output) for output in first_lines]
    diagnostics: dict[str, float | int] = {
        "first_line_truncation_rate": sum(
            "\\n" in response.lstrip() or "\n" in response.lstrip()
            for response in responses
        ) / len(responses),
        "long_first_line_rate_512_chars": sum(
            length >= LONG_FIRST_LINE_CHARS for length in first_line_lengths
        ) / len(first_line_lengths),
        "max_response_length": max(response_lengths, default=0),
        "max_first_line_length": max(first_line_lengths, default=0),
    }
    for quantile in LENGTH_QUANTILES:
        suffix = f"p{int(quantile * 100):02d}"
        diagnostics[f"response_length_{suffix}"] = _quantile(response_lengths, quantile)
        diagnostics[f"first_line_length_{suffix}"] = _quantile(first_line_lengths, quantile)
    return diagnostics


def _family_task_map(families: dict[str, list[str]]) -> dict[str, tuple[str, str]]:
    result = {}
    for family, languages in families.items():
        for language in languages:
            result[f"flores_{language}-eng_Latn"] = (family, "xx_en")
            result[f"flores_eng_Latn-{language}"] = (family, "en_xx")
    return result


def score_samples(eval_dir: Path, task_info: dict[str, tuple[str, str]]) -> dict[str, Any]:
    """Return filtered corpus ChrF plus scored and pre-filter length diagnostics."""
    rows = []
    response_groups: dict[str, list[str]] = defaultdict(list)
    first_line_groups: dict[str, list[str]] = defaultdict(list)
    for task, (family, direction) in task_info.items():
        sample_files = sorted(eval_dir.rglob(f"samples_{task}_*.jsonl"))
        if not sample_files:
            raise FileNotFoundError(f"No logged samples for {task} below {eval_dir}")
        refs, responses, outputs = [], [], []
        for line in sample_files[-1].read_text().splitlines():
            sample = json.loads(line)
            target = sample.get("target")
            if isinstance(target, list):
                target = target[0]
            if not isinstance(target, str):
                raise ValueError(f"Invalid reference in {sample_files[-1]}")
            refs.append(target)
            response = _response(sample)
            responses.append(response)
            outputs.append(truncate_translation(response))
        row = {
            "task": task, "family": family, "direction": direction,
            "chrf": sacrebleu.corpus_chrf(outputs, [refs]).score,
            "n_examples": len(refs),
            "empty_output_rate": sum(not output for output in outputs) / len(outputs),
            "mean_output_length": sum(len(output) for output in outputs) / len(outputs),
        }
        row.update(_output_diagnostics(responses, outputs))
        rows.append(row)
        for group in ("overall", f"family:{family}", f"direction:{direction}"):
            response_groups[group].extend(responses)
            first_line_groups[group].extend(outputs)
    result = aggregate_rows(rows)
    result["overall"].update(_output_diagnostics(
        response_groups["overall"], first_line_groups["overall"],
    ))
    for group, aggregate in result["aggregates"].items():
        aggregate.update(_output_diagnostics(response_groups[group], first_line_groups[group]))
    return result


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summary(items: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "macro_chrf": sum(x["chrf"] for x in items) / len(items),
            "empty_output_rate": sum(x["empty_output_rate"] for x in items) / len(items),
            "mean_output_length": sum(x["mean_output_length"] for x in items) / len(items),
            "n_tasks": len(items),
        }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"family:{row['family']}"].append(row)
        grouped[f"direction:{row['direction']}"].append(row)
    return {"tasks": rows, "overall": summary(rows), "aggregates": {key: summary(value) for key, value in grouped.items()}}


def add_dense_deltas(result: dict[str, Any], baseline: dict[str, Any]) -> None:
    """Attach the filtered-ChrF loss versus re-scored Dense generations."""
    baseline_by_task = {row["task"]: row for row in baseline["tasks"]}
    for row in result["tasks"]:
        row["dense_chrf"] = baseline_by_task[row["task"]]["chrf"]
        row["chrf_delta_vs_dense"] = row["chrf"] - row["dense_chrf"]
    result["vs_dense"] = aggregate_rows(result["tasks"])
    result["vs_dense"]["overall"]["macro_chrf_loss"] = -sum(
        row["chrf_delta_vs_dense"] for row in result["tasks"]
    ) / len(result["tasks"])


def _copy_hybrid(
    *, working: Any, donor: Any, start: int, end: int, beta: float,
) -> dict[str, int]:
    """Set target blocks to donor (or interpolate working toward donor), asserting provenance."""
    import torch

    from scripts.middle_layer_alpha_sweep import is_target_param

    working_state = working.state_dict()
    donor_state = donor.state_dict()
    target, other = 0, 0
    originals = {name: tensor.detach().cpu().clone() for name, tensor in working_state.items()}
    for name, tensor in working_state.items():
        is_target = is_target_param(name, start, end)
        if is_target:
            if name not in donor_state or donor_state[name].shape != tensor.shape:
                raise ValueError(f"Incompatible target tensor: {name}")
            source = donor_state[name].to(tensor.device)
            if beta == 1.0:
                tensor.copy_(source)
            else:
                tensor.copy_(tensor + beta * (source - tensor))
            target += 1
        else:
            other += 1
            if not torch.equal(tensor.detach().cpu(), originals[name]):
                raise AssertionError(f"Non-target tensor changed: {name}")
    # Exact provenance is only meaningful for beta=1 (reversion or full Dense transfer).
    if beta == 1.0:
        for name, tensor in working_state.items():
            expected = donor_state[name] if is_target_param(name, start, end) else originals[name]
            if not torch.equal(tensor.detach().cpu(), expected.detach().cpu()):
                raise AssertionError(f"Provenance check failed: {name}")
    return {"target_tensors": target, "non_target_tensors": other}


def run_variant(args: argparse.Namespace, *, tag: str, working_path: str, donor_path: str,
                families: dict[str, list[str]], beta: float, output_root: Path) -> dict[str, Any]:
    # Keep --help and family-selection validation available in a lightweight
    # environment; model/GPU dependencies are needed only for an actual run.
    import torch
    from transformers import AutoTokenizer

    from regularization import get_num_layers
    from scripts.middle_layer_alpha_sweep import load_text_model, run_lm_eval
    result_path = output_root / f"result_{tag}.json"
    if result_path.exists():
        print(f"[resume] using completed result: {result_path}", flush=True)
        return json.loads(result_path.read_text())
    task_info = _family_task_map(families)
    tasks = ",".join(task_info)
    checkpoint = Path(args.checkpoint_dir) / f"checkpoint_{tag}"
    eval_dir = output_root / f"eval_{tag}"
    print(f"[variant] {tag}: {len(task_info)} FLORES tasks, layers [{args.start},{args.end}), beta={beta}", flush=True)
    state_path = output_root / f"state_{tag}.json"
    state_path.write_text(json.dumps({"status": "running", "tag": tag}))
    working = load_text_model(working_path, args.device)
    donor = load_text_model(donor_path, args.device)
    num_layers = get_num_layers(working)
    if not (0 <= args.start < args.end <= num_layers):
        raise ValueError(f"Invalid range [{args.start},{args.end}) for {num_layers} layers")
    provenance = _copy_hybrid(working=working, donor=donor, start=args.start, end=args.end, beta=beta)
    checkpoint.mkdir(parents=True, exist_ok=True)
    working.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(working_path, trust_remote_code=True).save_pretrained(checkpoint)
    try:
        # ``run_lm_eval`` passes this stable path to lm-eval's --use_cache.
        # CachingLM commits every deterministic generation to SQLite, allowing a
        # requeue/retry to generate only requests missing from this window cache.
        run_lm_eval(checkpoint=checkpoint, tasks=tasks, output_dir=eval_dir,
                    cache_dir=output_root / f"response_cache_{tag}", batch_size=args.batch_size,
                    limit=args.limit or None, include_path=args.include_path or None, log_samples=True)
        result = score_samples(eval_dir, task_info)
        result.update({"tag": tag, "working_model": working_path, "donor_model": donor_path,
                       "layer_start": args.start, "layer_end": args.end, "beta": beta,
                       "paper_alpha": beta_to_alpha(beta),
                       "provenance": provenance, "eval_dir": str(eval_dir)})
        # Atomic replace makes a completed window idempotent on Slurm requeue or
        # a fresh retry with the same output directory.
        temporary_result = result_path.with_suffix(".tmp")
        temporary_result.write_text(json.dumps(result, indent=2))
        temporary_result.replace(result_path)
        state_path.write_text(json.dumps({"status": "completed", "tag": tag}))
        return result
    finally:
        del working, donor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not args.keep_checkpoints:
            shutil.rmtree(checkpoint, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["screen", "confirm", "transfer"], required=True)
    parser.add_argument("--dense_model", required=True)
    parser.add_argument("--base_model", default="google/gemma-3-4b-pt")
    parser.add_argument("--expert_model", default="", help="Required for transfer")
    parser.add_argument("--donor_model", default="", help="Transfer donor; defaults to --dense_model")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_dir", default="", help="NVMe directory for ephemeral hybrid checkpoints")
    parser.add_argument("--baseline_eval_dir", default="", help="Existing Dense samples to re-score with the same filter")
    parser.add_argument("--matched-dense-control", action="store_true",
                        help="For confirm, generate beta=0 Dense control before the intervention")
    parser.add_argument("--start", type=int, help="Required for confirm/transfer")
    parser.add_argument("--end", type=int, help="Required for confirm/transfer")
    parser.add_argument("--starts", default=",".join(map(str, SCREEN_STARTS)))
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--families", default="", help="Comma-separated family subset for confirm/transfer")
    parser.add_argument("--batch_size", default="8")
    parser.add_argument("--include_path", default="")
    parser.add_argument("--limit", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep_checkpoints", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint_dir:
        args.checkpoint_dir = str(root / "temporary_checkpoints")
    records = []
    if args.phase == "screen":
        if args.start is not None or args.end is not None:
            raise ValueError("screen derives ranges from --starts and --window")
        for start in (int(value) for value in args.starts.split(",") if value):
            args.start, args.end = start, start + args.window
            records.append(run_variant(args, tag=f"dense-revert-{start:02d}-{args.end:02d}", working_path=args.dense_model,
                                       donor_path=args.base_model, families=SCREEN_LANGS, beta=1.0, output_root=root))
    else:
        if args.start is None or args.end is None:
            raise ValueError("--start and --end are required outside the screen")
        if args.phase == "confirm":
            families = select_families(args.families) if args.families else FAM_LANGS
            if args.matched_dense_control:
                records.append(run_variant(
                    args, tag="dense-control", working_path=args.dense_model,
                    donor_path=args.base_model, families=families, beta=0.0, output_root=root,
                ))
            records.append(run_variant(args, tag=f"dense-revert-{args.start:02d}-{args.end:02d}", working_path=args.dense_model,
                                       donor_path=args.base_model, families=families, beta=args.beta, output_root=root))
        else:
            if not args.expert_model:
                raise ValueError("--expert_model is required for transfer")
            chosen = select_families(args.families or "indic,austronesian")
            donor = args.donor_model or args.dense_model
            if args.matched_dense_control:
                records.append(run_variant(args, tag="expert-control", working_path=args.expert_model,
                                           donor_path=donor, families=chosen, beta=0.0, output_root=root))
            records.append(run_variant(args, tag=f"expert-to-donor-b{args.beta:g}", working_path=args.expert_model,
                                       donor_path=donor, families=chosen, beta=args.beta, output_root=root))
    if args.matched_dense_control:
        baseline_tag = "dense-control" if args.phase == "confirm" else "expert-control"
        baseline = next(record for record in records if record["tag"] == baseline_tag)
        for record in records:
            if record is not baseline:
                add_dense_deltas(record, baseline)
    elif args.baseline_eval_dir:
        families = SCREEN_LANGS if args.phase == "screen" else (families if args.phase == "confirm" else chosen)
        baseline = score_samples(Path(args.baseline_eval_dir), _family_task_map(families))
        for record in records:
            add_dense_deltas(record, baseline)
    else:
        baseline = None
    with (root / "summary.json").open("w") as handle:
        json.dump({"phase": args.phase, "baseline": baseline, "records": records}, handle, indent=2)
    print(f"[done] {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
