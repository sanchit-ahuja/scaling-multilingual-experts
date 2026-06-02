#!/usr/bin/env python3
"""Evaluate soups of family experts after layer-group interpolation to base.

For each alpha, every expert contributes:

    theta_target(alpha) = theta_base + alpha * (theta_expert - theta_base)

on the selected layer group, while non-target layers remain the expert weights.
The script uniformly averages these interpolated experts into a single soup and
optionally evaluates it with lm-eval.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from middle_layer_alpha_sweep import (
    alpha_name,
    is_target_param,
    iter_alphas,
    load_text_model,
    parse_alphas,
    run_lm_eval,
    target_layer_range,
)
from regularization import get_num_layers


def selection_label(
    target_layers: str,
    *,
    first_layers: int,
    last_layers: int,
    num_layers: int,
) -> tuple[int | None, int | None]:
    if target_layers == "all":
        return None, None
    last_start = num_layers - last_layers
    return target_layer_range(
        target_layers,
        first_layers=first_layers,
        last_start=last_start,
        num_layers=num_layers,
    )


def is_selected_param(
    param_name: str,
    *,
    target_layers: str,
    layer_start: int | None,
    layer_end: int | None,
) -> bool:
    if target_layers == "all":
        return True
    assert layer_start is not None
    assert layer_end is not None
    return is_target_param(param_name, layer_start, layer_end)


def parse_experts(raw: str) -> list[str]:
    experts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(experts) < 2:
        raise ValueError("Expected at least two expert checkpoints")
    return experts


@torch.no_grad()
def create_interpolated_soup_checkpoint(
    *,
    base_model_path: str,
    expert_paths: list[str],
    out_dir: Path,
    alpha: float,
    target_layers: str,
    first_layers: int,
    last_layers: int,
    device: str,
) -> dict:
    if out_dir.exists() and (out_dir / "config.json").exists():
        return {"alpha": alpha, "checkpoint": str(out_dir), "status": "exists"}

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[alpha={alpha}] loading base model: {base_model_path}", flush=True)
    base = load_text_model(base_model_path, device)
    base_state = dict(base.named_parameters())

    print(f"[alpha={alpha}] initializing soup from: {expert_paths[0]}", flush=True)
    soup = load_text_model(expert_paths[0], device)
    soup_state = dict(soup.named_parameters())

    num_layers = get_num_layers(soup)
    layer_start, layer_end = selection_label(
        target_layers,
        first_layers=first_layers,
        last_layers=last_layers,
        num_layers=num_layers,
    )

    for param in soup.parameters():
        param.data.zero_()

    n_target_params = 0
    n_target_tensors = 0
    target_drift_sq = 0.0
    target_base_sq = 0.0
    weight = 1.0 / len(expert_paths)

    for expert_idx, expert_path in enumerate(expert_paths, start=1):
        print(
            f"[alpha={alpha}] loading expert {expert_idx}/{len(expert_paths)}: {expert_path}",
            flush=True,
        )
        expert = load_text_model(expert_path, device)
        expert_state = dict(expert.named_parameters())

        for name, expert_param in expert_state.items():
            soup_param = soup_state[name]
            if is_selected_param(
                name,
                target_layers=target_layers,
                layer_start=layer_start,
                layer_end=layer_end,
            ):
                base_param = base_state[name]
                diff = expert_param.data.float() - base_param.data.float()
                target_drift_sq += float(torch.sum(diff * diff).cpu())
                if expert_idx == 1:
                    target_base_sq += float(torch.sum(base_param.data.float() ** 2).cpu())
                    n_target_params += expert_param.numel()
                    n_target_tensors += 1
                interpolated = base_param.data + alpha * (expert_param.data - base_param.data)
                soup_param.data.add_(interpolated, alpha=weight)
            else:
                soup_param.data.add_(expert_param.data, alpha=weight)

        del expert, expert_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_target_drift_to_base = (
        (target_drift_sq / len(expert_paths)) ** 0.5 / (target_base_sq**0.5)
        if target_base_sq > 0
        else None
    )

    print(
        f"[alpha={alpha}] soup averaged {len(expert_paths)} experts; "
        f"interpolated {n_target_tensors} target tensors / {n_target_params:,} params "
        f"in {target_layers} selection "
        f"{'all parameters' if target_layers == 'all' else f'layers {layer_start}--{layer_end - 1}'}; "
        f"mean_original_target_drift_to_base={mean_target_drift_to_base}",
        flush=True,
    )

    soup.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(expert_paths[0], trust_remote_code=True)
    tokenizer.save_pretrained(out_dir)

    manifest = {
        "base_model": base_model_path,
        "expert_paths": expert_paths,
        "alpha": alpha,
        "target_layers": target_layers,
        "target_layer_start": layer_start,
        "target_layer_end": layer_end,
    }
    with (out_dir / "soup_alpha_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    del soup, base, soup_state, base_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "alpha": alpha,
        "checkpoint": str(out_dir),
        "status": "created",
        "target_layers": target_layers,
        "target_layer_start": layer_start,
        "target_layer_end": layer_end,
        "num_experts": len(expert_paths),
        "expert_paths": expert_paths,
        "interpolated_tensors": n_target_tensors,
        "interpolated_params": n_target_params,
        "mean_original_target_drift_to_base": mean_target_drift_to_base,
        "alpha_scaled_mean_target_drift_to_base": None
        if mean_target_drift_to_base is None
        else alpha * mean_target_drift_to_base,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/gemma-3-4b-pt")
    parser.add_argument(
        "--expert_models",
        required=True,
        help="Comma-separated expert checkpoint paths to uniformly average.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/drift_alpha_sweep/expert_soup_middle",
    )
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument(
        "--target_layers",
        choices=["first", "middle", "last", "all"],
        default="middle",
    )
    parser.add_argument("--first_layers", type=int, default=9)
    parser.add_argument("--last_layers", type=int, default=6)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--batch_size", default="8")
    parser.add_argument("--limit", default="")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--delete_after_eval", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
    expert_paths = parse_experts(args.expert_models)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.jsonl"

    for alpha in iter_alphas(parse_alphas(args.alphas)):
        ckpt_dir = out_root / f"alpha_{alpha_name(alpha)}"
        record = create_interpolated_soup_checkpoint(
            base_model_path=args.base_model,
            expert_paths=expert_paths,
            out_dir=ckpt_dir,
            alpha=alpha,
            target_layers=args.target_layers,
            first_layers=args.first_layers,
            last_layers=args.last_layers,
            device=args.device,
        )

        if not args.no_eval:
            record["eval"] = run_lm_eval(
                checkpoint=ckpt_dir,
                tasks=args.tasks,
                output_dir=out_root / f"eval_alpha_{alpha_name(alpha)}",
                cache_dir=out_root / f"cache_alpha_{alpha_name(alpha)}",
                batch_size=args.batch_size,
                limit=args.limit or None,
            )

        with summary_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if args.delete_after_eval and not args.no_eval:
            shutil.rmtree(ckpt_dir)

    print(f"[done] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
