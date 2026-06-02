#!/usr/bin/env python3
"""Create and optionally evaluate layer-group drift interpolation checkpoints.

The intervention keeps the CPT checkpoint's other layers fixed and replaces the
target layer group with an interpolation between the base and CPT weights:

    theta_target(alpha) = theta_base + alpha * (theta_cpt - theta_base)

alpha=0 reverts the target layer group; alpha=1 is the original CPT checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regularization import get_num_layers  # noqa: E402
from utils import load_model  # noqa: E402


def parse_alphas(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def alpha_name(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def layer_index(param_name: str) -> int | None:
    match = re.search(r"layers\.(\d+)\.", param_name)
    return int(match.group(1)) if match else None


def target_layer_range(
    target_layers: str,
    *,
    first_layers: int,
    last_start: int,
    num_layers: int,
) -> tuple[int, int]:
    if target_layers == "first":
        return 0, first_layers
    if target_layers == "middle":
        return first_layers, last_start
    if target_layers == "last":
        return last_start, num_layers
    raise ValueError(f"Unsupported target layer group: {target_layers}")


def is_target_param(param_name: str, layer_start: int, layer_end: int) -> bool:
    idx = layer_index(param_name)
    return idx is not None and layer_start <= idx < layer_end


def load_text_model(path: str, device: str):
    return load_model(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)


@torch.no_grad()
def create_alpha_checkpoint(
    *,
    base_model_path: str,
    cpt_model_path: str,
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
    print(f"[alpha={alpha}] loading CPT model: {cpt_model_path}", flush=True)
    cpt = load_text_model(cpt_model_path, device)
    print(f"[alpha={alpha}] loading base model: {base_model_path}", flush=True)
    base = load_text_model(base_model_path, device)

    num_layers = get_num_layers(cpt)
    last_start = num_layers - last_layers
    layer_start, layer_end = target_layer_range(
        target_layers,
        first_layers=first_layers,
        last_start=last_start,
        num_layers=num_layers,
    )
    n_params = 0
    n_tensors = 0
    drift_sq = 0.0
    base_sq = 0.0

    base_state = dict(base.named_parameters())
    for name, cpt_param in cpt.named_parameters():
        if not is_target_param(name, layer_start, layer_end):
            continue
        base_param = base_state[name]
        diff = cpt_param.data.float() - base_param.data.float()
        drift_sq += float(torch.sum(diff * diff).cpu())
        base_sq += float(torch.sum(base_param.data.float() ** 2).cpu())
        cpt_param.data.copy_(base_param.data + alpha * (cpt_param.data - base_param.data))
        n_params += cpt_param.numel()
        n_tensors += 1

    normalized_target_drift = (drift_sq**0.5) / (base_sq**0.5) if base_sq > 0 else None
    print(
        f"[alpha={alpha}] interpolated {n_tensors} tensors / {n_params:,} params "
        f"in {target_layers} layers {layer_start}--{layer_end - 1}; "
        f"original_target_drift={normalized_target_drift}",
        flush=True,
    )

    cpt.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(cpt_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(out_dir)

    del cpt, base, base_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    record = {
        "alpha": alpha,
        "checkpoint": str(out_dir),
        "status": "created",
        "target_layers": target_layers,
        "target_layer_start": layer_start,
        "target_layer_end": layer_end,
        "interpolated_tensors": n_tensors,
        "interpolated_params": n_params,
        "original_normalized_target_drift": normalized_target_drift,
        "alpha_scaled_normalized_target_drift": None
        if normalized_target_drift is None
        else alpha * normalized_target_drift,
    }
    if target_layers == "middle":
        record["original_normalized_middle_drift"] = normalized_target_drift
        record["alpha_scaled_normalized_middle_drift"] = (
            None if normalized_target_drift is None else alpha * normalized_target_drift
        )
    return record


def find_accelerate() -> str:
    candidates = [
        shutil.which("accelerate"),
        str(ROOT / ".venv" / "bin" / "accelerate"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Could not find accelerate on PATH or in .venv/bin")


def run_lm_eval(
    *,
    checkpoint: Path,
    tasks: str,
    output_dir: Path,
    cache_dir: Path,
    batch_size: str,
    limit: str | None,
    include_path: str | None = None,
    log_samples: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_accelerate(),
        "launch",
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={checkpoint}",
        "--tasks",
        tasks,
    ]
    if include_path:
        cmd.extend(["--include_path", include_path])
    if log_samples:
        cmd.append("--log_samples")
    cmd.extend([
        "--batch_size",
        batch_size,
        "--output_path",
        str(output_dir),
        "--use_cache",
        str(cache_dir),
    ])
    if limit:
        cmd.extend(["--limit", limit])

    print("[eval] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)

    result_paths = sorted(output_dir.rglob("results*.json"))
    if not result_paths:
        raise FileNotFoundError(f"No results*.json found under {output_dir}")

    with result_paths[-1].open() as f:
        raw = json.load(f)

    metrics = {}
    for task_name, task_metrics in raw.get("results", {}).items():
        metrics[task_name] = {
            key: value
            for key, value in task_metrics.items()
            if key in {"acc,none", "acc_norm,none", "chrf,none", "rougeL,none"}
        }
    return {"result_path": str(result_paths[-1]), "metrics": metrics}


def iter_alphas(alphas: Iterable[float]):
    for alpha in alphas:
        if alpha < 0 or alpha > 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        yield alpha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/gemma-3-4b-pt")
    parser.add_argument(
        "--cpt_model",
        default="data/checkpoints/gemma_4b_dense_25b/final",
    )
    parser.add_argument(
        "--output_dir",
        default="data/drift_alpha_sweep/dense_middle",
    )
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument(
        "--target_layers",
        choices=["first", "middle", "last"],
        default="middle",
        help="Layer group to interpolate between base and CPT weights.",
    )
    parser.add_argument("--first_layers", type=int, default=9)
    parser.add_argument("--last_layers", type=int, default=6)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--tasks",
        default=(
            "belebele_eng_Latn,belebele_hin_Deva,"
            "global_piqa_completions_eng_latn,global_piqa_completions_hin_deva"
        ),
    )
    parser.add_argument("--batch_size", default="16")
    parser.add_argument(
        "--include_path",
        default="",
        help="Optional lm-eval --include_path for local/custom task YAMLs.",
    )
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument(
        "--limit",
        default="200",
        help="lm-eval --limit for quick smoke runs; use empty string for full eval.",
    )
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--delete_after_eval", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.jsonl"

    for alpha in iter_alphas(parse_alphas(args.alphas)):
        ckpt_dir = out_root / f"alpha_{alpha_name(alpha)}"
        record = create_alpha_checkpoint(
            base_model_path=args.base_model,
            cpt_model_path=args.cpt_model,
            out_dir=ckpt_dir,
            alpha=alpha,
            target_layers=args.target_layers,
            first_layers=args.first_layers,
            last_layers=args.last_layers,
            device=args.device,
        )

        if not args.no_eval:
            eval_record = run_lm_eval(
                checkpoint=ckpt_dir,
                tasks=args.tasks,
                output_dir=out_root / f"eval_alpha_{alpha_name(alpha)}",
                cache_dir=out_root / f"cache_alpha_{alpha_name(alpha)}",
                batch_size=args.batch_size,
                limit=args.limit or None,
                include_path=args.include_path or None,
                log_samples=args.log_samples,
            )
            record["eval"] = eval_record

        with summary_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if args.delete_after_eval and not args.no_eval:
            shutil.rmtree(ckpt_dir)

    print(f"[done] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
