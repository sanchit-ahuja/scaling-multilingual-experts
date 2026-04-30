"""
Measure weight-space divergence between expert models and the base model.

Computes per-layer metrics:
1. Expert-to-base drift: how far each expert moved from base weights
2. Inter-expert divergence: how far experts are from each other (via task vectors)
3. Aggregated summary table for paper inclusion

Outputs JSON results + optional matplotlib plots.

Usage:
    # Basic: measure divergence for a set of experts
    python analyze_divergence.py \
        --base_path google/gemma-3-4b-pt \
        --expert_paths path/to/expert1 path/to/expert2 ... \
        --output_dir results/divergence

    # With labels and plotting
    python analyze_divergence.py \
        --base_path google/gemma-3-4b-pt \
        --expert_paths path/to/slavic path/to/germanic path/to/indic \
        --expert_labels slavic germanic indic \
        --output_dir results/divergence \
        --plot

    # Compare strategies (run once per strategy, results go in separate dirs)
    python analyze_divergence.py \
        --base_path google/gemma-3-4b-pt \
        --expert_paths checkpoints/dense/slavic checkpoints/dense/germanic \
        --expert_labels slavic germanic \
        --output_dir results/divergence/dense \
        --strategy_label "Dense (no reg)" \
        --plot
"""

import argparse
import json
import os
import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Weight loading (lightweight — no model instantiation, just state dicts)
# ---------------------------------------------------------------------------

def load_state_dict_from_path(path: str) -> Dict[str, torch.Tensor]:
    """Load a state dict from a checkpoint directory without instantiating a model."""
    import glob

    safetensors_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if safetensors_files:
        sd = {}
        for sf in safetensors_files:
            shard = load_file(sf, device="cpu")
            sd.update(shard)
        return sd

    pytorch_files = sorted(glob.glob(os.path.join(path, "pytorch_model*.bin")))
    if pytorch_files:
        sd = {}
        for pf in pytorch_files:
            shard = torch.load(pf, map_location="cpu", weights_only=True)
            sd.update(shard)
        return sd

    raise FileNotFoundError(f"No safetensors or pytorch_model files found in {path}")


def _remap_multimodal_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip 'language_model.' prefix from multimodal Gemma checkpoints."""
    needs_remap = any("language_model" in k for k in list(sd.keys())[:20])
    if not needs_remap:
        return sd
    remapped = {}
    for key, value in sd.items():
        if "vision_tower" in key or "multi_modal_projector" in key:
            continue
        new_key = key
        if ".language_model." in key:
            new_key = key.replace(".language_model.", ".", 1)
        elif key.startswith("language_model."):
            new_key = key.replace("language_model.", "", 1)
        remapped[new_key] = value
    return remapped


# ---------------------------------------------------------------------------
# Layer grouping
# ---------------------------------------------------------------------------

def extract_layer_idx(param_name: str) -> Optional[int]:
    match = re.search(r"layers\.(\d+)\.", param_name)
    return int(match.group(1)) if match else None


def is_attention_param(name: str) -> bool:
    return any(p in name for p in ["self_attn", "q_proj", "k_proj", "v_proj", "o_proj"])


def is_mlp_param(name: str) -> bool:
    return any(p in name for p in ["mlp", "gate_proj", "up_proj", "down_proj"])


def group_params_by_layer(
    sd: Dict[str, torch.Tensor],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Group parameters by transformer layer index. Non-layer params are skipped."""
    layers = defaultdict(dict)
    for name, tensor in sd.items():
        idx = extract_layer_idx(name)
        if idx is not None and torch.is_floating_point(tensor):
            layers[idx][name] = tensor
    return dict(layers)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine_similarity_flat(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    """Cosine similarity between two sets of parameters (flattened and concatenated)."""
    shared_keys = sorted(set(a.keys()) & set(b.keys()))
    if not shared_keys:
        return float("nan")
    vec_a = torch.cat([a[k].float().flatten() for k in shared_keys])
    vec_b = torch.cat([b[k].float().flatten() for k in shared_keys])
    return F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()


def l2_distance_flat(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    """Normalized L2 distance between two parameter sets."""
    shared_keys = sorted(set(a.keys()) & set(b.keys()))
    if not shared_keys:
        return float("nan")
    vec_a = torch.cat([a[k].float().flatten() for k in shared_keys])
    vec_b = torch.cat([b[k].float().flatten() for k in shared_keys])
    return (vec_a - vec_b).norm(2).item() / (vec_a.numel() ** 0.5)


def task_vector_cosine(
    expert_a: Dict[str, torch.Tensor],
    expert_b: Dict[str, torch.Tensor],
    base: Dict[str, torch.Tensor],
) -> float:
    """Cosine similarity between task vectors (expert - base) of two experts."""
    shared_keys = sorted(set(expert_a.keys()) & set(expert_b.keys()) & set(base.keys()))
    if not shared_keys:
        return float("nan")
    delta_a = torch.cat([(expert_a[k].float() - base[k].float()).flatten() for k in shared_keys])
    delta_b = torch.cat([(expert_b[k].float() - base[k].float()).flatten() for k in shared_keys])
    # Both task vectors are zero → experts are identical to base at this layer → similarity = 1
    if delta_a.norm() == 0 and delta_b.norm() == 0:
        return 1.0
    return F.cosine_similarity(delta_a.unsqueeze(0), delta_b.unsqueeze(0)).item()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(
    base_sd: Dict[str, torch.Tensor],
    expert_sds: List[Dict[str, torch.Tensor]],
    expert_labels: List[str],
) -> dict:
    """Run full divergence analysis. Returns a results dict."""

    base_layers = group_params_by_layer(base_sd)
    expert_layers_list = [group_params_by_layer(sd) for sd in expert_sds]
    all_layer_ids = sorted(base_layers.keys())
    num_layers = max(all_layer_ids) + 1 if all_layer_ids else 0

    print(f"Detected {num_layers} layers, {len(expert_sds)} experts")

    # --- 1. Expert-to-base drift per layer ---
    drift_results = {}  # expert_label -> {layer_id -> {cos_sim, l2_dist}}
    for ei, label in enumerate(expert_labels):
        drift_results[label] = {}
        for lid in all_layer_ids:
            if lid not in expert_layers_list[ei]:
                continue
            cos = cosine_similarity_flat(expert_layers_list[ei][lid], base_layers[lid])
            l2 = l2_distance_flat(expert_layers_list[ei][lid], base_layers[lid])
            drift_results[label][lid] = {"cos_sim_to_base": cos, "l2_dist_to_base": l2}
        print(f"  [{label}] avg cos_sim_to_base = "
              f"{sum(d['cos_sim_to_base'] for d in drift_results[label].values()) / len(drift_results[label]):.6f}")

    # --- 2. Pairwise inter-expert divergence per layer ---
    pair_results = {}  # "expert_a vs expert_b" -> {layer_id -> {cos_sim, task_vec_cos, l2_dist}}
    for (i, label_a), (j, label_b) in combinations(enumerate(expert_labels), 2):
        pair_key = f"{label_a} vs {label_b}"
        pair_results[pair_key] = {}
        for lid in all_layer_ids:
            if lid not in expert_layers_list[i] or lid not in expert_layers_list[j]:
                continue
            cos = cosine_similarity_flat(expert_layers_list[i][lid], expert_layers_list[j][lid])
            l2 = l2_distance_flat(expert_layers_list[i][lid], expert_layers_list[j][lid])
            tv_cos = task_vector_cosine(
                expert_layers_list[i][lid], expert_layers_list[j][lid], base_layers[lid]
            )
            pair_results[pair_key][lid] = {
                "cos_sim": cos,
                "l2_dist": l2,
                "task_vector_cos": tv_cos,
            }

    # --- 3. Aggregate: mean inter-expert divergence per layer (across all pairs) ---
    agg_per_layer = {}
    for lid in all_layer_ids:
        cos_vals, tv_vals, l2_vals = [], [], []
        for pair_data in pair_results.values():
            if lid in pair_data:
                cos_vals.append(pair_data[lid]["cos_sim"])
                tv_vals.append(pair_data[lid]["task_vector_cos"])
                l2_vals.append(pair_data[lid]["l2_dist"])
        if cos_vals:
            agg_per_layer[lid] = {
                "mean_pairwise_cos_sim": sum(cos_vals) / len(cos_vals),
                "mean_task_vector_cos": sum(tv_vals) / len(tv_vals),
                "mean_pairwise_l2_dist": sum(l2_vals) / len(l2_vals),
            }

    # --- 4. Global summary ---
    all_drift_cos = [
        d["cos_sim_to_base"]
        for expert_data in drift_results.values()
        for d in expert_data.values()
    ]
    all_pair_tv = [
        d["task_vector_cos"]
        for pair_data in pair_results.values()
        for d in pair_data.values()
    ]
    summary = {
        "num_experts": len(expert_sds),
        "num_layers": num_layers,
        "expert_labels": expert_labels,
        "mean_drift_cos_sim": sum(all_drift_cos) / len(all_drift_cos) if all_drift_cos else None,
        "mean_task_vector_cos": sum(all_pair_tv) / len(all_pair_tv) if all_pair_tv else None,
    }

    return {
        "summary": summary,
        "drift_per_expert": _convert_keys_to_str(drift_results),
        "pairwise_divergence": _convert_keys_to_str(pair_results),
        "aggregate_per_layer": _convert_keys_to_str(agg_per_layer),
    }


def _convert_keys_to_str(obj):
    """Recursively convert dict keys to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _convert_keys_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_keys_to_str(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, output_dir: str, strategy_label: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = results["aggregate_per_layer"]
    layers = sorted(int(k) for k in agg.keys())

    # --- Plot 1: Mean pairwise task-vector cosine similarity per layer ---
    fig, ax = plt.subplots(figsize=(10, 4))
    tv_cos = [agg[str(l)]["mean_task_vector_cos"] for l in layers]
    ax.plot(layers, tv_cos, marker=".", linewidth=1.5, label="Task-vector cosine sim")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    title = "Inter-expert task-vector alignment per layer"
    if strategy_label:
        title += f" ({strategy_label})"
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "task_vector_cosine_per_layer.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "task_vector_cosine_per_layer.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: Expert-to-base drift (cosine sim) per layer, one line per expert ---
    drift = results["drift_per_expert"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for expert_label, layer_data in drift.items():
        elayers = sorted(int(k) for k in layer_data.keys())
        vals = [layer_data[str(l)]["cos_sim_to_base"] for l in elayers]
        ax.plot(elayers, vals, marker=".", linewidth=1.2, alpha=0.7, label=expert_label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity to base")
    title = "Expert-to-base weight similarity per layer"
    if strategy_label:
        title += f" ({strategy_label})"
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "drift_per_layer.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "drift_per_layer.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: L2 distance from base per layer ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for expert_label, layer_data in drift.items():
        elayers = sorted(int(k) for k in layer_data.keys())
        vals = [layer_data[str(l)]["l2_dist_to_base"] for l in elayers]
        ax.plot(elayers, vals, marker=".", linewidth=1.2, alpha=0.7, label=expert_label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized L2 distance to base")
    title = "Expert-to-base weight drift per layer"
    if strategy_label:
        title += f" ({strategy_label})"
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "l2_drift_per_layer.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "l2_drift_per_layer.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# Granular per-expert analysis
# ---------------------------------------------------------------------------

def plot_per_expert_analysis(results_path: str, output_dir: str):
    """
    Granular per-expert divergence analysis from a saved divergence_results.json.

    Produces:
      1. per_pair_task_vector_cos.{pdf,png}  — one curve per expert pair
      2. per_pair_l2_drift.{pdf,png}         — per-pair L2 drift from base per layer
      3. heatmap_task_vector_cos.{pdf,png}   — N×N pairwise heatmap (mean across layers)
      4. heatmap_by_region.{pdf,png}         — N×N heatmaps split by early/mid/late layers
      5. per_expert_summary.json             — mean divergence from all others per expert
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(results_path) as f:
        results = json.load(f)

    experts = results["summary"]["expert_labels"]
    n = len(experts)
    pairwise = results["pairwise_divergence"]
    drift = results["drift_per_expert"]
    num_layers = results["summary"]["num_layers"]
    all_layers = list(range(num_layers))

    os.makedirs(output_dir, exist_ok=True)

    if n < 2:
        print("Only one expert — skipping pairwise plots.")
        return

    # ── 1. Per-pair task-vector cosine per layer ────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for pair_key, layer_data in pairwise.items():
        layers = sorted(int(k) for k in layer_data.keys())
        tv = [layer_data[str(l)]["task_vector_cos"] for l in layers]
        ax.plot(layers, tv, marker=".", linewidth=1.2, alpha=0.8, label=pair_key)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Task-vector cosine similarity")
    ax.set_title("Per-pair inter-expert task-vector alignment per layer")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_pair_task_vector_cos.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "per_pair_task_vector_cos.png"), dpi=150)
    plt.close(fig)

    # ── 1b. Per-expert mean TV cosine vs all others per layer (5 lines) ───────
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors
    for ei, exp in enumerate(experts):
        per_layer_vals = {}
        for j, other in enumerate(experts):
            if ei == j:
                continue
            pk = f"{exp} vs {other}" if f"{exp} vs {other}" in pairwise else f"{other} vs {exp}"
            if pk not in pairwise:
                continue
            for lid_str, ld in pairwise[pk].items():
                per_layer_vals.setdefault(int(lid_str), []).append(ld["task_vector_cos"])
        layers = sorted(per_layer_vals.keys())
        means = [sum(per_layer_vals[l]) / len(per_layer_vals[l]) for l in layers]
        ax.plot(layers, means, marker=".", linewidth=1.5, label=exp, color=colors[ei])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean task-vector cosine vs all other experts")
    ax.set_title("Per-expert mean inter-expert task-vector alignment per layer")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_expert_mean_tv_cos.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "per_expert_mean_tv_cos.png"), dpi=150)
    plt.close(fig)

    # ── 2. Per-expert L2 drift from base per layer ──────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for exp, layer_data in drift.items():
        layers = sorted(int(k) for k in layer_data.keys())
        l2 = [layer_data[str(l)]["l2_dist_to_base"] for l in layers]
        ax.plot(layers, l2, marker=".", linewidth=1.2, alpha=0.8, label=exp)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized L2 distance to base (log scale)")
    ax.set_title("Per-expert weight drift from base per layer")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_expert_l2_drift.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "per_expert_l2_drift.png"), dpi=150)
    plt.close(fig)

    # ── 3. Pairwise heatmap (mean task-vector cosine across all layers) ──────
    def _build_matrix(metric: str, layer_subset=None) -> np.ndarray:
        mat = np.full((n, n), np.nan)
        for i in range(n):
            mat[i, i] = 1.0
            for j in range(i + 1, n):
                pair_key = f"{experts[i]} vs {experts[j]}"
                if pair_key not in pairwise:
                    pair_key = f"{experts[j]} vs {experts[i]}"
                if pair_key not in pairwise:
                    continue
                ld = pairwise[pair_key]
                layers = [l for l in (layer_subset if layer_subset else all_layers) if str(l) in ld]
                if not layers:
                    continue
                val = sum(ld[str(l)][metric] for l in layers) / len(layers)
                mat[i, j] = mat[j, i] = val
        return mat

    def _plot_heatmap(mat: np.ndarray, title: str, path_stem: str, vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(max(5, n + 1), max(4, n)))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(experts, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(n)); ax.set_yticklabels(experts, fontsize=9)
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(path_stem + ".pdf", dpi=150)
        fig.savefig(path_stem + ".png", dpi=150)
        plt.close(fig)

    mat_all = _build_matrix("task_vector_cos")
    _plot_heatmap(
        mat_all,
        "Pairwise task-vector cosine (mean all layers)",
        os.path.join(output_dir, "heatmap_task_vector_cos"),
        vmin=-0.1, vmax=1.0,
    )

    # ── 4. Heatmaps by layer region (early / mid / late thirds) ─────────────
    thirds = num_layers // 3
    regions = {
        "early": list(range(0, thirds)),
        "mid":   list(range(thirds, 2 * thirds)),
        "late":  list(range(2 * thirds, num_layers)),
    }
    fig, axes = plt.subplots(1, 3, figsize=(max(14, 3 * (n + 2)), max(4, n + 1)))
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=-0.1, vmax=1.0)
    cmap = cm.RdYlGn
    for ax, (region_name, layer_ids) in zip(axes, regions.items()):
        mat = _build_matrix("task_vector_cos", layer_ids)
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(experts, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(experts, fontsize=8)
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{region_name.capitalize()} layers ({layer_ids[0]}–{layer_ids[-1]})")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Pairwise task-vector cosine by layer region", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heatmap_by_region.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "heatmap_by_region.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 5. Per-expert summary JSON ───────────────────────────────────────────
    summary_rows = []
    for i, exp in enumerate(experts):
        # mean task-vector cosine vs all other experts (all layers)
        tv_vals = []
        for j, other in enumerate(experts):
            if i == j:
                continue
            pk = f"{exp} vs {other}" if f"{exp} vs {other}" in pairwise else f"{other} vs {exp}"
            if pk not in pairwise:
                continue
            for ld in pairwise[pk].values():
                tv_vals.append(ld["task_vector_cos"])
        # mean drift from base (all layers)
        l2_vals = [v["l2_dist_to_base"] for v in drift.get(exp, {}).values()]
        cos_vals = [v["cos_sim_to_base"] for v in drift.get(exp, {}).values()]
        row = {
            "expert": exp,
            "mean_tv_cos_vs_others": round(sum(tv_vals) / len(tv_vals), 6) if tv_vals else None,
            "mean_l2_drift_from_base": round(sum(l2_vals) / len(l2_vals), 6) if l2_vals else None,
            "mean_cos_sim_to_base": round(sum(cos_vals) / len(cos_vals), 6) if cos_vals else None,
        }
        # per-region task-vector cosine vs others
        for region_name, layer_ids in regions.items():
            region_tv = []
            for j, other in enumerate(experts):
                if i == j:
                    continue
                pk = f"{exp} vs {other}" if f"{exp} vs {other}" in pairwise else f"{other} vs {exp}"
                if pk not in pairwise:
                    continue
                for lid in layer_ids:
                    if str(lid) in pairwise[pk]:
                        region_tv.append(pairwise[pk][str(lid)]["task_vector_cos"])
            row[f"mean_tv_cos_{region_name}"] = round(sum(region_tv) / len(region_tv), 6) if region_tv else None
        summary_rows.append(row)

    summary_path = os.path.join(output_dir, "per_expert_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    # Print table
    print(f"\n{'Expert':<15} {'TV-cos (all)':>12} {'TV-cos early':>13} {'TV-cos mid':>11} {'TV-cos late':>12} {'L2 drift':>10}")
    print("-" * 77)
    for r in summary_rows:
        print(
            f"{r['expert']:<15} "
            f"{r['mean_tv_cos_vs_others']:>12.4f} "
            f"{r['mean_tv_cos_early']:>13.4f} "
            f"{r['mean_tv_cos_mid']:>11.4f} "
            f"{r['mean_tv_cos_late']:>12.4f} "
            f"{r['mean_l2_drift_from_base']:>10.6f}"
        )

    print(f"\nAll outputs saved to {output_dir}")


# ---------------------------------------------------------------------------
# Multi-strategy comparison plot
# ---------------------------------------------------------------------------

def plot_strategy_comparison(result_dirs: List[str], labels: List[str], output_dir: str):
    """
    Overlay task-vector cosine curves from multiple strategy runs.

    Usage:
        python analyze_divergence.py --compare \
            --result_dirs results/divergence/dense results/divergence/l2sp results/divergence/freeze \
            --compare_labels "Dense" "L2-SP" "Freeze" \
            --output_dir results/divergence/comparison
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np
    os.makedirs(output_dir, exist_ok=True)

    # Collect per-strategy data first, then plot separately
    strategy_data = []
    for rdir, label in zip(result_dirs, labels):
        results_path = os.path.join(rdir, "divergence_results.json")
        with open(results_path) as f:
            results = json.load(f)
        agg = results["aggregate_per_layer"]
        layers = sorted(int(k) for k in agg.keys())
        pairwise = results["pairwise_divergence"]
        drift = results["drift_per_expert"]

        tv_mean, tv_lo, tv_hi = [], [], []
        for lid in layers:
            vals = [pairwise[pk][str(lid)]["task_vector_cos"]
                    for pk in pairwise if str(lid) in pairwise[pk]]
            if vals:
                m, s = np.mean(vals), np.std(vals)
                tv_mean.append(m); tv_lo.append(m - s); tv_hi.append(m + s)
            else:
                tv_mean.append(float("nan")); tv_lo.append(float("nan")); tv_hi.append(float("nan"))

        l2_mean, l2_lo, l2_hi = [], [], []
        for lid in layers:
            vals = [drift[exp][str(lid)]["l2_dist_to_base"]
                    for exp in drift if str(lid) in drift[exp]]
            if vals:
                m, s = np.mean(vals), np.std(vals)
                l2_mean.append(m); l2_lo.append(max(m - s, 1e-10)); l2_hi.append(m + s)
            else:
                l2_mean.append(float("nan")); l2_lo.append(float("nan")); l2_hi.append(float("nan"))

        strategy_data.append((label, layers, tv_mean, tv_lo, tv_hi, l2_mean, l2_lo, l2_hi))

    # ── Plot 1: Inter-expert task-vector cosine ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, layers, tv_mean, tv_lo, tv_hi, *_ in strategy_data:
        line, = ax.plot(layers, tv_mean, marker=".", linewidth=1.5, label=label)
        ax.fill_between(layers, tv_lo, tv_hi, alpha=0.15, color=line.get_color())
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Inter-expert task-vector alignment")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"comparison_task_vector_cos.{ext}"), dpi=150)
    plt.close(fig)

    # ── Plot 2: Mean expert-to-base drift (log scale) ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, layers, _, __, ___, l2_mean, l2_lo, l2_hi in strategy_data:
        line, = ax.plot(layers, l2_mean, marker=".", linewidth=1.5, label=label)
        ax.fill_between(layers, l2_lo, l2_hi, alpha=0.15, color=line.get_color())
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized L2 distance (log scale)")
    ax.set_title("Mean expert-to-base drift")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"comparison_l2_drift.{ext}"), dpi=150)
    plt.close(fig)

    print(f"Comparison plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure weight-space divergence between expert models."
    )
    sub = parser.add_subparsers(dest="command", help="Sub-command")

    # --- analyze sub-command ---
    p_analyze = sub.add_parser("analyze", help="Analyze divergence for one set of experts")
    p_analyze.add_argument("--base_path", type=str, required=True, help="Base model path")
    p_analyze.add_argument("--expert_paths", nargs="+", required=True, help="Expert checkpoint paths")
    p_analyze.add_argument("--expert_labels", nargs="+", default=None, help="Labels for experts")
    p_analyze.add_argument("--output_dir", type=str, required=True, help="Output directory")
    p_analyze.add_argument("--strategy_label", type=str, default="", help="Label for this strategy")
    p_analyze.add_argument("--plot", action="store_true", help="Generate plots")

    # --- compare sub-command ---
    p_compare = sub.add_parser("compare", help="Compare divergence across strategies")
    p_compare.add_argument("--result_dirs", nargs="+", required=True, help="Result directories from analyze runs")
    p_compare.add_argument("--compare_labels", nargs="+", required=True, help="Labels for each strategy")
    p_compare.add_argument("--output_dir", type=str, required=True, help="Output directory for comparison")

    # --- per_expert sub-command ---
    p_per = sub.add_parser("per_expert", help="Granular per-expert divergence analysis from saved results")
    p_per.add_argument("--results_path", type=str, required=True,
                       help="Path to divergence_results.json (from a prior analyze run)")
    p_per.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for per-expert plots and summary")

    args = parser.parse_args()

    if args.command == "per_expert":
        plot_per_expert_analysis(args.results_path, args.output_dir)
        return

    if args.command == "compare":
        plot_strategy_comparison(args.result_dirs, args.compare_labels, args.output_dir)
        return

    if args.command is None:
        parser.print_help()
        return

    # --- Analyze ---
    labels = args.expert_labels or [f"expert_{i}" for i in range(len(args.expert_paths))]
    assert len(labels) == len(args.expert_paths), "Number of labels must match number of experts"

    print(f"Loading base model from {args.base_path}...")
    base_sd = load_state_dict_from_path(args.base_path)
    base_sd = _remap_multimodal_keys(base_sd)

    expert_sds = []
    for path in args.expert_paths:
        print(f"Loading expert from {path}...")
        sd = load_state_dict_from_path(path)
        sd = _remap_multimodal_keys(sd)
        expert_sds.append(sd)

    results = analyze(base_sd, expert_sds, labels)
    results["summary"]["strategy_label"] = args.strategy_label

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "divergence_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    if args.plot:
        plot_results(results, args.output_dir, args.strategy_label)


if __name__ == "__main__":
    main()
