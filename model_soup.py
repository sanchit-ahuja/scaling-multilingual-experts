import argparse
import json
import os
from datetime import datetime

import torch
from transformers import AutoTokenizer

from utils import load_model


def _read_expert_list(expert_paths, experts_file):
    paths = []
    if expert_paths:
        paths.extend(expert_paths)

    if experts_file:
        with open(experts_file, "r") as handle:
            for line in handle:
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                paths.append(value)

    # De-duplicate while preserving order
    seen = set()
    ordered = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _accumulate_state_dict(sum_sd, current_sd, index):
    for key, value in current_sd.items():
        if torch.is_floating_point(value):
            if index == 0:
                sum_sd[key] = value.float().clone()
            else:
                sum_sd[key] += value.float()
        else:
            if index == 0:
                sum_sd[key] = value.clone()


def _average_state_dict(sum_sd, num_experts):
    avg_sd = {}
    for key, value in sum_sd.items():
        if torch.is_floating_point(value):
            avg_sd[key] = value / float(num_experts)
        else:
            avg_sd[key] = value
    return avg_sd


def create_expert_soup(
    expert_paths,
    output_dir,
    alpha=1.0,
    anchor_path=None,
    tokenizer_path=None,
    dtype=torch.bfloat16,
):
    if not expert_paths:
        raise ValueError("No expert checkpoints provided.")

    os.makedirs(output_dir, exist_ok=True)

    tokenizer_source = tokenizer_path or expert_paths[0]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    print(f"Loading {len(expert_paths)} experts...")
    sum_sd = {}
    for index, path in enumerate(expert_paths):
        model = load_model(path, dtype=dtype)
        state_dict = model.state_dict()
        _accumulate_state_dict(sum_sd, state_dict, index)
        del model

    expert_mean = _average_state_dict(sum_sd, len(expert_paths))

    if anchor_path:
        print(f"Applying anchor blend with alpha={alpha}...")
        anchor_model = load_model(anchor_path, dtype=dtype)
        anchor_sd = anchor_model.state_dict()
        blended_sd = {}
        for key in expert_mean.keys():
            if key not in anchor_sd:
                blended_sd[key] = expert_mean[key]
                continue
            if torch.is_floating_point(expert_mean[key]):
                blended_sd[key] = (1.0 - alpha) * anchor_sd[key].float() + alpha * expert_mean[key]
            else:
                blended_sd[key] = anchor_sd[key]
        state_dict_to_save = blended_sd
        model_for_save = anchor_model
    else:
        if alpha != 1.0:
            print("Warning: alpha is ignored without an anchor model.")
        state_dict_to_save = expert_mean
        model_for_save = load_model(expert_paths[0], dtype=dtype)

    missing, unexpected = model_for_save.load_state_dict(state_dict_to_save, strict=False)
    if missing or unexpected:
        print(f"Warning: missing={len(missing)}, unexpected={len(unexpected)}")

    model_for_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "experts": expert_paths,
        "num_experts": len(expert_paths),
        "alpha": alpha,
        "anchor_path": anchor_path,
        "tokenizer_path": tokenizer_source,
        "dtype": str(dtype),
    }
    with open(os.path.join(output_dir, "soup_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"Saved soup to {output_dir}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a fixed expert-only model soup.")
    parser.add_argument("--experts", nargs="*", default=[], help="Expert checkpoint paths.")
    parser.add_argument("--experts_file", type=str, default=None, help="File with expert paths.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for soup.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Blend alpha (anchor only).")
    parser.add_argument("--anchor_path", type=str, default=None, help="Optional anchor checkpoint.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer source.")
    args = parser.parse_args()

    experts = _read_expert_list(args.experts, args.experts_file)
    create_expert_soup(
        expert_paths=experts,
        output_dir=args.output_dir,
        alpha=args.alpha,
        anchor_path=args.anchor_path,
        tokenizer_path=args.tokenizer_path,
    )