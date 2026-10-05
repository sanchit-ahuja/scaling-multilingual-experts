#!/usr/bin/env python3
"""Create a persistent Expert hybrid for the confirmed FLORES layer band."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regularization import get_num_layers  # noqa: E402
from scripts.middle_layer_alpha_sweep import is_target_param, load_text_model  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_model", required=True)
    parser.add_argument("--donor_model", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest_path = args.output_dir / "transfer_manifest.json"
    expected = {
        "expert_model": args.expert_model, "donor_model": args.donor_model,
        "layer_start": args.start, "layer_end": args.end, "beta": args.beta,
    }
    if (args.output_dir / "config.json").exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all(manifest.get(key) == value for key, value in expected.items()):
            print(f"[resume] using existing hybrid: {args.output_dir}", flush=True)
            return
        raise ValueError(f"Existing hybrid manifest does not match requested config: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expert = load_text_model(args.expert_model, args.device)
    donor = load_text_model(args.donor_model, args.device)
    n_layers = get_num_layers(expert)
    if not (0 <= args.start < args.end <= n_layers):
        raise ValueError(f"Invalid layer range [{args.start}, {args.end}) for {n_layers} layers")

    expert_state, donor_state = expert.state_dict(), donor.state_dict()
    originals = {name: tensor.detach().cpu().clone() for name, tensor in expert_state.items()}
    target_tensors = 0
    for name, tensor in expert_state.items():
        if is_target_param(name, args.start, args.end):
            donor_tensor = donor_state.get(name)
            if donor_tensor is None or donor_tensor.shape != tensor.shape:
                raise ValueError(f"Incompatible donor tensor: {name}")
            # Preserve byte-exact donor provenance for beta=1. Arithmetic
            # interpolation in bfloat16 need not equal the donor bit-for-bit.
            if args.beta == 1.0:
                tensor.copy_(donor_tensor.to(tensor.device))
            else:
                tensor.copy_(tensor + args.beta * (donor_tensor.to(tensor.device) - tensor))
            target_tensors += 1
        elif not torch.equal(tensor.detach().cpu(), originals[name]):
            raise AssertionError(f"Non-target tensor changed: {name}")
    if args.beta == 1.0:
        for name, tensor in expert_state.items():
            expected_tensor = donor_state[name] if is_target_param(name, args.start, args.end) else originals[name]
            if not torch.equal(tensor.detach().cpu(), expected_tensor.detach().cpu()):
                raise AssertionError(f"Provenance check failed: {name}")

    expert.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(args.expert_model, trust_remote_code=True).save_pretrained(args.output_dir)
    expected["target_tensors"] = target_tensors
    manifest_path.write_text(json.dumps(expected, indent=2))
    print(f"[done] {args.output_dir} ({target_tensors} target tensors)", flush=True)
    del expert, donor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
