#!/usr/bin/env python3
"""Run a resumable lm-eval evaluation for one already-created checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.middle_layer_alpha_sweep import run_lm_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", default="32")
    parser.add_argument("--include_path", default="")
    args = parser.parse_args()
    summary = args.output_dir / "summary.json"
    if summary.exists():
        print(f"[resume] using completed evaluation: {summary}", flush=True)
        return
    result = run_lm_eval(
        checkpoint=args.checkpoint, tasks=args.tasks, output_dir=args.output_dir / "raw",
        cache_dir=args.output_dir / "response_cache", batch_size=args.batch_size, limit=None,
        include_path=args.include_path or None, log_samples=False,
    )
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2))
    print(f"[done] {summary}", flush=True)


if __name__ == "__main__":
    main()
