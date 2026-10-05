#!/usr/bin/env python3
"""Resumably score one persistent checkpoint on a held-in FLORES family."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.flores_jobs import FAM_LANGS, HELDOUT_FAM_LANGS, tasks_for  # noqa: E402
from scripts.flores_layer_localization import score_samples, _family_task_map  # noqa: E402
from scripts.middle_layer_alpha_sweep import run_lm_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--family", required=True, choices=[name.lower() for name in FAM_LANGS])
    parser.add_argument("--split", choices=["heldin", "heldout"], default="heldin")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", default="32")
    parser.add_argument("--include_path", default="")
    args = parser.parse_args()
    family_name = args.family.title() if args.family != "austronesian" else "Austronesian"
    language_map = HELDOUT_FAM_LANGS if args.split == "heldout" else FAM_LANGS
    families = {family_name: language_map[family_name]}
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        print(f"[resume] using completed result: {result_path}", flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_lm_eval(
        checkpoint=args.checkpoint, tasks=",".join(tasks_for(families[family_name])),
        output_dir=args.output_dir / "raw", cache_dir=args.output_dir / "response_cache",
        batch_size=args.batch_size, limit=None, include_path=args.include_path or None,
        log_samples=True,
    )
    result = score_samples(args.output_dir / "raw", _family_task_map(families))
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(result_path)
    print(f"[done] {result_path}", flush=True)


if __name__ == "__main__":
    main()
