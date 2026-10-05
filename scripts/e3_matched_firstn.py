#!/usr/bin/env python3
"""Matched first_n baseline for E3: read the released (first_n) per-language accuracies
from the existing held-in campaign result JSONs, so the seed-robustness delta compares
random-sampler vs first_n over the *identical* task set (removes any language-set/grouping
mismatch with the E1 pivot, e.g. galician or the es/pt/fr regional variants).

Returns {(pretrained_path, benchmark): {base_task_name: acc}} where base_task_name has no
`_rs` suffix. When several first_n runs exist for a checkpoint, prefer the one covering the
most tasks, then the newest.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

CAMPAIGNS = {
    "belebele": [
        "/work/nvme/bfzp/training_lang_in_domain_results/belebele_in_domain",
        "/work/nvme/bfzp/in_domain_romance_archive",
    ],
    "global_piqa": [
        "/work/nvme/bfzp/training_lang_in_domain_results/piqa_in_domain",
        "/work/nvme/bfzp/in_domain_romance_archive/piqa",
        "/work/nvme/bfzp/in_domain_romance_archive",
    ],
}
PREFIX = {"belebele": "belebele_", "global_piqa": "global_piqa_completions_"}


def _pretrained(cfg) -> str:
    ma = cfg.get("config", {}).get("model_args", "")
    if isinstance(ma, dict):
        return ma.get("pretrained", "")
    m = re.search(r"pretrained=([^,]+)", ma or "")
    return m.group(1) if m else ""


def build_index(benchmark: str) -> dict:
    """{pretrained: {task: acc}} choosing the run with most tasks (newest tie-break)."""
    pref = PREFIX[benchmark]
    best: dict[str, tuple] = {}   # pretrained -> (ntasks, path_mtime, accs)
    seen = set()
    for root in CAMPAIGNS[benchmark]:
        for jf in glob.glob(f"{root}/**/results_*.json", recursive=True):
            if benchmark == "belebele" and "piqa" in jf:
                continue
            if jf in seen:
                continue
            seen.add(jf)
            try:
                d = json.loads(Path(jf).read_text())
            except Exception:
                continue
            accs = {k: v["acc,none"] for k, v in d.get("results", {}).items()
                    if k.startswith(pref) and "acc,none" in v}
            if not accs:
                continue
            pre = _pretrained(d)
            if not pre:
                continue
            key = (len(accs), Path(jf).stat().st_mtime)
            if pre not in best or key > best[pre][0]:
                best[pre] = (key, accs)
    return {pre: accs for pre, (key, accs) in best.items()}


def matched_firstn(benchmark: str) -> dict:
    return build_index(benchmark)


if __name__ == "__main__":
    for b in ("belebele", "global_piqa"):
        idx = matched_firstn(b)
        print(f"{b}: {len(idx)} checkpoints indexed")
        for pre, accs in sorted(idx.items()):
            print(f"   {pre}: {len(accs)} tasks")
