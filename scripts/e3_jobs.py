#!/usr/bin/env python3
"""Enumerate E3 seed-sweep jobs: one line per (checkpoint x benchmark).

Line format (pipe-separated): tag|pretrained|benchmark|include_path|comma_task_list

Family experts (expert / expert-reverted / freeze / layer-reg) are evaluated only on
their own family's held-in languages; the shared models (base / dense / dense-reverted /
soup) are evaluated on all held-in languages. Held-in task subsets per family are parsed
from the family-comment blocks of the released group yamls so the cells match the paper.
Checkpoint paths were resolved from the existing held-in result JSONs (Dense/D-Rev = v2,
which reproduce the E1 table exactly; Romance from the in_domain_romance_archive campaign).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CKPT = "/work/nvme/bfzp/checkpoints"
RS = ROOT / "rebuttal" / "rs_tasks"
BEL_GRP = Path("/u/sahuja1/lm-evaluation-harness/lm_eval/tasks/belebele/_belebele.yaml")
PIQA_GRP = Path("/u/sahuja1/lm-evaluation-harness/lm_eval/tasks/global_piqa/completions/_global_piqa.yaml")

COMMENT_TO_FAMILY = {
    "balto-slavic": "Slavic", "slavic": "Slavic",
    "germanic": "Germanic",
    "indo-aryan": "Indic", "dravidian": "Indic",
    "austronesian": "Austronesian",
    "romance": "Romance",
}

# (strategy, family) -> checkpoint path. Shared strategies use family "All".
EXPERTS = {
    "Slavic": ("slavic_gemma_4b_expert/final", "slavic_gemma_4b_expert/reverted",
               "gemma_4b_slavic_freeze/final", "gemma_4b_slavic_layer_reg/final"),
    "Germanic": ("germanic_gemma_4b_expert/final", "germanic_gemma_4b_expert/reverted",
                 "gemma_4b_germanic_freeze/final", "gemma_4b_germanic_layer_reg/final"),
    "Indic": ("Indic_gemma_4b_expert/checkpoint-7000", "Indic_gemma_4b_expert/reverted-7000",
              "gemma_4b_indic_freeze/final", "gemma_4b_indic_layer_reg/final"),
    "Austronesian": ("austronesian_gemma_4b_expert/final", "austronesian_gemma_4b_expert/reverted-9000",
                     "gemma_4b_austronesian_freeze/final", "gemma_4b_austronesian_layer_reg/final"),
    "Romance": ("romance_gemma_4b_expert/final", "romance_gemma_4b_expert/reverted",
                "gemma_4b_romance_freeze/final", "gemma_4b_romance_layer_reg/final"),
}
STRAT_NAMES = ("expert", "expert-reverted", "freeze", "layer-reg")
SHARED = {
    "base": "google/gemma-3-4b-pt",
    "dense": "gemma_4b_dense_25b_v2/final",
    "dense-reverted": "gemma_4b_dense_25b_v2/reverted",
    "soup": "gemma_4b_expert_soup",
}


def heldin_by_family(group_yaml: Path, task_re: str) -> dict:
    """{family: [task,...]} from the active 'Held-in languages' block, keyed by family comment."""
    lines = group_yaml.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if "Held-in languages" in l)
    end = next(i for i, l in enumerate(lines[start:], start)
               if l.strip().startswith("aggregate_metric_list"))
    fam = None
    out = {}
    for l in lines[start:end]:
        s = l.strip()
        cm = re.match(r"#\s*([A-Za-z\-]+)", s)
        if cm and cm.group(1).lower() in COMMENT_TO_FAMILY:
            fam = COMMENT_TO_FAMILY[cm.group(1).lower()]
            continue
        m = re.search(task_re, l)
        if m and fam:
            out.setdefault(fam, []).append(m.group(1))
    return out


def resolve(path: str) -> str:
    return path if "/" in path and not path.startswith(("google/", "Qwen/")) and Path(path).parts[0] != "google" \
        else path
    # (kept simple; abs paths below)


def full(path: str) -> str:
    return path if path.startswith("google/") else f"{CKPT}/{path}"


def main() -> None:
    bel = heldin_by_family(BEL_GRP, r"-\s*(belebele_\S+)")
    piqa = heldin_by_family(PIQA_GRP, r"task:\s*(global_piqa_completions_\S+)")
    # append _rs and drop the completions "task_alias" leftovers
    bel = {f: [t + "_rs" for t in ts] for f, ts in bel.items()}
    piqa = {f: [t + "_rs" for t in ts] for f, ts in piqa.items()}
    bel_all = [t for ts in bel.values() for t in ts]
    piqa_all = [t for ts in piqa.values() for t in ts]

    inc = {"belebele": str(RS / "belebele_rs"), "global_piqa": str(RS / "global_piqa_rs")}
    subsets = {"belebele": bel, "global_piqa": piqa}
    allt = {"belebele": bel_all, "global_piqa": piqa_all}

    jobs = []
    for fam, paths in EXPERTS.items():
        for strat, p in zip(STRAT_NAMES, paths):
            for bench in ("belebele", "global_piqa"):
                tasks = subsets[bench].get(fam, [])
                tag = f"{strat}_{fam}_{bench}"
                jobs.append((tag, full(p), bench, inc[bench], ",".join(tasks)))
    for strat, p in SHARED.items():
        for bench in ("belebele", "global_piqa"):
            tag = f"{strat}_All_{bench}"
            jobs.append((tag, full(p), bench, inc[bench], ",".join(allt[bench])))

    for tag, p, bench, incp, tasks in jobs:
        print(f"{tag}|{p}|{bench}|{incp}|{tasks}")


if __name__ == "__main__":
    main()
