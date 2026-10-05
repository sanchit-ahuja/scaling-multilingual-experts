#!/usr/bin/env python3
"""Generate random-sampler ("_rs") variants of the held-in Belebele / Global-PIQA
tasks for the E3 few-shot seed-robustness study (Concern 1).

The released tasks use `fewshot_config.sampler: first_n` (deterministic first-2 demos),
so `--seed` has no effect on the demonstrations. To measure demonstration-sampling
variance without disturbing the canonical tasks/numbers, we emit *separate* tasks with
`sampler: default` (random, seed-controlled) into a self-contained directory that is
loaded only via `--include_path`. Task names get an `_rs` suffix so they never collide
with the released `belebele_*` / `global_piqa_completions_*` tasks.

Held-in task lists: Belebele via scripts/belebele_heldin_tasks.py (28 langs); Global-PIQA
via the active "Held-in languages" block of the released group yaml (30 subsets).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.belebele_heldin_tasks import heldin_tasks as belebele_heldin  # noqa: E402

LM_TASKS = Path("/u/sahuja1/lm-evaluation-harness/lm_eval/tasks")
BEL_SRC = LM_TASKS / "belebele"
PIQA_SRC = LM_TASKS / "global_piqa/completions"
OUT = ROOT / "rebuttal" / "rs_tasks"


def piqa_heldin_subsets() -> list[str]:
    grp = (PIQA_SRC / "_global_piqa.yaml").read_text().splitlines()
    start = next(i for i, l in enumerate(grp) if "Held-in languages" in l)
    end = next(i for i, l in enumerate(grp) if l.strip().startswith("aggregate_metric_list"))
    block = "\n".join(grp[start:end])
    tasks = re.findall(r"-\s*task:\s*global_piqa_completions_(\S+)", block)
    return tasks  # e.g. mkd_cyrl, spa_latn_mexi, ...


def write_belebele():
    dst = OUT / "belebele_rs"
    dst.mkdir(parents=True, exist_ok=True)
    # random-sampler template
    tmpl = (BEL_SRC / "_default_template_yaml").read_text()
    tmpl = tmpl.replace("sampler: first_n", "sampler: default")
    (dst / "_default_template_rs_yaml").write_text(tmpl)
    tasks = belebele_heldin()  # belebele_afr_Latn, ...
    for t in tasks:
        lang = t[len("belebele_"):]           # afr_Latn
        (dst / f"belebele_{lang}_rs.yaml").write_text(
            "dataset_name: " + lang + "\n"
            "fewshot_split: test\n"
            "include: _default_template_rs_yaml\n"
            f"task: belebele_{lang}_rs\n"
            "test_split: test\n"
        )
    return dst, [f"belebele_{t[len('belebele_'):]}_rs" for t in tasks]


def write_piqa():
    dst = OUT / "global_piqa_rs"
    dst.mkdir(parents=True, exist_ok=True)
    tmpl = (PIQA_SRC / "_template").read_text()
    tmpl = tmpl.replace("sampler: first_n", "sampler: default")
    (dst / "_template_rs").write_text(tmpl)
    subs = piqa_heldin_subsets()
    for s in subs:
        (dst / f"global_piqa_completions_{s}_rs.yaml").write_text(
            "dataset_name: " + s + "\n"
            "include: _template_rs\n"
            f"task: global_piqa_completions_{s}_rs\n"
        )
    return dst, [f"global_piqa_completions_{s}_rs" for s in subs]


def main():
    bdir, btasks = write_belebele()
    pdir, ptasks = write_piqa()
    (OUT / "belebele_rs_tasks.txt").write_text(",".join(btasks) + "\n")
    (OUT / "global_piqa_rs_tasks.txt").write_text(",".join(ptasks) + "\n")
    print(f"[belebele_rs] {len(btasks)} tasks -> {bdir}")
    print(f"[global_piqa_rs] {len(ptasks)} tasks -> {pdir}")
    print(f"task lists written under {OUT}")


if __name__ == "__main__":
    main()
