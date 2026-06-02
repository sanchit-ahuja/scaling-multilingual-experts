#!/usr/bin/env python3
"""Shared FLORES sample post-processing used for offline scoring audits."""

from __future__ import annotations

import re
from pathlib import Path


SAMPLE_PAIR_RE = re.compile(r"samples_flores_([^-]+)-([^_]+_[^_]+)_")


def truncate_translation(generation: str) -> str:
    """Apply the paper truncation rule to one generated translation."""
    if not generation:
        return generation
    generation = generation.lstrip()
    cuts = [
        index
        for index in (generation.find("\\n"), generation.find("\n"))
        if index >= 0
    ]
    if cuts:
        generation = generation[: min(cuts)]
    return generation.strip()


def parse_sample_pair(path: str | Path) -> tuple[str, str, str, str] | None:
    """Return source, target, direction, and non-English language code."""
    match = SAMPLE_PAIR_RE.match(Path(path).name)
    if not match:
        return None
    source, target = match.groups()
    if source.startswith("eng_"):
        return source, target, "en_xx", target
    if target.startswith("eng_"):
        return source, target, "xx_en", source
    return None
