"""Shared constants for x-elm-v2 project."""

from typing import Dict, List

# In-domain training languages per family (held-out languages are excluded)
LANGS: Dict[str, List[str]] = {
    "slavic": ["mk", "hr", "ru", "sk", "sr", "uk"],
    "germanic": ["af", "fy", "lb", "da", "nl", "en"],
    "indic": ["bn", "hi", "kn", "ml", "mr", "ne", "ta", "te"],
    "austronesian": ["sm", "jv", "ceb", "fil", "id", "ms"],
    "romance": ["es", "pt", "fr", "gl", "it", "ro"],
}

# Alias for convenience
LANGUAGE_FAMILIES = list(LANGS.keys())

# Data budget constants
TOTAL_BUDGET_TOKENS = 100_000_000_000  # 100B tokens
TOKENS_PER_FAMILY = 25_000_000_000  # 25B tokens per family
