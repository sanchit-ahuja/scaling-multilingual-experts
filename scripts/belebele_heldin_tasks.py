#!/usr/bin/env python3
"""Print held-in Belebele task names for languages covered by the benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.constants import LANGS  # noqa: E402


BELEBELE_CODE = {
    "mk": "mkd_Cyrl",
    "hr": "hrv_Latn",
    "ru": "rus_Cyrl",
    "sk": "slk_Latn",
    "sr": "srp_Cyrl",
    "uk": "ukr_Cyrl",
    "af": "afr_Latn",
    "da": "dan_Latn",
    "nl": "nld_Latn",
    "en": "eng_Latn",
    "bn": "ben_Beng",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "jv": "jav_Latn",
    "ceb": "ceb_Latn",
    "fil": "tgl_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "ro": "ron_Latn",
}


def heldin_tasks() -> list[str]:
    return [
        f"belebele_{BELEBELE_CODE[language]}"
        for languages in LANGS.values()
        for language in languages
        if language in BELEBELE_CODE
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--separator", default=",")
    args = parser.parse_args()
    print(args.separator.join(heldin_tasks()))


if __name__ == "__main__":
    main()
