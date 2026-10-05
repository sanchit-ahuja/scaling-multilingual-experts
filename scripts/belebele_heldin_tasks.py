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

HELDOUT_BELEBELE_CODE = {
    "bg": "bul_Cyrl",
    "cs": "ces_Latn",
    "lt": "lit_Latn",
    "pl": "pol_Latn",
    "sl": "slv_Latn",
    "lv": "lvs_Latn",
    "de": "deu_Latn",
    "is": "isl_Latn",
    "no": "nob_Latn",
    "sv": "swe_Latn",
    "as": "asm_Beng",
    "gu": "guj_Gujr",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "sd": "snd_Arab",
    "si": "sin_Sinh",
    "ur": "urd_Arab",
    "ilo": "ilo_Latn",
    "mi": "mri_Latn",
    "su": "sun_Latn",
    "war": "war_Latn",
    "mg": "plt_Latn",
    "ca": "cat_Latn",
}


def heldin_tasks(family: str | None = None) -> list[str]:
    if family is not None:
        family = family.lower()
        if family not in LANGS:
            raise ValueError(f"Unknown family: {family}")
        language_groups = [LANGS[family]]
    else:
        language_groups = LANGS.values()
    return [
        f"belebele_{BELEBELE_CODE[language]}"
        for languages in language_groups
        for language in languages
        if language in BELEBELE_CODE
    ]


def heldout_tasks(family: str | None = None) -> list[str]:
    if family is not None:
        family = family.lower()
        if family not in LANGS:
            raise ValueError(f"Unknown family: {family}")
        languages = {
            "slavic": ["bg", "cs", "lt", "pl", "sl", "lv"],
            "germanic": ["de", "is", "no", "sv"],
            "indic": ["as", "gu", "or", "pa", "sd", "si", "ur"],
            "austronesian": ["ilo", "mi", "su", "war", "mg"],
            "romance": ["ca"],
        }[family]
    else:
        languages = [language for group in (
            ["bg", "cs", "lt", "pl", "sl", "lv"],
            ["de", "is", "no", "sv"],
            ["as", "gu", "or", "pa", "sd", "si", "ur"],
            ["ilo", "mi", "su", "war", "mg"],
            ["ca"],
        ) for language in group]
    return [f"belebele_{HELDOUT_BELEBELE_CODE[language]}" for language in languages]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--separator", default=",")
    parser.add_argument("--family", choices=sorted(LANGS))
    parser.add_argument("--split", choices=["heldin", "heldout"], default="heldin")
    args = parser.parse_args()
    task_fn = heldout_tasks if args.split == "heldout" else heldin_tasks
    print(args.separator.join(task_fn(args.family)))


if __name__ == "__main__":
    main()
