#!/usr/bin/env python3
"""Enumerate FLORES seed-sweep jobs (Concern 1 core set): one line per checkpoint.

Line format (pipe-separated): tag|pretrained|comma_task_list

Core set chosen for the FLORES claims: Dense and Dense-Reverted (both directions, all 30
held-in languages = 60 tasks) plus the 5 family Experts (own family's languages x 2
directions). Held-in FLORES = 30 langs x {xx->en, en->xx}. Checkpoint paths + the seed-1234
baseline come from the existing FLORES result JSONs (v2 Dense reproduces the paper).
"""
from __future__ import annotations

CKPT = "/work/nvme/bfzp/checkpoints"

# FLORES held-in languages per family (full script-tagged codes)
FAM_LANGS = {
    "Slavic": ["mkd_Cyrl", "hrv_Latn", "rus_Cyrl", "slk_Latn", "srp_Cyrl", "ukr_Cyrl"],
    "Germanic": ["afr_Latn", "ltz_Latn", "dan_Latn", "nld_Latn"],
    "Indic": ["ben_Beng", "hin_Deva", "kan_Knda", "mal_Mlym", "mar_Deva", "npi_Deva",
              "tam_Taml", "tel_Telu"],
    "Austronesian": ["smo_Latn", "jav_Latn", "ceb_Latn", "tgl_Latn", "ind_Latn", "zsm_Latn"],
    "Romance": ["spa_Latn", "por_Latn", "fra_Latn", "glg_Latn", "ita_Latn", "ron_Latn"],
}

# Held-out languages used for transfer evaluation. The two-letter project
# codes map to FLORES script-tagged codes below.
HELDOUT_FAM_LANGS = {
    "Slavic": ["bul_Cyrl", "ces_Latn", "lit_Latn", "pol_Latn", "slv_Latn", "lvs_Latn"],
    "Germanic": ["deu_Latn", "isl_Latn", "nob_Latn", "swe_Latn"],
    "Indic": ["asm_Beng", "guj_Gujr", "ory_Orya", "pan_Guru", "snd_Arab", "sin_Sinh", "urd_Arab"],
    "Austronesian": ["ilo_Latn", "mri_Latn", "sun_Latn", "war_Latn", "plt_Latn"],
    "Romance": ["cat_Latn"],
}

EXPERT_PATH = {
    "Slavic": "slavic_gemma_4b_expert/final",
    "Germanic": "germanic_gemma_4b_expert/final",
    "Indic": "Indic_gemma_4b_expert/checkpoint-7000",
    "Austronesian": "austronesian_gemma_4b_expert/final",
    "Romance": "romance_gemma_4b_expert/final",
}
# family-scoped layer-aware strategies (own-family tasks x 2 directions), like the experts
FREEZE_PATH = {f: f"gemma_4b_{f.lower()}_freeze/final" for f in EXPERT_PATH}
LREG_PATH = {f: f"gemma_4b_{f.lower()}_layer_reg/final" for f in EXPERT_PATH}
SHARED = {
    "dense": "gemma_4b_dense_25b_v2/final",
    "dense-reverted": "gemma_4b_dense_25b_v2/reverted",
}


def tasks_for(langs):
    out = []
    for lg in langs:
        out.append(f"flores_{lg}-eng_Latn")     # xx -> en
        out.append(f"flores_eng_Latn-{lg}")     # en -> xx
    return out


def all_heldin_langs():
    return [lg for langs in FAM_LANGS.values() for lg in langs]


def jobs():
    out = []
    all_tasks = tasks_for(all_heldin_langs())
    for strat, p in SHARED.items():
        out.append((f"{strat}_All", f"{CKPT}/{p}", all_tasks))
    for strat, table in (("expert", EXPERT_PATH), ("freeze", FREEZE_PATH), ("layer-reg", LREG_PATH)):
        for fam, p in table.items():
            out.append((f"{strat}_{fam}", f"{CKPT}/{p}", tasks_for(FAM_LANGS[fam])))
    return out


if __name__ == "__main__":
    for tag, p, tasks in jobs():
        print(f"{tag}|{p}|{','.join(tasks)}")
