#!/usr/bin/env python3
"""
Aggregate downstream evaluation results from all Gemma 4B checkpoint evaluations.

Scans --results_base for baseline_lm_eval_* and perplexity_eval_* result
directories and produces a consolidated CSV with one row per (checkpoint, task)
containing all relevant metrics.

Columns:
  checkpoint, strategy, family, benchmark, task, language, direction,
  acc, bleu, chrf, ter, perplexity, loss

Usage:
    python aggregate_downstream_results.py
    python aggregate_downstream_results.py --output_dir /path/to/output
    python aggregate_downstream_results.py --benchmarks belebele flores perplexity
"""

import json
import argparse
import os
import re
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from configs.constants import LANGS


# =============================================================================
# Checkpoint metadata — maps directory name patterns to (strategy, family)
# =============================================================================

CHECKPOINT_METADATA = {
    # Expert models
    "austronesian_gemma_4b_expert_final": ("expert", "Austronesian"),
    "austronesian_gemma_4b_expert_reverted-9000": ("expert-reverted", "Austronesian"),
    "germanic_gemma_4b_expert_final": ("expert", "Germanic"),
    "germanic_gemma_4b_expert_reverted": ("expert-reverted", "Germanic"),
    "Indic_gemma_4b_expert_checkpoint-7000": ("expert", "Indic"),
    "Indic_gemma_4b_expert_reverted-7000": ("expert-reverted", "Indic"),
    "romance_gemma_4b_expert_final": ("expert", "Romance"),
    "romance_gemma_4b_expert_reverted": ("expert-reverted", "Romance"),
    "slavic_gemma_4b_expert_final": ("expert", "Slavic"),
    "slavic_gemma_4b_expert_reverted": ("expert-reverted", "Slavic"),

    # Freeze models
    "gemma_4b_austronesian_freeze_final": ("freeze", "Austronesian"),
    "gemma_4b_slavic_freeze_final": ("freeze", "Slavic"),
    "gemma_4b_germanic_freeze_final": ("freeze", "Germanic"),
    "gemma_4b_indic_freeze_final": ("freeze", "Indic"),
    "gemma_4b_romance_freeze_final": ("freeze", "Romance"),

    # Layer-reg models
    "gemma_4b_austronesian_layer_reg_final": ("layer-reg", "Austronesian"),
    "gemma_4b_germanic_layer_reg_final": ("layer-reg", "Germanic"),
    "gemma_4b_indic_layer_reg_final": ("layer-reg", "Indic"),
    "gemma_4b_romance_layer_reg_final": ("layer-reg", "Romance"),
    "gemma_4b_slavic_layer_reg_final": ("layer-reg", "Slavic"),

    # Dense model
    "gemma_4b_dense_25b_final": ("dense", "All"),
    "gemma_4b_dense_25b_reverted": ("dense-reverted", "All"),
    "checkpoints_gemma_4b_expert_soup": ("expert-soup", "All"),

    # Base model
    "google_gemma-3-4b-pt": ("base", "N/A"),
}


# =============================================================================
# Helper functions
# =============================================================================

def detect_benchmark(results: dict) -> Optional[str]:
    """Detect benchmark type from lm_eval results JSON."""
    for task_name in results.get('results', {}):
        if 'belebele_' in task_name:
            return 'belebele'
        elif 'global_piqa_completions_' in task_name:
            return 'global_piqa_completions'
        elif 'flores' in task_name.lower():
            return 'flores'
    return None


def extract_checkpoint_name(dir_name: str) -> str:
    """Extract checkpoint identifier from result directory name.

    Strips both lm_eval and perplexity prefixes, plus benchmark suffixes.
    Also strips the leading Family_ prefix that the batch script prepends.
    """
    # Remove prefix
    name = dir_name.replace("baseline_lm_eval_", "").replace("perplexity_eval_", "")

    # Remove benchmark suffix
    for benchmark in ['_belebele', '_global_piqa_completions', '_flores']:
        if name.endswith(benchmark):
            name = name[:-len(benchmark)]
            break

    # Try to match against known checkpoint metadata keys
    # The batch script prepends Family_ to the checkpoint name, so we may
    # need to strip that prefix to match the metadata keys
    if name not in CHECKPOINT_METADATA:
        # Try stripping a leading Family_ prefix (e.g., "Germanic_germanic_gemma_4b_expert_final")
        for key in CHECKPOINT_METADATA:
            if name.endswith(key):
                return key

    return name


def get_metadata(checkpoint_name: str) -> Tuple[str, str]:
    """Get (strategy, family) for a checkpoint name."""
    if checkpoint_name in CHECKPOINT_METADATA:
        return CHECKPOINT_METADATA[checkpoint_name]

    # Fuzzy matching for unexpected naming patterns
    for key, (strategy, family) in CHECKPOINT_METADATA.items():
        if key in checkpoint_name or checkpoint_name in key:
            return (strategy, family)

    return ("unknown", "unknown")


def parse_flores_direction(task_name: str) -> Tuple[str, str]:
    """Parse source and target language from a Flores task name.

    Flores tasks look like: basque_bench_flores_en-eu, flores_eng_Latn-hin_Deva, etc.
    Returns (direction, target_lang) where direction is 'en_xx' or 'xx_en'.
    """
    # Match patterns like "en-eu", "eu-en", "eng_Latn-hin_Deva"
    pair_match = re.search(r'([a-z]{2,3}(?:_[A-Za-z]{4})?)-([a-z]{2,3}(?:_[A-Za-z]{4})?)\s*$', task_name)
    if pair_match:
        src, tgt = pair_match.group(1), pair_match.group(2)
        # Normalize: strip script suffixes like _Latn, _Deva
        src_base = src.split('_')[0]
        tgt_base = tgt.split('_')[0]

        if src_base in ('en', 'eng'):
            return ('en_xx', tgt_base)
        elif tgt_base in ('en', 'eng'):
            return ('xx_en', src_base)
        else:
            return (f'{src_base}_{tgt_base}', tgt_base)

    return ('unknown', 'unknown')


# =============================================================================
# Result parsers
# =============================================================================

def parse_lm_eval_json(json_path: Path) -> List[dict]:
    """Parse a single lm_eval results.json into rows with all available metrics."""
    rows = []

    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ⚠ Error reading {json_path}: {e}")
        return rows

    benchmark = detect_benchmark(data)
    if not benchmark:
        return rows

    # Get checkpoint name — walk up from the JSON to the result dir
    # JSON may be nested: result_dir / model_subdir / results.json
    result_dir = json_path.parent
    while result_dir.parent.name != '' and not (
        result_dir.name.startswith("baseline_lm_eval_") or
        result_dir.name.startswith("perplexity_eval_")
    ):
        result_dir = result_dir.parent

    checkpoint_name = extract_checkpoint_name(result_dir.name)
    strategy, family = get_metadata(checkpoint_name)

    for task_name, metrics in data.get('results', {}).items():
        # Skip aggregate tasks
        if task_name in ('belebele', 'global_piqa_completions', 'flores'):
            continue

        row = {
            'checkpoint': checkpoint_name,
            'strategy': strategy,
            'family': family,
            'benchmark': benchmark,
            'task': task_name,
        }

        if benchmark == 'flores':
            direction, lang = parse_flores_direction(task_name)
            row['direction'] = direction
            row['language'] = lang
            row['bleu'] = metrics.get('bleu,none')
            row['chrf'] = metrics.get('chrf,none')
            row['ter'] = metrics.get('ter,none')

        elif benchmark in ('belebele', 'global_piqa_completions'):
            # Extract language code from task name
            lang_match = re.search(
                r'(?:belebele|global_piqa_completions)_([a-z]{2,3})(?:_[A-Za-z]{4})?$',
                task_name
            )
            row['language'] = lang_match.group(1) if lang_match else task_name
            row['direction'] = None
            row['acc'] = metrics.get('acc,none', metrics.get('acc_norm,none'))

        # Round numeric values
        for col in ('acc', 'bleu', 'chrf', 'ter'):
            if col in row and row[col] is not None:
                row[col] = round(float(row[col]), 4)

        rows.append(row)

    return rows


def parse_perplexity_json(json_path: Path) -> List[dict]:
    """Parse a per_language_eval_results.json from train.py into rows."""
    rows = []

    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ⚠ Error reading {json_path}: {e}")
        return rows

    result_dir_name = json_path.parent.name
    checkpoint_name = extract_checkpoint_name(result_dir_name)
    strategy, family = get_metadata(checkpoint_name)

    for entry in data:
        lang = entry.get('language', 'unknown')
        ppl = entry.get('perplexity')
        loss = entry.get('eval_loss')

        # Skip NaN perplexity
        if ppl is not None and isinstance(ppl, float) and math.isnan(ppl):
            continue

        rows.append({
            'checkpoint': checkpoint_name,
            'strategy': strategy,
            'family': family,
            'benchmark': 'perplexity',
            'task': f"perplexity_{entry.get('family', 'unknown')}_{lang}",
            'language': lang,
            'direction': None,
            'perplexity': round(float(ppl), 4) if ppl is not None else None,
            'loss': round(float(loss), 4) if loss is not None else None,
        })

    return rows


# =============================================================================
# Scanner
# =============================================================================

def scan_results(results_base: Path, benchmarks: List[str]) -> pd.DataFrame:
    """Scan results directory for all evaluation outputs."""
    all_rows = []

    for entry in sorted(results_base.rglob("*")):
        if not entry.is_dir():
            continue

        is_lm_eval = entry.name.startswith("baseline_lm_eval_")
        is_perplexity = entry.name.startswith("perplexity_eval_")

        if not is_lm_eval and not is_perplexity:
            continue

        # Filter by benchmark if specified
        if benchmarks:
            if is_perplexity and 'perplexity' not in benchmarks:
                continue
            if is_lm_eval and not any(b in entry.name for b in benchmarks if b != 'perplexity'):
                continue

        if is_perplexity:
            results_json = entry / "per_language_eval_results.json"
            if not results_json.exists():
                continue
            rows = parse_perplexity_json(results_json)
        else:
            # lm_eval nests results inside a model-path subdirectory
            candidates = list(entry.rglob("results*.json"))
            if not candidates:
                continue
            results_json = sorted(candidates)[-1]
            rows = parse_lm_eval_json(results_json)

        if rows:
            # Skip checkpoints not in CHECKPOINT_METADATA
            if rows[0].get('strategy') == 'unknown':
                print(f"  ⊘ {entry.name}: skipped (unknown checkpoint)")
                continue
            print(f"  ✓ {entry.name}: {len(rows)} results")
            all_rows.extend(rows)

    if not all_rows:
        print("No results found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Ensure all metric columns exist
    for col in ('acc', 'bleu', 'chrf', 'ter', 'perplexity', 'loss'):
        if col not in df.columns:
            df[col] = None

    # Reorder columns
    col_order = ['checkpoint', 'strategy', 'family', 'benchmark', 'task',
                 'language', 'direction', 'acc', 'bleu', 'chrf', 'ter',
                 'perplexity', 'loss']
    df = df[[c for c in col_order if c in df.columns]]

    return df


# =============================================================================
# Language code mapping (3-letter belebele/flores → 2-letter LANGS codes)
# =============================================================================

# Map 3-letter ISO 639-3 / belebele codes to the 2-letter codes used in LANGS
LANG_CODE_MAP = {
    # Training languages
    'mkd': 'mk', 'hrv': 'hr', 'rus': 'ru', 'slk': 'sk', 'srp': 'sr', 'ukr': 'uk',
    'afr': 'af', 'fry': 'fy', 'ltz': 'lb', 'dan': 'da', 'nld': 'nl', 'deu': 'de', 'eng': 'en',
    'ben': 'bn', 'hin': 'hi', 'kan': 'kn', 'mal': 'ml', 'mar': 'mr', 'npi': 'ne', 'tam': 'ta', 'tel': 'te',
    'smo': 'sm', 'jav': 'jv', 'ceb': 'ceb', 'tgl': 'fil', 'ind': 'id', 'zsm': 'ms',
    'spa': 'es', 'por': 'pt', 'fra': 'fr', 'gle': 'ga', 'ita': 'it', 'ron': 'ro',
    # Held-out languages
    'bul': 'bg', 'ces': 'cs', 'pol': 'pl', 'slv': 'sl', 'lit': 'lt', 'lvs': 'lv',
    'swe': 'sv', 'nob': 'no', 'isl': 'is',
    'asm': 'as', 'guj': 'gu', 'ory': 'or', 'pan': 'pa', 'sin': 'si', 'urd': 'ur', 'snd': 'sd',
    'ilo': 'ilo', 'war': 'war', 'plt': 'mg', 'mri': 'mi', 'sun': 'su',
    'cat': 'ca',
}

# Build reverse lookup: 2-letter code → language family name
_LANG_TO_FAMILY = {}
for fam, codes in LANGS.items():
    for code in codes:
        _LANG_TO_FAMILY[code] = fam.capitalize()


def get_language_family(lang_code: str) -> str:
    """Map a language code (2- or 3-letter) to its family name."""
    code2 = LANG_CODE_MAP.get(lang_code, lang_code)
    return _LANG_TO_FAMILY.get(code2, 'Other')


# =============================================================================
# Pivot
# =============================================================================

# Strategies that are shared across all families (not family-specific)
SHARED_FAMILIES = {'All', 'N/A'}


def _get_metric_info(benchmark: str):
    """Return (value_col, col_suffix) for a benchmark."""
    if benchmark in ('belebele', 'global_piqa_completions'):
        return 'acc', 'Acc'
    elif benchmark == 'perplexity':
        return 'perplexity', 'PPL'
    else:
        return 'acc', 'Val'


def _add_lang_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add lang_2 and Language_Family columns to a DataFrame."""
    df = df.copy()
    df['lang_2'] = df['language'].map(lambda x: LANG_CODE_MAP.get(x, x))
    df['Language_Family'] = df['lang_2'].map(lambda x: get_language_family(x))
    return df


def _do_pivot(df: pd.DataFrame, value_col: str, col_suffix: str) -> pd.DataFrame:
    """Core pivot: rows = (Language_Family, Lang), columns = strategy."""
    pivot = df.pivot_table(
        index=['Language_Family', 'lang_2'],
        columns='strategy',
        values=value_col,
        aggfunc='first',
    ).reset_index()

    pivot.rename(columns={'lang_2': 'Lang'}, inplace=True)

    # Rename strategy columns to e.g. Base_Acc, Expert_Acc, ...
    strategy_cols = [c for c in pivot.columns if c not in ('Language_Family', 'Lang')]
    rename_map = {}
    for s in strategy_cols:
        label = s.replace('-', '_').title().replace(' ', '_')
        rename_map[s] = f"{label}_{col_suffix}"
    pivot.rename(columns=rename_map, inplace=True)

    pivot.sort_values(['Language_Family', 'Lang'], inplace=True)
    return pivot


def _do_pivot_flores(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot flores with separate columns per direction (ChrF only)."""
    parts = []
    for direction in ('en_xx', 'xx_en'):
        sub = df[df['direction'] == direction]
        if sub.empty:
            continue
        piv = sub.pivot_table(
            index=['Language_Family', 'lang_2'],
            columns='strategy',
            values='chrf',
            aggfunc='first',
        )
        piv = piv.rename(columns=lambda s: f"{s.replace('-','_').title().replace(' ','_')}_{direction}_ChrF")
        parts.append(piv)

    if not parts:
        return pd.DataFrame()

    merged = parts[0]
    for p in parts[1:]:
        merged = merged.join(p, how='outer')
    merged = merged.reset_index().rename(columns={'lang_2': 'Lang'})
    merged.sort_values(['Language_Family', 'Lang'], inplace=True)
    return merged


def pivot_per_family(df: pd.DataFrame, benchmark: str, output_dir: Path, timestamp: str):
    """Produce one pivoted CSV per language family.

    Each CSV contains rows for ALL languages, with columns for:
      - Shared models: base, dense, dense-reverted
      - Family-specific models: expert, expert-reverted, freeze, layer-reg
    """
    if df.empty:
        return

    df = _add_lang_columns(df)

    # Determine all language families that have family-specific models
    family_specific = df[~df['family'].isin(SHARED_FAMILIES)]['family'].unique()

    for fam in sorted(family_specific):
        # Filter: shared models + this family's models
        fam_df = df[(df['family'].isin(SHARED_FAMILIES)) | (df['family'] == fam)]

        if fam_df.empty:
            continue

        if benchmark == 'flores':
            pivoted = _do_pivot_flores(fam_df)
        else:
            value_col, col_suffix = _get_metric_info(benchmark)
            pivoted = _do_pivot(fam_df, value_col, col_suffix)

        if pivoted.empty:
            continue

        pivot_path = output_dir / f"{benchmark}_{fam.lower()}_pivoted_{timestamp}.csv"
        pivoted.to_csv(pivot_path, index=False)
        print(f"\n  ✓ {benchmark} / {fam}: {pivot_path}")
        print(pivoted.to_string(index=False, max_cols=12))


# =============================================================================
# Output
# =============================================================================

def print_summary(df: pd.DataFrame):
    """Print a summary table of coverage."""
    if df.empty:
        return

    print("\n" + "=" * 80)
    print("COVERAGE SUMMARY")
    print("=" * 80)

    coverage = df.groupby(['checkpoint', 'strategy', 'family'])['benchmark'].apply(
        lambda x: ', '.join(sorted(x.unique()))
    ).reset_index()

    for _, row in coverage.iterrows():
        print(f"  {row['checkpoint']:<50} {row['strategy']:<15} {row['family']:<12} {row['benchmark']}")

    print(f"\nTotal unique checkpoints: {coverage.shape[0]}")
    print(f"Total result rows: {len(df)}")

    # Per-benchmark counts
    print("\nPer-benchmark breakdown:")
    for bm, group in df.groupby('benchmark'):
        n_ckpts = group['checkpoint'].nunique()
        n_langs = group['language'].nunique()
        print(f"  {bm:<30} {n_ckpts:>3} checkpoints  {n_langs:>3} languages  {len(group):>5} rows")


def main():
    parser = argparse.ArgumentParser(description='Aggregate downstream eval results')
    default_base = os.environ.get('RESULTS_BASE', os.environ.get('DATA_ROOT', './results'))
    parser.add_argument('--results_base', default=default_base,
                        help='Base directory containing result folders (env: RESULTS_BASE, DATA_ROOT)')
    parser.add_argument('--output_dir', default=f'{default_base}/aggregated_results',
                        help='Output directory for aggregated CSV')
    parser.add_argument('--benchmarks', nargs='*', default=[],
                        help='Filter to specific benchmarks (e.g., belebele global_piqa_completions flores perplexity)')
    parser.add_argument('--pivot', action='store_true',
                        help='Also produce a pivoted CSV with strategies as columns')

    args = parser.parse_args()

    results_base = Path(args.results_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"downstream_eval_results_{timestamp}.csv"

    print("Scanning for evaluation results...")
    print(f"  Base: {results_base}")
    if args.benchmarks:
        print(f"  Filtering to: {args.benchmarks}")
    print()

    df = scan_results(results_base, args.benchmarks)

    if not df.empty:
        df.to_csv(output_path, index=False)
        print(f"\n✓ Wrote {len(df)} rows to {output_path}")

    print_summary(df)

    # Pivoted output (per family)
    if args.pivot and not df.empty:
        print("\n" + "=" * 80)
        print("PIVOTED OUTPUT (per family)")
        print("=" * 80)
        for bm in df['benchmark'].unique():
            bm_df = df[df['benchmark'] == bm]
            pivot_per_family(bm_df, bm, output_dir, timestamp)


if __name__ == '__main__':
    main()
