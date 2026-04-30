#!/usr/bin/env python3
"""
Process FLORES evaluation results and create a CSV for Google Sheets.
Compares base model, checkpoint-7000, and reverted-7000 models.
"""

import json
import argparse
import pandas as pd
from pathlib import Path

# Language family mapping
LANGS = {
    "Slavic": ["mk", "hr", "ru", "sk", "sr", "uk"],
    "Germanic": ["af", "fy", "lb", "da", "nl", "de", "en"],
    "Indic": ["bn", "hi", "kn", "ml", "mr", "ne", "ta", "te"],
    "Austronesian": ["sm", "jv", "ceb", "fil", "id", "ms"],
    "Romance": ["es", "pt", "fr", "ga", "it", "ro"],
}

# ISO 639-3 to ISO 639-1 mapping
LANG_CODE_MAPPING = {
    "mkd_Cyrl": "mk", "hrv_Latn": "hr", "rus_Cyrl": "ru", "slk_Latn": "sk", "srp_Cyrl": "sr", "ukr_Cyrl": "uk",
    "afr_Latn": "af", "ltz_Latn": "lb", "dan_Latn": "da", "nld_Latn": "nl", "deu_Latn": "de",
    "ben_Beng": "bn", "hin_Deva": "hi", "kan_Knda": "kn", "mal_Mlym": "ml", "mar_Deva": "mr", 
    "npi_Deva": "ne", "tam_Taml": "ta", "tel_Telu": "te",
    "smo_Latn": "sm", "jav_Latn": "jv", "ceb_Latn": "ceb", "tgl_Latn": "fil", "ind_Latn": "id", "zsm_Latn": "ms",
    "spa_Latn": "es", "por_Latn": "pt", "fra_Latn": "fr", "gle_Latn": "ga", "ita_Latn": "it", "ron_Latn": "ro",
}

LANG_TO_FAMILY = {lang: family for family, langs in LANGS.items() for lang in langs}


def extract_scores(results, direction, lang_639_3, model_name):
    """Extract scores for a language-direction-model combination."""
    task = f"flores_eng_Latn-{lang_639_3}" if direction == "en_to_xx" else f"flores_{lang_639_3}-eng_Latn"
    if task in results.get('results', {}):
        bleu = results['results'][task].get('bleu,none', 'N/A')
        chrf = results['results'][task].get('chrf,none', 'N/A')
        bleu_str = f"{bleu:.2f}" if isinstance(bleu, (int, float)) else "N/A"
        chrf_str = f"{chrf:.2f}" if isinstance(chrf, (int, float)) else "N/A"
        return f"{bleu_str}/{chrf_str}"
    return "N/A/N/A"


def process_results(base_path, checkpoint_path, reverted_path, output_path):
    """Process FLORES results and create CSV with two tables."""
    # Load results
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)
    with open(reverted_path) as f:
        reverted = json.load(f)
    base = json.load(open(base_path)) if base_path else None
    
    # Get all languages
    langs_639_3 = [t.replace('flores_eng_Latn-', '') for t in checkpoint.get('results', {}).keys() 
                   if t.startswith('flores_eng_Latn-') and t.replace('flores_eng_Latn-', '') in LANG_CODE_MAPPING]
    
    data = []
    for lang_639_3 in sorted(langs_639_3):
        lang = LANG_CODE_MAPPING[lang_639_3]
        family = LANG_TO_FAMILY.get(lang, "Unknown")
        
        for direction in ["en_to_xx", "xx_to_en"]:
            data.append({
                'Language': lang,
                'Family': family,
                'Direction': direction,
                'Base': extract_scores(base, direction, lang_639_3, 'base') if base else "N/A/N/A",
                'Checkpoint': extract_scores(checkpoint, direction, lang_639_3, 'checkpoint'),
                'Reverted': extract_scores(reverted, direction, lang_639_3, 'reverted'),
            })
    
    # Create DataFrame and split by direction
    df = pd.DataFrame(data)
    en_to_xx = df[df['Direction'] == 'en_to_xx'].drop('Direction', axis=1).sort_values(['Family', 'Language'])
    xx_to_en = df[df['Direction'] == 'xx_to_en'].drop('Direction', axis=1).sort_values(['Family', 'Language'])
    
    # Write to CSV
    with open(output_path, 'w') as f:
        f.write("# English to Target Language (en→xx) - Format: BLEU/chrF\n")
        en_to_xx.to_csv(f, index=False)
        f.write("\n# Target Language to English (xx→en) - Format: BLEU/chrF\n")
        xx_to_en.to_csv(f, index=False)
    
    print(f"✓ CSV created: {output_path}")
    print(f"✓ Processed {len(langs_639_3)} languages")
    print(f"\nTwo tables: en→xx and xx→en")
    print(f"Format: BLEU/chrF (e.g., '25.30/37.04')")


def main():
    parser = argparse.ArgumentParser(description='Process FLORES results into CSV')
    parser.add_argument('--base', help='Base model results JSON (optional)')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint-7000 results JSON')
    parser.add_argument('--reverted', required=True, help='Reverted-7000 results JSON')
    parser.add_argument('--output', default='flores_comparison.csv', help='Output CSV path')
    args = parser.parse_args()
    
    for path, name in [(args.checkpoint, 'checkpoint'), (args.reverted, 'reverted')]:
        if not Path(path).exists():
            print(f"Error: {name} file not found: {path}")
            return
    
    if args.base and not Path(args.base).exists():
        print(f"Error: base file not found: {args.base}")
        return
    
    process_results(args.base, args.checkpoint, args.reverted, args.output)


if __name__ == '__main__':
    main()
