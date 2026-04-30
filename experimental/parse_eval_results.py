#!/usr/bin/env python3
"""
Parse perplexity and downstream evaluation results into a consolidated CSV.
Combines train.py per-language PPL results with lm_eval belebele results.
"""

import json
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Import language families from constants
sys.path.append('/u/sahuja1/x-elm-v2')
from configs.constants import LANGS


def parse_ppl_results(ppl_output: str) -> Dict[str, Dict[str, float]]:
    """
    Parse per-language PPL results from train.py output.
    
    Expected format: "family/lang: Loss=X.XXXX, PPL=XX.XX"
    
    Returns:
        Dict[family][lang] = ppl_score
    """
    ppl_results = {}
    
    # Look for lines like "Indic/bn: Loss=2.3456, PPL=10.44"
    pattern = r"([^/]+)/([^:]+):\s+Loss=[\d.]+,\s+PPL=([\d.]+)"
    
    for line in ppl_output.split('\n'):
        match = re.search(pattern, line.strip())
        if match:
            family, lang, ppl = match.groups()
            if family not in ppl_results:
                ppl_results[family] = {}
            ppl_results[family][lang] = float(ppl)
    
    return ppl_results


def parse_belebele_results(belebele_json_path: str) -> Dict[str, float]:
    """
    Parse belebele results from lm_eval JSON output.
    
    Returns:
        Dict[lang_code] = accuracy_score
    """
    belebele_results = {}
    
    if not Path(belebele_json_path).exists():
        print(f"Warning: Belebele results file not found: {belebele_json_path}")
        return belebele_results
    
    try:
        with open(belebele_json_path, 'r') as f:
            results = json.load(f)
        
        # Parse belebele results - actual format from lm_eval
        for task_name, metrics in results.get('results', {}).items():
            if 'belebele_' in task_name and task_name != 'belebele':  # Skip the overall belebele entry
                # Extract language code from task name
                # Format: "belebele_eng_Latn", "belebele_hin_Deva", etc.
                lang_match = re.search(r'belebele_([a-z]{3})_', task_name)
                if lang_match:
                    lang_iso3 = lang_match.group(1)
                    # Get accuracy metric (use "acc,none" as primary)
                    accuracy = metrics.get('acc,none', metrics.get('acc_norm,none', None))
                    if accuracy is not None:
                        belebele_results[lang_iso3] = float(accuracy)
        
        print(f"Parsed belebele results for {len(belebele_results)} languages")
        
    except Exception as e:
        print(f"Error parsing belebele results: {e}")
    
    return belebele_results


def create_lang_mapping() -> Dict[str, str]:
    """
    Create mapping from 2-letter language codes (constants.py) to 3-letter ISO codes (belebele).
    """
    # Common 2-to-3 letter mappings for languages in constants.py
    lang_mapping = {
        # Germanic
        'en': 'eng', 'de': 'deu', 'nl': 'nld', 'af': 'afr', 'da': 'dan',
        'fy': None, 'lb': None,  # Not commonly in belebele
        
        # Romance  
        'es': 'spa', 'fr': 'fra', 'it': 'ita', 'pt': 'por', 'ro': 'ron',
        'ga': 'gle',
        
        # Slavic
        'ru': 'rus', 'uk': 'ukr', 'hr': 'hrv', 'sr': 'srp', 'sk': 'slk',
        'mk': 'mkd',
        
        # Indic
        'hi': 'hin', 'bn': 'ben', 'ta': 'tam', 'te': 'tel', 'ml': 'mal',
        'kn': 'kan', 'mr': 'mar', 'ne': 'npi',  # Fixed: nep -> npi for Nepali
        
        # Austronesian  
        'id': 'ind', 'ms': 'zsm', 'fil': 'tgl', 'jv': 'jav', 'ceb': 'ceb',  # Fixed: msa->zsm, fil->tgl
        'sm': 'smo',
    }
    
    return lang_mapping


def generate_csv(ppl_results: Dict[str, Dict[str, float]], 
                belebele_results: Dict[str, float],
                output_path: str) -> None:
    """
    Generate consolidated CSV with format: Language_Family, Lang, PPL_score, Accuracy_score
    """
    lang_mapping = create_lang_mapping()
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Language_Family', 'Lang', 'PPL_score', 'Accuracy_score'])
        
        for family, languages in LANGS.items():
            for lang in languages:
                # Get PPL score
                ppl_score = ppl_results.get(family, {}).get(lang, 'N/A')
                
                # Get belebele accuracy
                lang_iso3 = lang_mapping.get(lang)
                if lang_iso3 and lang_iso3 in belebele_results:
                    accuracy_score = belebele_results[lang_iso3]
                else:
                    accuracy_score = 'N/A'
                
                writer.writerow([family, lang, ppl_score, accuracy_score])
    
    print(f"Generated CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Parse evaluation results into CSV')
    parser.add_argument('--ppl_output', required=True, help='Path to PPL evaluation output file')
    parser.add_argument('--belebele_json', required=True, help='Path to belebele JSON results')
    parser.add_argument('--output_csv', required=True, help='Output CSV path')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    
    # Parse PPL results
    print(f"Reading PPL results from: {args.ppl_output}")
    if Path(args.ppl_output).exists():
        with open(args.ppl_output, 'r') as f:
            ppl_output_text = f.read()
        ppl_results = parse_ppl_results(ppl_output_text)
    else:
        print(f"Warning: PPL output file not found: {args.ppl_output}")
        ppl_results = {}
    
    if args.debug:
        print("PPL Results:")
        for family, langs in ppl_results.items():
            print(f"  {family}: {langs}")
    
    # Parse belebele results
    print(f"Reading belebele results from: {args.belebele_json}")
    belebele_results = parse_belebele_results(args.belebele_json)
    
    if args.debug:
        print("Belebele Results:")
        for lang, acc in belebele_results.items():
            print(f"  {lang}: {acc}")
    
    # Generate CSV
    generate_csv(ppl_results, belebele_results, args.output_csv)
    print(f"Results consolidated in: {args.output_csv}")


if __name__ == '__main__':
    main()