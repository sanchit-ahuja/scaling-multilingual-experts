#!/usr/bin/env python3
"""
Combine multiple evaluation results (belebele, piqa, etc.) into a single CSV for comparison.
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys
import re

# Import language families from constants
sys.path.append('/u/sahuja1/x-elm-v2')
from configs.constants import LANGS


def detect_benchmark_type(data: dict) -> Optional[str]:
    """
    Auto-detect the benchmark type from the JSON structure.
    
    Returns:
        'belebele', 'piqa', or None if unknown
    """
    results = data.get('results', {})
    
    for task_name in results.keys():
        if 'belebele_' in task_name:
            return 'belebele'
        elif 'global_piqa_completions_' in task_name:
            return 'piqa'
    
    return None


def extract_lang_code(task_name: str, benchmark_type: str) -> Optional[str]:
    """
    Extract language code from task name based on benchmark type.
    
    Args:
        task_name: Task name like "belebele_eng_Latn" or "global_piqa_completions_hin_deva"
        benchmark_type: 'belebele' or 'piqa'
    
    Returns:
        3-letter ISO language code or None
    """
    if benchmark_type == 'belebele':
        # Format: "belebele_eng_Latn"
        match = re.search(r'belebele_([a-z]{3})_', task_name)
        if match:
            return match.group(1)
    elif benchmark_type == 'piqa':
        # Format: "global_piqa_completions_hin_deva" or "global_piqa_completions_eng_latn"
        # Also handle: "global_piqa_completions_spa_latn_mexi"
        match = re.search(r'global_piqa_completions_([a-z]{3})_', task_name)
        if match:
            return match.group(1)
    
    return None


def parse_benchmark_json(json_path: str, benchmark_type: Optional[str] = None) -> Tuple[Dict[str, float], str]:
    """
    Parse benchmark results from lm_eval JSON output.
    
    Args:
        json_path: Path to JSON results file
        benchmark_type: 'belebele', 'piqa', or None for auto-detect
    
    Returns:
        Tuple of (Dict[lang_code] = accuracy_score, detected_benchmark_type)
    """
    results = {}
    
    if not Path(json_path).exists():
        print(f"Warning: File not found: {json_path}")
        return results, 'unknown'
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Auto-detect benchmark type if not specified
        if benchmark_type is None:
            benchmark_type = detect_benchmark_type(data)
            if benchmark_type is None:
                print(f"Warning: Could not detect benchmark type for {json_path}")
                return results, 'unknown'
        
        # Parse results based on benchmark type
        for task_name, metrics in data.get('results', {}).items():
            # Skip aggregate tasks (without language suffix)
            if benchmark_type == 'belebele' and task_name == 'belebele':
                continue
            if benchmark_type == 'piqa' and task_name == 'global_piqa_completions':
                continue
            
            # Extract language code
            lang_iso3 = extract_lang_code(task_name, benchmark_type)
            if lang_iso3:
                # Use acc,none as primary metric (consistent with belebele)
                accuracy = metrics.get('acc,none', None)
                if accuracy is not None:
                    results[lang_iso3] = float(accuracy)
        
        print(f"Parsed {len(results)} languages from {Path(json_path).name} (benchmark: {benchmark_type})")
        
    except Exception as e:
        print(f"Error parsing {json_path}: {e}")
        return results, 'unknown'
    
    return results, benchmark_type


def create_lang_mapping() -> Dict[str, tuple]:
    """
    Create mapping from 3-letter ISO codes to (2-letter code, family).
    """
    iso3_to_info = {}
    
    # Mapping from 2-letter to 3-letter codes
    lang_2to3 = {
        # Germanic
        'en': 'eng', 'de': 'deu', 'nl': 'nld', 'af': 'afr', 'da': 'dan',
        
        # Romance  
        'es': 'spa', 'fr': 'fra', 'it': 'ita', 'pt': 'por', 'ro': 'ron',
        'ga': 'gle',
        
        # Slavic
        'ru': 'rus', 'uk': 'ukr', 'hr': 'hrv', 'sr': 'srp', 'sk': 'slk',
        'mk': 'mkd',
        
        # Indic
        'hi': 'hin', 'bn': 'ben', 'ta': 'tam', 'te': 'tel', 'ml': 'mal',
        'kn': 'kan', 'mr': 'mar', 'ne': 'npi',
        
        # Austronesian  
        'id': 'ind', 'ms': 'zsm', 'fil': 'tgl', 'jv': 'jav', 'ceb': 'ceb',
        'sm': 'smo',
    }
    
    # Build reverse mapping with family info
    for family, langs in LANGS.items():
        for lang_2letter in langs:
            if lang_2letter in lang_2to3:
                lang_3letter = lang_2to3[lang_2letter]
                iso3_to_info[lang_3letter] = (lang_2letter, family)
    
    return iso3_to_info


def generate_comparison_csv(base_results: Dict[str, float],
                            expert_results: Dict[str, float],
                            reverted_results: Dict[str, float],
                            output_path: str,
                            benchmark_type: str = 'belebele') -> None:
    """
    Generate CSV comparing all three models.
    Format: Language_Family, Lang, Lang_ISO3, Base_Acc, Expert_Acc, Reverted_Acc, Expert_Gain, Revert_Loss
    
    Args:
        base_results: Base model results
        expert_results: Expert checkpoint results
        reverted_results: Reverted expert results
        output_path: Output CSV path
        benchmark_type: Type of benchmark for display purposes
    """
    lang_mapping = create_lang_mapping()
    
    # Get all unique language codes across all results
    all_langs = set(base_results.keys()) | set(expert_results.keys()) | set(reverted_results.keys())
    
    rows = []
    for lang_iso3 in sorted(all_langs):
        if lang_iso3 in lang_mapping:
            lang_2letter, family = lang_mapping[lang_iso3]
        else:
            # Unknown language - skip or use defaults
            lang_2letter = lang_iso3
            family = 'Unknown'
        
        base_acc = base_results.get(lang_iso3, float('nan'))
        expert_acc = expert_results.get(lang_iso3, float('nan'))
        reverted_acc = reverted_results.get(lang_iso3, float('nan'))
        
        # Calculate deltas
        try:
            expert_gain = expert_acc - base_acc if base_acc == base_acc else float('nan')
            revert_loss = reverted_acc - expert_acc if expert_acc == expert_acc else float('nan')
        except:
            expert_gain = float('nan')
            revert_loss = float('nan')
        
        rows.append({
            'Language_Family': family,
            'Lang': lang_2letter,
            'Lang_ISO3': lang_iso3,
            'Base_Acc': f'{base_acc:.4f}' if base_acc == base_acc else 'N/A',
            'Expert_Acc': f'{expert_acc:.4f}' if expert_acc == expert_acc else 'N/A',
            'Reverted_Acc': f'{reverted_acc:.4f}' if reverted_acc == reverted_acc else 'N/A',
            'Expert_Gain': f'{expert_gain:+.4f}' if expert_gain == expert_gain else 'N/A',
            'Revert_Loss': f'{revert_loss:+.4f}' if revert_loss == revert_loss else 'N/A',
        })
    
    # Sort by family, then language
    rows.sort(key=lambda x: (x['Language_Family'], x['Lang']))
    
    # Write CSV
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Language_Family', 'Lang', 'Lang_ISO3', 'Base_Acc', 
                     'Expert_Acc', 'Reverted_Acc', 'Expert_Gain', 'Revert_Loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nGenerated comparison CSV: {output_path}")
    print(f"Total languages: {len(rows)}")
    
    # Print summary statistics
    print("\n=== Summary by Family ===")
    families = {}
    for row in rows:
        family = row['Language_Family']
        if family not in families:
            families[family] = {'count': 0, 'expert_gains': [], 'revert_losses': []}
        families[family]['count'] += 1
        if row['Expert_Gain'] != 'N/A':
            families[family]['expert_gains'].append(float(row['Expert_Gain']))
        if row['Revert_Loss'] != 'N/A':
            families[family]['revert_losses'].append(float(row['Revert_Loss']))
    
    for family in sorted(families.keys()):
        info = families[family]
        avg_expert_gain = sum(info['expert_gains']) / len(info['expert_gains']) if info['expert_gains'] else 0
        avg_revert_loss = sum(info['revert_losses']) / len(info['revert_losses']) if info['revert_losses'] else 0
        print(f"{family:15s}: {info['count']:2d} langs | Avg Expert Gain: {avg_expert_gain:+.4f} | Avg Revert Loss: {avg_revert_loss:+.4f}")


def main():
    parser = argparse.ArgumentParser(description='Combine evaluation results (belebele, piqa, etc.) into comparison CSV')
    parser.add_argument('--base', required=True, help='Path to base model JSON results')
    parser.add_argument('--expert', required=True, help='Path to expert checkpoint JSON results')
    parser.add_argument('--reverted', required=True, help='Path to reverted expert JSON results')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--benchmark', choices=['belebele', 'piqa', 'auto'], default='auto',
                       help='Benchmark type (default: auto-detect)')
    
    args = parser.parse_args()
    
    print("Parsing evaluation results...")
    
    # Parse results with auto-detection or specified benchmark type
    benchmark_type = None if args.benchmark == 'auto' else args.benchmark
    
    base_results, base_type = parse_benchmark_json(args.base, benchmark_type)
    expert_results, expert_type = parse_benchmark_json(args.expert, benchmark_type)
    reverted_results, reverted_type = parse_benchmark_json(args.reverted, benchmark_type)
    
    # Verify all files are the same benchmark type
    detected_types = {base_type, expert_type, reverted_type}
    if len(detected_types) > 1:
        print(f"Warning: Mixed benchmark types detected: {detected_types}")
    
    # Use the most common type (or first non-unknown)
    final_benchmark_type = base_type
    if base_type == 'unknown':
        final_benchmark_type = expert_type if expert_type != 'unknown' else reverted_type
    
    generate_comparison_csv(base_results, expert_results, reverted_results, args.output, final_benchmark_type)


if __name__ == '__main__':
    main()
