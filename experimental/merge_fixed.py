#!/usr/bin/env python3
"""
Fixed version of merge_experiment_results.py that properly handles all languages.
"""

import pandas as pd
import argparse
import numpy as np
import os
from pathlib import Path
from datetime import datetime


def parse_new_csv(csv_path):
    """Parse the new experiment CSV."""
    return pd.read_csv(csv_path)


def parse_master_csv_simple(master_path):
    """Simple parsing of master CSV - just read the sections we need."""
    
    with open(master_path, 'r') as f:
        lines = f.readlines()
    
    # Find the belebele per-language section
    belebele_data = []
    ppl_data = []
    
    # Look for the section with "Language Family,Lang,Baseline"
    belebele_start = None
    for i, line in enumerate(lines):
        if 'Language Family,Lang,Baseline' in line:
            belebele_start = i
            break
    
    # Look for the section with "Language,Baseline" (PPL section)
    ppl_start = None
    for i, line in enumerate(lines):
        if line.startswith('Language,Baseline') and 'PPL' in ''.join(lines[max(0, i-3):i]):
            ppl_start = i
            break
    
    # Parse belebele section
    if belebele_start is not None:
        header = lines[belebele_start].strip().split(',')
        for i in range(belebele_start + 1, len(lines)):
            line = lines[i].strip()
            if not line:
                break
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3 and parts[1]:  # Has language
                # Handle missing family (use previous row's family)
                if not parts[0] and belebele_data:
                    parts[0] = belebele_data[-1]['Language_Family']
                if parts[0]:  # Only add if we have a family
                    row_data = {'Language_Family': parts[0], 'Language': parts[1]}
                    # Add experiment columns
                    for j, exp_name in enumerate(header[2:]):
                        if j + 2 < len(parts):
                            row_data[exp_name] = parts[j + 2] if parts[j + 2] not in ['', 'N/A'] else np.nan
                    belebele_data.append(row_data)
    
    # Parse PPL section
    if ppl_start is not None:
        header = lines[ppl_start].strip().split(',')
        for i in range(ppl_start + 1, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith('AVERAGE'):
                break
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2 and parts[0]:  # Has language
                # Clean language name
                lang = parts[0].split(' (')[0] if ' (' in parts[0] else parts[0]
                row_data = {'Language': lang}
                # Add experiment columns
                for j, exp_name in enumerate(header[1:]):
                    if j + 1 < len(parts):
                        row_data[exp_name] = parts[j + 1] if parts[j + 1] not in ['', 'N/A'] else np.nan
                ppl_data.append(row_data)
    
    return pd.DataFrame(belebele_data), pd.DataFrame(ppl_data)


def merge_all_data(master_belebele, master_ppl, new_df, experiment_name):
    """Merge all data sources into unified DataFrames."""
    
    # Define language code to full name mapping for Indic languages
    indic_mapping = {
        'bn': 'Bengali', 'hi': 'Hindi', 'kn': 'Kannada', 'ml': 'Malayalam',
        'mr': 'Marathi', 'ne': 'Nepali', 'ta': 'Tamil', 'te': 'Telugu'
    }
    
    # Reverse mapping from full name to code
    indic_name_to_code = {v: k for k, v in indic_mapping.items()}
    
    # Get all unique languages and their families
    lang_family_map = {}
    all_languages = set()
    
    # Process master belebele data first (this contains accuracy data)
    if not master_belebele.empty:
        for _, row in master_belebele.iterrows():
            lang_code = row['Language']
            family = row['Language_Family']
            
            all_languages.add(lang_code)
            lang_family_map[lang_code] = family
    
    # Process master PPL data (convert full names to codes for Indic languages)
    if not master_ppl.empty:
        for _, row in master_ppl.iterrows():
            lang_full = row['Language']
            
            # Convert full name to code if it's an Indic language, otherwise use as-is
            if lang_full in indic_name_to_code:
                canonical_lang = indic_name_to_code[lang_full]  # Use the code
                family = 'Indic'
            else:
                canonical_lang = lang_full
                # Try to find family from existing data
                if canonical_lang in lang_family_map:
                    family = lang_family_map[canonical_lang]
                else:
                    family = 'Unknown'
            
            all_languages.add(canonical_lang)
            lang_family_map[canonical_lang] = family
    
    # Process new experiment data
    for _, row in new_df.iterrows():
        lang_code = row['Lang']
        family = row['Language_Family']
        
        all_languages.add(lang_code)
        lang_family_map[lang_code] = family
    
    # Convert to sorted list and create base dataframes
    all_languages = sorted(list(all_languages))
    base_data = []
    for lang in all_languages:
        base_data.append({
            'Language_Family': lang_family_map[lang],
            'Language': lang
        })
    
    unified_belebele = pd.DataFrame(base_data)
    unified_ppl = pd.DataFrame(base_data)
    
    # Add ALL columns from master belebele data (accuracy data) to belebele dataframe
    if not master_belebele.empty:
        for col in master_belebele.columns:
            if col not in ['Language_Family', 'Language']:
                unified_belebele[col] = np.nan
                for _, row in master_belebele.iterrows():
                    lang = row['Language']
                    mask = unified_belebele['Language'] == lang
                    if mask.any():
                        val = pd.to_numeric(row[col], errors='coerce')
                        if not pd.isna(val):
                            unified_belebele.loc[mask, col] = val
    
    # Add ALL columns from master PPL data to PPL dataframe
    if not master_ppl.empty:
        for col in master_ppl.columns:
            if col != 'Language':
                unified_ppl[col] = np.nan
                for _, row in master_ppl.iterrows():
                    lang_full = row['Language']
                    
                    # Convert full name to code if it's an Indic language
                    if lang_full in indic_name_to_code:
                        canonical_lang = indic_name_to_code[lang_full]
                    else:
                        canonical_lang = lang_full
                    
                    mask = unified_ppl['Language'] == canonical_lang
                    if mask.any():
                        val = pd.to_numeric(row[col], errors='coerce')
                        if not pd.isna(val):
                            unified_ppl.loc[mask, col] = val
    
    # Add new experiment data
    unified_belebele[experiment_name] = np.nan
    unified_ppl[experiment_name] = np.nan
    
    for _, row in new_df.iterrows():
        lang = row['Lang']
        mask = unified_belebele['Language'] == lang
        
        if mask.any():
            if pd.notna(row['Accuracy_score']):
                unified_belebele.loc[mask, experiment_name] = row['Accuracy_score']
            
        mask_ppl = unified_ppl['Language'] == lang
        if mask_ppl.any():
            if pd.notna(row['PPL_score']):
                unified_ppl.loc[mask_ppl, experiment_name] = row['PPL_score']
    
    return unified_belebele, unified_ppl


def save_google_sheets_csv(belebele_df, ppl_df, output_path):
    """Save in Google Sheets friendly format."""
    
    with open(output_path, 'w') as f:
        # Header
        f.write(f"# Multilingual LLM Evaluation Results\\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"# Total languages: {len(belebele_df)}\\n")
        f.write("\\n")
        
        # Accuracy results
        f.write("=== ACCURACY RESULTS (Belebele Task) ===\\n")
        belebele_df.to_csv(f, index=False, na_rep='')
        f.write("\\n\\n")
        
        # PPL results
        f.write("=== PERPLEXITY RESULTS (Lower is Better) ===\\n")
        ppl_df.to_csv(f, index=False, na_rep='')


def main():
    parser = argparse.ArgumentParser(description="Merge experiment results")
    parser.add_argument("new_csv", help="Path to new experiment CSV")
    parser.add_argument("experiment_name", help="Name for the experiment")
    parser.add_argument("--master", 
                       default="/work/nvme/bfzp/comprehensive_evals/all_indic_l2_results.csv",
                       help="Master CSV path")
    parser.add_argument("--output-dir",
                       default="/work/nvme/bfzp/comprehensive_evals/",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Generate output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"merged_results_{timestamp}.csv")
    
    print(f"🔄 Merging '{args.experiment_name}' with master results...")
    print(f"📊 New: {args.new_csv}")
    print(f"📋 Master: {args.master}")
    print(f"💾 Output: {output_path}")
    print("-" * 70)
    
    try:
        # Parse data
        new_df = parse_new_csv(args.new_csv)
        master_belebele, master_ppl = parse_master_csv_simple(args.master)
        
        print(f"Master belebele: {len(master_belebele)} rows")
        print(f"Master PPL: {len(master_ppl)} rows") 
        print(f"New experiment: {len(new_df)} rows")
        
        # Merge data
        unified_belebele, unified_ppl = merge_all_data(
            master_belebele, master_ppl, new_df, args.experiment_name
        )
        
        # Save results
        save_google_sheets_csv(unified_belebele, unified_ppl, output_path)
        
        print(f"\\n✅ Success! Results saved to: {output_path}")
        print(f"📊 Total languages: {len(unified_belebele)}")
        print(f"📈 Belebele experiments: {len(unified_belebele.columns) - 2}")
        print(f"📉 PPL experiments: {len(unified_ppl.columns) - 2}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())