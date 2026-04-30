"""
Combine evaluation results from multiple CSV files into separate PPL and Belebele tables.
Outputs a consolidated CSV ready for Google Sheets import.
"""

import pandas as pd
import argparse
from pathlib import Path


def combine_eval_results(csv_files, output_path="combined_results.csv"):
    """
    Combine multiple evaluation CSV files into PPL and Accuracy tables.
    
    Args:
        csv_files: List of paths to CSV files
        output_path: Path to save the combined output
    """
    # Dictionary to store data by model
    model_data = {}
    
    # Read each CSV file
    for csv_path in csv_files:
        # Extract model name from filename (everything before _comprehensive)
        model_name = Path(csv_path).stem.split('_comprehensive')[0]
        
        # Read the CSV
        df = pd.read_csv(csv_path)
        model_data[model_name] = df
        print(f"Loaded {model_name}: {len(df)} rows")
    
    # Get all unique language families and languages
    all_langs = set()
    for df in model_data.values():
        for _, row in df.iterrows():
            all_langs.add((row['Language_Family'], row['Lang']))
    
    # Sort languages by family then language code
    all_langs = sorted(all_langs, key=lambda x: (x[0], x[1]))
    
    # Create PPL table
    ppl_rows = []
    for family, lang in all_langs:
        row_data = {
            'Language_Family': family,
            'Language': lang
        }
        for model_name, df in model_data.items():
            # Find the row for this language
            mask = (df['Language_Family'] == family) & (df['Lang'] == lang)
            if mask.any():
                ppl_value = df.loc[mask, 'PPL_score'].values[0]
                row_data[f'{model_name}_PPL'] = ppl_value
            else:
                row_data[f'{model_name}_PPL'] = 'N/A'
        ppl_rows.append(row_data)
    
    ppl_df = pd.DataFrame(ppl_rows)
    
    # Create Accuracy (Belebele) table
    acc_rows = []
    for family, lang in all_langs:
        row_data = {
            'Language_Family': family,
            'Language': lang
        }
        for model_name, df in model_data.items():
            # Find the row for this language
            mask = (df['Language_Family'] == family) & (df['Lang'] == lang)
            if mask.any():
                acc_value = df.loc[mask, 'Accuracy_score'].values[0]
                row_data[f'{model_name}_Accuracy'] = acc_value
            else:
                row_data[f'{model_name}_Accuracy'] = 'N/A'
        acc_rows.append(row_data)
    
    acc_df = pd.DataFrame(acc_rows)
    
    # Combine both tables with a separator row
    separator = pd.DataFrame([{col: '' for col in ppl_df.columns}])
    header_row = pd.DataFrame([{'Language_Family': '=== BELEBELE ACCURACY SCORES ==='}])
    
    # Combine: PPL table + separator + header + Accuracy table
    combined_df = pd.concat([
        ppl_df,
        separator,
        header_row,
        acc_df
    ], ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Combined results saved to: {output_path}")
    print(f"\nPPL Table: {len(ppl_df)} languages")
    print(f"Accuracy Table: {len(acc_df)} languages")
    print(f"\nModels included:")
    for model_name in model_data.keys():
        print(f"  - {model_name}")
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description='Combine evaluation CSV files for Google Sheets import'
    )
    parser.add_argument(
        'csv_files',
        nargs='+',
        help='CSV files to combine'
    )
    parser.add_argument(
        '-o', '--output',
        default='combined_eval_results.csv',
        help='Output CSV file (default: combined_eval_results.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    for csv_file in args.csv_files:
        if not Path(csv_file).exists():
            print(f"Error: File not found: {csv_file}")
            return
    
    combine_eval_results(args.csv_files, args.output)


if __name__ == '__main__':
    main()
