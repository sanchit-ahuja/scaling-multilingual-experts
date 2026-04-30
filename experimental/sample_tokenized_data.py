import argparse
import json
from pathlib import Path
from datasets import load_from_disk
import shutil
from tqdm import tqdm
import multiprocessing
from functools import partial

# Language families from train_gemma.py
LANGS = {
    "Slavic": ["mk", "hr", "ru", "sk", "sr", "uk"],
    "Germanic": ["af", "fy", "lb", "da", "nl", "en"],
    "Indic": ["bn", "hi", "kn", "ml", "mr", "ne", "ta", "te"],
    "Austronesian": ["sm", "jv", "ceb", "fil", "id", "ms"],
    "Romance": ["es", "pt", "fr", "ga", "it", "ro"],
}

def load_tokenization_stats(family_path):
    """Load tokenization stats from family directory."""
    stats_file = family_path / "tokenization_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            return json.load(f)
    return None

def count_tokens_batch(examples):
    """Count tokens in a batch of examples."""
    return {"token_count": [len(ids) for ids in examples['input_ids']]}

def count_tokens_in_dataset(dataset, num_proc):
    """Count total tokens using multiprocessing."""
    # Add token counts to dataset
    dataset_with_counts = dataset.map(
        count_tokens_batch,
        batched=True,
        num_proc=num_proc,
        desc="Counting tokens",
        remove_columns=[col for col in dataset.column_names if col != 'input_ids']
    )
    
    # Sum all token counts
    total_tokens = sum(dataset_with_counts['token_count'])
    return total_tokens

def sample_dataset_by_tokens(dataset, target_tokens, current_tokens, num_proc, rand_seed=42):
    """Sample dataset to approximately match target tokens using binary search."""
    if current_tokens <= target_tokens:
        return dataset, current_tokens
    
    # Calculate approximate sampling fraction
    sample_fraction = target_tokens / current_tokens
    
    # Shuffle dataset first
    shuffled_dataset = dataset.shuffle(seed=rand_seed)
    
    # Binary search for the right number of samples
    left, right = 1, len(dataset)
    best_num_samples = right
    best_token_count = current_tokens
    
    print(f"    Binary search for optimal sample size (target: {target_tokens:,} tokens)...")
    
    while left <= right:
        mid = (left + right) // 2
        
        # Sample mid examples
        sample = shuffled_dataset.select(range(mid))
        
        # Count tokens in sample
        sample_tokens = count_tokens_in_dataset(sample, num_proc)
        
        print(f"      Trying {mid:,} samples: {sample_tokens:,} tokens")
        
        if abs(sample_tokens - target_tokens) < abs(best_token_count - target_tokens):
            best_num_samples = mid
            best_token_count = sample_tokens
        
        if sample_tokens < target_tokens:
            left = mid + 1
        else:
            right = mid - 1
    
    # Return best sample
    final_sample = shuffled_dataset.select(range(best_num_samples))
    print(f"    Selected {best_num_samples:,} samples with {best_token_count:,} tokens")
    
    return final_sample, best_token_count

def sample_language_data(lang_path, target_tokens, num_proc, rand_seed=42):
    """Sample approximately target_tokens from a language dataset."""
    dataset = load_from_disk(str(lang_path))
    
    # Count tokens using multiprocessing
    print(f"    Counting tokens in {lang_path.name}...")
    total_tokens = count_tokens_in_dataset(dataset, num_proc)
    
    if total_tokens <= target_tokens:
        print(f"    Language has {total_tokens:,} tokens (≤ target {target_tokens:,}), keeping all data")
        return dataset, total_tokens
    
    print(f"    Total tokens: {total_tokens:,}, Target: {target_tokens:,}")
    print(f"    Sampling ratio: {target_tokens/total_tokens*100:.2f}%")
    
    # Use binary search to find optimal sample size
    sampled_dataset, actual_tokens = sample_dataset_by_tokens(
        dataset, target_tokens, total_tokens, num_proc, rand_seed
    )
    
    print(f"    Final: {len(sampled_dataset):,}/{len(dataset):,} examples, {actual_tokens:,} tokens")
    
    return sampled_dataset, actual_tokens

def process_family(family, input_dir, output_dir, tokens_per_family, num_proc, rand_seed=42):
    """Process one language family."""
    print(f"\n{'='*60}")
    print(f"Processing {family} family")
    print(f"{'='*60}")
    
    family_input_path = input_dir / family
    family_output_path = output_dir / family
    
    if not family_input_path.exists():
        print(f"Warning: Family path not found: {family_input_path}")
        return
    
    # Load stats to get available languages
    stats = load_tokenization_stats(family_input_path)
    if stats:
        available_langs = stats['languages']
        print(f"Languages from stats: {available_langs}")
    else:
        # Fallback: check directories
        train_path = family_input_path / "train"
        available_langs = [d.name for d in train_path.iterdir() if d.is_dir()]
        print(f"Languages from directory: {available_langs}")
    
    num_langs = len(available_langs)
    tokens_per_lang = tokens_per_family // num_langs
    
    print(f"Target: {tokens_per_family:,} tokens total")
    print(f"Distribution: {tokens_per_lang:,} tokens per language ({num_langs} languages)")
    print(f"Using {num_proc} CPU cores for processing")
    
    # Create output directories
    (family_output_path / "train").mkdir(parents=True, exist_ok=True)
    (family_output_path / "valid").mkdir(parents=True, exist_ok=True)
    
    family_stats = {
        "family": family,
        "languages": available_langs,
        "target_tokens_per_language": tokens_per_lang,
        "target_total_tokens": tokens_per_family,
        "train_sizes": {},
        "train_token_counts": {},
        "valid_sizes": {},
        "tokenizer": stats['tokenizer'] if stats else "google/gemma-3-1b-pt"
    }
    
    # Process each language
    for lang in available_langs:
        print(f"\n  Processing {lang}...")
        
        # Process train split - sample to target tokens
        train_input_path = family_input_path / "train" / lang
        train_output_path = family_output_path / "train" / lang
        
        if train_input_path.exists():
            sampled_train, actual_tokens = sample_language_data(
                train_input_path, tokens_per_lang, num_proc, rand_seed
            )
            
            # Save with multiprocessing
            print(f"    Saving {len(sampled_train):,} train examples...")
            sampled_train.save_to_disk(str(train_output_path), num_proc=num_proc)
            
            family_stats['train_sizes'][lang] = len(sampled_train)
            family_stats['train_token_counts'][lang] = actual_tokens
            print(f"    ✓ Saved train data")
        else:
            print(f"    Warning: Train path not found: {train_input_path}")
        
        # Copy valid split as-is
        valid_input_path = family_input_path / "valid" / lang
        valid_output_path = family_output_path / "valid" / lang
        
        if valid_input_path.exists():
            print(f"    Copying validation data...")
            shutil.copytree(valid_input_path, valid_output_path, dirs_exist_ok=True)
            valid_ds = load_from_disk(str(valid_input_path))
            family_stats['valid_sizes'][lang] = len(valid_ds)
            print(f"    ✓ Copied {len(valid_ds):,} valid examples")
        else:
            print(f"    Warning: Valid path not found: {valid_input_path}")
    
    # Save family stats
    stats_output_path = family_output_path / "tokenization_stats.json"
    with open(stats_output_path, 'w') as f:
        json.dump(family_stats, f, indent=2)
    
    # Print summary
    total_train_tokens = sum(family_stats['train_token_counts'].values())
    avg_deviation = abs(total_train_tokens - tokens_per_family) / tokens_per_family * 100
    
    print(f"\n{family} Summary:")
    print(f"  Target tokens: {tokens_per_family:,}")
    print(f"  Actual tokens: {total_train_tokens:,}")
    print(f"  Deviation: {avg_deviation:.2f}%")
    print(f"  Total train examples: {sum(family_stats['train_sizes'].values()):,}")
    print(f"  Total valid examples: {sum(family_stats['valid_sizes'].values()):,}")
    
    return family_stats

def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Determine number of processes
    if args.num_proc is None:
        num_proc = max(1, int(multiprocessing.cpu_count() * 0.75))
    else:
        num_proc = args.num_proc
    
    print("="*60)
    print("Sampling Tokenized Data - 5B Tokens per Family")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tokens per family: {args.tokens_per_family:,}")
    print(f"Random seed: {args.rand_seed}")
    print(f"CPU cores to use: {num_proc}/{multiprocessing.cpu_count()}")
    print(f"Families to process: {args.families if args.families else 'all'}")
    print("="*60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which families to process
    families_to_process = args.families if args.families else list(LANGS.keys())
    
    # Process each family
    all_family_stats = {}
    for family in families_to_process:
        family_stats = process_family(
            family=family,
            input_dir=input_dir,
            output_dir=output_dir,
            tokens_per_family=args.tokens_per_family,
            num_proc=num_proc,
            rand_seed=args.rand_seed
        )
        if family_stats:
            all_family_stats[family] = family_stats
    
    # Create overall stats
    overall_stats = {
        "total_families": len(families_to_process),
        "families": families_to_process,
        "tokens_per_family": args.tokens_per_family,
        "rand_seed": args.rand_seed,
        "num_proc": num_proc,
        "family_summaries": {}
    }
    
    for family, stats in all_family_stats.items():
        overall_stats["family_summaries"][family] = {
            "total_train_tokens": sum(stats['train_token_counts'].values()),
            "total_train_examples": sum(stats['train_sizes'].values()),
            "total_valid_examples": sum(stats['valid_sizes'].values()),
            "num_languages": len(stats['languages'])
        }
    
    with open(output_dir / "overall_stats.json", 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for family, summary in overall_stats["family_summaries"].items():
        print(f"{family}:")
        print(f"  Tokens: {summary['total_train_tokens']:,}")
        print(f"  Train examples: {summary['total_train_examples']:,}")
        print(f"  Valid examples: {summary['total_valid_examples']:,}")
        print(f"  Languages: {summary['num_languages']}")
    print("="*60)
    print(f"Sampled data saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample 5B tokens per language family uniformly across languages"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to input tokenized data directory (e.g., /work/nvme/bfzp/madlad-tokenized-dataset)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for sampled data (e.g., /work/nvme/bfzp/madlad-tokenized-5B)"
    )
    parser.add_argument(
        "--tokens_per_family",
        type=int,
        default=5_000_000_000,
        help="Number of tokens per family (default: 5B)"
    )
    parser.add_argument(
        "--families",
        nargs="*",
        choices=list(LANGS.keys()),
        help="Specific families to process (default: all)"
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of CPU cores to use (default: 75%% of available cores)"
    )
    
    args = parser.parse_args()
    main(args)