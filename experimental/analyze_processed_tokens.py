#!/usr/bin/env python3
"""
Analyze processed MADLAD dataset and calculate exact total tokens per language.
Uses Gemma tokenizer for accurate token counting.
Outputs results to CSV file.
"""

import os
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
import psutil

# Configuration
# PROCESSED_DATA_PATH = "/projects/lilac_lab/sanchit/processed-madlad-dataset"
PROCESSED_DATA_PATH = "/scratch/ahuja.sanc/cleanup-madlad-dataset"
# OUTPUT_CSV_PATH = "/projects/lilac_lab/sanchit/token_analysis_new.csv"
OUTPUT_CSV_PATH = "/scratch/ahuja.sanc/token_analysis_cleanup.csv"
# DETAILED_OUTPUT_PATH = "/projects/lilac_lab/sanchit/token_analysis_detailed_new.json"
DETAILED_OUTPUT_PATH = "/scratch/ahuja.sanc/token_analysis_detailed_cleanup.json"
TOKENIZER_PATH = "google/gemma-3-1b-pt"
# CHECKPOINT_PATH = "/projects/lilac_lab/sanchit/token_analysis_checkpoint_new.json"
CHECKPOINT_PATH = "/scratch/ahuja.sanc/token_analysis_checkpoint_cleanup.json"

# Language families


LANG_FAMILIES = {
    # "Austronesian": ["sm", "jv", "ceb", "fil", "id", "ms"],
    # "Indic": ["hi", "kn", "ml", "mr", "ne", "ta", "te"],
    "Indic": ["ne", "ta", "te"]
    # "Germanic": ["af", "da", "fy", "lb"],
    # "Romance": ["ga", "ro"],
    # "Indic": ["hi", "kn", "ml", "mr", "ne", "ta", "te"],
    # "Austronesian": ["ceb", "fil", "jv"],
}

# Processing settings
BATCH_SIZE = 10000  # Process texts in batches for tokenization
MAX_MEMORY_USAGE = 0.8  # 80% of available memory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("token_analysis.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class MemoryManager:
    """Memory management utilities."""

    @staticmethod
    def get_memory_usage():
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total / (1024**3),  # GB
            "available": memory.available / (1024**3),  # GB
            "percent": memory.percent,
            "used": memory.used / (1024**3),  # GB
        }

    @staticmethod
    def should_gc():
        """Check if garbage collection is needed."""
        import gc

        memory = MemoryManager.get_memory_usage()
        if memory["percent"] > MAX_MEMORY_USAGE * 100:
            gc.collect()
            return True
        return False


def count_exact_tokens(texts: List[str], tokenizer) -> int:
    """Count exact number of tokens using the tokenizer."""
    total_tokens = 0

    for text in texts:
        if text and isinstance(text, str):
            # Tokenize and count tokens
            tokens = tokenizer(text, truncation=False, add_special_tokens=False)
            total_tokens += len(tokens["input_ids"])

    return total_tokens


def analyze_language_tokens(family: str, lang: str, dataset, tokenizer) -> Dict:
    """Analyze exact token count for a specific language using Gemma tokenizer."""
    logger.info(f"Analyzing {family}/{lang}...")

    total_tokens = 0
    num_examples = len(dataset)

    # Process in batches to avoid memory issues
    batch_size = BATCH_SIZE

    for i in tqdm(range(0, num_examples, batch_size), desc=f"{family}/{lang}"):
        batch_end = min(i + batch_size, num_examples)
        batch = dataset[i:batch_end]

        # Get text column
        if "text" in batch:
            texts = batch["text"]
        else:
            # Try to find text column
            text_columns = [col for col in batch.keys() if "text" in col.lower()]
            if text_columns:
                texts = batch[text_columns[0]]
            else:
                logger.warning(f"No text column found for {family}/{lang}")
                texts = []

        # Ensure texts is a list
        if not isinstance(texts, list):
            texts = [texts]

        # Count exact tokens for batch
        batch_tokens = count_exact_tokens(texts, tokenizer)
        total_tokens += batch_tokens

        # Memory management
        if (i // batch_size) % 10 == 0:  # Check every 10 batches
            MemoryManager.should_gc()

    avg_tokens_per_example = total_tokens / num_examples if num_examples > 0 else 0

    result = {
        "family": family,
        "language": lang,
        "num_examples": num_examples,
        "total_tokens": total_tokens,
        "avg_tokens_per_example": round(avg_tokens_per_example, 2),
    }

    logger.info(
        f"{family}/{lang}: {num_examples:,} examples, {total_tokens:,} tokens, {avg_tokens_per_example:.2f} avg tokens/example"
    )

    return result


def save_checkpoint(all_results, family_totals, language_totals, completed_families):
    """Save checkpoint data to resume if interrupted."""
    checkpoint_data = {
        "completed_families": list(completed_families),
        "all_results": all_results,
        "family_totals": {
            family: dict(totals) for family, totals in family_totals.items()
        },
        "language_totals": {
            lang: dict(totals) for lang, totals in language_totals.items()
        },
    }

    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    logger.info(f"Checkpoint saved: {CHECKPOINT_PATH}")


def load_checkpoint():
    """Load checkpoint data if exists."""
    if Path(CHECKPOINT_PATH).exists():
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                data = json.load(f)

            # Convert dicts back to defaultdicts
            family_totals = defaultdict(lambda: {"examples": 0, "tokens": 0})
            for family, totals in data["family_totals"].items():
                family_totals[family] = totals

            language_totals = defaultdict(
                lambda: {"examples": 0, "tokens": 0, "families": []}
            )
            for lang, totals in data["language_totals"].items():
                language_totals[lang] = totals

            logger.info(
                f"Checkpoint loaded. Completed families: {data['completed_families']}"
            )
            return (
                data["all_results"],
                family_totals,
                language_totals,
                set(data["completed_families"]),
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    return (
        [],
        defaultdict(lambda: {"examples": 0, "tokens": 0}),
        defaultdict(lambda: {"examples": 0, "tokens": 0, "families": []}),
        set(),
    )


def save_intermediate_results(all_results, family_totals, language_totals):
    """Save intermediate results to CSV and JSON."""
    logger.info(f"Saving intermediate results...")

    # Save CSV
    with open(OUTPUT_CSV_PATH, "w", newline="") as csvfile:
        fieldnames = [
            "family",
            "language",
            "num_examples",
            "total_tokens",
            "avg_tokens_per_example",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in sorted(all_results, key=lambda x: (x["family"], x["language"])):
            writer.writerow(result)

        # Add summary rows
        writer.writerow({})
        writer.writerow(
            {
                "family": "FAMILY TOTALS",
                "language": "",
                "num_examples": "",
                "total_tokens": "",
                "avg_tokens_per_example": "",
            }
        )

        for family, totals in sorted(family_totals.items()):
            avg_tokens = (
                totals["tokens"] / totals["examples"] if totals["examples"] > 0 else 0
            )
            writer.writerow(
                {
                    "family": family,
                    "language": "TOTAL",
                    "num_examples": totals["examples"],
                    "total_tokens": totals["tokens"],
                    "avg_tokens_per_example": round(avg_tokens, 2),
                }
            )

    # Save detailed JSON
    grand_total_examples = sum(t["examples"] for t in family_totals.values())
    grand_total_tokens = sum(t["tokens"] for t in family_totals.values())
    grand_avg = (
        grand_total_tokens / grand_total_examples if grand_total_examples > 0 else 0
    )

    detailed_output = {
        "tokenizer": TOKENIZER_PATH,
        "per_language_results": all_results,
        "family_totals": {
            family: dict(totals) for family, totals in family_totals.items()
        },
        "language_totals": {
            lang: dict(totals) for lang, totals in language_totals.items()
        },
        "grand_totals": {
            "total_examples": grand_total_examples,
            "total_tokens": grand_total_tokens,
            "avg_tokens_per_example": round(grand_avg, 2),
            "num_families": len(family_totals),
            "num_languages": len(language_totals),
        },
        "status": "in_progress",
    }

    with open(DETAILED_OUTPUT_PATH, "w") as f:
        json.dump(detailed_output, f, indent=2)

    logger.info(
        f"Intermediate results saved to {OUTPUT_CSV_PATH} and {DETAILED_OUTPUT_PATH}"
    )


def analyze_all_datasets():
    """Analyze all processed datasets and calculate exact token counts."""
    logger.info("Starting exact token analysis of processed datasets...")
    logger.info(f"Processed data path: {PROCESSED_DATA_PATH}")
    logger.info(f"Loading tokenizer: {TOKENIZER_PATH}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        logger.info(f"Tokenizer loaded successfully: {tokenizer.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return

    # Load checkpoint if exists
    all_results, family_totals, language_totals, completed_families = load_checkpoint()

    processed_path = Path(PROCESSED_DATA_PATH)

    if not processed_path.exists():
        logger.error(f"Processed data path does not exist: {processed_path}")
        return

    # Iterate through each family
    for family, languages in LANG_FAMILIES.items():
        # Skip if already completed
        if family in completed_families:
            logger.info(f"Skipping {family} (already completed)")
            continue

        family_path = processed_path / family

        if not family_path.exists():
            logger.warning(f"Family path not found: {family_path}")
            continue

        logger.info(f"\nProcessing {family} family...")

        # Try to load as DatasetDict
        try:
            dataset_dict = DatasetDict.load_from_disk(str(family_path))
            logger.info(
                f"Loaded {family} as DatasetDict with languages: {list(dataset_dict.keys())}"
            )

            # Analyze each language in the dataset
            for lang in dataset_dict.keys():
                if lang in languages:
                    result = analyze_language_tokens(
                        family, lang, dataset_dict[lang], tokenizer
                    )
                    all_results.append(result)

                    # Update totals
                    family_totals[family]["examples"] += result["num_examples"]
                    family_totals[family]["tokens"] += result["total_tokens"]
                    language_totals[lang]["examples"] += result["num_examples"]
                    language_totals[lang]["tokens"] += result["total_tokens"]
                    language_totals[lang]["families"].append(family)
                else:
                    logger.warning(
                        f"Language {lang} found in dataset but not in config"
                    )

        except Exception as e:
            logger.error(f"Failed to load {family} as DatasetDict: {e}")

            # Try loading individual language datasets
            for lang in languages:
                lang_path = family_path / lang
                if lang_path.exists():
                    try:
                        dataset = load_from_disk(str(lang_path))
                        result = analyze_language_tokens(
                            family, lang, dataset, tokenizer
                        )
                        all_results.append(result)

                        # Update totals
                        family_totals[family]["examples"] += result["num_examples"]
                        family_totals[family]["tokens"] += result["total_tokens"]
                        language_totals[lang]["examples"] += result["num_examples"]
                        language_totals[lang]["tokens"] += result["total_tokens"]
                        language_totals[lang]["families"].append(family)

                    except Exception as lang_error:
                        logger.error(f"Failed to load {family}/{lang}: {lang_error}")

        # Mark family as completed
        completed_families.add(family)

        # Save checkpoint and intermediate results after each family
        save_checkpoint(all_results, family_totals, language_totals, completed_families)
        save_intermediate_results(all_results, family_totals, language_totals)

        logger.info(
            f"✓ Completed {family} family. Progress: {len(completed_families)}/{len(LANG_FAMILIES)}"
        )

    # Write final results to CSV
    logger.info(f"\nWriting final results to CSV: {OUTPUT_CSV_PATH}")

    with open(OUTPUT_CSV_PATH, "w", newline="") as csvfile:
        fieldnames = [
            "family",
            "language",
            "num_examples",
            "total_tokens",
            "avg_tokens_per_example",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in sorted(all_results, key=lambda x: (x["family"], x["language"])):
            writer.writerow(result)

        # Add summary rows
        writer.writerow({})  # Empty row
        writer.writerow(
            {
                "family": "FAMILY TOTALS",
                "language": "",
                "num_examples": "",
                "total_tokens": "",
                "avg_tokens_per_example": "",
            }
        )

        for family, totals in sorted(family_totals.items()):
            avg_tokens = (
                totals["tokens"] / totals["examples"] if totals["examples"] > 0 else 0
            )
            writer.writerow(
                {
                    "family": family,
                    "language": "TOTAL",
                    "num_examples": totals["examples"],
                    "total_tokens": totals["tokens"],
                    "avg_tokens_per_example": round(avg_tokens, 2),
                }
            )

        # Add grand total
        writer.writerow({})  # Empty row
        grand_total_examples = sum(t["examples"] for t in family_totals.values())
        grand_total_tokens = sum(t["tokens"] for t in family_totals.values())
        grand_avg = (
            grand_total_tokens / grand_total_examples if grand_total_examples > 0 else 0
        )

        writer.writerow(
            {
                "family": "GRAND TOTAL",
                "language": "ALL",
                "num_examples": grand_total_examples,
                "total_tokens": grand_total_tokens,
                "avg_tokens_per_example": round(grand_avg, 2),
            }
        )

    logger.info(f"CSV file created: {OUTPUT_CSV_PATH}")

    # Create final detailed JSON output
    detailed_output = {
        "tokenizer": TOKENIZER_PATH,
        "per_language_results": all_results,
        "family_totals": {
            family: dict(totals) for family, totals in family_totals.items()
        },
        "language_totals": {
            lang: dict(totals) for lang, totals in language_totals.items()
        },
        "grand_totals": {
            "total_examples": grand_total_examples,
            "total_tokens": grand_total_tokens,
            "avg_tokens_per_example": round(grand_avg, 2),
            "num_families": len(family_totals),
            "num_languages": len(language_totals),
        },
        "status": "completed",
    }

    with open(DETAILED_OUTPUT_PATH, "w") as f:
        json.dump(detailed_output, f, indent=2)

    logger.info(f"Detailed JSON output created: {DETAILED_OUTPUT_PATH}")

    # Clean up checkpoint file
    if Path(CHECKPOINT_PATH).exists():
        os.remove(CHECKPOINT_PATH)
        logger.info("Checkpoint file removed (analysis completed)")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXACT TOKEN ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tokenizer used: {TOKENIZER_PATH}")
    logger.info(f"Total families processed: {len(family_totals)}")
    logger.info(f"Total languages processed: {len(language_totals)}")
    logger.info(f"Total examples: {grand_total_examples:,}")
    logger.info(f"Total tokens (exact): {grand_total_tokens:,}")
    logger.info(f"Average tokens per example: {grand_avg:.2f}")
    logger.info("=" * 60)

    logger.info("\nFamily breakdown:")
    for family, totals in sorted(family_totals.items()):
        logger.info(
            f"  {family}: {totals['tokens']:,} tokens ({totals['examples']:,} examples)"
        )

    logger.info("\nToken analysis completed successfully!")


def main():
    """Main function."""
    logger.info(f"Initial memory: {MemoryManager.get_memory_usage()}")

    try:
        analyze_all_datasets()
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise

    logger.info(f"Final memory: {MemoryManager.get_memory_usage()}")


if __name__ == "__main__":
    main()
