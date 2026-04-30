#!/usr/bin/env python3
"""
Dataset bucket processing script for language families.
Ensures balanced representation across languages within each family.
Features: Memory management, resume capability, optimized tokenization.
"""

import os
import json
import logging
import pickle
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing as mp
import threading

import numpy as np
from datasets import DatasetDict, load_from_disk, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import psutil

# Configuration
LANGS = {
    #    "Austronesian": ["sm", "jv", "ceb", "fil", "id", "ms"],
    #    "Indic": ["bn", "hi", "kn", "ml", "mr", "ne", "ta", "te"],
    #    "Slavic": ["mk", "hr", "ru", "sk", "sr", "uk"],
    #    "Romance": ["es", "pt", "ro", "ga", "fr", "it"],
    "Germanic": ["af", "fy", "lb", "da", "nl", "de", "en"],
}

INPUT_BASE_PATH = "/projects/lilac_lab/sanchit/madlad-dataset"
OUTPUT_BASE_PATH = "/projects/lilac_lab/sanchit/processed-madlad-dataset"
CHECKPOINT_PATH = "/projects/lilac_lab/sanchit/processing_checkpoints"
TOKENIZER_PATH = "google/gemma-3-1b-pt"
TOTAL_BUDGET_TOKENS = 100_000_000_000  # 100B tokens
TOKENS_PER_FAMILY = 25_000_000_000  # 25B tokens per family

# Memory management settings
BATCH_SIZE = 20000  # Process texts in batches
MAX_MEMORY_USAGE = 0.8  # 80% of available memory
MIN_FREE_MEMORY_GB = 4  # Keep at least 4GB free

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dataset_processing.log"), logging.StreamHandler()],
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
        memory = MemoryManager.get_memory_usage()
        return (
            memory["percent"] > MAX_MEMORY_USAGE * 100
            or memory["available"] < MIN_FREE_MEMORY_GB
        )

    @staticmethod
    def force_gc():
        """Force garbage collection."""
        gc.collect()
        logger.info(f"Memory after GC: {MemoryManager.get_memory_usage()}")


class CheckpointManager:
    """Manage processing checkpoints for resume capability."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.checkpoint_dir / "progress.json"
        self.dataset_sizes_file = self.checkpoint_dir / "dataset_sizes.pkl"

    def save_progress(self, family: str, status: str, **kwargs):
        """Save processing progress."""
        progress = self.load_progress()
        progress[family] = {
            "status": status,
            "timestamp": str(np.datetime64("now")),
            **kwargs,
        }
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    def load_progress(self) -> Dict:
        """Load processing progress."""
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                return json.load(f)
        return {}

    def save_dataset_sizes(self, family: str, dataset_sizes: Dict[str, int]):
        """Save dataset sizes for a family."""
        all_sizes = self.load_dataset_sizes()
        all_sizes[family] = dataset_sizes
        with open(self.dataset_sizes_file, "wb") as f:
            pickle.dump(all_sizes, f)

    def load_dataset_sizes(self) -> Dict:
        """Load dataset sizes."""
        if self.dataset_sizes_file.exists():
            with open(self.dataset_sizes_file, "rb") as f:
                return pickle.load(f)
        return {}

    def is_family_completed(self, family: str) -> bool:
        """Check if a family is already completed."""
        progress = self.load_progress()
        return progress.get(family, {}).get("status") == "completed"

    def get_completed_families(self) -> List[str]:
        """Get list of completed families."""
        progress = self.load_progress()
        return [
            family
            for family, info in progress.items()
            if info.get("status") == "completed"
        ]


def estimate_avg_tokens_per_example(
    dataset: Dataset, sample_size: int = 1000, text_column: str = "text"
) -> float:
    """Estimate average tokens per example using character-based heuristic."""
    try:
        # Sample random examples
        sample_indices = np.random.choice(
            len(dataset), size=min(sample_size, len(dataset)), replace=False
        )
        sample_data = dataset.select(sample_indices)

        # Get text data
        if isinstance(sample_data[text_column], list):
            texts = sample_data[text_column]
        else:
            texts = [sample_data[text_column]]

        # Use character-based estimation (rough approximation: 1 token ≈ 4 characters)
        char_counts = [len(str(text)) for text in texts if text]
        if not char_counts:
            return 50.0  # Default fallback

        avg_chars = np.mean(char_counts)
        estimated_avg_tokens = avg_chars / 4.0  # Rough approximation

        logger.info(
            f"Estimated avg tokens per example: {estimated_avg_tokens:.1f} (based on {len(char_counts)} samples)"
        )
        return max(estimated_avg_tokens, 10.0)  # Minimum 10 tokens per example

    except Exception as e:
        logger.warning(f"Error in token estimation: {e}, using default")
        return 50.0  # Default fallback


def get_language_dataset_sizes(
    family: str, languages: List[str], checkpoint_manager: CheckpointManager
) -> Dict[str, Dict]:
    """Get dataset sizes (number of examples) for each language in a family."""
    # Check if sizes are already cached
    cached_sizes = checkpoint_manager.load_dataset_sizes()
    if family in cached_sizes:
        logger.info(f"Using cached dataset sizes for {family} family")
        return cached_sizes[family]

    family_path = Path(INPUT_BASE_PATH) / family
    dataset_info = {}

    logger.info(f"Analyzing dataset sizes for {family} family...")

    if not family_path.exists():
        logger.error(f"Family dataset path not found: {family_path}")
        return dataset_info

    try:
        # First, try to load as a DatasetDict (all languages in one folder)
        try:
            dataset_dict = DatasetDict.load_from_disk(str(family_path))
            logger.info(
                f"Loaded {family} family dataset as DatasetDict: {list(dataset_dict.keys())}"
            )
        except Exception as e1:
            logger.info(f"Failed to load as DatasetDict: {e1}")
            # If that fails, try loading each language as separate datasets
            dataset_dict = {}
            for lang in languages:
                lang_path = family_path / lang
                if lang_path.exists():
                    try:
                        dataset = Dataset.load_from_disk(str(lang_path))
                        dataset_dict[lang] = dataset
                        logger.info(f"Loaded {family}/{lang} as individual dataset")
                    except Exception as e2:
                        logger.warning(f"Failed to load {family}/{lang}: {e2}")
                        continue
                else:
                    logger.warning(f"Language path not found: {lang_path}")

            if not dataset_dict:
                logger.error(f"No datasets could be loaded for {family} family")
                return dataset_info

        # Process each language
        for lang in languages:
            if lang not in dataset_dict:
                logger.warning(f"Language {lang} not found in {family} family dataset")
                dataset_info[lang] = {
                    "num_examples": 0,
                    "estimated_avg_tokens": 0,
                    "estimated_total_tokens": 0,
                }
                continue

            try:
                dataset = dataset_dict[lang]
                num_examples = len(dataset)

                # Estimate average tokens per example
                avg_tokens = estimate_avg_tokens_per_example(dataset)
                estimated_total_tokens = num_examples * avg_tokens

                dataset_info[lang] = {
                    "num_examples": num_examples,
                    "estimated_avg_tokens": avg_tokens,
                    "estimated_total_tokens": int(estimated_total_tokens),
                }

                logger.info(
                    f"{family}/{lang}: {num_examples:,} examples, ~{avg_tokens:.1f} tokens/example, ~{estimated_total_tokens:,.0f} total tokens"
                )

            except Exception as e:
                logger.error(f"Error processing {family}/{lang}: {e}")
                dataset_info[lang] = {
                    "num_examples": 0,
                    "estimated_avg_tokens": 0,
                    "estimated_total_tokens": 0,
                }

        # Cache the dataset sizes
        checkpoint_manager.save_dataset_sizes(family, dataset_info)

    except Exception as e:
        logger.error(f"Error loading {family} family dataset: {e}")
        for lang in languages:
            dataset_info[lang] = {
                "num_examples": 0,
                "estimated_avg_tokens": 0,
                "estimated_total_tokens": 0,
            }

    return dataset_info


def sample_dataset_by_examples(dataset: Dataset, target_examples: int) -> Dataset:
    """Sample dataset to target number of examples."""
    if len(dataset) <= target_examples:
        logger.info(
            f"Dataset already within limit: {len(dataset)} <= {target_examples}"
        )
        return dataset

    # Use uniform sampling
    indices = np.random.choice(len(dataset), size=target_examples, replace=False)
    indices = sorted(indices)

    # Select efficiently using HuggingFace's optimized select
    sampled_dataset = dataset.select(indices)

    logger.info(
        f"Sampled {len(sampled_dataset)} examples from {len(dataset)} (target: {target_examples})"
    )

    MemoryManager.force_gc()
    return sampled_dataset


def process_language_family(
    family: str, languages: List[str], checkpoint_manager: CheckpointManager
) -> bool:
    """Process a single language family with checkpointing."""
    try:
        # Check if already completed
        if checkpoint_manager.is_family_completed(family):
            logger.info(f"{family} family already completed, skipping...")
            return True

        checkpoint_manager.save_progress(family, "started")
        logger.info(f"Starting processing for {family} family...")

        # Get dataset sizes for each language
        checkpoint_manager.save_progress(family, "analyzing_sizes")
        dataset_info = get_language_dataset_sizes(family, languages, checkpoint_manager)

        # Find languages with valid data
        valid_languages = {
            lang: info
            for lang, info in dataset_info.items()
            if info["num_examples"] > 0
        }

        if not valid_languages:
            logger.error(f"No valid datasets found for {family} family")
            checkpoint_manager.save_progress(family, "failed", error="no_valid_data")
            return False

        # Calculate target tokens per language to reach exactly TOKENS_PER_FAMILY total
        # Each language gets equal share: TOKENS_PER_FAMILY / number_of_languages
        target_tokens_per_lang = TOKENS_PER_FAMILY // len(valid_languages)

        logger.info(f"{family} family - {len(valid_languages)} valid languages")
        logger.info(
            f"{family} family - target tokens per language: {target_tokens_per_lang:,}"
        )
        logger.info(f"{family} family - total family budget: {TOKENS_PER_FAMILY:,}")

        # Load family dataset (handle both DatasetDict and individual dataset structures)
        checkpoint_manager.save_progress(family, "loading_dataset")
        family_path = Path(INPUT_BASE_PATH) / family

        try:
            # Try loading as DatasetDict first
            dataset_dict = DatasetDict.load_from_disk(str(family_path))
            logger.info(f"Loaded {family} as DatasetDict")
        except Exception:
            # Load individual datasets
            dataset_dict = {}
            for lang in languages:
                if lang in valid_languages:
                    lang_path = family_path / lang
                    if lang_path.exists():
                        try:
                            dataset = Dataset.load_from_disk(str(lang_path))
                            dataset_dict[lang] = dataset
                            logger.info(f"Loaded {family}/{lang} from {lang_path}")
                        except Exception as e:
                            logger.error(f"Failed to load {family}/{lang}: {e}")

            if not dataset_dict:
                logger.error(f"Failed to load any datasets for {family} family")
                checkpoint_manager.save_progress(
                    family, "failed", error="failed_to_load_datasets"
                )
                return False

        # Process each language
        checkpoint_manager.save_progress(family, "processing_languages")
        processed_datasets = {}

        for lang in languages:
            if lang not in valid_languages:
                logger.warning(f"Skipping {family}/{lang} - no valid data")
                continue

            if lang not in dataset_dict:
                logger.warning(f"Language {lang} not found in dataset")
                continue

            logger.info(f"Processing {family}/{lang}...")

            # Calculate target number of examples to achieve exactly target_tokens_per_lang tokens
            lang_info = dataset_info[lang]
            avg_tokens = lang_info["estimated_avg_tokens"]
            target_examples = int(target_tokens_per_lang / avg_tokens)

            max_possible_examples = lang_info["num_examples"]

            # Check if we need to upsample (target > available)
            dataset = dataset_dict[lang]

            if target_examples <= max_possible_examples:
                # Downsample: uniform sampling without replacement
                logger.info(
                    f"{family}/{lang} - downsampling from {max_possible_examples:,} to {target_examples:,} examples"
                )
                sampled_dataset = sample_dataset_by_examples(dataset, target_examples)
            else:
                # Upsample: uniform sampling with replacement
                logger.info(
                    f"{family}/{lang} - upsampling from {max_possible_examples:,} to {target_examples:,} examples"
                )
                # Sample with replacement to reach target
                indices = np.random.choice(
                    max_possible_examples, size=target_examples, replace=True
                )
                indices = sorted(indices)
                sampled_dataset = dataset.select(indices)
                logger.info(
                    f"Upsampled {family}/{lang}: {len(sampled_dataset)} examples (with replacement)"
                )

            processed_datasets[lang] = sampled_dataset

            # Force garbage collection after each language
            MemoryManager.force_gc()

        # Save processed family dataset
        checkpoint_manager.save_progress(family, "saving_results")
        output_family_path = Path(OUTPUT_BASE_PATH) / family
        output_family_path.mkdir(parents=True, exist_ok=True)

        # Create and save DatasetDict
        family_dataset_dict = DatasetDict(processed_datasets)
        family_dataset_dict.save_to_disk(str(output_family_path))
        logger.info(f"Saved processed dataset: {output_family_path}")

        # Calculate actual estimated token counts for verification
        actual_estimated_tokens = {}
        token_deviation_percentages = {}
        total_family_tokens = 0

        for lang, dataset in processed_datasets.items():
            lang_info = dataset_info[lang]
            actual_tokens = len(dataset) * lang_info["estimated_avg_tokens"]
            actual_estimated_tokens[lang] = int(actual_tokens)
            total_family_tokens += actual_tokens

            # Calculate deviation from target
            deviation = (
                abs(actual_tokens - target_tokens_per_lang)
                / target_tokens_per_lang
                * 100
            )
            token_deviation_percentages[lang] = round(deviation, 2)

            logger.info(
                f"{family}/{lang} final: {len(dataset):,} examples, estimated {actual_tokens:,.0f} tokens ({deviation:.1f}% from target)"
            )

        # Log summary of deviations
        if token_deviation_percentages:
            max_deviation = max(token_deviation_percentages.values())
            avg_deviation = sum(token_deviation_percentages.values()) / len(
                token_deviation_percentages
            )
            logger.info(
                f"{family} family token balance - Max deviation: {max_deviation:.1f}%, Avg deviation: {avg_deviation:.1f}%"
            )
            logger.info(
                f"{family} family total tokens: {total_family_tokens:,.0f} (target: {TOKENS_PER_FAMILY:,})"
            )
        else:
            max_deviation = 0
            avg_deviation = 0

        # Save family processing stats
        stats = {
            "family": family,
            "languages": languages,
            "dataset_info": dataset_info,
            "target_tokens_per_lang": target_tokens_per_lang,
            "family_token_budget": TOKENS_PER_FAMILY,
            "processed_languages": list(processed_datasets.keys()),
            "final_sizes": {
                lang: len(dataset) for lang, dataset in processed_datasets.items()
            },
            "actual_estimated_tokens": actual_estimated_tokens,
            "total_family_tokens": int(total_family_tokens),
            "token_deviation_percentages": token_deviation_percentages,
            "max_deviation_percent": max_deviation,
            "avg_deviation_percent": round(avg_deviation, 2),
            "memory_stats": MemoryManager.get_memory_usage(),
        }

        stats_path = output_family_path / "processing_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        checkpoint_manager.save_progress(
            family,
            "completed",
            processed_languages=list(processed_datasets.keys()),
            stats=stats,
        )

        logger.info(f"Completed processing for {family} family")
        return True

    except Exception as e:
        logger.error(f"Error processing {family} family: {e}")
        checkpoint_manager.save_progress(family, "failed", error=str(e))
        return False


def estimate_avg_tokens_per_example(
    dataset: Dataset, sample_size: int = 15000, text_column: str = "text"
) -> float:
    """Estimate average tokens per example using character-based heuristic with larger sample."""
    try:
        # Use larger sample size for better accuracy
        sample_indices = np.random.choice(
            len(dataset), size=min(sample_size, len(dataset)), replace=False
        )
        sample_data = dataset.select(sample_indices)

        # Get text data
        if isinstance(sample_data[text_column], list):
            texts = sample_data[text_column]
        else:
            texts = [sample_data[text_column]]

        # Use character-based estimation with better approximation
        # Research suggests 1 token ≈ 3.5-4 characters for multilingual text
        char_counts = [len(str(text)) for text in texts if text]
        if not char_counts:
            return 50.0  # Default fallback

        avg_chars = np.mean(char_counts)
        # Use 3.7 characters per token (more accurate for multilingual data)
        estimated_avg_tokens = avg_chars / 3.7

        logger.info(
            f"Estimated avg tokens per example: {estimated_avg_tokens:.1f} (based on {len(char_counts)} samples, avg {avg_chars:.1f} chars)"
        )
        return max(estimated_avg_tokens, 10.0)  # Minimum 10 tokens per example

    except Exception as e:
        logger.warning(f"Error in token estimation: {e}, using default")
        return 50.0  # Default fallback


def main():
    """Main processing function with resume capability."""
    logger.info("Starting dataset bucket processing...")
    logger.info(f"Total budget: {TOTAL_BUDGET_TOKENS:,} tokens")
    logger.info(f"Budget per family: {TOKENS_PER_FAMILY:,} tokens")
    logger.info(f"Initial memory: {MemoryManager.get_memory_usage()}")

    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(CHECKPOINT_PATH)

    # Create output directory
    Path(OUTPUT_BASE_PATH).mkdir(parents=True, exist_ok=True)

    # Check which families are already completed
    completed_families = checkpoint_manager.get_completed_families()
    remaining_families = {
        family: languages
        for family, languages in LANGS.items()
        if family not in completed_families
    }

    if completed_families:
        logger.info(f"Already completed families: {completed_families}")

    if not remaining_families:
        logger.info("All families already completed!")
        return

    logger.info(f"Remaining families to process: {list(remaining_families.keys())}")

    # Process families sequentially to better manage memory
    results = {}

    for family, languages in remaining_families.items():
        logger.info(f"Processing family: {family}")
        try:
            success = process_language_family(family, languages, checkpoint_manager)
            results[family] = success
            logger.info(
                f"Family {family} completed: {'SUCCESS' if success else 'FAILED'}"
            )

            # Force cleanup between families
            MemoryManager.force_gc()

        except Exception as e:
            logger.error(f"Family {family} failed with exception: {e}")
            results[family] = False

    # Add completed families to results
    for family in completed_families:
        results[family] = True

    # Summary
    successful = sum(results.values())
    total = len(results)
    logger.info(f"Processing completed: {successful}/{total} families successful")

    # Save overall stats
    overall_stats = {
        "total_budget_tokens": TOTAL_BUDGET_TOKENS,
        "tokens_per_family": TOKENS_PER_FAMILY,
        "language_families": LANGS,
        "processing_results": results,
        "successful_families": successful,
        "total_families": total,
        "completed_families": completed_families,
        "final_memory": MemoryManager.get_memory_usage(),
    }

    stats_path = Path(OUTPUT_BASE_PATH) / "overall_stats.json"
    with open(stats_path, "w") as f:
        json.dump(overall_stats, f, indent=2)

    logger.info(f"Overall statistics saved to: {stats_path}")
    logger.info("Dataset bucket processing completed!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
