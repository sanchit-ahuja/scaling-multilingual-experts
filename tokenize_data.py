import json
import logging
import os
import argparse
from pathlib import Path
from typing import List

from datasets import load_dataset
from transformers import AutoTokenizer
import psutil

# Configuration — resolved from environment with sensible fallbacks.
# See scripts/setup_env.sh.example for the full env-var setup.
OUTPUT_BASE_PATH = os.environ.get("TOKENIZED_DATA", "./data/tokenized")
LOCAL_DATA_FILE = os.environ.get("MADLAD_LOCAL_JSONL", "")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "google/gemma-3-4b-pt")

# Default token target per language (5B total / 6 Romance langs ≈ 833M each)
DEFAULT_TOKEN_TARGET = 833_333_333

# Number of docs used to estimate avg tokens/doc before committing to full tokenization
ESTIMATION_SAMPLE_SIZE = 2_000

# In-domain training languages per family (mirrors configs/constants.py).
LANGS = {
    "Slavic":       ["mk", "hr", "ru", "sk", "sr", "uk"],
    "Germanic":     ["af", "fy", "lb", "da", "nl", "en"],
    "Indic":        ["bn", "hi", "kn", "ml", "mr", "ne", "ta", "te"],
    "Austronesian": ["sm", "jv", "ceb", "fil", "id", "ms"],
    "Romance":      ["es", "pt", "fr", "gl", "it", "ro"],
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("tokenization.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def tokenize_function(examples, tokenizer, text_field="text"):
    """Tokenize text examples and add EOS token."""
    tokenized = tokenizer(
        examples[text_field],
        truncation=True,
        max_length=2047,  # Leave room for EOS token
        padding=False,    # Don't pad, let trainer handle it
    )

    # Add EOS token to the end of each sequence
    eos_token_id = tokenizer.eos_token_id
    tokenized["input_ids"] = [ids + [eos_token_id] for ids in tokenized["input_ids"]]
    tokenized["attention_mask"] = [mask + [1] for mask in tokenized["attention_mask"]]

    return tokenized


def process_language_from_hf(
    lang: str,
    family: str,
    tokenizer,
    output_base: Path,
    token_target: int,
):
    """
    Download `lang` from MADLAD-400, check token sufficiency, then tokenize to target.

    Steps:
      1. Download full split from HuggingFace.
      2. Tokenize a small sample to estimate avg tokens/doc.
      3. Warn and exit if estimated total < token_target.
      4. Select ceil(token_target / avg_tpd * 1.1) docs (10% buffer) and tokenize.
      5. Trim to the exact token target via cumulative sum.
      6. 95/5 train–valid split and save alongside existing family data.
    """
    num_proc = max(1, psutil.cpu_count() - 2)

    # ── Step 1: Download ───────────────────────────────────────────────────────
    logger.info(f"Loading {lang} from local file: {LOCAL_DATA_FILE}")
    dataset = load_dataset("json", data_files=LOCAL_DATA_FILE, split="train")
    total_docs = len(dataset)
    logger.info(f"  {total_docs:,} documents available for {lang}")

    # ── Step 2: Estimate total tokens ──────────────────────────────────────────
    sample_size = min(ESTIMATION_SAMPLE_SIZE, total_docs)
    logger.info(f"  Estimating tokens from {sample_size:,}-doc sample...")
    sample = dataset.shuffle(seed=42).select(range(sample_size))
    sample_tok = sample.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        batch_size=500,
        remove_columns=sample.column_names,
        num_proc=num_proc,
        desc=f"Estimating tokens [{lang}]",
    )
    total_sample_tokens = sum(len(ids) for ids in sample_tok["input_ids"])
    avg_tpd = total_sample_tokens / sample_size
    estimated_total = int(avg_tpd * total_docs)

    logger.info(f"  Avg tokens/doc: {avg_tpd:.1f}")
    logger.info(
        f"  Estimated total tokens: {estimated_total:,}  |  target: {token_target:,}"
    )

    # ── Step 3: Sufficiency check ──────────────────────────────────────────────
    if estimated_total < token_target:
        shortfall = token_target - estimated_total
        logger.error(
            f"\n{'!'*60}\n"
            f"INSUFFICIENT DATA for '{lang}':\n"
            f"  Estimated available: {estimated_total:,} tokens\n"
            f"  Required target:     {token_target:,} tokens\n"
            f"  Shortfall:           ~{shortfall:,} tokens "
            f"({shortfall / token_target * 100:.1f}%)\n"
            f"Stopping — no data written.\n"
            f"{'!'*60}"
        )
        return False

    # ── Step 4: Select docs with 10% buffer and tokenize ──────────────────────
    docs_needed = min(int(token_target / avg_tpd * 1.1) + 1_000, total_docs)
    logger.info(
        f"  Selecting {docs_needed:,} docs (target / avg_tpd × 1.1 buffer) for tokenization..."
    )
    selected = dataset.shuffle(seed=42).select(range(docs_needed))
    tokenized = selected.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        batch_size=1_000,
        remove_columns=selected.column_names,
        num_proc=num_proc,
        desc=f"Tokenizing {lang}",
    )

    # ── Step 5: 95/5 split and save ────────────────────────────────────────────
    split_ds = tokenized.train_test_split(test_size=0.05, seed=42)

    family_out = output_base / family
    train_path = family_out / "train" / lang
    valid_path = family_out / "valid" / lang
    train_path.mkdir(parents=True, exist_ok=True)
    valid_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Saving train split → {train_path}")
    split_ds["train"].save_to_disk(str(train_path), num_proc=num_proc)
    logger.info(f"  Saving valid split → {valid_path}")
    split_ds["test"].save_to_disk(str(valid_path), num_proc=num_proc)

    logger.info(
        f"  Done: {len(split_ds['train']):,} train docs, {len(split_ds['test']):,} valid docs"
    )

    # ── Save stats ─────────────────────────────────────────────────────────────
    stats = {
        "language": lang,
        "family": family,
        "source": LOCAL_DATA_FILE,
        "token_target_approx": token_target,
        "docs_selected": docs_needed,
        "train_docs": len(split_ds["train"]),
        "valid_docs": len(split_ds["test"]),
        "avg_tokens_per_doc_estimated": round(avg_tpd, 2),
        "estimated_total_tokens_selected": int(avg_tpd * docs_needed),
        "estimated_total_tokens_available": estimated_total,
        "tokenizer": TOKENIZER_PATH,
    }
    stats_path = family_out / f"tokenization_stats_{lang}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"  Stats saved → {stats_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize languages from MADLAD-400 (HuggingFace) to a token count target"
    )
    parser.add_argument(
        "--token-target",
        type=int,
        default=DEFAULT_TOKEN_TARGET,
        help=(
            f"Token count target per language "
            f"(default: {DEFAULT_TOKEN_TARGET:,})"
        ),
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=list(LANGS.keys()),
        default=None,
        help="Process only these families (default: all). Useful for re-running failures.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=f"Override the output base directory (default: {OUTPUT_BASE_PATH}).",
    )
    args = parser.parse_args()

    logger.info("Starting tokenization process...")
    logger.info(f"Source: {HF_DATASET}")
    logger.info(f"Output base: {args.output_path or OUTPUT_BASE_PATH}")
    logger.info(f"Tokenizer: {TOKENIZER_PATH}")
    logger.info(f"Token target per language: {args.token_target:,}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    logger.info(f"Tokenizer loaded: vocab size = {len(tokenizer):,}")

    output_base = Path(args.output_path if args.output_path else OUTPUT_BASE_PATH)

    families_to_run = {
        k: v for k, v in LANGS.items()
        if args.families is None or k in args.families
    }

    results = {}
    for family, languages in families_to_run.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Family: {family}  ({len(languages)} language(s): {', '.join(languages)})")
        logger.info(f"{'='*60}\n")

        for lang in languages:
            success = process_language_from_hf(
                lang, family, tokenizer, output_base, args.token_target
            )
            results[lang] = success
            status = "✓" if success else "✗"
            logger.info(f"{status} {family}/{lang} {'completed' if success else 'failed'}")

    logger.info(f"\n{'='*60}")
    logger.info("Tokenization Summary")
    logger.info(f"{'='*60}")
    successful = sum(results.values())
    logger.info(f"Languages succeeded: {successful}/{len(results)}")
    for lang, ok in results.items():
        logger.info(f"  {'✓' if ok else '✗'} {lang}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
