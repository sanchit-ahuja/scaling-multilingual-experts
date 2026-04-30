import argparse
import torch
from torch.nn import functional as F
import random
import math
import numpy as np
import gc
import os
from tqdm import tqdm
from pathlib import Path
import multiprocessing
import glob
import time

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk, concatenate_datasets

# Model options for Gemma 3
MODEL_OPTIONS = ["google/gemma-3-1b-pt", "google/gemma-3-3b-pt", "google/gemma-3-9b-pt", "Qwen/Qwen3-1.7B-Base"]

# Language families and their languages (based on your tokenize_data.py)
LANGS = {
    "Slavic": ["mk", "hr", "ru", "sk", "sr", "uk"],
    "Germanic": ["af", "fy", "lb", "da", "nl", "en"],
    "Indic": ["bn", "hi", "kn", "ml", "mr", "ne", "ta", "te"],
    "Austronesian": ["sm", "jv", "ceb", "fil", "id", "ms"],
    "Romance": ["es", "pt", "fr", "ga", "it", "ro"],
}

ALL_LANGS = [lang for langs in LANGS.values() for lang in langs]


def set_training_args(args):
    num_proc = (
        args.num_proc if args.num_proc is not None else multiprocessing.cpu_count()
    )
    num_proc = max(1, int(num_proc * 0.75))
    print(
        f"Using {num_proc} processes for data loading (out of {multiprocessing.cpu_count()} available cores)"
    )

    training_args = TrainingArguments(
        output_dir=args.serialization_dir,
        overwrite_output_dir=True,
        do_train=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bsz,
        per_device_eval_batch_size=args.valid_bsz,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_checkpointing=args.use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True} if args.use_gradient_checkpointing else None,  # Changed to True - faster
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        prediction_loss_only=True,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        save_total_limit=10,
        seed=args.rand_seed,
        bf16=True,
        bf16_full_eval=True,
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        torch_compile=False,
        report_to=["tensorboard", "wandb"],
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
    )
    return training_args, num_proc


def load_tokenized_data(families, languages, data_prefix, rand_seed, data_fraction=1.0):
    """
    Load pre-tokenized data from family-based directory structure.

    Args:
        families: List of language families to load
        languages: List of specific languages to load (optional filter)
        data_prefix: Path to tokenized data directory
        rand_seed: Random seed for shuffling
        data_fraction: Fraction of data to use (0.0 to 1.0), useful for testing
    """
    train_data = []
    valid_data = []

    data_path = Path(data_prefix)

    for family in families:
        family_train_path = data_path / family / "train"
        family_valid_path = data_path / family / "valid"

        if not family_train_path.exists():
            print(f"Warning: Family path not found: {family_train_path}")
            continue

        # Get all languages in this family
        available_langs = [d.name for d in family_train_path.iterdir() if d.is_dir()]

        # Filter by requested languages if specified
        if languages:
            langs_to_load = [lang for lang in available_langs if lang in languages]
        else:
            langs_to_load = available_langs

        print(f"Loading {family} family: {langs_to_load}")

        for lang in langs_to_load:
            try:
                train_lang_path = family_train_path / lang
                valid_lang_path = family_valid_path / lang

                if train_lang_path.exists():
                    t = load_from_disk(str(train_lang_path))

                    # Sample data if data_fraction < 1.0
                    if data_fraction < 1.0:
                        num_samples = max(1, int(len(t) * data_fraction))
                        t = t.shuffle(seed=rand_seed).select(range(num_samples))
                        print(
                            f"  Sampled {num_samples}/{len(load_from_disk(str(train_lang_path)))} examples ({data_fraction*100:.1f}%)"
                        )

                    train_data.append(t)
                    print(f"  Loaded {family}/{lang} train: {len(t)} examples")

                if valid_lang_path.exists():
                    v = load_from_disk(str(valid_lang_path))

                    # Sample validation data if data_fraction < 1.0
                    if data_fraction < 1.0:
                        num_samples = max(1, int(len(v) * data_fraction))
                        v = v.shuffle(seed=rand_seed).select(range(num_samples))
                        print(
                            f"  Sampled {num_samples}/{len(load_from_disk(str(valid_lang_path)))} examples ({data_fraction*100:.1f}%)"
                        )

                    valid_data.append(v)
                    print(f"  Loaded {family}/{lang} valid: {len(v)} examples")

            except Exception as e:
                print(f"Error loading {family}/{lang}: {e}")
                continue

    if not train_data:
        raise ValueError(
            "No training data loaded! Check your data paths and language selections."
        )

    # Concatenate and shuffle
    train_ds = concatenate_datasets(train_data).shuffle(seed=rand_seed)
    valid_ds = concatenate_datasets(valid_data).shuffle(seed=rand_seed)

    print(
        f"\nTotal training examples: {len(train_ds) if hasattr(train_ds, '__len__') else 'infinite'}"
    )
    print(f"Total validation examples: {len(valid_ds)}")
    print(f"Data fraction used: {data_fraction*100:.1f}%")

    return train_ds, valid_ds


def find_latest_checkpoint(serialization_dir):
    """Find the latest checkpoint in the serialization directory."""
    checkpoint_dirs = glob.glob(os.path.join(serialization_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None

    # Sort by checkpoint number
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoint_dirs[-1]


def evaluate_checkpoint(checkpoint_path, tokenizer, valid_ds, args):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        use_cache=False,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    
    eval_args = TrainingArguments(
        output_dir=args.serialization_dir,
        per_device_eval_batch_size=args.valid_bsz,
        bf16=True,
        bf16_full_eval=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to=[],
        do_train=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=valid_ds,
        data_collator=data_collator,
    )

    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss", float("nan"))
    
    if not math.isnan(eval_loss):
        perplexity = math.exp(eval_loss)
        print(f"\nCheckpoint: {checkpoint_path}")
        print(f"Overall Validation loss: {eval_loss:.4f}")
        print(f"Overall Perplexity: {perplexity:.4f}")
    else:
        print(f"\nCheckpoint: {checkpoint_path}")
        print(f"Overall Validation loss: {eval_loss}")
        print(f"Overall Perplexity: N/A")

    return eval_metrics


def evaluate_checkpoint_per_language(checkpoint_path, tokenizer, families, data_prefix, args):
    """Evaluate checkpoint separately for each language."""
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        use_cache=False,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    
    eval_args = TrainingArguments(
        output_dir=args.serialization_dir,
        per_device_eval_batch_size=args.valid_bsz,
        bf16=True,
        bf16_full_eval=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to=[],
        do_train=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    results = []
    
    for family in families:
        family_langs = LANGS.get(family, [])
        
        for lang in family_langs:
            try:
                valid_lang_path = Path(data_prefix) / family / "valid" / lang
                
                if not valid_lang_path.exists():
                    print(f"Skipping {family}/{lang}: path not found")
                    continue
                
                lang_ds = load_from_disk(str(valid_lang_path))
                
                if args.data_fraction < 1.0:
                    num_samples = max(1, int(len(lang_ds) * args.data_fraction))
                    lang_ds = lang_ds.shuffle(seed=args.rand_seed).select(range(num_samples))
                
                print(f"\nEvaluating {family}/{lang} ({len(lang_ds)} examples)...")
                
                trainer = Trainer(
                    model=model,
                    args=eval_args,
                    eval_dataset=lang_ds,
                    data_collator=data_collator,
                )
                
                eval_metrics = trainer.evaluate()
                eval_loss = eval_metrics.get("eval_loss", float("nan"))
                
                if not math.isnan(eval_loss):
                    perplexity = math.exp(eval_loss)
                    print(f"  Loss: {eval_loss:.4f}, Perplexity: {perplexity:.4f}")
                    results.append({
                        "family": family,
                        "language": lang,
                        "eval_loss": eval_loss,
                        "perplexity": perplexity,
                        "num_examples": len(lang_ds)
                    })
                else:
                    print(f"  Loss: {eval_loss}, Perplexity: N/A")
                    results.append({
                        "family": family,
                        "language": lang,
                        "eval_loss": eval_loss,
                        "perplexity": float("nan"),
                        "num_examples": len(lang_ds)
                    })
                    
            except Exception as e:
                print(f"Error evaluating {family}/{lang}: {e}")
                continue
    
    return results


def print_gpu_memory(msg):
    """Print GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(
            f"{msg} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, "
            f"Peak: {max_allocated:.2f}GB, Total: {total:.2f}GB"
        )

def main(args):
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = 2048
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer max length: {tokenizer.model_max_length}")
    print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

    # Check if we should only evaluate existing checkpoint
    if args.eval_only:
        checkpoint_path = args.checkpoint_path

        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(args.serialization_dir)
            if checkpoint_path is None:
                print(f"No checkpoints found in {args.serialization_dir}")
                return
            print(f"Found latest checkpoint: {checkpoint_path}")

        families = args.families if args.families else list(LANGS.keys())
        
        if args.per_language_eval:
            results = evaluate_checkpoint_per_language(
                checkpoint_path, tokenizer, families, args.data_prefix, args
            )
            
            print("\n" + "="*60)
            print("PER-LANGUAGE EVALUATION RESULTS")
            print("="*60)
            for r in results:
                print(f"{r['family']}/{r['language']}: Loss={r['eval_loss']:.4f}, PPL={r['perplexity']:.2f}")
        else:
            _, valid_ds = load_tokenized_data(
                families=families,
                languages=args.languages,
                data_prefix=args.data_prefix,
                rand_seed=args.rand_seed,
                data_fraction=args.data_fraction,
            )
            
            evaluate_checkpoint(checkpoint_path, tokenizer, valid_ds, args)
        
        return

    # Continue with normal training flow
    print(f"Loading model from {args.model_name}...")
    
    torch.cuda.empty_cache()
    gc.collect()
    print_gpu_memory("Before model load")
    
    if args.initialization_dir:
        model = AutoModelForCausalLM.from_pretrained(
            args.initialization_dir,
            use_cache=False,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="cuda",  # Add this
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            use_cache=False,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="cuda",  # Add this
        )

    print_gpu_memory("After model load")
    
    # Only enable gradient checkpointing if needed for memory
    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing: ENABLED (slower but less memory)")
    else:
        print("Gradient checkpointing: DISABLED (faster but more memory)")
    
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    print(f"Model loaded: {model.config}")
    print(f"\nVocab size: {model.config.vocab_size}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load training data
    train_ds, valid_ds = load_tokenized_data(
        families=list(LANGS.keys()) if args.families is None else args.families,
        languages=args.languages,
        data_prefix=args.data_prefix,
        rand_seed=args.rand_seed,
        data_fraction=args.data_fraction,
    )

    print(f"Train dataset columns: {train_ds.column_names}")
    print(f"Dataset size: {len(train_ds)} train, {len(valid_ds)} valid")

    # Set up training arguments
    training_args, num_proc = set_training_args(args)

    # Create data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # DEBUG: Time trainer creation
    print("\n=== Timing trainer creation ===")
    t0 = time.time()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
    )
    
    t1 = time.time()
    print(f"Trainer creation took: {t1-t0:.2f}s")
    print_gpu_memory("After trainer creation")

    # DEBUG: Time a single forward pass manually
    print("\n=== Timing single batch ===")
    model.train()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    print_gpu_memory("Before batch loading")
    
    dataloader = trainer.get_train_dataloader()
    batch_iter = iter(dataloader)
    
    # Warmup: Run 3 batches without timing
    print("Warming up (3 batches)...")
    for _ in range(3):
        batch = next(batch_iter)
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # Now time 5 batches
    print("Timing 5 batches after warmup...")
    forward_times = []
    backward_times = []
    
    for i in range(5):
        batch = next(batch_iter)
        batch = {k: v.cuda() for k, v in batch.items()}
        
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
        torch.cuda.synchronize()
        t1 = time.time()
        forward_times.append((t1 - t0) * 1000)
        
        torch.cuda.synchronize()
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.time()
        backward_times.append((t1 - t0) * 1000)
        
        model.zero_grad()
    
    print(f"Forward times (ms): {forward_times}")
    print(f"Backward times (ms): {backward_times}")
    print(f"Avg Forward: {sum(forward_times)/len(forward_times):.2f}ms")
    print(f"Avg Backward: {sum(backward_times)/len(backward_times):.2f}ms")
    print(f"Avg Total: {(sum(forward_times)+sum(backward_times))/len(forward_times):.2f}ms")
    print_gpu_memory("After timing")
    print("=" * 40)

    # Train model
    print("\nStarting training...")
    print_gpu_memory("Before training")
    
    resume_from_checkpoint = (
        args.resume_from_checkpoint if hasattr(args, "resume_from_checkpoint") else None
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Evaluate
    print("\nRunning final evaluation...")
    eval_metrics = trainer.evaluate(valid_ds)
    print(f"Eval metrics: {eval_metrics}")

    eval_loss = eval_metrics.get("eval_loss", float("nan"))
    if not math.isnan(eval_loss):
        perplexity = math.exp(eval_loss)
        print(f"Final validation loss: {eval_loss:.4f}")
        print(f"Final perplexity: {perplexity:.4f}")

    # Save final model
    print(f"\nSaving model to {args.serialization_dir}/final...")
    trainer.save_model(f"{args.serialization_dir}/final")
    print("Model saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Gemma 3 on multilingual tokenized data"
    )

    # Model and paths
    parser.add_argument(
        "--serialization_dir",
        default="/scratch/user/gemma_checkpoints",
        type=str,
        help="Path to save model checkpoints",
    )
    parser.add_argument(
        "--initialization_dir",
        type=str,
        help="Path to starting model checkpoint (optional)",
    )
    parser.add_argument(
        "--model_name",
        default="google/gemma-3-1b-pt",
        choices=MODEL_OPTIONS,
        help="Gemma model to use",
    )
    parser.add_argument(
        "--data_prefix",
        type=str,
        required=True,
        help="Path to tokenized data directory (e.g., /scratch/user/madlad-tokenized-dataset)",
    )

    # Evaluation mode
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate existing checkpoint without training",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Specific checkpoint path to evaluate (if not provided, uses latest checkpoint)",
    )
    parser.add_argument(
        "--per_language_eval",
        action="store_true",
        help="Evaluate separately for each language",
    )

    # Language selection
    parser.add_argument(
        "--families",
        nargs="*",
        choices=list(LANGS.keys()),
        help="Language families to train on (default: all)",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        choices=ALL_LANGS,
        help="Specific languages to train on (default: all from selected families)",
    )

    # Data sampling
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=0.001,
        help="Fraction of data to use (0.0 to 1.0). Use 0.01 for 1%% for testing (default: 1.0)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume training from last checkpoint if available",
    )

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--grad_accum", type=int, default=8, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1200, help="Maximum training steps"
    )
    parser.add_argument(
        "--train_bsz", type=int, default=16, help="Training batch size per device"
    )
    parser.add_argument(
        "--valid_bsz", type=int, default=16, help="Validation batch size per device"
    )
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="Evaluation frequency"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Logging frequency"
    )
    parser.add_argument(
        "--save_steps", type=int, default=100, help="Checkpoint save frequency"
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (slower but uses less memory)",
    )

    # Other
    parser.add_argument("--rand_seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for dataset packing (default: auto-detect 75%% of available cores)",
    )

    args = parser.parse_args()
    # Set random seeds
    torch.manual_seed(args.rand_seed)
    os.environ["PYTHONHASHSEED"] = str(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("=" * 60)
    print("Gemma 3 Multilingual Pretraining")
    print("=" * 60)
    print(f"Arguments: {args}")
    print("=" * 60)

    main(args)

# EOF
