"""
Continual Pre-training script using TRL's SFTTrainer with sequence packing.
Supports multilingual training on pre-tokenized data.
"""

import argparse
import gc
import glob
import json
import math
import multiprocessing
import os
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from regularization import (
    L2SPRegularizer,
    LayerRangeL2SPRegularizer,
    RegularizedTrainer,
    LayerFreezer,
    WeightReverter,
    create_regularizer,
    get_num_layers,
)

# Language families and their languages
LANGS = {
    "Slavic": ["mk", "hr", "ru", "sk", "sr", "uk"],
    "Germanic": ["af", "fy", "lb", "da", "nl", "de", "en"],
    "Indic": ["bn", "hi", "kn", "ml", "mr", "ne", "ta", "te"],
    "Austronesian": ["sm", "jv", "ceb", "fil", "id", "ms"],
    "Romance": ["es", "pt", "fr", "ga", "it", "ro"],
}


class ThroughputCallback(TrainerCallback):
    """Callback to log training throughput (tokens/second)."""
    
    def __init__(self, max_length: int, train_batch_size: int, grad_accum: int):
        self.max_length = max_length
        self.grad_accum = grad_accum
        self.train_batch_size = train_batch_size
        self.tokens_per_step = None  # Will be calculated on train_begin
        self.step_start_time = None
        self.total_tokens = 0
        self.training_start_time = None
        self.first_batch_logged = False
        self.is_main_process = True
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start_time = time.time()
        self.total_tokens = 0
        
        # Get world size from environment or args
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_main_process = (local_rank == 0)
        
        # Correct calculation: batch × accum × seq_len × num_gpus
        self.tokens_per_step = (
            self.train_batch_size * 
            self.grad_accum * 
            self.max_length * 
            world_size
        )
        
        if self.is_main_process and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[Memory] Training start - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            print(f"[Config] World size: {world_size}")
            print(f"[Config] Tokens per step: {self.tokens_per_step:,}")
            print(f"[Config] Effective batch size: {self.train_batch_size * self.grad_accum * world_size}")
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
        
        if not self.first_batch_logged and self.is_main_process and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[Memory] First step begin - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def on_step_end(self, args, state, control, **kwargs):
        if not self.first_batch_logged and self.is_main_process and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[Memory] First step end - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            self.first_batch_logged = True
        
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            step_tokens_per_sec = self.tokens_per_step / step_time
            self.total_tokens += self.tokens_per_step
            
            # Only log from main process
            if self.is_main_process and state.global_step % args.logging_steps == 0:
                elapsed = time.time() - self.training_start_time
                avg_tokens_per_sec = self.total_tokens / elapsed if elapsed > 0 else 0
                
                print(f"[Throughput] Step {state.global_step}: "
                      f"{step_tokens_per_sec:,.0f} tok/s (step), "
                      f"{avg_tokens_per_sec:,.0f} tok/s (avg), "
                      f"{step_time:.2f}s/step")
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.is_main_process and self.training_start_time is not None:
            total_time = time.time() - self.training_start_time
            avg_tokens_per_sec = self.total_tokens / total_time if total_time > 0 else 0
            print(f"\n[Throughput Summary]")
            print(f"  Total tokens: {self.total_tokens:,}")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Average throughput: {avg_tokens_per_sec:,.0f} tokens/second")


class BatchSizeVerifierCallback(TrainerCallback):
    """Callback to verify actual batch sizes during training."""
    
    def __init__(self):
        self.logged_first_batch = False
    
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"\n[BatchSize Config]")
        print(f"  per_device_train_batch_size: {args.per_device_train_batch_size}")
        print(f"  gradient_accumulation_steps: {args.gradient_accumulation_steps}")
        print(f"  effective_batch_size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")


class BatchInspectorTrainer(SFTTrainer):
    """Custom trainer that logs batch information."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_logged = False
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Log batch info on first call
        if not self._batch_logged:
            print(f"\n[Batch Verification]")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: type={type(value)}")
            
            if 'input_ids' in inputs:
                batch_size, seq_len = inputs['input_ids'].shape
                print(f"\n  Actual batch_size: {batch_size}")
                print(f"  Actual seq_length: {seq_len}")
                print(f"  Tokens per micro-batch: {batch_size * seq_len:,}")
            
            self._batch_logged = True
        
        return super().training_step(model, inputs, num_items_in_batch)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_gpu_memory(msg: str):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[GPU] {msg} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def load_tokenized_data(
    families: List[str],
    data_prefix: str,
    rand_seed: int,
    data_fraction: float = 1.0,
    max_eval_samples: int = 10000,  # Add this parameter
) -> tuple[Dataset, Dataset, dict]:
    """
    Load pre-tokenized data from family-based directory structure.
    
    Args:
        families: List of language families to load
        data_prefix: Path to tokenized data directory
        rand_seed: Random seed for shuffling
        data_fraction: Fraction of data to use (0.0 to 1.0)
        max_eval_samples: Maximum number of evaluation samples (to speed up eval)
    
    Returns:
        Tuple of (train_dataset, valid_dataset, stats)
    """
    train_data = []
    valid_data = []
    data_path = Path(data_prefix)
    
    stats = {"families": {}, "total_train": 0, "total_valid": 0}
    
    for family in families:
        family_train_path = data_path / family / "train"
        family_valid_path = data_path / family / "valid"
        
        if not family_train_path.exists():
            print(f"Warning: Family path not found: {family_train_path}")
            continue
        
        # Get all languages in this family
        available_langs = [d.name for d in family_train_path.iterdir() if d.is_dir()]
        print(f"\nLoading {family} family: {available_langs}")
        
        family_stats = {"languages": {}}
        
        for lang in available_langs:
            try:
                train_lang_path = family_train_path / lang
                valid_lang_path = family_valid_path / lang
                
                lang_stats = {}
                
                # Load train data
                if train_lang_path.exists():
                    t = load_from_disk(str(train_lang_path))
                    original_size = len(t)
                    
                    if data_fraction < 1.0:
                        num_samples = max(1, int(len(t) * data_fraction))
                        t = t.shuffle(seed=rand_seed).select(range(num_samples))
                        print(f"  {lang} train: sampled {num_samples:,}/{original_size:,} ({data_fraction*100:.1f}%)")
                    else:
                        print(f"  {lang} train: {len(t):,} examples")
                    
                    train_data.append(t)
                    lang_stats["train"] = len(t)
                
                # Load valid data
                if valid_lang_path.exists():
                    v = load_from_disk(str(valid_lang_path))
                    original_size = len(v)
                    
                    if data_fraction < 1.0:
                        num_samples = max(1, int(len(v) * data_fraction))
                        v = v.shuffle(seed=rand_seed).select(range(num_samples))
                        print(f"  {lang} valid: sampled {num_samples:,}/{original_size:,} ({data_fraction*100:.1f}%)")
                    else:
                        print(f"  {lang} valid: {len(v):,} examples")
                    
                    valid_data.append(v)
                    lang_stats["valid"] = len(v)
                
                family_stats["languages"][lang] = lang_stats
                
            except Exception as e:
                print(f"Error loading {family}/{lang}: {e}")
                continue
        
        stats["families"][family] = family_stats
    
    if not train_data:
        raise ValueError("No training data loaded! Check your data paths.")
    
    # Concatenate datasets
    print("\nConcatenating datasets...")
    train_ds = concatenate_datasets(train_data)
    valid_ds = concatenate_datasets(valid_data) if valid_data else None
    
    # Limit eval dataset size for faster evaluation
    if valid_ds and max_eval_samples and len(valid_ds) > max_eval_samples:
        print(f"Limiting eval dataset: {len(valid_ds):,} -> {max_eval_samples:,} samples")
        valid_ds = valid_ds.shuffle(seed=rand_seed).select(range(max_eval_samples))
    
    stats["total_train"] = len(train_ds)
    stats["total_valid"] = len(valid_ds) if valid_ds else 0
    
    print(f"Total examples: {len(train_ds):,} train, {len(valid_ds) if valid_ds else 0:,} valid")
    
    return train_ds, valid_ds, stats


def find_latest_checkpoint(serialization_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the serialization directory."""
    # Check for final checkpoint first
    final_path = os.path.join(serialization_dir, "final")
    if os.path.isdir(final_path):
        return final_path
    
    # Otherwise find the latest numbered checkpoint
    checkpoint_dirs = glob.glob(os.path.join(serialization_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoint_dirs[-1]


def create_sft_config(args) -> SFTConfig:
    """Create SFTConfig from arguments."""
    
    # Determine number of workers - be conservative to avoid overhead
    if args.num_proc is not None:
        num_workers = args.num_proc
    else:
        # Use a reasonable default, not all CPUs
        num_workers = min(8, max(1, multiprocessing.cpu_count() // 4))
    
    print(f"Using {num_workers} dataloader workers")
    
    config = SFTConfig(
        # Output
        output_dir=args.serialization_dir,
        overwrite_output_dir=True,
        
        # Training
        do_train=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_bsz,
        per_device_eval_batch_size=args.valid_bsz,
        gradient_accumulation_steps=args.grad_accum,
        
        # Optimization
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        
        # Memory optimization
        gradient_checkpointing=args.use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.use_gradient_checkpointing else None,
        
        # Precision
        bf16=True,
        bf16_full_eval=True,
        
        # Evaluation & Logging
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        
        # Data loading - conservative settings
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2 if num_workers > 0 else None,
        
        # Reporting
        report_to=["tensorboard", "wandb"],
        run_name=args.run_name,
        
        # Misc
        seed=args.seed,
        prediction_loss_only=True,
        
        # SFT specific - TRL handles packing
        max_length=args.max_length,
        packing=args.packing,
        packing_strategy=args.packing_strategy,
        dataset_num_proc=num_workers,
        dataset_text_field=None,
        
        # DDP settings
        ddp_find_unused_parameters=False,
        
        # Performance
        torch_compile=args.torch_compile,
    )
    
    return config


def evaluate_checkpoint(checkpoint_path: str, tokenizer, valid_ds, args):
    """Evaluate a single checkpoint."""
    print(f"\nEvaluating checkpoint: {checkpoint_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    
    # Simple eval config
    eval_config = SFTConfig(
        output_dir=args.serialization_dir,
        per_device_eval_batch_size=args.valid_bsz,
        bf16=True,
        bf16_full_eval=True,
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        do_train=False,
        do_eval=True,
        max_length=args.max_length,
        packing=args.packing,
        report_to=[],
    )
    
    trainer = SFTTrainer(
        model=model,
        args=eval_config,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
    )
    
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss", float("nan"))
    
    if not math.isnan(eval_loss):
        perplexity = math.exp(eval_loss)
        print(f"Validation loss: {eval_loss:.4f}")
        print(f"Perplexity: {perplexity:.4f}")
    else:
        print(f"Validation loss: {eval_loss}")
        print(f"Perplexity: N/A")
    
    return eval_metrics


def evaluate_checkpoint_per_language(
    checkpoint_path: str, 
    tokenizer, 
    families: List[str], 
    data_prefix: str, 
    args
) -> List[dict]:
    """Evaluate checkpoint separately for each language."""
    print(f"\nEvaluating checkpoint per language: {checkpoint_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    
    # Use regular Trainer, not SFTTrainer for pre-tokenized data
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    
    eval_args = TrainingArguments(
        output_dir=args.serialization_dir,
        per_device_eval_batch_size=args.valid_bsz,
        bf16=True,
        bf16_full_eval=True,
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        report_to=[],
        do_train=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    results = []
    data_path = Path(data_prefix)
    
    for family in families:
        family_langs = LANGS.get(family, [])
        
        for lang in family_langs:
            try:
                valid_lang_path = data_path / family / "valid" / lang
                
                if not valid_lang_path.exists():
                    print(f"Skipping {family}/{lang}: path not found")
                    continue
                
                lang_ds = load_from_disk(str(valid_lang_path))
                
                if args.data_fraction < 1.0:
                    num_samples = max(1, int(len(lang_ds) * args.data_fraction))
                    lang_ds = lang_ds.shuffle(seed=args.seed).select(range(num_samples))
                
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
                import traceback
                traceback.print_exc()
                continue
    
    return results


def revert_checkpoint_weights(
    base_model_path: str,
    checkpoint_path: str,
    output_path: str,
    first_n: int = 6,
    last_n: int = 4,
    num_layers: Optional[int] = None,
    attention_only: bool = False,
) -> None:
    """
    Apply weight reversion to an already trained checkpoint.
    
    This loads a base model and a trained checkpoint, then reverts the middle
    layers of the trained checkpoint to the base model weights (Train-then-Revert
    strategy applied post-hoc).
    
    Args:
        base_model_path: Path to the base/original model
        checkpoint_path: Path to the trained checkpoint to revert
        output_path: Path to save the reverted model
        first_n: Number of first layers to keep trained
        last_n: Number of last layers to keep trained  
        num_layers: Total number of transformer layers (auto-detected if None)
        attention_only: If True, only revert MLP weights (keep trained attention)
    """
    print("\n" + "=" * 60)
    print("WEIGHT REVERSION MODE")
    print("=" * 60)
    print(f"Base model: {base_model_path}")
    print(f"Trained checkpoint: {checkpoint_path}")
    print(f"Output path: {output_path}")
    print(f"First layers (keep trained): {first_n}")
    print(f"Last layers (keep trained): {last_n}")
    print(f"Attention only: {attention_only}")
    print("=" * 60)
    
    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    
    # Auto-detect number of layers
    detected_num_layers = get_num_layers(base_model, override=num_layers)
    
    # Create reverter and register base model weights
    reverter = WeightReverter(
        first_n=first_n,
        last_n=last_n,
        num_layers=detected_num_layers,
        attention_only=attention_only,
    )
    reverter.register(base_model)
    
    # Free base model from GPU memory (weights are stored on CPU)
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load trained checkpoint
    print("\nLoading trained checkpoint...")
    trained_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    
    # Apply reversion
    reverter.revert(trained_model)
    reverter.clear()  # Free CPU memory
    
    # Save reverted model
    print(f"\nSaving reverted model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    trained_model.save_pretrained(output_path)
    
    # Also copy tokenizer if it exists in checkpoint
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    for tf in tokenizer_files:
        src = Path(checkpoint_path) / tf
        if src.exists():
            import shutil
            shutil.copy(src, Path(output_path) / tf)
            print(f"Copied {tf}")
    
    # If no tokenizer in checkpoint, try to copy from base model
    if not (Path(output_path) / "tokenizer.json").exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            tokenizer.save_pretrained(output_path)
            print("Saved tokenizer from base model")
        except Exception as e:
            print(f"Warning: Could not save tokenizer: {e}")
    
    # Save reversion config for reference
    revert_config = {
        "base_model_path": base_model_path,
        "checkpoint_path": checkpoint_path,
        "first_n": first_n,
        "last_n": last_n,
        "num_layers": detected_num_layers,
        "attention_only": attention_only,
        "middle_layers_reverted": f"{first_n}-{detected_num_layers - last_n - 1}",
    }
    config_path = Path(output_path) / "reversion_config.json"
    with open(config_path, "w") as f:
        json.dump(revert_config, f, indent=2)
    print(f"Saved reversion config to: {config_path}")
    
    print("\n" + "=" * 60)
    print("WEIGHT REVERSION COMPLETE")
    print("=" * 60)
    print(f"Reverted model saved to: {output_path}")
    print(f"Middle layers {first_n}-{detected_num_layers - last_n - 1} restored to base weights")
    print(f"First {first_n} and last {last_n} layers retain trained weights")
    print("=" * 60 + "\n")


def main(args):
    # Set seeds
    set_seed(args.seed)
    
    # Check if we should only revert weights on existing checkpoint
    if args.revert_checkpoint_only:
        checkpoint_path = args.checkpoint_path
        
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(args.serialization_dir)
            if checkpoint_path is None:
                print(f"No checkpoints found in {args.serialization_dir}")
                return
            print(f"Found latest checkpoint: {checkpoint_path}")
        
        # Determine output path
        if args.revert_output_path:
            output_path = args.revert_output_path
        else:
            # Default: create a 'reverted' subdirectory
            output_path = str(Path(checkpoint_path).parent / "reverted")
        
        revert_checkpoint_weights(
            base_model_path=args.model_name,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            first_n=args.first_layers,
            last_n=args.last_layers,
            num_layers=args.num_layers,
            attention_only=args.attention_only,
        )
        return
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = args.max_length
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Vocab size: {len(tokenizer):,}")
    print(f"Max length: {tokenizer.model_max_length}")
    print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    
    # Determine families to load
    families = args.families if args.families else list(LANGS.keys())
    print(f"\nFamilies to load: {families}")
    
    # Check if we should only evaluate existing checkpoint
    if args.eval_only:
        checkpoint_path = args.checkpoint_path
        
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(args.serialization_dir)
            if checkpoint_path is None:
                print(f"No checkpoints found in {args.serialization_dir}")
                return
            print(f"Found latest checkpoint: {checkpoint_path}")
        
        if args.per_language_eval:
            results = evaluate_checkpoint_per_language(
                checkpoint_path, tokenizer, families, args.data_prefix, args
            )
            
            print("\n" + "="*60)
            print("PER-LANGUAGE EVALUATION RESULTS")
            print("="*60)
            for r in results:
                print(f"{r['family']}/{r['language']}: Loss={r['eval_loss']:.4f}, PPL={r['perplexity']:.2f}")
            
            # Save results
            results_path = Path(args.serialization_dir) / "per_language_eval_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {results_path}")
        else:
            _, valid_ds, _ = load_tokenized_data(
                families=families,
                data_prefix=args.data_prefix,
                rand_seed=args.seed,
                data_fraction=args.data_fraction,
                max_eval_samples=args.max_eval_samples if args.max_eval_samples > 0 else None,
            )
            
            evaluate_checkpoint(checkpoint_path, tokenizer, valid_ds, args)
        
        return
    
    # Initialize W&B run name if not provided
    if args.run_name is None:
        families_str = "-".join(args.families) if args.families else "all"
        args.run_name = f"cpt-{args.model_name.split('/')[-1]}-{families_str}"
    
    print("=" * 60)
    print("Continual Pre-training with TRL")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Families: {args.families if args.families else 'all'}")
    print(f"Data fraction: {args.data_fraction * 100:.1f}%")
    print(f"Max length: {args.max_length}")
    print(f"Packing: {args.packing} (strategy: {args.packing_strategy})")
    print(f"torch.compile: {args.torch_compile}")
    print("=" * 60)
    
    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    train_ds, valid_ds, data_stats = load_tokenized_data(
        families=families,
        data_prefix=args.data_prefix,
        rand_seed=args.seed,
        data_fraction=args.data_fraction,
        max_eval_samples=args.max_eval_samples if args.max_eval_samples > 0 else None,
    )
    
    # Save data stats
    os.makedirs(args.serialization_dir, exist_ok=True)
    stats_path = Path(args.serialization_dir) / "data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(data_stats, f, indent=2)
    print(f"Data stats saved to: {stats_path}")
    
    # Load model
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    
    torch.cuda.empty_cache()
    gc.collect()
    print_gpu_memory("Before model load")
    
    model_path = args.initialization_dir if args.initialization_dir else args.model_name
    print(f"Loading model from: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    
    print_gpu_memory("After model load")
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Auto-detect number of layers if not specified
    num_layers = get_num_layers(model, override=args.num_layers)
    last_layer_start = num_layers - args.last_layers
    
    # Initialize layer freezer and weight reverter
    freezer = None
    reverter = None
    
    if args.freeze_middle:
        print("\n" + "=" * 60)
        print("SETTING UP LAYER FREEZING (Partial SFT/CPT)")
        print("=" * 60)
        freezer = LayerFreezer(
            first_n=args.first_layers,
            last_n=args.last_layers,
            num_layers=num_layers,
            attention_only=args.attention_only,
        )
        freezer.freeze(model)
        
        # Recalculate trainable params after freezing
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters after freezing: {trainable_params:,}")
    
    if args.revert_middle:
        print("\n" + "=" * 60)
        print("SETTING UP WEIGHT REVERSION (Train-then-Revert)")
        print("=" * 60)
        reverter = WeightReverter(
            first_n=args.first_layers,
            last_n=args.last_layers,
            num_layers=num_layers,
            attention_only=args.attention_only,
        )
        reverter.register(model)
    
    # Create training config
    sft_config = create_sft_config(args)
    
    # Calculate effective batch size
    effective_batch_size = args.train_bsz * args.grad_accum
    tokens_per_step = effective_batch_size * args.max_length
    print(f"\nEffective batch size: {effective_batch_size}")
    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Total training tokens (approx): {args.max_steps * tokens_per_step:,}")
    
    # Create callbacks
    throughput_callback = ThroughputCallback(
        max_length=args.max_length,
        train_batch_size=args.train_bsz,
        grad_accum=args.grad_accum,
    )
    batch_verifier_callback = BatchSizeVerifierCallback()
    
    # Create regularizer if requested
    regularizer = None
    if args.regularization_type != "none":
        print(f"\n=== Setting up {args.regularization_type.upper()} Regularization ===")
        
        # For EWC, we need to compute Fisher information first
        fisher_diag = None
        if args.regularization_type == "ewc":
            if args.fisher_path and os.path.exists(args.fisher_path):
                print(f"Loading pre-computed Fisher from {args.fisher_path}")
                fisher_diag = torch.load(args.fisher_path)
            else:
                print(f"Computing Fisher information with {args.fisher_samples} samples...")
                # Create a dataloader for Fisher computation
                fisher_loader = torch.utils.data.DataLoader(
                    train_ds.select(range(min(args.fisher_samples, len(train_ds)))),
                    batch_size=args.per_device_train_batch_size,
                    shuffle=True,
                )
                
                # Compute Fisher using the model AFTER loading
                temp_regularizer = EWCRegularizer(
                    lambda_ewc=args.lambda_l2,
                    exclude_patterns=args.l2_exclude_patterns,
                    include_patterns=args.l2_include_patterns,
                )
                fisher_diag = temp_regularizer.compute_fisher(model, fisher_loader)
                
                # Save Fisher for future use
                fisher_save_path = os.path.join(args.serialization_dir, "fisher_info.pt")
                torch.save(fisher_diag, fisher_save_path)
                print(f"Saved Fisher information to {fisher_save_path}")
        
        regularizer = create_regularizer(
            regularization_type=args.regularization_type,
            base_model=model,
            lambda_reg=args.lambda_l2,
            exclude_patterns=args.l2_exclude_patterns,
            include_patterns=args.l2_include_patterns,
            fisher_diag=fisher_diag,
            teacher_model=model if args.regularization_type == "kl" else None,
            temperature=args.kl_temperature,
            # Layer-range L2-SP parameters
            lambda_first=args.lambda_first_layers,
            lambda_middle=args.lambda_middle_layers,
            lambda_last=args.lambda_last_layers,
            first_layer_end=args.first_layers,
            last_layer_start=last_layer_start,
            num_layers=num_layers,
            attention_only=args.attention_only,
        )
        
        # IMPORTANT: Register base model weights AFTER model is loaded
        # but BEFORE training starts
        print(f"Registering base model weights for regularization...")
        regularizer.register_base_model(model)
        print(f"=== Regularization Setup Complete ===\n")

    # Create trainer (use RegularizedTrainer for forgetting mitigation)
    print("\n" + "=" * 60)
    print("CREATING TRAINER")
    print("=" * 60)
    
    trainer = RegularizedTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        regularizer=regularizer,
        callbacks=[throughput_callback, batch_verifier_callback],
    )
    
    print_gpu_memory("After trainer creation")
    
    # Check for checkpoint to resume from
    resume_from = None
    if args.resume_from_checkpoint:
        if isinstance(args.resume_from_checkpoint, str) and os.path.isdir(args.resume_from_checkpoint):
            resume_from = args.resume_from_checkpoint
        else:
            resume_from = find_latest_checkpoint(args.serialization_dir)
        
        if resume_from:
            print(f"Resuming from checkpoint: {resume_from}")
        else:
            print("No checkpoint found to resume from, starting fresh")
    
    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print_gpu_memory("Before training")
    
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training metrics: {train_result.metrics}")
    print_gpu_memory("After training")
    
    # Apply Train-then-Revert strategy if requested
    if reverter is not None:
        print("\n" + "=" * 60)
        print("APPLYING WEIGHT REVERSION")
        print("=" * 60)
        reverter.revert(model)
        reverter.clear()  # Free CPU memory
    
    # Unfreeze layers before saving (for complete checkpoint)
    if freezer is not None:
        print("\n[Unfreezing layers before saving checkpoint]")
        freezer.unfreeze(model)
    
    # Final evaluation
    if valid_ds and args.eval_steps > 0:
        print("\nRunning final evaluation...")
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss", float("nan"))
        
        if not math.isnan(eval_loss):
            perplexity = math.exp(eval_loss)
            print(f"Final validation loss: {eval_loss:.4f}")
            print(f"Final perplexity: {perplexity:.4f}")
        
        # Save eval metrics
        eval_path = Path(args.serialization_dir) / "final_eval_metrics.json"
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
    
    # Save final model
    final_path = Path(args.serialization_dir) / "final"
    print(f"\nSaving final model to: {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    # Save training args
    args_path = final_path / "training_args.json"
    os.makedirs(final_path, exist_ok=True)
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continual pre-training with TRL SFTTrainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model and paths
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-1b-pt",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--initialization_dir",
        type=str,
        default=None,
        help="Path to initialize model from (overrides model_name for weights)",
    )
    parser.add_argument(
        "--data_prefix",
        type=str,
        default=None,
        help="Path to tokenized data directory (required for training/eval)",
    )
    parser.add_argument(
        "--serialization_dir",
        type=str,
        default=None,
        help="Path to save checkpoints and outputs (required for training)",
    )
    
    # Data selection
    parser.add_argument(
        "--families",
        nargs="*",
        choices=list(LANGS.keys()),
        default=None,
        help="Language families to train on (default: all)",
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="Fraction of data to use (0.0 to 1.0)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=10000,
        help="Maximum number of eval samples (default: 10000, use -1 for all)",
    )

    # Sequence handling
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        default=True,
        help="Enable sequence packing (default: True)",
    )
    parser.add_argument(
        "--no_packing",
        action="store_true",
        help="Disable sequence packing",
    )
    parser.add_argument(
        "--packing_strategy",
        type=str,
        default="wrapped",
        choices=["wrapped", "bfd"],
        help="Packing strategy: 'wrapped' (simple concat) or 'bfd' (best-fit-decreasing)",
    )
    
    # Training hyperparameters
    parser.add_argument("--train_bsz", type=int, default=16, help="Training batch size per device")
    parser.add_argument("--valid_bsz", type=int, default=16, help="Validation batch size per device")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    
    # Checkpointing and logging
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency (0 to disable)")
    parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint save frequency")
    parser.add_argument("--save_total_limit", type=int, default=10, help="Maximum checkpoints to keep")
    
    # Memory optimization
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (slower but uses less memory)",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Enable torch.compile for faster training (has warmup overhead)",
    )
    
    # Resumption
    parser.add_argument(
        "--resume_from_checkpoint",
        nargs="?",
        const=True,
        default=False,
        help="Resume from checkpoint (optionally specify path)",
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
        default=None,
        help="Specific checkpoint path to evaluate (if not provided, uses latest checkpoint)",
    )
    parser.add_argument(
        "--per_language_eval",
        action="store_true",
        help="Evaluate separately for each language",
    )
    
    # Weight reversion mode (post-hoc Train-then-Revert)
    parser.add_argument(
        "--revert_checkpoint_only",
        action="store_true",
        help="Only apply weight reversion to existing checkpoint (no training). "
             "Reverts middle layers to base model weights (Train-then-Revert strategy).",
    )
    parser.add_argument(
        "--revert_output_path",
        type=str,
        default=None,
        help="Output path for reverted model (default: checkpoint_path/../reverted)",
    )
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of dataloader/dataset workers")
    parser.add_argument("--run_name", type=str, default=None, help="Weights & Biases run name")
    
    # Regularization arguments
    parser.add_argument(
        "--regularization_type",
        type=str,
        default="none",
        choices=["none", "l2sp", "layer_range_l2sp", "ewc", "kl"],
        help="Type of regularization to use for catastrophic forgetting mitigation",
    )
    parser.add_argument(
        "--lambda_l2",
        type=float,
        default=0.01,
        help="L2-SP regularization strength (default: 0.01)",
    )
    parser.add_argument(
        "--l2_exclude_patterns",
        nargs="*",
        default=None,
        help="Regex patterns for layer names to EXCLUDE from L2 regularization (e.g., 'embed' 'lm_head')",
    )
    parser.add_argument(
        "--l2_include_patterns",
        nargs="*",
        default=None,
        help="Regex patterns for layer names to INCLUDE in L2 regularization (if set, only matching layers are regularized)",
    )
    
    # EWC-specific arguments
    parser.add_argument(
        "--fisher_samples",
        type=int,
        default=1000,
        help="Number of samples to compute Fisher information for EWC (default: 1000)",
    )
    parser.add_argument(
        "--fisher_path",
        type=str,
        default=None,
        help="Path to pre-computed Fisher information file for EWC",
    )
    
    # KL distillation-specific arguments
    parser.add_argument(
        "--kl_temperature",
        type=float,
        default=2.0,
        help="Temperature for KL distillation softmax (default: 2.0, higher = softer distributions)",
    )
    
    # Layer-range L2-SP arguments
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Number of transformer layers (auto-detected if not specified)",
    )
    parser.add_argument(
        "--first_layers",
        type=int,
        default=6,
        help="Number of first layers to protect/train (default: 6)",
    )
    parser.add_argument(
        "--last_layers",
        type=int,
        default=4,
        help="Number of last layers to protect/train (default: 4)",
    )
    parser.add_argument(
        "--lambda_first_layers",
        type=float,
        default=0.1,
        help="Regularization strength for first layers (default: 0.1)",
    )
    parser.add_argument(
        "--lambda_middle_layers",
        type=float,
        default=0.001,
        help="Regularization strength for middle layers (default: 0.001)",
    )
    parser.add_argument(
        "--lambda_last_layers",
        type=float,
        default=0.1,
        help="Regularization strength for last layers (default: 0.1)",
    )
    parser.add_argument(
        "--attention_only",
        action="store_true",
        help="Only apply regularization/freezing/reversion to attention (not MLP)",
    )
    
    # Layer freezing and reversion strategies
    parser.add_argument(
        "--freeze_middle",
        action="store_true",
        help="Freeze middle layers during training (Partial SFT/CPT strategy)",
    )
    parser.add_argument(
        "--revert_middle",
        action="store_true",
        help="Revert middle layers to base weights after training (Train-then-Revert strategy)",
    )
    
    args = parser.parse_args()
    
    # Handle --no_packing flag
    if args.no_packing:
        args.packing = False
    
    # Validate required arguments based on mode
    if args.revert_checkpoint_only:
        # Revert mode only needs model_name (base model) and checkpoint_path
        if args.checkpoint_path is None and args.serialization_dir is None:
            parser.error("--revert_checkpoint_only requires either --checkpoint_path or --serialization_dir")
    elif args.eval_only:
        # Eval mode needs data_prefix and either checkpoint_path or serialization_dir
        if args.data_prefix is None:
            parser.error("--eval_only requires --data_prefix")
        if args.checkpoint_path is None and args.serialization_dir is None:
            parser.error("--eval_only requires either --checkpoint_path or --serialization_dir")
    else:
        # Training mode needs data_prefix and serialization_dir
        if args.data_prefix is None:
            parser.error("Training requires --data_prefix")
        if args.serialization_dir is None:
            parser.error("Training requires --serialization_dir")
    
    main(args)
