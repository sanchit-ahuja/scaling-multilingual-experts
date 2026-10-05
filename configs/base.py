"""Base configuration dataclasses for pyrallis-based config management."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for model loading and initialization."""
    
    # Model identifier (HuggingFace name or local path)
    model_name: str = "google/gemma-3-1b-pt"
    
    # Optional: load weights from a different directory (for continual pretraining)
    initialization_dir: Optional[str] = None
    
    # Memory optimization
    use_gradient_checkpointing: bool = False
    torch_compile: bool = False


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Path to tokenized data directory
    data_prefix: str = ""
    
    # Language families to train on (None = all families)
    families: Optional[List[str]] = None
    
    # Fraction of data to use (for debugging/testing)
    data_fraction: float = 1.0
    
    # Maximum evaluation samples (-1 for all)
    max_eval_samples: int = 10000
    
    # Sequence handling
    max_length: int = 2048
    packing: bool = True
    packing_strategy: str = "wrapped"  # "wrapped" or "bfd"
    
    # Data loading workers
    num_proc: int = 4


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    # Batch sizes
    train_bsz: int = 16
    valid_bsz: int = 16
    grad_accum: int = 8
    
    # Training schedule
    max_steps: int = 1000
    warmup_steps: int = 100
    
    # Optimizer settings
    lr: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Random seed
    seed: int = 42


@dataclass
class RegularizerConfig:
    """Configuration for regularization to prevent catastrophic forgetting.
    
    Supports two regularization types:
    - "l2sp": Standard L2-SP with single lambda for all layers
    - "layer_range_l2sp": Different lambdas for first/middle/last layers
    - "none": No regularization
    
    For layer_range_l2sp, use lambda_first/middle/last.
    For l2sp, only lambda_l2 is used.
    """
    
    # Regularization type: "none", "l2sp", or "layer_range_l2sp"
    regularization_type: str = "none"
    
    # L2-SP parameters (used when type="l2sp")
    lambda_l2: float = 0.01
    
    # Layer-range L2-SP parameters (used when type="layer_range_l2sp")
    lambda_first_layers: float = 0.1
    lambda_middle_layers: float = 0.001
    lambda_last_layers: float = 0.1
    
    # Layer range boundaries
    num_layers: Optional[int] = None  # Auto-detected if not specified
    first_layers: int = 6  # Number of first layers to protect
    last_layers: int = 4   # Number of last layers to protect
    
    # Pattern-based filtering
    exclude_patterns: Optional[List[str]] = None
    include_patterns: Optional[List[str]] = None
    
    # Attention-only regularization
    attention_only: bool = False
    
    # Layer freezing strategy (Partial SFT/CPT)
    freeze_middle: bool = False
    
    # Weight reversion strategy (Train-then-Revert)
    revert_middle: bool = False


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing and logging."""
    
    # Output directory for checkpoints
    serialization_dir: str = "./checkpoints"
    
    # Logging frequency
    logging_steps: int = 10
    
    # Evaluation frequency (0 to disable)
    eval_steps: int = 500
    
    # Checkpoint saving
    save_steps: int = 500
    save_total_limit: int = 10
    
    # Resume from checkpoint
    resume_from_checkpoint: bool = False
    checkpoint_path: Optional[str] = None
    
    # Weights & Biases run name
    run_name: Optional[str] = None
    
    # Weights & Biases project and entity
    wandb_project: Optional[str] = None  # W&B project name
    wandb_entity: Optional[str] = None   # W&B team/entity name
    
    # Early stopping
    early_stopping_patience: int = 0  # 0 to disable early stopping
    early_stopping_threshold: float = 0.0  # Minimum improvement required
    early_stopping_metric: str = "eval_loss"  # Metric to monitor


@dataclass
class EvalConfig:
    """Configuration for evaluation mode."""
    
    # Only evaluate, don't train
    eval_only: bool = False
    
    # Evaluate each language separately
    per_language_eval: bool = False

    # Evaluate the explicit held-out language split instead of training
    # languages. Requires the corresponding held-out tokenized dataset.
    heldout: bool = False


@dataclass
class RevertConfig:
    """Configuration for post-hoc weight reversion."""
    revert_checkpoint_only: bool = False
    checkpoint_path: Optional[str] = None
    revert_output_path: Optional[str] = None
    base_model_path: Optional[str] = None


@dataclass
class Config:
    """Main configuration combining all sub-configs.
    
    Usage with pyrallis:
        @pyrallis.wrap()
        def main(cfg: Config):
            ...
    
    Or load from YAML:
        cfg = pyrallis.load(Config, open("config.yaml"))
    """
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    regularizer: RegularizerConfig = field(default_factory=RegularizerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    revert: RevertConfig = field(default_factory=RevertConfig)
    
    def to_flat_dict(self) -> dict:
        """Convert nested config to flat dictionary (for backward compatibility)."""
        flat = {}
        for section_name in ["model", "data", "training", "regularizer", "checkpoint", "eval", "revert"]:
            section = getattr(self, section_name)
            for key, value in vars(section).items():
                flat[key] = value
        return flat
