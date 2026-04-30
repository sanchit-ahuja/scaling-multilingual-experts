# x-elm-v2 Copilot Instructions

## Project Overview

Multilingual LLM continual pre-training research project focused on preventing catastrophic forgetting when adapting Gemma-3 to new language families.

## Architecture

### Core Components
- **[train.py](../train.py)** - Main training script using TRL's SFTTrainer with sequence packing
- **[regularization.py](../regularization.py)** - Anti-forgetting techniques: L2-SP, LayerRangeL2SP, LayerFreezer, WeightReverter
- **[configs/base.py](../configs/base.py)** - Pyrallis dataclasses for typed configuration
- **[configs/constants.py](../configs/constants.py)** - Language families (LANGS dict) and budget constants

### Configuration System
Uses **pyrallis** for config management. Configs can be:
```bash
# YAML file
python train.py --config_path configs/yaml/train_gemma_dense.yaml

# CLI overrides
python train.py --config_path configs/yaml/base.yaml --training.lr 1e-4

# Pure CLI (no config file)
python train.py --data.data_prefix /path/to/data --checkpoint.serialization_dir /output
```

Config structure in [configs/base.py](../configs/base.py): `Config` → `ModelConfig`, `DataConfig`, `TrainingConfig`, `RegularizerConfig`, `CheckpointConfig`, `EvalConfig`, `RevertConfig`

### Data Pipeline
1. **Raw data** (e.g., MADLAD) → `tokenize_data.py` → Pre-tokenized datasets (Arrow format)
2. **Training** loads from `{data_prefix}/{family}/train/{lang}/` structure

Language families defined in `configs/constants.py`:
```python
LANGS = {"Slavic": [...], "Germanic": [...], "Indic": [...], "Austronesian": [...], "Romance": [...]}
```

## Key Patterns

### Regularization Strategies
Three anti-forgetting approaches in [regularization.py](../regularization.py):

1. **L2-SP** - Penalize weight drift from base model
   ```python
   regularizer = L2SPRegularizer(lambda_l2=0.01)
   regularizer.register_base_model(model)  # MUST call before training
   ```

2. **LayerRangeL2SP** - Different lambdas for first/middle/last layers (protects multilingual layers)
   ```yaml
   regularizer:
     regularization_type: "layer_range_l2sp"
     lambda_first_layers: 0.1   # Strong protection
     lambda_middle_layers: 0.001  # Weak (reasoning layers)
     lambda_last_layers: 0.1
   ```

3. **LayerFreezer / WeightReverter** - Freeze or revert middle layers (Partial SFT/CPT, Train-then-Revert)

### Custom Trainer
`RegularizedTrainer` extends TRL's `SFTTrainer` to add regularization loss during training. Always use this when `regularization_type != "none"`.

### Model Soup
[model_soup.py](../model_soup.py) - Weight interpolation between base and expert: `(1-α)*base + α*expert`

## Model-Specific Notes

The supported base model is **Gemma-3-4B** (`google/gemma-3-4b-pt`, 34 decoder layers). The publicly released checkpoint is multimodal; `utils.load_model()` strips the vision tower and loads only the text sub-network.

`regularization.py` auto-detects `num_layers` by inspecting `model.model.layers`, so it works unchanged on any standard decoder-only transformer (LLaMA, Mistral, etc.) if a user wants to adapt the pipeline.

When switching models, update:
- `model.model_name` in config
- `regularizer.num_layers` (auto-detected; override only if needed)
- `TOKENIZER_PATH` in `tokenize_data.py`

## Development Workflow

### Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### SLURM Jobs
All SLURM scripts in `slurm/`. Source common setup first:
```bash
source slurm/common.sh  # Sets HF_HOME, CUDA, NCCL vars
```

Key scripts:
- `slurm_train.sh` - Single GPU training
- `slurm_train_single_experts.sh` - Train per-family experts

### Adding New Configs
1. Create YAML in `configs/yaml/` (see `train_gemma_layer_range.yaml` for a template)
2. Override only what differs from `base.yaml`

## Weights & Biases Integration

Training automatically logs to W&B. Key conventions:
- **Run naming**: Set `checkpoint.run_name` in config, e.g., `"indic-layer-range-l2sp"`
- **Auto-generated names**: If not set, uses `cpt-{model}-{families}` pattern
- **Logged metrics**: `loss/ce`, `loss/{regularizer_type}`, `loss/total`, throughput
- **Local logs**: Stored in `wandb/run-*/`

To disable W&B, set `WANDB_MODE=disabled` before running.

## Common Gotchas

- **Register regularizer**: Always call `regularizer.register_base_model(model)` AFTER loading model, BEFORE training
- **Data path structure**: Must be `{prefix}/{family}/train/{lang}/` with Arrow datasets
- **`num_layers` auto-detection**: Uses `model.model.layers` or falls back to parsing param names
- **Evaluation samples**: Set `max_eval_samples` to limit eval time (default 10000)
- **Tokenizer mismatch**: Ensure `tokenize_data.py` uses same tokenizer as training model

## File Conventions

- Training configs: `configs/yaml/*.yaml`
- SLURM scripts: `slurm/slurm_*.sh`
- W&B logs: `wandb/run-*/`
