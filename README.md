# Parameter Alignment Mitigates Catastrophic Forgetting in Multilingual Expert Language Models

This repository implements the experiments in *Parameter Alignment Mitigates Catastrophic Forgetting in Multilingual Expert Language Models*.

---

### Base model

All experiments in the paper use [`google/gemma-3-4b-pt`](https://huggingface.co/google/gemma-3-4b-pt) (4B params, 34 layers). The multimodal vision tower is auto-stripped by `utils.py` so only the text sub-network is loaded and trained.

---

## Repository layout

```
x-elm-v2/
├── train.py                   # Main training + evaluation entry point (pyrallis CLI)
├── tokenize_data.py           # Tokenize MADLAD data per family into Arrow datasets
├── model_soup.py              # Average weights across expert checkpoints
├── utils.py                   # Model loading (incl. Gemma multimodal strip)
├── regularization.py          # L2-SP / Layer-Range L2-SP / Freeze / Revert
├── analyze_divergence.py      # Weight-space divergence analysis across strategies
├── upload_to_hf.py            # Push trained checkpoints to HuggingFace Hub
├── aggregate_downstream_results.py  # Aggregate lm-eval + perplexity results into CSVs
├── combine_eval_results.py    # Combine per-benchmark CSVs into comparison tables
├── generate_plots.py          # Produce paper-ready heatmaps + radar plots
├── configs/
│   ├── base.py                # pyrallis dataclasses (Config, ModelConfig, …)
│   ├── constants.py           # LANGS, family / token-budget constants
│   └── yaml/                  # Experiment YAMLs (base, train_gemma_*, revert_gemma_*)
├── slurm/                     # SLURM job scripts (templated for NCSA Delta)
│   └── common.sh              # Shared setup: venv, HF cache, env vars
├── scripts/
│   └── setup_env.sh.example   # Template for .env / environment variables
├── experimental/              # Archived scripts from earlier research cycles
└── requirements.txt
```

---

## Installation

### 1. Python + virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.10+ is recommended (3.11 tested). A recent CUDA toolkit (12.x) is required for GPU training.

### 2. flash-attention (optional but recommended)

`flash-attn` is not installed by `requirements.txt` because it needs the Torch and CUDA toolchain to be present at build time. Install it after `pip install -r requirements.txt`:

```bash
pip install flash-attn>=2.6.0 --no-build-isolation
```

On clusters with custom builds, point at the prebuilt wheel instead.

### 3. Environment variables

Copy the template and edit for your environment:

```bash
cp scripts/setup_env.sh.example .env
# edit .env with your DATA_ROOT, HF_TOKEN, etc.
```

The variables you'll typically set:

| Variable            | Purpose                                    | Default                          |
| ------------------- | ------------------------------------------ | -------------------------------- |
| `PROJECT_ROOT`      | Path to this repo                          | `$HOME/x-elm-v2`                 |
| `DATA_ROOT`         | Root for data + checkpoints + HF cache     | `$PROJECT_ROOT/data`             |
| `TOKENIZED_DATA`    | Tokenized dataset root                     | `$DATA_ROOT/tokenized`           |
| `CHECKPOINTS_ROOT`  | Checkpoint root                            | `$DATA_ROOT/checkpoints`         |
| `HF_HOME`           | HuggingFace cache                          | `$DATA_ROOT/hf_cache`            |
| `HF_TOKEN`          | Required for gated models (e.g. `gemma-3`) | —                                |

`slurm/common.sh` auto-loads `.env` if present.

---

## Quick start

### Tokenize

Download MADLAD-400 for the configured languages and save 95/5 train/valid Arrow splits per family:

```bash
python tokenize_data.py \
    --samples-per-family 10000 \
    --output-path $TOKENIZED_DATA
```

Output layout:

```
$TOKENIZED_DATA/
  ├── slavic/train/{mk,hr,ru,...}/
  ├── slavic/valid/{mk,hr,ru,...}/
  ├── germanic/...
  └── ...
```

### Train an expert

```bash
python train.py --config_path configs/yaml/train_gemma_single_expert.yaml \
    --data.families "[Slavic]" \
    --checkpoint.serialization_dir $CHECKPOINTS_ROOT/slavic_gemma_4b_expert \
    --checkpoint.run_name slavic-expert
```

Swap the YAML for `train_gemma_dense.yaml` (no regularization), `train_gemma_freeze.yaml` (layer freezing), or `train_gemma_layer_range.yaml` (layer-range L2-SP) to get a different strategy.

### Evaluate

Per-language perplexity on a single checkpoint:

```bash
python train.py --config_path configs/yaml/train_gemma_single_expert.yaml \
    --eval.eval_only true \
    --eval.per_language_eval true \
    --checkpoint.checkpoint_path $CHECKPOINTS_ROOT/slavic_gemma_4b_expert/final
```

Downstream evaluation (Belebele, PIQA, FLORES) uses a patched `lm-eval-harness` — see the SLURM section below.

### Revert middle-layer weights post-hoc

Given a trained checkpoint, restore its middle layers to their base-model values (Train-then-Revert):

```bash
python train.py --config_path configs/yaml/revert_gemma_checkpoint.yaml \
    --revert.checkpoint_path $CHECKPOINTS_ROOT/slavic_gemma_4b_expert/final \
    --revert.revert_output_path $CHECKPOINTS_ROOT/slavic_gemma_4b_expert/reverted
```

### Soup the experts

Uniform average of per-family experts:

```bash
python model_soup.py \
    --experts $CHECKPOINTS_ROOT/slavic_gemma_4b_expert/final \
              $CHECKPOINTS_ROOT/germanic_gemma_4b_expert/final \
              $CHECKPOINTS_ROOT/indic_gemma_4b_expert/final \
              $CHECKPOINTS_ROOT/austronesian_gemma_4b_expert/final \
              $CHECKPOINTS_ROOT/romance_gemma_4b_expert/final \
    --output_dir $CHECKPOINTS_ROOT/gemma_4b_expert_soup \
    --alpha 1.0 \
    --anchor_path google/gemma-3-4b-pt
```

---

## Configuration

Config is managed via [pyrallis](https://github.com/eladrich/pyrallis) dataclasses in `configs/base.py`. There are seven config groups:

| Group         | What it controls                                               |
| ------------- | -------------------------------------------------------------- |
| `model`       | Base model, gradient checkpointing, torch.compile              |
| `data`        | Data prefix, families, packing, max length                     |
| `training`    | Batch sizes, LR, warmup, max steps, seed                       |
| `regularizer` | L2-SP / Layer-Range L2-SP / Freeze settings                    |
| `checkpoint`  | Serialization dir, eval/save cadence, resume, early stopping   |
| `eval`        | `eval_only`, `per_language_eval`                               |
| `revert`      | Post-hoc weight reversion inputs/outputs                       |

Any YAML field can be overridden from the CLI via dot-notation:

```bash
python train.py --config_path configs/yaml/train_gemma_dense.yaml \
    --training.lr 1e-4 \
    --training.max_steps 5000 \
    --checkpoint.run_name my-dense-run
```

Pure-CLI mode is also supported (no YAML):

```bash
python train.py \
    --model.model_name google/gemma-3-4b-pt \
    --data.data_prefix $TOKENIZED_DATA \
    --checkpoint.serialization_dir ./checkpoints/my-run
```

### How each strategy maps to config

| Strategy                | Config value                                          | YAML                                 |
| ----------------------- | ----------------------------------------------------- | ------------------------------------ |
| Dense CPT (baseline)    | `regularizer.regularization_type: "none"`             | `train_gemma_dense.yaml`             |
| Family Expert (baseline)| `regularizer.regularization_type: "none"` + single-family `data.families` | `train_gemma_single_expert.yaml` |
| Layer Freezing          | `regularizer.freeze_middle: true`                     | `train_gemma_freeze.yaml`            |
| Layer-Range L2-SP       | `regularizer.regularization_type: "layer_range_l2sp"` | `train_gemma_layer_range.yaml`       |
| Dense / Expert-Reverted | `revert.revert_checkpoint_only: true`                 | `revert_gemma_checkpoint.yaml`       |
| Expert Soup             | Uniform weight averaging                              | `model_soup.py` (post-hoc)           |

Layer boundaries (`first_layers` / `last_layers`) protect the middle layers while letting the outer layers adapt. For **Gemma-3-4B** (34 layers), the middle 19 layers are constrained and the outer layers are trainable. The exact split is set in the YAML (`regularizer.first_layers`, `regularizer.last_layers`).

### Language families

| Family       | Training languages                                                              | Held-out relatives                                         |
| ------------ | ------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Slavic       | mk, hr, ru, sk, sr, uk                                                          | bg, cs, pl, sl, lt, lv                                     |
| Germanic     | af, fy, lb, da, nl, en                                                          | sv, no, is                                                 |
| Indic        | bn, hi, kn, ml, mr, ne, ta, te                                                  | as, gu, or, pa, si, ur, sd                                 |
| Austronesian | sm, jv, ceb, fil, id, ms                                                        | ilo, war, mg, mi, su                                       |
| Romance      | es, pt, fr, gl, it, ro                                                          | ca                                                         |

Training data is drawn from [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400) with a budget of **5B tokens per family** (25B total), distributed equally across the member languages of each family.

---

## SLURM usage

All scripts in `slurm/` are **templated for the NCSA Delta cluster** (partition `ghx4`, account `bfzp-dtai-gh`, H100 GPUs). To use them on a different cluster, edit the `#SBATCH` directives at the top of each script — partition, account, gres, time, and memory — to match your environment.

Common launch patterns:

```bash
# Train one family specialist
sbatch --export=FAMILY=Germanic,CONFIG_TYPE=single_expert \
       slurm/slurm_train_single_experts.sh

# Tokenize data (CPU-only)
sbatch slurm/slurm_tokenize_data.sh

# Per-language perplexity eval on one checkpoint
sbatch slurm/slurm_perplexity_eval.sh \
       --checkpoint_path $CHECKPOINTS_ROOT/germanic_gemma_4b_expert/final

# Batch-submit downstream evals across all checkpoints
bash slurm/slurm_batch_downstream_eval.sh belebele
bash slurm/slurm_batch_downstream_eval.sh flores
bash slurm/slurm_batch_downstream_eval.sh global_piqa_completions
bash slurm/slurm_batch_downstream_eval.sh perplexity

# Weight-space divergence analysis across strategies
bash slurm/slurm_divergence_analysis.sh
```

All scripts source `slurm/common.sh` for venv activation, HF-token login, and path defaults. Override paths by setting them before `sbatch`:

```bash
DATA_ROOT=/my/fast/storage sbatch slurm/slurm_train_single_experts.sh
```

---

## Reproducing the paper

All seven systems (2 baselines + 5 alignment strategies) share the same setup: `google/gemma-3-4b-pt`, 2,048-token sequences, bfloat16 + gradient checkpointing, 95/5 train-valid split, early stopping with patience 6 at 500-step intervals.

1. **Tokenize** MADLAD-400 at 5B tokens/family:
   ```bash
   python tokenize_data.py --samples-per-family 10000 --output-path $TOKENIZED_DATA
   ```
2. **Train the baselines**:
   - *Dense CPT* on all 32 training languages: `train_gemma_dense.yaml`, LR=`5e-5` , up to 50k steps.
   - *Family Expert* (one per family, five total): `train_gemma_single_expert.yaml` with `--data.families "[<Family>]"`, LR=`2e-5`, ~17k steps (~1 epoch).
3. **Train the alignment strategies** per family:
   - *Layer Freezing*: `train_gemma_freeze.yaml`.
   - *Layer-Range L2-SP*: `train_gemma_layer_range.yaml` with $\lambda_{\mathrm{first}}{=}\lambda_{\mathrm{last}}{=}0.001$, $\lambda_{\mathrm{mid}}{=}0.05$.
4. **Apply post-hoc strategies**:
   - *Dense-Reverted* / *Expert-Reverted*: `revert_gemma_checkpoint.yaml` on each trained checkpoint.
   - *Expert Soup*: uniformly average the five non-reverted experts via `model_soup.py`.
5. **Evaluate** every resulting checkpoint (2-shot, all 32 languages + held-out):
   ```bash
   bash slurm/slurm_batch_downstream_eval.sh belebele
   bash slurm/slurm_batch_downstream_eval.sh flores
   bash slurm/slurm_batch_downstream_eval.sh global_piqa_completions
   bash slurm/slurm_batch_downstream_eval.sh perplexity
   ```
6. **Aggregate** results:
   ```bash
   python aggregate_downstream_results.py
   python combine_eval_results.py
   ```
7. **Plot** figures:
   ```bash
   python generate_plots.py
   ```
8. **(Optional) Inspect weight-space divergence** across strategies:
   ```bash
   bash slurm/slurm_divergence_analysis.sh
   ```

---

## Logging

Training logs to [Weights & Biases](https://wandb.ai/) when a `run_name` is set in the config. To disable:

```bash
export WANDB_MODE=disabled
```

Logs are otherwise emitted to stdout and captured by SLURM to `logs/`.

---

## Citation

If you use this codebase, please cite:

<!-- add cite after arxiv -->


---

## License

_TBD — license will be decided at release time (likely MIT or Apache-2.0)._

---

## Acknowledgements

_TBD — compute and collaborator acknowledgements will be added before the camera-ready release._
