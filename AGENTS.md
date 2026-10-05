# AGENTS.md

## Project overview

This repository contains research code for continual pre-training and evaluating
multilingual Gemma experts. Most workflows are expensive: they may download
gated models or datasets, allocate GPUs, submit SLURM jobs, write large
checkpoints, or push artifacts to external services.

## Environment

- Use Python 3.12 and the repository-local `.venv`.
- Create the environment with `uv venv --python 3.12`, then install dependencies
  with `uv pip install -r requirements.txt`.
- Install `flash-attn` separately only when the available CUDA and Torch build
  are compatible.
- Use `scripts/setup_env.sh.example` as the template for local environment
  variables. Never commit `.env`, Hugging Face tokens, W&B keys, or cluster
  credentials.

## Repository map

- `train.py`: main pyrallis-based training, evaluation, and checkpoint-reversion
  entry point.
- `tokenize_data.py`: MADLAD tokenization into per-family Arrow datasets.
- `regularization.py`: L2-SP, layer-range L2-SP, freezing, and weight reversion.
- `utils.py`: model loading, including text-only loading for multimodal Gemma.
- `model_soup.py`: post-hoc checkpoint averaging.
- `configs/base.py` and `configs/yaml/`: configuration dataclasses and experiment
  presets.
- `slurm/`: portable launch scripts. Shared defaults live in
  `slurm/common.sh`.
- `scripts/`: sweep, plotting, and task-list helper scripts.
- `docs/causal_analysis.md`: causal interpolation reproduction notes.

## Working rules

- Preserve the existing pyrallis config structure. Add configuration fields to
  the dataclasses and relevant YAML presets together.
- Treat model IDs, language families, layer counts, checkpoint paths, token
  budgets, and SLURM resources as experiment-sensitive values. Do not change
  them casually.
- Keep `configs/constants.py` and any duplicated language-family mappings in
  sync when editing language coverage.
- Keep SLURM scripts portable through environment overrides. Do not hard-code
  personal paths, tokens, or new cluster-specific values in Python modules.
- Before creating, modifying, or submitting a SLURM job, ask the user for the
  target cluster/site, account or project, partition and QoS, GPU type and
  count, CPU and memory requirements, walltime, storage paths, and environment
  or module setup if those details have not already been provided. Do not
  guess these values or submit a job before the user confirms them.
- Keep site-specific settings outside the public repository where practical:
  pass them through `sbatch` options, environment variables, or a local
  untracked configuration file. Never commit credentials or private cluster
  paths.
- Do not read or print secret values from `.env`.
- Do not launch tokenization, model downloads, training, evaluation sweeps,
  `sbatch`, Hugging Face uploads, or W&B runs unless the user explicitly asks.
  These operations are expensive and may use external services.
- Do not modify generated artifacts under `data/`, `checkpoints/`, `logs/`,
  `results/`, `plots/`, or `divergence_results/`.

## Validation

There is no committed test suite yet. For ordinary Python edits, run:

```bash
python -m compileall \
  train.py tokenize_data.py model_soup.py utils.py regularization.py \
  aggregate_downstream_results.py combine_eval_results.py generate_plots.py \
  analyze_divergence.py upload_to_hf.py configs scripts
```

When the development dependencies are installed, also run:

```bash
ruff check .
```

For changes to a CLI, run its `--help` path when that does not import or
download a gated model. State clearly when full GPU, dataset, or SLURM
validation was not run.
