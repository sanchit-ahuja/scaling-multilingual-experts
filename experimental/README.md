# Experimental / Archived Scripts

The scripts in this directory were used at various points during the research
cycle but are **not** part of the current public pipeline. They are preserved
for reference and may or may not still run without modification.

## Contents

### Superseded / pre-Gemma (XGLM-era) code
- `main.py` — placeholder entry point
- `eval.py` — standalone XGLM-era evaluation script
- `data_preprocessing.py` — XGLM-era preprocessing
- `tokenize_calculation.py` — XGLM-era token accounting
- `vocab_reinitalization.py` — vocabulary reinit experiment (never integrated)
- `feature_analysis.py` — lang2vec-based analysis
- `cleanup_data.py` — older data cleanup utility
- `sample_tokenized_data.py` — dataset subsampling helper
- `train_trl.py` — earlier training design, superseded by top-level `train.py`
- `merge_fixed.py` — earlier results merge, superseded by `aggregate_downstream_results.py`
- `train_archive/` — snapshots of earlier training scripts

### Post-processing / results utilities
- `combine_belebele_results.py`
- `process_flores_results.py`
- `parse_eval_results.py`

### Data pipeline (MADLAD-specific one-offs)
- `process_madlad.py`
- `analyze_processed_tokens.py`
- `create_dataset_buckets.py`

### Language-clustering exploration
- `clustering/` — k-means / random language clustering experiments

### Companion SLURM scripts (`slurm/`)
SLURM wrappers for the scripts above:
`slurm_comprehensive_eval.sh`, `slurm_count_tokens*.sh`, `slurm_process_data*.sh`,
`slurm_create_dataset_buckets.sh`, `slurm_copy_ds*.sh`, `slurm_install_flash_attn3.sh`.

## Use at your own risk

These scripts contain cluster-specific paths and may reference environment
assumptions that no longer hold. Treat them as historical artifacts, not
supported entry points.
