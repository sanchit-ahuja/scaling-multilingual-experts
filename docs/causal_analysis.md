# Causal Layer-Drift Analysis

This document reproduces the interpolation experiments used to test whether
layer-group movement causes catastrophic forgetting after dense continual
pre-training (CPT).

## Intervention

For one target layer group at a time, keep the dense CPT checkpoint fixed
except for the selected layers and construct:

```text
theta_target(alpha) = theta_base + alpha * (theta_cpt - theta_base)
```

`alpha=0` fully restores the selected layers to the base model. `alpha=1`
recovers the original dense CPT checkpoint. For Gemma-3-4B, the default split
is first layers `0-8`, middle layers `9-27`, and last layers `28-33`.

The primary causal comparison is the dense first/middle/last sweep. The
expert-soup sweep is included as a secondary check.

## Harness dependency

Use the project [`lm-evaluation-harness`
fork](https://github.com/sanchit-ahuja/lm-evaluation-harness) in editable mode:

```bash
git clone https://github.com/sanchit-ahuja/lm-evaluation-harness
uv pip install -e ./lm-evaluation-harness
```

Its required FLORES-specific changes are documented beside the task
definitions:

```text
lm_eval/tasks/flores/README.md
```

Point the sweep helper at that checkout:

```bash
export FLORES_DIR=/path/to/lm-evaluation-harness/lm_eval/tasks/flores
```

For FLORES, the harness must generate with relaxed `until: []`, then apply the
registered `flores_first_line` filter before BLEU and ChrF aggregation. This
avoids scoring prompt continuations while preserving translations from models
that emit leading whitespace or a leading newline.

## Dense Belebele sweep

Set paths in `.env`, then submit one job for each layer group:

```bash
for target in first middle last; do
  sbatch --export=ALL,TARGET_LAYERS="$target",TASKS=BELEBELE_HELDIN,OUTPUT_DIR="$DATA_ROOT/drift_alpha_sweep/dense_${target}_belebele_all" \
    slurm/slurm_middle_layer_alpha_sweep.sh
done
```

Each output directory contains `summary.jsonl`. Regenerate the CSVs and plots:

```bash
python scripts/plot_belebele_layer_group_sweeps.py \
  --first-summary "$DATA_ROOT/drift_alpha_sweep/dense_first_belebele_all/summary.jsonl" \
  --middle-summary "$DATA_ROOT/drift_alpha_sweep/dense_middle_belebele_all/summary.jsonl" \
  --last-summary "$DATA_ROOT/drift_alpha_sweep/dense_last_belebele_all/summary.jsonl"
```

## Dense FLORES sweep

Run FLORES with `--log_samples` enabled. New harness runs compute paper-facing
ChrF directly using the FLORES first-line filter. Sample logs are still needed
to audit old runs and compare raw versus normalized generations.

```bash
for target in first middle last; do
  sbatch --export=ALL,TARGET_LAYERS="$target",TASKS=FLORES_HELDIN,LOG_SAMPLES=true,OUTPUT_DIR="$DATA_ROOT/drift_alpha_sweep/dense_${target}_flores_heldin" \
    slurm/slurm_middle_layer_alpha_sweep.sh
done
```

Plot the metrics recorded by the harness:

```bash
python scripts/plot_flores_layer_group_sweeps.py \
  --flores-dir "$FLORES_DIR" \
  --first-summary "$DATA_ROOT/drift_alpha_sweep/dense_first_flores_heldin/summary.jsonl" \
  --middle-summary "$DATA_ROOT/drift_alpha_sweep/dense_middle_flores_heldin/summary.jsonl" \
  --last-summary "$DATA_ROOT/drift_alpha_sweep/dense_last_flores_heldin/summary.jsonl"
```

For runs produced before the harness filter was added, re-score the logged
samples offline with the same truncation rule and plot the audited results:

```bash
python scripts/retruncate_flores_alpha_sweep.py \
  --root "$DATA_ROOT/drift_alpha_sweep" \
  --output-dir results
python scripts/plot_flores_fixed_alpha_sweep.py \
  --task-csv results/flores_alpha_sweep_fixed_task.csv
```

## Expert-soup check

The expert-soup script interpolates each family expert toward the base model on
the selected layers, then uniformly averages the experts. Override
`EXPERT_MODELS` when checkpoint names differ from the defaults.

```bash
sbatch --export=ALL,TARGET_LAYERS=middle,TASKS=BELEBELE_HELDIN,OUTPUT_DIR="$DATA_ROOT/drift_alpha_sweep/expert_soup_middle_belebele_all" \
  slurm/slurm_expert_soup_alpha_sweep.sh
sbatch --export=ALL,TARGET_LAYERS=all,TASKS=BELEBELE_HELDIN,OUTPUT_DIR="$DATA_ROOT/drift_alpha_sweep/expert_soup_all_belebele_all" \
  slurm/slurm_expert_soup_alpha_sweep.sh

python scripts/plot_expert_soup_alpha_sweeps.py \
  --middle_summary "$DATA_ROOT/drift_alpha_sweep/expert_soup_middle_belebele_all/summary.jsonl" \
  --all_summary "$DATA_ROOT/drift_alpha_sweep/expert_soup_all_belebele_all/summary.jsonl"
```