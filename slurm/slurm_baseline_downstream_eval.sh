#!/bin/bash
#SBATCH --job-name=baseline_lm_eval
#SBATCH --output=logs/baseline_lm_eval_%j.out
#SBATCH --error=logs/baseline_lm_eval_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G

# =============================================================================
# NOTE: The SBATCH directives above are templated for NCSA Delta. Edit the
#       partition, account, gres, time, etc. for your own cluster before use.
# =============================================================================

# Source common setup (venv, HF login, env vars)
source "$(dirname "$0")/common.sh"
hf_login

export LMEVAL_LOG_LEVEL="DEBUG"

# Parse command line arguments
CHECKPOINT_PATH=""
DATASET=""
FAMILY=""
BATCH_SIZE="32"

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --family)
            FAMILY="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT_PATH" ] || [ -z "$DATASET" ]; then
    echo "Error: --checkpoint_path and --dataset are required"
    echo "Usage: sbatch $0 --checkpoint_path <path> --dataset <dataset> [--family <family>] [--batch_size <size>]"
    echo "Example: sbatch $0 --checkpoint_path \$CHECKPOINTS_ROOT/Indic_gemma_4b_expert/reverted-7000 --dataset global_piqa_completions --family Indic"
    exit 1
fi

# Extract a unique identifier from checkpoint path for output naming
# Remove leading path and replace / with _
CHECKPOINT_NAME=$(basename $(dirname "$CHECKPOINT_PATH"))_$(basename "$CHECKPOINT_PATH")
# For base models like google/gemma-3-4b-pt, handle differently
if [[ "$CHECKPOINT_PATH" == *"/"*"/"* ]] && [[ "$CHECKPOINT_PATH" != "/"* ]]; then
    # This is likely a HF model ID (e.g., google/gemma-3-4b-pt)
    CHECKPOINT_NAME=$(echo "$CHECKPOINT_PATH" | tr '/' '_')
fi

# Build output and cache path components
RESULTS_BASE="${RESULTS_BASE:-${DATA_ROOT}}"
if [ -n "$FAMILY" ]; then
    OUTPUT_PATH="${RESULTS_BASE}/baseline_lm_eval_${FAMILY}_${CHECKPOINT_NAME}_${DATASET}"
    CACHE_PATH="${RESULTS_BASE}/${FAMILY}_${CHECKPOINT_NAME}_${DATASET}_cache"
else
    OUTPUT_PATH="${RESULTS_BASE}/baseline_lm_eval_${CHECKPOINT_NAME}_${DATASET}"
    CACHE_PATH="${RESULTS_BASE}/${CHECKPOINT_NAME}_${DATASET}_cache"
fi

echo "Running evaluation with:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Dataset: $DATASET"
echo "  Family: ${FAMILY:-<not specified>}"
echo "  Batch size: $BATCH_SIZE"
echo "  Output path: $OUTPUT_PATH"
echo "  Cache path: $CACHE_PATH"

# Run evaluation
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=$CHECKPOINT_PATH \
    --tasks $DATASET \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --output_path "$OUTPUT_PATH" \
    --use_cache "$CACHE_PATH"
