#!/bin/bash
#SBATCH --job-name=perplexity_eval
#SBATCH --output=logs/perplexity_eval_%j.out
#SBATCH --error=logs/perplexity_eval_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --requeue

# =============================================================================
# Per-language Perplexity Evaluation Worker Script
#
# Evaluates a single checkpoint on all 5 language families using train.py.
# Produces per_language_eval_results.json with per-language loss & perplexity.
#
# Usage (submitted by slurm_batch_downstream_eval.sh):
#   sbatch slurm_perplexity_eval.sh --checkpoint_path <path>
#
# NOTE: SBATCH directives above are templated for NCSA Delta. Edit for your
#       own cluster (partition, account, gres, time, etc.).
# =============================================================================

set -euo pipefail

# Slurm copies this script to a spool directory; use absolute paths rather
# than resolving common.sh relative to $0. This evaluation uses local
# checkpoints and the tokenized dataset, so the established ARM64 venv is
# sufficient and avoids an unnecessary Hugging Face login.
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${VENV:-${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/.venv}/bin/activate"
export DATA_ROOT="${DATA_ROOT:-${DATA_ROOT:-data}}"
export TOKENIZED_DATA="${TOKENIZED_DATA:-${DATA_ROOT}/tokenized}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"
mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

# Parse command line arguments
CHECKPOINT_PATH=""
FAMILY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --family)
            FAMILY="$2"
            shift 2
            ;;
        --dataset)
            # Accepted but ignored — not applicable for perplexity
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: --checkpoint_path is required"
    echo "Usage: sbatch $0 --checkpoint_path <path>"
    exit 1
fi

# Configuration
DATA_PREFIX="${DATA_PREFIX:-${TOKENIZED_DATA}}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-4b-pt}"

# Build output directory name (same naming convention as batch script)
CHECKPOINT_NAME=$(basename $(dirname "$CHECKPOINT_PATH"))_$(basename "$CHECKPOINT_PATH")
# Handle HuggingFace model IDs (e.g., google/gemma-3-4b-pt)
if [[ "$CHECKPOINT_PATH" == *"/"*"/"* ]] && [[ "$CHECKPOINT_PATH" != "/"* ]]; then
    CHECKPOINT_NAME=$(echo "$CHECKPOINT_PATH" | tr '/' '_')
fi

OUTPUT_DIR="${RESULTS_BASE:-${DATA_ROOT}}/perplexity_eval_${CHECKPOINT_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "PERPLEXITY EVALUATION"
echo "=========================================="
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Data prefix: ${DATA_PREFIX}"
echo "  Model name: ${MODEL_NAME}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "=========================================="

python "${PROJECT_ROOT}/train.py" \
    --data.data_prefix ${DATA_PREFIX} \
    --checkpoint.serialization_dir ${OUTPUT_DIR} \
    --checkpoint.checkpoint_path ${CHECKPOINT_PATH} \
    --data.families "[${FAMILY:-slavic,germanic,indic,romance,austronesian}]" \
    --model.model_name ${MODEL_NAME} \
    --training.valid_bsz 8 \
    --eval.eval_only true \
    --eval.per_language_eval true \
    --data.data_fraction 0.01 \
    --data.max_length 2048 \
    --data.packing true \
    2>&1 | tee ${OUTPUT_DIR}/eval.log

EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Perplexity evaluation failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

echo ""
echo "=========================================="
echo "PERPLEXITY EVALUATION COMPLETE"
echo "Results: ${OUTPUT_DIR}/per_language_eval_results.json"
echo "=========================================="
