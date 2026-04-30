#!/bin/bash
#SBATCH --job-name=comprehensive_eval
#SBATCH --output=logs/comprehensive_eval_%j.out
#SBATCH --error=logs/comprehensive_eval_%j.err
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --account=bfzp-dtai-gh
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G

# Source common setup
source /u/sahuja1/x-elm-v2/slurm/common.sh

# Parse command line arguments
CHECKPOINT_PATH=${1}
TASKS=${2:-"flores"}  # Default to belebele if not provided
DATA_PREFIX=${3:-"/work/nvme/bfzp/madlad-tokenized-5B"}
MODEL_NAME=${4:-"google/gemma-3-4b-pt"}

# Validate inputs
if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint path is required"
    echo "Usage: sbatch $0 <CHECKPOINT_PATH> [TASKS] [DATA_PREFIX] [MODEL_NAME]"
    echo "Example: sbatch $0 /path/to/checkpoint belebele"
    exit 1
fi

# Check if it's a HuggingFace model or local path
if [[ "${CHECKPOINT_PATH}" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$ ]]; then
    echo "Detected HuggingFace checkpoint: ${CHECKPOINT_PATH}"
    # For HF checkpoints, we don't need to validate the path exists locally
elif [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Local checkpoint directory not found: ${CHECKPOINT_PATH}"
    exit 1
fi

# Configuration
CHECKPOINT_NAME=$(basename ${CHECKPOINT_PATH})
OUTPUT_DIR="/work/nvme/bfzp/comprehensive_gemma_evals"
TASKS_SANITIZED=$(echo ${TASKS} | tr ',' '_')  # Replace commas with underscores

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "========================================"
echo "COMPREHENSIVE EVALUATION"
echo "========================================"
echo "Checkpoint: ${CHECKPOINT_NAME}"
echo "Path: ${CHECKPOINT_PATH}"
echo "Tasks: ${TASKS}"
echo "Data prefix: ${DATA_PREFIX}"
echo "Model name: ${MODEL_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "========================================"

# Step 1: Run per-language perplexity evaluation
echo ""
echo "========================================"
echo "STEP 1: PERPLEXITY EVALUATION"
echo "========================================"

# Check if any ppl log exists for this checkpoint
if ls ${OUTPUT_DIR}/${CHECKPOINT_NAME}_ppl_*.log 1> /dev/null 2>&1; then
    PPL_OUTPUT=$(ls -t ${OUTPUT_DIR}/${CHECKPOINT_NAME}_ppl_*.log | head -1)
    echo "Perplexity evaluation already exists, skipping: ${PPL_OUTPUT}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    PPL_OUTPUT="${OUTPUT_DIR}/${CHECKPOINT_NAME}_ppl_${TIMESTAMP}.log"
    
    python /u/sahuja1/x-elm-v2/train.py \
        --data.data_prefix ${DATA_PREFIX} \
        --checkpoint.serialization_dir $(dirname ${CHECKPOINT_PATH}) \
        --checkpoint.checkpoint_path ${CHECKPOINT_PATH} \
        --data.families "[Slavic,Germanic,Indic,Romance,Austronesian]" \
        --model.model_name ${MODEL_NAME} \
        --training.valid_bsz 8 \
        --eval.eval_only true \
        --eval.per_language_eval true \
        --data.data_fraction 0.01 \
        --data.max_length 2048 \
        --data.packing true \
        2>&1 | tee ${PPL_OUTPUT}

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Perplexity evaluation failed"
        exit 1
    fi

    echo "Perplexity evaluation complete. Output saved to: ${PPL_OUTPUT}"
fi

# Step 2: Run downstream task evaluation (belebele)
echo ""
echo "========================================"
echo "STEP 2: DOWNSTREAM TASK EVALUATION"
echo "========================================"

# Check if any downstream log exists for this checkpoint and tasks
if ls ${OUTPUT_DIR}/${CHECKPOINT_NAME}_${TASKS_SANITIZED}_*.log 1> /dev/null 2>&1; then
    DOWNSTREAM_LOG=$(ls -t ${OUTPUT_DIR}/${CHECKPOINT_NAME}_${TASKS_SANITIZED}_*.log | head -1)
    echo "Downstream evaluation already exists, skipping: ${DOWNSTREAM_LOG}"
    
    # Extract timestamp from existing log to find output directory
    EXISTING_TIMESTAMP=$(echo ${DOWNSTREAM_LOG} | sed -n 's/.*_\([0-9]\{8\}_[0-9]\{6\}\)\.log$/\1/p')
    if [ -n "${EXISTING_TIMESTAMP}" ]; then
        DOWNSTREAM_OUTPUT_DIR="${OUTPUT_DIR}/${CHECKPOINT_NAME}_${TASKS_SANITIZED}_${EXISTING_TIMESTAMP}"
    else
        # Fallback: try to find the directory
        DOWNSTREAM_OUTPUT_DIR=$(ls -td ${OUTPUT_DIR}/${CHECKPOINT_NAME}_${TASKS_SANITIZED}_* 2>/dev/null | grep -v '\.log$' | head -1)
    fi
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    DOWNSTREAM_LOG="${OUTPUT_DIR}/${CHECKPOINT_NAME}_${TASKS_SANITIZED}_${TIMESTAMP}.log"
    DOWNSTREAM_OUTPUT_DIR="${OUTPUT_DIR}/${CHECKPOINT_NAME}_${TASKS_SANITIZED}_${TIMESTAMP}"
    
    lm_eval --model hf \
        --model_args pretrained=${CHECKPOINT_PATH} \
        --tasks ${TASKS} \
        --device cuda:0 \
        --batch_size auto:4 \
        --output_path ${DOWNSTREAM_OUTPUT_DIR} \
        --log_samples \
        2>&1 | tee ${DOWNSTREAM_LOG}

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Downstream evaluation failed"
        exit 1
    fi

    echo "Downstream evaluation complete. Output saved to: ${DOWNSTREAM_OUTPUT_DIR}"
fi

# Step 3: Parse and combine results
echo ""
echo "========================================"
echo "STEP 3: PARSING RESULTS"
echo "========================================"

# Generate final CSV path
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FINAL_CSV="${OUTPUT_DIR}/${CHECKPOINT_NAME}_comprehensive_${TIMESTAMP}.csv"

# Find the results JSON file
BELEBELE_JSON=""
if [ -f "${DOWNSTREAM_OUTPUT_DIR}/results.json" ]; then
    BELEBELE_JSON="${DOWNSTREAM_OUTPUT_DIR}/results.json"
elif ls "${DOWNSTREAM_OUTPUT_DIR}"/results_*.json 1> /dev/null 2>&1; then
    BELEBELE_JSON=$(ls -t "${DOWNSTREAM_OUTPUT_DIR}"/results_*.json | head -1)
else
    echo "WARNING: No JSON results file found in ${DOWNSTREAM_OUTPUT_DIR}"
    BELEBELE_JSON="/dev/null"  # Create empty file as fallback
fi

echo "Using belebele results: ${BELEBELE_JSON}"

# Run the parser
python /u/sahuja1/x-elm-v2/parse_eval_results.py \
    --ppl_output ${PPL_OUTPUT} \
    --belebele_json ${BELEBELE_JSON} \
    --output_csv ${FINAL_CSV} \
    --debug

if [ $? -ne 0 ]; then
    echo "ERROR: Results parsing failed"
    exit 1
fi

# Step 4: Display summary
echo ""
echo "========================================"
echo "EVALUATION COMPLETE"
echo "========================================"
echo "Checkpoint evaluated: ${CHECKPOINT_NAME}"
echo "Tasks: ${TASKS}"
echo ""
echo "Output files:"
echo "  PPL results: ${PPL_OUTPUT}"
echo "  Downstream results: ${DOWNSTREAM_OUTPUT_DIR}"
echo "  Final CSV: ${FINAL_CSV}"
echo ""
echo "Final results preview:"
echo "========================================"
head -20 ${FINAL_CSV}
echo "========================================"
echo ""
echo "Total languages evaluated: $(tail -n +2 ${FINAL_CSV} | wc -l)"
echo "Languages with PPL scores: $(tail -n +2 ${FINAL_CSV} | grep -v 'N/A' | grep -E '[0-9]+\.[0-9]+' | wc -l)"
echo "Languages with accuracy scores: $(tail -n +2 ${FINAL_CSV} | awk -F',' '$4 != \"N/A\"' | wc -l)"

echo ""
echo "Evaluation pipeline complete!"
echo "All results saved to: ${OUTPUT_DIR}"