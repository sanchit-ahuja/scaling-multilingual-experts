#!/bin/bash
#SBATCH --job-name=cross_family_lm_freezing_eval
#SBATCH --output=cross_family_lm_freezing_eval_%j.out
#SBATCH --error=cross_family_lm_freezing_eval_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G

# Source common setup (cuda, venv, HF login, env vars)
source "$(dirname "$0")/common.sh"
hf_login

# Parse command line arguments
TASKS=${1:-"belebele"}  # Default to belebele if not provided

# Configuration
# ALL_FAMILIES=("Slavic" "Germanic" "Indic" "Austronesian" "Romance")
ALL_FAMILIES=("Indic")
# ALL_FAMILIES=("dense_run")
CHECKPOINT_BASE="${CHECKPOINTS_ROOT:-${DATA_ROOT}/checkpoints}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-4b-pt}"
OUTPUT_DIR="${DATA_ROOT}/regularization_evals"
TASKS_SANITIZED=$(echo ${TASKS} | tr ',' '_')  # Replace commas with underscores for filename
OUTPUT_CSV="${OUTPUT_DIR}/intermediate_layer_l2_regularization_${TASKS_SANITIZED}_$(date +%Y%m%d_%H%M%S).csv"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Initialize CSV with header
echo "trained_family,checkpoint_name,checkpoint_path,task,metric,value,timestamp" > ${OUTPUT_CSV}
echo "Evaluating on tasks: ${TASKS}"
echo "Results will be saved to: ${OUTPUT_CSV}"
echo "========================================"

# Iterate over each trained model
for TRAINED_FAMILY in "${ALL_FAMILIES[@]}"; do
    echo ""
    echo "========================================"
    echo "Evaluating model trained on: ${TRAINED_FAMILY}"
    echo "========================================"
    
    # Handle Base model case - use the original pretrained model
    if [ "${TRAINED_FAMILY}" == "Base" ] || [ "${TRAINED_FAMILY}" == "Baseline" ]; then
        CHECKPOINTS=("${MODEL_NAME}")
        echo "Using base model: ${MODEL_NAME}"
    else
        SERIALIZATION_DIR="${CHECKPOINT_BASE}/${TRAINED_FAMILY}_layer_range_l2_checkpoints"
        
        # Check if checkpoint directory exists
        if [ ! -d "${SERIALIZATION_DIR}" ]; then
            echo "WARNING: Directory not found: ${SERIALIZATION_DIR}"
            continue
        fi
        
        # Find latest checkpoint or final checkpoint
        CHECKPOINTS=()
        
        # Prefer final checkpoint if it exists
        if [ -d "${SERIALIZATION_DIR}/final" ]; then
            CHECKPOINTS=("${SERIALIZATION_DIR}/final")
            echo "Using final checkpoint"
        else
            # Otherwise, get the latest checkpoint (sorted numerically)
            LATEST_CHECKPOINT=$(find ${SERIALIZATION_DIR} -type d -name "checkpoint-*" | sort -V | tail -1)
            if [ -n "${LATEST_CHECKPOINT}" ]; then
                CHECKPOINTS=("${LATEST_CHECKPOINT}")
                echo "Using latest checkpoint: $(basename ${LATEST_CHECKPOINT})"
            fi
        fi
        
        if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
            echo "WARNING: No checkpoints found in ${SERIALIZATION_DIR}"
            continue
        fi
    fi

    CHECKPOINTS=("${SERIALIZATION_DIR}/checkpoint-7500")
    
    # Evaluate each checkpoint
    for CHECKPOINT_PATH in "${CHECKPOINTS[@]}"; do
        CHECKPOINT_NAME=$(basename ${CHECKPOINT_PATH})
        echo ""
        echo "----------------------------------------"
        echo "Evaluating checkpoint: ${CHECKPOINT_NAME}"
        echo "Path: ${CHECKPOINT_PATH}"
        echo "----------------------------------------"
        
        # Run lm_eval for this checkpoint
        EVAL_OUTPUT_FILE="${OUTPUT_DIR}/${TRAINED_FAMILY}_${CHECKPOINT_NAME}_${TASKS_SANITIZED}_$(date +%Y%m%d_%H%M%S)"
        
        lm_eval --model hf \
            --model_args pretrained=${CHECKPOINT_PATH} \
            --tasks ${TASKS} \
            --device cuda:0 \
            --batch_size auto:4 \
            --output_path ${EVAL_OUTPUT_FILE} \
            --log_samples \
            2>&1 | tee ${OUTPUT_DIR}/${TRAINED_FAMILY}_${CHECKPOINT_NAME}_${TASKS_SANITIZED}.log
        
        # Parse results from JSON if available
        if ls "${EVAL_OUTPUT_FILE}"/results_*.json 1> /dev/null 2>&1; then
            RESULTS_FILE=$(ls -t "${EVAL_OUTPUT_FILE}"/results_*.json | head -1)
            python3 << EOF
import json
import csv
from datetime import datetime

try:
    with open("${EVAL_OUTPUT_FILE}/results.json", 'r') as f:
        results = json.load(f)
    
    with open("${OUTPUT_CSV}", 'a') as csvfile:
        writer = csv.writer(csvfile)
        for task, metrics in results.get('results', {}).items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.writerow([
                        "${TRAINED_FAMILY}",
                        "${CHECKPOINT_NAME}",
                        "${CHECKPOINT_PATH}",
                        task,
                        metric_name,
                        value,
                        datetime.now().isoformat()
                    ])
    print(f"Successfully parsed results for {CHECKPOINT_NAME}")
except Exception as e:
    print(f"Error parsing results for ${CHECKPOINT_NAME}: {e}")
EOF
        else
            echo "WARNING: Results file not found: ${EVAL_OUTPUT_FILE}/results.json"
        fi
    done
done

echo ""
echo "========================================"
echo "Cross-family lm_eval complete!"
echo "Tasks evaluated: ${TASKS}"
echo "Results saved to: ${OUTPUT_CSV}"
echo "Full outputs in: ${OUTPUT_DIR}"
echo "========================================"
echo ""
echo "Summary of results:"
cat ${OUTPUT_CSV}