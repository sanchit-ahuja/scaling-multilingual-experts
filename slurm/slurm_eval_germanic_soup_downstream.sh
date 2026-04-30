#!/bin/bash
#SBATCH --job-name=soup_lm_eval
#SBATCH --output=soup_lm_eval_%j.out
#SBATCH --error=soup_lm_eval_%j.err
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --account=bfzp-dtai-gh
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G

# Source common setup (cuda, venv, HF login, env vars)
source "$(dirname "$0")/common.sh"
hf_login

# Parse command line arguments
TASKS=${1:-"belebele"}  # Default to belebele if not provided

# Configuration
CHECKPOINT_BASE="${CHECKPOINTS_ROOT:-${DATA_ROOT}/checkpoints}"
OUTPUT_BASE="${DATA_ROOT}/lm_eval_reg_l2_soup_results"
TASKS_SANITIZED=$(echo ${TASKS} | tr ',' '_')

# Language families to evaluate
# FAMILIES=("germanic" "austronesian" "indic" "slavic" "romance")
FAMILIES=("Indic")

# Model soup directories
# SOUP_MODELS=("soup_alpha_0.3" "soup_alpha_0.5" "soup_alpha_0.7")
SOUP_MODELS=("soup_alpha_0.25" "soup_alpha_0.35")

# Create master output CSV for all families
MASTER_OUTPUT_DIR="${OUTPUT_BASE}/indic_reg_l2"
mkdir -p ${MASTER_OUTPUT_DIR}
MASTER_CSV="${MASTER_OUTPUT_DIR}/all_families_soup_models_${TASKS_SANITIZED}_$(date +%Y%m%d_%H%M%S).csv"

# Initialize master CSV with header
echo "family,model_name,checkpoint_path,task,metric,value,timestamp" > ${MASTER_CSV}
echo "Evaluating soup models on tasks: ${TASKS}"
echo "Families: ${FAMILIES[*]}"
echo "Master results will be saved to: ${MASTER_CSV}"
echo "========================================"

# Evaluate each family
for FAMILY in "${FAMILIES[@]}"; do
    echo ""
    echo "########################################"
    echo "# Processing family: ${FAMILY}"
    echo "########################################"
    
    FAMILY_CHECKPOINT_BASE="${CHECKPOINT_BASE}/${FAMILY}_l2_reg_soup_checkpoints"
    FAMILY_OUTPUT_DIR="${OUTPUT_BASE}/${FAMILY}_l2_reg_soups"
    FAMILY_CSV="${FAMILY_OUTPUT_DIR}/l2_reg_soup_models_${TASKS_SANITIZED}_$(date +%Y%m%d_%H%M%S).csv"
    
    # Create family output directory
    mkdir -p ${FAMILY_OUTPUT_DIR}
    
    # Initialize family CSV with header
    echo "model_name,checkpoint_path,task,metric,value,timestamp" > ${FAMILY_CSV}
    
    # Evaluate each soup model for this family
    for SOUP_MODEL in "${SOUP_MODELS[@]}"; do
        CHECKPOINT_PATH="${FAMILY_CHECKPOINT_BASE}/${SOUP_MODEL}"
        
        echo ""
        echo "========================================"
        echo "Evaluating: ${FAMILY}/${SOUP_MODEL}"
        echo "Path: ${CHECKPOINT_PATH}"
        echo "========================================"
        
        if [ ! -d "${CHECKPOINT_PATH}" ]; then
            echo "WARNING: Directory not found: ${CHECKPOINT_PATH}"
            continue
        fi
        
        # Run lm_eval
        EVAL_OUTPUT_FILE="${FAMILY_OUTPUT_DIR}/${SOUP_MODEL}_${TASKS_SANITIZED}_$(date +%Y%m%d_%H%M%S)"
        
        lm_eval --model hf \
            --model_args pretrained=${CHECKPOINT_PATH} \
            --tasks ${TASKS} \
            --device cuda:0 \
            --batch_size auto:4 \
            --output_path ${EVAL_OUTPUT_FILE} \
            --log_samples \
            2>&1 | tee ${FAMILY_OUTPUT_DIR}/${SOUP_MODEL}_${TASKS_SANITIZED}.log
        
        # Parse results
        if [ -f "${EVAL_OUTPUT_FILE}/results.json" ]; then
            python3 << EOF
import json
import csv
from datetime import datetime

try:
    with open("${EVAL_OUTPUT_FILE}/results.json", 'r') as f:
        results = json.load(f)
    
    # Write to family CSV
    with open("${FAMILY_CSV}", 'a') as csvfile:
        writer = csv.writer(csvfile)
        for task, metrics in results.get('results', {}).items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.writerow([
                        "${SOUP_MODEL}",
                        "${CHECKPOINT_PATH}",
                        task,
                        metric_name,
                        value,
                        datetime.now().isoformat()
                    ])
    
    # Write to master CSV (includes family column)
    with open("${MASTER_CSV}", 'a') as csvfile:
        writer = csv.writer(csvfile)
        for task, metrics in results.get('results', {}).items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.writerow([
                        "${FAMILY}",
                        "${SOUP_MODEL}",
                        "${CHECKPOINT_PATH}",
                        task,
                        metric_name,
                        value,
                        datetime.now().isoformat()
                    ])
    print(f"Successfully parsed results for ${FAMILY}/${SOUP_MODEL}")
except Exception as e:
    print(f"Error parsing results: {e}")
EOF
        else
            echo "WARNING: Results file not found: ${EVAL_OUTPUT_FILE}/results.json"
        fi
    done
    
    echo ""
    echo "Family ${FAMILY} evaluation complete. Results saved to: ${FAMILY_CSV}"
done

echo ""
echo "########################################"
echo "# All family evaluations complete!"
echo "########################################"
echo "Master results saved to: ${MASTER_CSV}"
echo ""
echo "Summary of all results:"
cat ${MASTER_CSV}