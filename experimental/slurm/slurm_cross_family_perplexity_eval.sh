#!/bin/bash
#SBATCH --job-name=cross_family_perplexity_eval
#SBATCH --output=cross_family_perplexity_eval_%j.out
#SBATCH --error=cross_family_perplexity_eval_%j.err
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --account=bfzp-dtai-gh
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G

module load cuda/12.6.1
source /u/sahuja1/x-elm-v2/.venv/bin/activate

export HF_HOME="/work/nvme/bfzp/hf_cache"

if [ -f /u/sahuja1/x-elm-v2/.env ]; then
    export $(grep -v '^#' /u/sahuja1/x-elm-v2/.env | xargs)
fi

hf auth login --token ${HF_TOKEN}

# Configuration
# ALL_FAMILIES=("Base" "Slavic" "Germanic" "Indic" "Romance" "Austronesian")
# EVAL_FAMILIES=("Austronesian")
EVAL_FAMILIES=("Slavic" "Germanic" "Indic" "Romance" "Austronesian")
ALL_FAMILIES=("Indic")
# ALL_FAMILIES=("denser_run")
DATA_PREFIX="/work/nvme/bfzp/madlad-qwen-tokenized-5B"
CHECKPOINT_BASE="/work/nvme/bfzp"
MODEL_NAME="Qwen/Qwen3-1.7B-Base"
OUTPUT_CSV="/work/nvme/bfzp/cross_family_qwen_perplexity_regularization_10000_lambda_0.001_checkpoints_$(date +%Y%m%d_%H%M%S).csv"

# Initialize CSV with header
echo "trained_family,checkpoint_name,eval_family,language,checkpoint_path,eval_loss,perplexity,num_examples,timestamp" > ${OUTPUT_CSV}
echo "Results will be saved to: ${OUTPUT_CSV}"
echo "========================================"

# Function to check if a configuration has already been evaluated
is_already_evaluated() {
    local trained_family=$1
    local checkpoint_name=$2
    local eval_family=$3
    local language=$4
    
    # Check if CSV exists and has this combination
    if [ -f "${OUTPUT_CSV}" ]; then
        grep -q "^${trained_family},${checkpoint_name},${eval_family},${language}," "${OUTPUT_CSV}"
        return $?
    fi
    return 1
}

# Iterate over each trained model
for TRAINED_FAMILY in "${ALL_FAMILIES[@]}"; do
    echo ""
    echo "========================================"
    echo "Evaluating model trained on: ${TRAINED_FAMILY}"
    echo "========================================"
    
    # Handle Base model case - use original Qwen model
    if [ "${TRAINED_FAMILY}" == "Base" ]; then
        CHECKPOINTS=("${MODEL_NAME}")
        CHECKPOINT_NAMES=("base")
        SERIALIZATION_DIR="${CHECKPOINT_BASE}/Base_eval_results"
        mkdir -p ${SERIALIZATION_DIR}
        echo "Using base model: ${MODEL_NAME}"
    else
        SERIALIZATION_DIR="${CHECKPOINT_BASE}/Indic_l2_reg_qwen_checkpoints"
        
        # Check if checkpoint directory exists
        if [ ! -d "${SERIALIZATION_DIR}" ]; then
            echo "WARNING: Directory not found: ${SERIALIZATION_DIR}"
            for EVAL_FAMILY in "${ALL_FAMILIES[@]}"; do
                if [ "${EVAL_FAMILY}" != "Base" ]; then
                    echo "${TRAINED_FAMILY},NOT_FOUND,${EVAL_FAMILY},N/A,NOT_FOUND,N/A,N/A,0,$(date -Iseconds)" >> ${OUTPUT_CSV}
                fi
            done
            continue
        fi
        
        # Find all checkpoint directories (sorted numerically)
        CHECKPOINTS=($(find ${SERIALIZATION_DIR} -type d -name "checkpoint-*" | sort -V))
        
        # Add final checkpoint if it exists
        if [ -d "${SERIALIZATION_DIR}/final" ]; then
            CHECKPOINTS+=("${SERIALIZATION_DIR}/final")
        fi
        
        if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
            echo "WARNING: No checkpoints found in ${SERIALIZATION_DIR}"
            for EVAL_FAMILY in "${EVAL_FAMILIES[@]}"; do
                if [ "${EVAL_FAMILY}" != "Base" ]; then
                    echo "${TRAINED_FAMILY},NO_CHECKPOINT,${EVAL_FAMILY},N/A,NO_CHECKPOINT,N/A,N/A,0,$(date -Iseconds)" >> ${OUTPUT_CSV}
                fi
            done
            continue
        fi
        
        echo "Found ${#CHECKPOINTS[@]} checkpoints to evaluate"
        
        # Extract checkpoint names
        CHECKPOINTS=("${SERIALIZATION_DIR}/checkpoint-10000") #last checkpoint in the checkpoints array
        CHECKPOINT_NAMES=()
        for cp in "${CHECKPOINTS[@]}"; do
            CHECKPOINT_NAMES+=("$(basename ${cp})")
        done
    fi
    
    # Evaluate each checkpoint
    for i in "${!CHECKPOINTS[@]}"; do
        CHECKPOINT_PATH="${CHECKPOINTS[$i]}"
        CHECKPOINT_NAME="${CHECKPOINT_NAMES[$i]}"
        
        echo ""
        echo "========================================"
        echo "Checkpoint: ${CHECKPOINT_NAME}"
        echo "Path: ${CHECKPOINT_PATH}"
        echo "========================================"
        
        # Evaluate on ALL families (including the one it was trained on)
        for EVAL_FAMILY in "${EVAL_FAMILIES[@]}"; do
            # Skip Base as evaluation family (it's not a language family)
            if [ "${EVAL_FAMILY}" == "Base" ]; then
                echo "Skipping Base as evaluation family (not a language family)"
                continue
            fi
            
            echo ""
            echo "----------------------------------------"
            echo "Evaluating on family: ${EVAL_FAMILY}"
            echo "----------------------------------------"
            
            # Run per-language evaluation on this family
            EVAL_OUTPUT=$(python train_trl.py \
                --data_prefix ${DATA_PREFIX} \
                --serialization_dir ${SERIALIZATION_DIR} \
                --checkpoint_path ${CHECKPOINT_PATH} \
                --families ${EVAL_FAMILY} \
                --model_name ${MODEL_NAME} \
                --valid_bsz 16 \
                --eval_only \
                --per_language_eval \
                --data_fraction 0.01 \
                --max_length 2048 \
                --packing \
                2>&1)
            
            echo "${EVAL_OUTPUT}"
            
            # Parse per-language results (format: family/lang: Loss=X.XXXX, PPL=XX.XX)
            echo "${EVAL_OUTPUT}" | grep -oP ".+/.+: Loss=[\d.]+, PPL=[\d.]+" | while read line; do
                LANG=$(echo "$line" | grep -oP "/.+(?=:)" | sed 's|/||')
                LOSS=$(echo "$line" | grep -oP "Loss=\K[\d.]+")
                PPL=$(echo "$line" | grep -oP "PPL=\K[\d.]+")
                
                # Check if this configuration was already evaluated
                if is_already_evaluated "${TRAINED_FAMILY}" "${CHECKPOINT_NAME}" "${EVAL_FAMILY}" "${LANG}"; then
                    echo "Skipping already evaluated: ${TRAINED_FAMILY}/${CHECKPOINT_NAME} on ${EVAL_FAMILY}/${LANG}"
                else
                    echo "${TRAINED_FAMILY},${CHECKPOINT_NAME},${EVAL_FAMILY},${LANG},${CHECKPOINT_PATH},${LOSS},${PPL},N/A,$(date -Iseconds)" >> ${OUTPUT_CSV}
                fi
            done
        done
    done
done

echo ""
echo "========================================"
echo "Cross-family evaluation complete!"
echo "Results saved to: ${OUTPUT_CSV}"
echo "========================================"
echo ""
echo "Summary of results:"
cat ${OUTPUT_CSV}