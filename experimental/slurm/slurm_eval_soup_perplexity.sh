#!/bin/bash
#SBATCH --job-name=soup_perplexity_eval
#SBATCH --output=soup_perplexity_eval_%j.out
#SBATCH --error=soup_perplexity_eval_%j.err
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

# Parse command line arguments
EVAL_FAMILIES=${1:-"Indic"}  # Default to Indic if not provided

# Configuration
CHECKPOINT_BASE="/work/nvme/bfzp/Indic_l2_reg_soup_checkpoints"
OUTPUT_DIR="/work/nvme/bfzp/lm_eval_reg_l2_soup_results/indic_l2_reg_soups"
DATA_PREFIX="/work/nvme/bfzp/madlad-qwen-tokenized-5B"
MODEL_NAME="Qwen/Qwen3-1.7B-Base"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="${OUTPUT_DIR}/soup_models_perplexity_${TIMESTAMP}.csv"

# Model soup directories
SOUP_MODELS=("soup_alpha_0.25" "soup_alpha_0.35" "soup_alpha_0.3" "soup_alpha_0.5" "soup_alpha_0.7")

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Initialize CSV with header
echo "model_name,eval_family,language,checkpoint_path,eval_loss,perplexity,num_examples,timestamp" > ${OUTPUT_CSV}
echo "Evaluating perplexity on families: ${EVAL_FAMILIES}"
echo "Results will be saved to: ${OUTPUT_CSV}"
echo "========================================"

# Evaluate each soup model
for SOUP_MODEL in "${SOUP_MODELS[@]}"; do
    CHECKPOINT_PATH="${CHECKPOINT_BASE}/${SOUP_MODEL}"
    
    echo ""
    echo "========================================"
    echo "Evaluating: ${SOUP_MODEL}"
    echo "Path: ${CHECKPOINT_PATH}"
    echo "========================================"
    
    if [ ! -d "${CHECKPOINT_PATH}" ]; then
        echo "WARNING: Directory not found: ${CHECKPOINT_PATH}"
        continue
    fi
    
    # Evaluate on specified families
    IFS=',' read -ra FAMILY_ARRAY <<< "${EVAL_FAMILIES}"
    for EVAL_FAMILY in "${FAMILY_ARRAY[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Evaluating perplexity on family: ${EVAL_FAMILY}"
        echo "----------------------------------------"
        
        # Run per-language evaluation on this family
        PERPLEXITY_OUTPUT=$(python train_trl.py \
            --data_prefix ${DATA_PREFIX} \
            --serialization_dir ${OUTPUT_DIR} \
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
        
        echo "${PERPLEXITY_OUTPUT}"
        
        # Parse per-language results (format: family/lang: Loss=X.XXXX, PPL=XX.XX)
        echo "${PERPLEXITY_OUTPUT}" | grep -oP ".+/.+: Loss=[\d.]+, PPL=[\d.]+" | while read line; do
            LANG=$(echo "$line" | grep -oP "/.+(?=:)" | sed 's|/||')
            LOSS=$(echo "$line" | grep -oP "Loss=\K[\d.]+")
            PPL=$(echo "$line" | grep -oP "PPL=\K[\d.]+")
            
            echo "${SOUP_MODEL},${EVAL_FAMILY},${LANG},${CHECKPOINT_PATH},${LOSS},${PPL},N/A,$(date -Iseconds)" >> ${OUTPUT_CSV}
        done
    done
done

echo ""
echo "========================================"
echo "Soup model perplexity evaluation complete!"
echo "Results saved to: ${OUTPUT_CSV}"
echo "========================================"
echo ""
echo "Summary of results:"
cat ${OUTPUT_CSV}