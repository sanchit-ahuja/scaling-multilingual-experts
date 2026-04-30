#!/bin/bash
#SBATCH --job-name=eval_all_families
#SBATCH --output=eval_all_families_%j.out
#SBATCH --error=eval_all_families_%j.err
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --account=bfzp-dtai-gh
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G

module load cuda/12.4.0
source /u/sahuja1/x-elm-v2/.venv/bin/activate

export HF_HOME="/work/nvme/bfzp/hf_cache"

if [ -f /u/sahuja1/x-elm-v2/.env ]; then
    export $(grep -v '^#' /u/sahuja1/x-elm-v2/.env | xargs)
fi

hf auth login --token ${HF_TOKEN}

# Configuration
FAMILIES=("Slavic" "Germanic" "Indic" "Austronesian" "Romance")
DATA_PREFIX="/work/nvme/bfzp/madlad-tokenized-dataset"
CHECKPOINT_BASE="/work/nvme/bfzp"
MODEL_NAME="google/gemma-3-1b-pt"
OUTPUT_CSV="/work/nvme/bfzp/language_evaluation_results_$(date +%Y%m%d_%H%M%S).csv"

# Initialize CSV with header
echo "family,language,checkpoint_path,eval_loss,perplexity,num_examples,timestamp" > ${OUTPUT_CSV}
echo "Results will be saved to: ${OUTPUT_CSV}"
echo "========================================"

# Evaluate each family
for FAMILY in "${FAMILIES[@]}"; do
    echo ""
    echo "========================================"
    echo "Evaluating family: ${FAMILY}"
    echo "========================================"
    
    SERIALIZATION_DIR="${CHECKPOINT_BASE}/${FAMILY}_full_checkpoints"
    
    # Check if checkpoint directory exists
    if [ ! -d "${SERIALIZATION_DIR}" ]; then
        echo "WARNING: Directory not found: ${SERIALIZATION_DIR}"
        echo "${FAMILY},N/A,NOT_FOUND,N/A,N/A,0,$(date -Iseconds)" >> ${OUTPUT_CSV}
        continue
    fi
    
    # Run per-language evaluation
    EVAL_OUTPUT=$(python train_gemma.py \
        --data_prefix ${DATA_PREFIX} \
        --serialization_dir ${SERIALIZATION_DIR} \
        --families ${FAMILY} \
        --model_name ${MODEL_NAME} \
        --valid_bsz 16 \
        --eval_only \
        --per_language_eval \
        --data_fraction 0.1 \
        2>&1)
    
    echo "${EVAL_OUTPUT}"
    
    # Parse checkpoint path
    CHECKPOINT_PATH=$(echo "${EVAL_OUTPUT}" | grep -oP "Found latest checkpoint: \K.*|Checkpoint: \K.*" | head -1)
    CHECKPOINT_PATH=${CHECKPOINT_PATH:-"N/A"}
    
    # Parse per-language results (format: family/lang: Loss=X.XXXX, PPL=XX.XX)
    echo "${EVAL_OUTPUT}" | grep -oP ".+/.+: Loss=[\d.]+, PPL=[\d.]+" | while read line; do
        LANG=$(echo "$line" | grep -oP "/.+(?=:)" | sed 's|/||')
        LOSS=$(echo "$line" | grep -oP "Loss=\K[\d.]+")
        PPL=$(echo "$line" | grep -oP "PPL=\K[\d.]+")
        
        echo "${FAMILY},${LANG},${CHECKPOINT_PATH},${LOSS},${PPL},N/A,$(date -Iseconds)" >> ${OUTPUT_CSV}
    done
done

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_CSV}"
echo "========================================"
cat ${OUTPUT_CSV}
