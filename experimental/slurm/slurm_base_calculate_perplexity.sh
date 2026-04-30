#!/bin/bash
#SBATCH --job-name=eval_all_families
#SBATCH --output=eval_all_families_%j.out
#SBATCH --error=eval_all_families_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=d1028
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

module load cuda/12.8.0
module load anaconda3/2024.06
source /home/ahuja.sanc/x-elm-v2/.venv/bin/activate

export HF_HOME="/scratch/ahuja.sanc/hf_cache"

# Configuration
FAMILIES=("Slavic" "Germanic" "Indic" "Austronesian" "Romance")
DATA_PREFIX="/scratch/ahuja.sanc/madlad-tokenized-dataset"
MODEL_NAME="google/gemma-3-1b-pt"
OUTPUT_CSV="/scratch/ahuja.sanc/base_gemma_eval_results_$(date +%Y%m%d_%H%M%S).csv"
TEMP_DIR="/scratch/ahuja.sanc/base_gemma_eval_temp"

mkdir -p ${TEMP_DIR}

# Initialize CSV with header
echo "family,language,model,eval_loss,perplexity,num_examples,timestamp" > ${OUTPUT_CSV}
echo "Results will be saved to: ${OUTPUT_CSV}"
echo "========================================"

# Evaluate each family
for FAMILY in "${FAMILIES[@]}"; do
    echo ""
    echo "========================================"
    echo "Evaluating base Gemma on family: ${FAMILY}"
    echo "========================================"
    
    # Run per-language evaluation using base model
    EVAL_OUTPUT=$(python train_gemma.py \
        --data_prefix ${DATA_PREFIX} \
        --serialization_dir ${TEMP_DIR} \
        --families ${FAMILY} \
        --model_name ${MODEL_NAME} \
        --valid_bsz 8 \
        --train_bsz 8 \
        --eval_only \
        --per_language_eval \
        --checkpoint_path ${MODEL_NAME} \
        --data_fraction 0.1 \
        2>&1)
    
    echo "${EVAL_OUTPUT}"
    
    # Parse per-language results (format: family/lang: Loss=X.XXXX, PPL=XX.XX)
    echo "${EVAL_OUTPUT}" | grep -oP ".+/.+: Loss=[\d.]+, PPL=[\d.]+" | while read line; do
        LANG=$(echo "$line" | grep -oP "/.+(?=:)" | sed 's|/||')
        LOSS=$(echo "$line" | grep -oP "Loss=\K[\d.]+")
        PPL=$(echo "$line" | grep -oP "PPL=\K[\d.]+")
        
        echo "${FAMILY},${LANG},${MODEL_NAME},${LOSS},${PPL},N/A,$(date -Iseconds)" >> ${OUTPUT_CSV}
    done
done

echo ""
echo "========================================"
echo "Base Gemma evaluation complete!"
echo "Results saved to: ${OUTPUT_CSV}"
echo "========================================"
cat ${OUTPUT_CSV}

# Cleanup temp dir
rm -rf ${TEMP_DIR}
