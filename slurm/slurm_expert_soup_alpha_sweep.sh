#!/bin/bash
#SBATCH --job-name=soup_alpha_sweep
#SBATCH --output=logs/soup_alpha_sweep_%j.out
#SBATCH --error=logs/soup_alpha_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$(dirname "$0")/common.sh"
cd "${PROJECT_ROOT}"

BASE_MODEL="${BASE_MODEL:-google/gemma-3-4b-pt}"
EXPERT_MODELS="${EXPERT_MODELS:-${CHECKPOINTS_ROOT}/slavic_gemma_4b_expert/final,${CHECKPOINTS_ROOT}/germanic_gemma_4b_expert/final,${CHECKPOINTS_ROOT}/Indic_gemma_4b_expert/checkpoint-7000,${CHECKPOINTS_ROOT}/austronesian_gemma_4b_expert/final,${CHECKPOINTS_ROOT}/romance_gemma_4b_expert/final}"
TARGET_LAYERS="${TARGET_LAYERS:-middle}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/drift_alpha_sweep/expert_soup_${TARGET_LAYERS}_belebele_all_${SLURM_JOB_ID}}"
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1}"
FIRST_LAYERS="${FIRST_LAYERS:-9}"
LAST_LAYERS="${LAST_LAYERS:-6}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DELETE_AFTER_EVAL="${DELETE_AFTER_EVAL:-true}"
TASKS="${TASKS:-BELEBELE_HELDIN}"

if [ "$TASKS" = "BELEBELE_HELDIN" ]; then
    TASKS="$(python scripts/belebele_heldin_tasks.py)"
fi

DELETE_FLAG=""
if [ "$DELETE_AFTER_EVAL" = "true" ]; then
    DELETE_FLAG="--delete_after_eval"
fi

echo "============================================================"
echo "Expert-soup layer alpha sweep"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Base model:    ${BASE_MODEL}"
echo "Experts:       ${EXPERT_MODELS}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Alphas:        ${ALPHAS}"
echo "Target layers: ${TARGET_LAYERS}"
echo "Layer split:   first=${FIRST_LAYERS}, last=${LAST_LAYERS}"
echo "Tasks:         ${TASKS}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Limit:         <none; full task datasets>"
echo "Delete ckpts:  ${DELETE_AFTER_EVAL}"
echo "============================================================"

python scripts/expert_soup_alpha_sweep.py \
    --base_model "${BASE_MODEL}" \
    --expert_models "${EXPERT_MODELS}" \
    --output_dir "${OUTPUT_DIR}" \
    --alphas "${ALPHAS}" \
    --target_layers "${TARGET_LAYERS}" \
    --first_layers "${FIRST_LAYERS}" \
    --last_layers "${LAST_LAYERS}" \
    --device cuda \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --limit "" \
    ${DELETE_FLAG}

echo "Done. Summary: ${OUTPUT_DIR}/summary.jsonl"
