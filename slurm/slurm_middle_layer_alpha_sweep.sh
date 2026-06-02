#!/bin/bash
#SBATCH --job-name=middle_alpha_sweep
#SBATCH --output=logs/middle_alpha_sweep_%j.out
#SBATCH --error=logs/middle_alpha_sweep_%j.err
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --account=bfzp-dtai-gh
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$(dirname "$0")/common.sh"
cd "${PROJECT_ROOT}"

BASE_MODEL="${BASE_MODEL:-google/gemma-3-4b-pt}"
CPT_MODEL="${CPT_MODEL:-${CHECKPOINTS_ROOT}/gemma_4b_dense_25b/final}"
TARGET_LAYERS="${TARGET_LAYERS:-middle}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/drift_alpha_sweep/dense_${TARGET_LAYERS}_full_${SLURM_JOB_ID}}"
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1}"
FIRST_LAYERS="${FIRST_LAYERS:-9}"
LAST_LAYERS="${LAST_LAYERS:-6}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TASKS="${TASKS:-belebele_eng_Latn,belebele_hin_Deva,global_piqa_completions_eng_latn,global_piqa_completions_hin_deva}"
INCLUDE_PATH="${INCLUDE_PATH:-}"
LOG_SAMPLES="${LOG_SAMPLES:-false}"
DELETE_AFTER_EVAL="${DELETE_AFTER_EVAL:-true}"

if [ "$TASKS" = "FLORES_HELDIN" ]; then
    TASKS="$(python scripts/flores_heldin_tasks.py --flores-dir "${FLORES_DIR:-${HOME}/lm-evaluation-harness/lm_eval/tasks/flores}")"
fi
if [ "$TASKS" = "BELEBELE_HELDIN" ]; then
    TASKS="$(python scripts/belebele_heldin_tasks.py)"
fi

LOG_SAMPLES_FLAG=""
if [ "$LOG_SAMPLES" = "true" ]; then
    LOG_SAMPLES_FLAG="--log_samples"
fi

DELETE_FLAG=""
if [ "$DELETE_AFTER_EVAL" = "true" ]; then
    DELETE_FLAG="--delete_after_eval"
fi

echo "============================================================"
echo "Layer-group alpha sweep"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Base model:    ${BASE_MODEL}"
echo "CPT model:     ${CPT_MODEL}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Alphas:        ${ALPHAS}"
echo "Target layers: ${TARGET_LAYERS}"
echo "Layer split:   first=${FIRST_LAYERS}, last=${LAST_LAYERS}"
echo "Tasks:         ${TASKS}"
echo "Include path:  ${INCLUDE_PATH:-<none>}"
echo "Log samples:   ${LOG_SAMPLES}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Limit:         <none; full task datasets>"
echo "Delete ckpts:  ${DELETE_AFTER_EVAL}"
echo "============================================================"

python scripts/middle_layer_alpha_sweep.py \
    --base_model "${BASE_MODEL}" \
    --cpt_model "${CPT_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --alphas "${ALPHAS}" \
    --target_layers "${TARGET_LAYERS}" \
    --first_layers "${FIRST_LAYERS}" \
    --last_layers "${LAST_LAYERS}" \
    --device cuda \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --include_path "${INCLUDE_PATH}" \
    ${LOG_SAMPLES_FLAG} \
    --limit "" \
    ${DELETE_FLAG}

echo "Done. Summary: ${OUTPUT_DIR}/summary.jsonl"
