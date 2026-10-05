#!/bin/bash
#SBATCH --job-name=sliding_window
#SBATCH --output=logs/sliding_window_%j.out
#SBATCH --error=logs/sliding_window_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G

set -euo pipefail

# The runnable venv lives in the repository; this repo (scaling-multilingual-experts)
# holds the scripts. Activate the venv explicitly, then run from this repo.
# NOTE: $0 is the SLURM spool copy at runtime, so hardcode the repo path.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[[ -n "${CUDA_MODULE:-}" ]] && module load "${CUDA_MODULE}"
source "${VENV:-${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/.venv}/bin/activate"
export DATA_ROOT="${DATA_ROOT:-${DATA_ROOT:-data}}"
export CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${DATA_ROOT:-data}/checkpoints}"
export HF_HOME="${HF_HOME:-${DATA_ROOT:-data}/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd "${REPO}"

BASE_MODEL="${BASE_MODEL:-google/gemma-3-4b-pt}"
CPT_MODEL="${CPT_MODEL:-${CHECKPOINTS_ROOT}/gemma_4b_dense_25b_v2/final}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/sliding_window_belebele_${SLURM_JOB_ID}}"
WINDOW="${WINDOW:-6}"
STRIDE="${STRIDE:-1}"
START_FROM="${START_FROM:-0}"
START_TO="${START_TO:--1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT="${LIMIT:-}"                       # set e.g. 100 for a smoke test
TASKS="$(python scripts/belebele_heldin_tasks.py)"

echo "=== sliding window: W=${WINDOW} stride=${STRIDE} limit='${LIMIT}' ==="
echo "CPT=${CPT_MODEL}"
echo "OUT=${OUTPUT_DIR}"
python scripts/sliding_window_interp.py \
    --base_model "${BASE_MODEL}" \
    --cpt_model "${CPT_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --window "${WINDOW}" \
    --stride "${STRIDE}" \
    --start_from "${START_FROM}" \
    --start_to "${START_TO}" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --limit "${LIMIT}" \
    --device cuda
echo "Done: ${OUTPUT_DIR}/summary.jsonl"
