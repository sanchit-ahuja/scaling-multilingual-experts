#!/bin/bash
#SBATCH --job-name=boundary_ppl
#SBATCH --output=logs/boundary_ppl_%j.out
#SBATCH --error=logs/boundary_ppl_%j.err
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --account=bfzp-dtai-gh
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G

set -euo pipefail

REPO="/u/sahuja1/scaling-multilingual-experts"
module load cuda/12.6.1 2>/dev/null || true
source /u/sahuja1/x-elm-v2/.venv/bin/activate
export DATA_ROOT="${DATA_ROOT:-/work/nvme/bfzp}"
export CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-/work/nvme/bfzp/checkpoints}"
export HF_HOME="${HF_HOME:-/work/nvme/bfzp/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd "${REPO}"

BASE_MODEL="${BASE_MODEL:-google/gemma-3-4b-pt}"
CPT_MODEL="${CPT_MODEL:-${CHECKPOINTS_ROOT}/gemma_4b_dense_25b_v2/final}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/boundary_ppl_sweep_${SLURM_JOB_ID}}"
DATA_PREFIX="${DATA_PREFIX:-/work/nvme/bfzp/madlad-tokenized-5B}"
FAMILIES="${FAMILIES:-slavic,germanic,indic,romance,austronesian}"
BOUNDARIES="${BOUNDARIES:-9,6;6,6;12,6;9,4;9,8;6,4;12,8;15,6;9,10}"
DATA_FRACTION="${DATA_FRACTION:-0.01}"

echo "=== boundary perplexity sweep ==="
echo "CPT=${CPT_MODEL}"
echo "DATA_PREFIX=${DATA_PREFIX}"
echo "BOUNDARIES=${BOUNDARIES}"
echo "OUT=${OUTPUT_DIR}"
python scripts/boundary_perplexity_sweep.py \
    --base_model "${BASE_MODEL}" \
    --cpt_model "${CPT_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --boundaries "${BOUNDARIES}" \
    --data_prefix "${DATA_PREFIX}" \
    --families "${FAMILIES}" \
    --data_fraction "${DATA_FRACTION}" \
    --device cuda
echo "Done: ${OUTPUT_DIR}/summary.jsonl"
