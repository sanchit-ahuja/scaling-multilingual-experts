#!/bin/bash
#SBATCH --job-name=gemma-early-window
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00

# Train a family Expert from Base while protecting exactly layers [5,11).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-${PROJECT_ROOT}/.venv}"
FAMILY="${FAMILY:?Set FAMILY (e.g. Slavic)}"
FAMILY_KEY="${FAMILY,,}"
MODE="${MODE:?Set MODE to l2sp or freeze}"
DATA_PREFIX="${DATA_PREFIX:-${DATA_ROOT:-data}/madlad-tokenized-5B}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DATA_ROOT:-data}/checkpoints}"

source "${VENV}/bin/activate"
cd "${REPO}"
mkdir -p logs
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-${DATA_ROOT:-data}/hf_cache}"

case "${MODE}" in
  l2sp)
    CONFIG="configs/yaml/train_gemma_early_window_l2sp.yaml"
    RUN_TAG="${FAMILY_KEY}-gemma-4b-expert-early-window-l2sp"
    ;;
  freeze)
    CONFIG="configs/yaml/train_gemma_early_window_freeze.yaml"
    RUN_TAG="${FAMILY_KEY}-gemma-4b-expert-early-window-freeze"
    ;;
  *)
    echo "MODE must be l2sp or freeze, got ${MODE}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${CHECKPOINT_ROOT}/${RUN_TAG}"
echo "job=${SLURM_JOB_ID} family=${FAMILY_KEY} mode=${MODE} output=${OUTPUT_DIR}"

accelerate launch train.py \
  --config_path "${CONFIG}" \
  --data.families "[${FAMILY_KEY}]" \
  --data.data_prefix "${DATA_PREFIX}" \
  --checkpoint.serialization_dir "${OUTPUT_DIR}" \
  --checkpoint.run_name "${RUN_TAG}"

printf '{"experiment_name":"%s","job_id":"%s","job_type":"cpt","status":"completed"}\n' \
  "${RUN_TAG}" "${SLURM_JOB_ID}" > "logs/${RUN_TAG}_${SLURM_JOB_ID}.done.json"
