#!/bin/bash
#SBATCH --job-name=gemma-ew-matched
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ghx4
#SBATCH --account=bfzp-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00

# Train from Base to a matched 0.35-epoch exposure. Re-entering this script for
# the same arm resumes the newest numbered Trainer checkpoint after a time limit.
set -euo pipefail

REPO="/u/sahuja1/scaling-multilingual-experts"
VENV="${VENV:-/u/sahuja1/x-elm-v2/.venv}"
FAMILY="${FAMILY:?Set FAMILY (e.g. Slavic)}"
FAMILY_KEY="${FAMILY,,}"
MODE="${MODE:?Set MODE to l2sp or freeze}"
DATA_PREFIX="${DATA_PREFIX:-/work/nvme/bfzp/madlad-tokenized-5B}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/work/nvme/bfzp/checkpoints}"

source "${VENV}/bin/activate"
cd "${REPO}"
mkdir -p logs
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-/work/nvme/bfzp/hf_cache}"

case "${MODE}" in
  l2sp)
    CONFIG="configs/yaml/train_gemma_early_window_l2sp_matched.yaml"
    RUN_TAG="${FAMILY_KEY}-gemma-4b-expert-early-window-matched-l2sp"
    ;;
  freeze)
    CONFIG="configs/yaml/train_gemma_early_window_freeze_matched.yaml"
    RUN_TAG="${FAMILY_KEY}-gemma-4b-expert-early-window-matched-freeze"
    ;;
  *)
    echo "MODE must be l2sp or freeze, got ${MODE}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${CHECKPOINT_ROOT}/${RUN_TAG}"
if [[ -d "${OUTPUT_DIR}/final" ]]; then
  echo "final checkpoint already exists: ${OUTPUT_DIR}/final"
  exit 0
fi

echo "job=${SLURM_JOB_ID} family=${FAMILY_KEY} mode=${MODE} output=${OUTPUT_DIR}"
echo "progress is visible in ${OUTPUT_DIR}/checkpoint-*/trainer_state.json"

accelerate launch train.py \
  --config_path "${CONFIG}" \
  --data.families "[${FAMILY_KEY}]" \
  --data.data_prefix "${DATA_PREFIX}" \
  --checkpoint.serialization_dir "${OUTPUT_DIR}" \
  --checkpoint.run_name "${RUN_TAG}" \
  --checkpoint.resume_from_checkpoint true

printf '{"experiment_name":"%s","job_id":"%s","job_type":"cpt","status":"completed"}\n' \
  "${RUN_TAG}" "${SLURM_JOB_ID}" > "logs/${RUN_TAG}_${SLURM_JOB_ID}.done.json"
