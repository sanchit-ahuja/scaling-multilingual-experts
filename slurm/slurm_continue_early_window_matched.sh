#!/bin/bash
#SBATCH --job-name=gemma-ew-continue
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

# Continue a pilot's numbered Trainer checkpoint to the original Expert's
# effective epoch exposure. Do not point at final/: it lacks Trainer state.
set -euo pipefail

REPO="/u/sahuja1/scaling-multilingual-experts"
VENV="${VENV:-/u/sahuja1/x-elm-v2/.venv}"
FAMILY="${FAMILY:?Set FAMILY (Slavic or Romance)}"
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
  l2sp) CONFIG="configs/yaml/train_gemma_early_window_l2sp.yaml" ;;
  freeze) CONFIG="configs/yaml/train_gemma_early_window_freeze.yaml" ;;
  *) echo "MODE must be l2sp or freeze, got ${MODE}" >&2; exit 2 ;;
esac

case "${FAMILY_KEY}" in
  slavic) TARGET_STEPS=18000 ;;
  romance) TARGET_STEPS=21000 ;;
  *) echo "FAMILY must be Slavic or Romance, got ${FAMILY}" >&2; exit 2 ;;
esac

RUN_TAG="${FAMILY_KEY}-gemma-4b-expert-early-window-${MODE}"
OUTPUT_DIR="${CHECKPOINT_ROOT}/${RUN_TAG}"
RESUME_FROM=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
  | sort -V | tail -n 1)
if [[ -z "${RESUME_FROM}" ]]; then
  echo "No numbered Trainer checkpoint in ${OUTPUT_DIR}" >&2
  exit 1
fi
RESUME_FROM="${OUTPUT_DIR}/${RESUME_FROM}"
CURRENT_STEP=$("${VENV}/bin/python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["global_step"])' \
  "${RESUME_FROM}/trainer_state.json")

if (( CURRENT_STEP >= TARGET_STEPS )); then
  echo "already checkpointed at step ${CURRENT_STEP}, target is ${TARGET_STEPS}; nothing to continue"
  exit 0
fi

echo "job=${SLURM_JOB_ID} family=${FAMILY_KEY} mode=${MODE} resume=${RESUME_FROM} step=${CURRENT_STEP}/${TARGET_STEPS}"
echo "progress is visible in ${OUTPUT_DIR}/checkpoint-*/trainer_state.json"

accelerate launch train.py \
  --config_path "${CONFIG}" \
  --data.families "[${FAMILY_KEY}]" \
  --data.data_prefix "${DATA_PREFIX}" \
  --training.max_steps "${TARGET_STEPS}" \
  --checkpoint.serialization_dir "${OUTPUT_DIR}" \
  --checkpoint.run_name "${RUN_TAG}-continued" \
  --checkpoint.early_stopping_patience 0 \
  --checkpoint.early_stopping_threshold 0.0 \
  --checkpoint.save_steps 500 \
  --checkpoint.resume_from_checkpoint true \
  --checkpoint.checkpoint_path "${RESUME_FROM}"

printf '{"experiment_name":"%s","job_id":"%s","job_type":"cpt","status":"completed"}\n' \
  "${RUN_TAG}-continued" "${SLURM_JOB_ID}" > "logs/${RUN_TAG}-continued_${SLURM_JOB_ID}.done.json"
