#!/bin/bash
#SBATCH --job-name=expert-revert-window
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --time=4:00:00

# Materialize an exact Expert-to-Base reversion in a requested half-open range.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-${PROJECT_ROOT}/.venv}"
FAMILY="${FAMILY:?Set FAMILY (e.g. Slavic)}"
FAMILY_KEY="${FAMILY,,}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DATA_ROOT:-data}/checkpoints}"
START="${START:-5}"
END="${END:-11}"

if ! [[ "${START}" =~ ^[0-9]+$ && "${END}" =~ ^[0-9]+$ ]] || (( START < 0 || END > 34 || START >= END )); then
  echo "Invalid layer range [${START},${END}); expected 0 <= START < END <= 34" >&2
  exit 2
fi

TAG="reverted-${START}-$((${END} - 1))"

source "${VENV}/bin/activate"
cd "${REPO}"
mkdir -p logs
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-${DATA_ROOT:-data}/hf_cache}"

# These are the canonical saved Expert checkpoints. Indic is a numbered
# checkpoint rather than a final export, so do not infer paths generically.
case "${FAMILY_KEY}" in
  slavic|germanic|romance|austronesian)
    INPUT_PATH="${CHECKPOINT_ROOT}/${FAMILY_KEY}_gemma_4b_expert/final"
    OUTPUT_PATH="${CHECKPOINT_ROOT}/${FAMILY_KEY}_gemma_4b_expert/${TAG}"
    ;;
  indic)
    INPUT_PATH="${CHECKPOINT_ROOT}/Indic_gemma_4b_expert/checkpoint-7000"
    OUTPUT_PATH="${CHECKPOINT_ROOT}/Indic_gemma_4b_expert/${TAG}"
    ;;
  *)
    echo "Unsupported FAMILY=${FAMILY}; expected Slavic, Germanic, Romance, Indic, or Austronesian" >&2
    exit 2
    ;;
esac

if [[ -f "${OUTPUT_PATH}/reversion_config.json" ]]; then
  echo "exact reversion already exists: ${OUTPUT_PATH}"
  exit 0
fi

echo "job=${SLURM_JOB_ID} family=${FAMILY_KEY} range=[${START},${END}) input=${INPUT_PATH} output=${OUTPUT_PATH}"
python train.py \
  --config_path configs/yaml/revert_gemma_checkpoint.yaml \
  --revert.checkpoint_path "${INPUT_PATH}" \
  --revert.revert_output_path "${OUTPUT_PATH}" \
  --regularizer.first_layers "${START}" \
  --regularizer.last_layers "$((34 - END))" \
  --regularizer.attention_only false

printf '{"experiment_name":"expert-revert-window","job_id":"%s","job_type":"revert","status":"completed"}\n' \
  "${SLURM_JOB_ID}" > "logs/expert-revert-05-11_${SLURM_JOB_ID}.done.json"
