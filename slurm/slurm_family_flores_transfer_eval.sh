#!/bin/bash
#SBATCH --job-name=flores-transfer-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ghx4
#SBATCH --account=bfzp-dtai-gh
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --time=10:00:00
#SBATCH --requeue
set -euo pipefail
REPO=/u/sahuja1/scaling-multilingual-experts
source /u/sahuja1/x-elm-v2/.venv/bin/activate
export HF_HOME=${HF_HOME:-/work/nvme/bfzp/hf_cache}
export RESULTS_ROOT=${RESULTS_ROOT:-/work/hdd/bfzp/${USER}}
mkdir -p "${REPO}/logs"
cd "${REPO}"
: "${FAMILY:?Set FAMILY}" "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}" "${TAG:?Set TAG}"
FAMILY_KEY="${FAMILY,,}"
python scripts/eval_flores_checkpoint.py --checkpoint "${CHECKPOINT_PATH}" --family "${FAMILY_KEY}" \
  --output_dir "${RESULTS_ROOT}/flores_layer_localization/transfer_evals/${TAG}/flores" \
  --batch_size "${BATCH_SIZE:-32}" --include_path "${FLORES_DIR:-}"
printf '{"experiment_name":"flores-transfer-eval","job_id":"%s","job_type":"eval","status":"completed"}\n' "$SLURM_JOB_ID" > "logs/flores-transfer-eval_${SLURM_JOB_ID}.done.json"
