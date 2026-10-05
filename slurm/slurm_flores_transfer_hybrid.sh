#!/bin/bash
#SBATCH --job-name=flores-transfer-hybrid
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --time=2:00:00
#SBATCH --requeue
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${VENV:-${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/.venv}/bin/activate"
export DATA_ROOT=${DATA_ROOT:-${DATA_ROOT:-data}}
export CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:-${DATA_ROOT}/checkpoints}
export HF_HOME=${HF_HOME:-${DATA_ROOT}/hf_cache}
export RESULTS_ROOT=${RESULTS_ROOT:-${RESULTS_ROOT:-results}/${USER}}
mkdir -p "${REPO}/logs"
cd "${REPO}"
: "${FAMILY:?Set FAMILY}" "${BETA:?Set BETA}" "${EXPERT_MODEL:?Set EXPERT_MODEL}" "${DONOR_MODEL:?Set DONOR_MODEL}"
TAG="${TAG:-${FAMILY,,}_l27_33_b${BETA/./p}}"
HYBRID_DIR="${HYBRID_DIR:-${RESULTS_ROOT}/flores_layer_localization/hybrid_experts/${TAG}}"
python scripts/create_flores_transfer_hybrid.py \
  --expert_model "${EXPERT_MODEL}" --donor_model "${DONOR_MODEL}" \
  --output_dir "${HYBRID_DIR}" --start 27 --end 33 --beta "${BETA}"
printf '{"experiment_name":"flores-transfer-hybrid","job_id":"%s","job_type":"eval","status":"completed"}\n' "$SLURM_JOB_ID" > "logs/flores-transfer-hybrid_${SLURM_JOB_ID}.done.json"
