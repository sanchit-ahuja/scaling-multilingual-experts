#!/bin/bash
# Resubmit pending [10,16) evaluations after shortening their wall-time limits.
# All five reversion jobs completed before this script is used, so evaluations
# are submitted directly against their materialized checkpoints.
set -euo pipefail

REPO=/u/sahuja1/scaling-multilingual-experts
CHECKPOINT_ROOT=/work/nvme/bfzp/checkpoints
DATA_PREFIX=/work/nvme/bfzp/madlad-tokenized-5B
RESULTS_ROOT=/work/hdd/bfzp/${USER}

submit_family() {
  local family="$1"
  local checkpoint tag
  if [[ "$family" == indic ]]; then
    checkpoint="${CHECKPOINT_ROOT}/Indic_gemma_4b_expert/reverted-10-15"
  else
    checkpoint="${CHECKPOINT_ROOT}/${family}_gemma_4b_expert/reverted-10-15"
  fi
  tag="${family}-expert-reverted-10-16"

  sbatch --time=6:00:00 \
    --export="ALL,FAMILY=${family},CHECKPOINT_PATH=${checkpoint},TAG=${tag},RESULTS_ROOT=${RESULTS_ROOT}" \
    slurm/slurm_family_flores_transfer_eval.sh
  if [[ "$family" != slavic ]]; then
    sbatch --time=1:00:00 \
      --export="ALL,FAMILY=${family},CHECKPOINT_PATH=${checkpoint},TAG=${tag},RESULTS_ROOT=${RESULTS_ROOT}" \
      slurm/slurm_family_belebele_eval.sh
  fi
  sbatch --time=0:30:00 \
    --export="ALL,DATA_PREFIX=${DATA_PREFIX}" \
    slurm/slurm_perplexity_eval.sh --checkpoint_path "$checkpoint" --family "$family"
}

cd "$REPO"
submit_family slavic
submit_family germanic
submit_family indic
submit_family austronesian
submit_family romance
