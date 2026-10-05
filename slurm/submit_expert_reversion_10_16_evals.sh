#!/bin/bash
# Submit all-family exact Expert→Base [10,16) reversions and dependent held-in evaluations.
set -euo pipefail

REPO=/u/sahuja1/scaling-multilingual-experts
CHECKPOINT_ROOT=/work/nvme/bfzp/checkpoints
DATA_PREFIX=/work/nvme/bfzp/madlad-tokenized-5B
RESULTS_ROOT=/work/hdd/bfzp/${USER}
FAMILIES=(slavic germanic indic austronesian romance)

checkpoint_path() {
  local family="$1"
  if [[ "$family" == indic ]]; then
    printf '%s/Indic_gemma_4b_expert/reverted-10-15' "$CHECKPOINT_ROOT"
  else
    printf '%s/%s_gemma_4b_expert/reverted-10-15' "$CHECKPOINT_ROOT" "$family"
  fi
}

cd "$REPO"
for family in "${FAMILIES[@]}"; do
  checkpoint="$(checkpoint_path "$family")"
  tag="${family}-expert-reverted-10-16"
  reversion_id="$(sbatch --parsable --time=1:00:00 --export="ALL,FAMILY=${family},START=10,END=16" \
    slurm/slurm_revert_expert_early_window.sh)"

  sbatch --time=6:00:00 --dependency="afterok:${reversion_id}" \
    --export="ALL,FAMILY=${family},CHECKPOINT_PATH=${checkpoint},TAG=${tag},RESULTS_ROOT=${RESULTS_ROOT}" \
    slurm/slurm_family_flores_transfer_eval.sh
  sbatch --time=1:00:00 --dependency="afterok:${reversion_id}" \
    --export="ALL,FAMILY=${family},CHECKPOINT_PATH=${checkpoint},TAG=${tag},RESULTS_ROOT=${RESULTS_ROOT}" \
    slurm/slurm_family_belebele_eval.sh
  sbatch --time=0:30:00 --dependency="afterok:${reversion_id}" \
    --export="ALL,DATA_PREFIX=${DATA_PREFIX}" \
    slurm/slurm_perplexity_eval.sh --checkpoint_path "$checkpoint" --family "$family"
done
