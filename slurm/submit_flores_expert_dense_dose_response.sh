#!/usr/bin/env bash
# Submit matched Expert→Dense [27,33) FLORES dose-response evaluations.
#
# Uses the localization driver for every FLORES point so source Expert (beta=0)
# and hybrid generations receive the same first-line and length diagnostics.
# Existing beta=.5/1 PPL and Belebele results are retained; this launcher adds
# safeguards only for the new intermediate beta=.25/.75 hybrid checkpoints.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-${RESULTS_ROOT:-results}/${USER}}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${DATA_ROOT:-data}/checkpoints}"
DENSE_MODEL="${DENSE_MODEL:-${CHECKPOINTS_ROOT}/gemma_4b_dense_25b_v2/final}"
INDIC_EXPERT_MODEL="${INDIC_EXPERT_MODEL:-${CHECKPOINTS_ROOT}/Indic_gemma_4b_expert/checkpoint-7000}"
AUSTRO_EXPERT_MODEL="${AUSTRO_EXPERT_MODEL:-${CHECKPOINTS_ROOT}/austronesian_gemma_4b_expert/final}"
BETAS="${BETAS:-0,0.25,0.5,0.75,1.0}"
INTERMEDIATE_BETAS="${INTERMEDIATE_BETAS:-0.25,0.75}"
DATA_PREFIX="${DATA_PREFIX:-${DATA_ROOT:-data}/madlad-tokenized-5B}"

tag_beta() {
  # Decimal beta spellings are valid path components and match the existing
  # persistent hybrid names (for example, ``b0.25``).
  printf '%s' "$1"
}

submit_flores() {
  local family="$1" expert="$2" beta="$3" beta_tag run_tag output_dir
  beta_tag="$(tag_beta "${beta}")"
  run_tag="flores-expert-dense-dose-${family}-b${beta_tag}"
  output_dir="${RESULTS_ROOT}/flores_layer_localization/${run_tag}"
  sbatch --parsable --job-name="${run_tag}" \
    --export="ALL,PHASE=transfer,START=27,END=33,FAMILIES=${family},BETA=${beta},EXPERT_MODEL=${expert},DENSE_MODEL=${DENSE_MODEL},RUN_TAG=${run_tag},OUTPUT_DIR=${output_dir}" \
    "${REPO}/slurm/slurm_flores_layer_localization.sh"
}

submit_intermediate_safeguards() {
  local family="$1" expert="$2" beta="$3" beta_tag tag hybrid_dir hybrid_job
  beta_tag="$(tag_beta "${beta}")"
  tag="${family}-expert-dense-l27-33-b${beta_tag}"
  hybrid_dir="${RESULTS_ROOT}/flores_layer_localization/expert_dense_dose_hybrids/${tag}"
  hybrid_job="$(sbatch --parsable --job-name="flores-dose-hybrid-${family}-b${beta_tag}" \
    --export="ALL,FAMILY=${family},BETA=${beta},EXPERT_MODEL=${expert},DONOR_MODEL=${DENSE_MODEL},TAG=${tag},HYBRID_DIR=${hybrid_dir}" \
    "${REPO}/slurm/slurm_flores_transfer_hybrid.sh")"
  local ppl_job belebele_job
  ppl_job="$(sbatch --parsable --dependency="afterok:${hybrid_job}" --job-name="flores-dose-ppl-${family}-b${beta_tag}" \
    --export="ALL,DATA_PREFIX=${DATA_PREFIX},RESULTS_BASE=${RESULTS_ROOT}" \
    "${REPO}/slurm/slurm_perplexity_eval.sh" --checkpoint_path "${hybrid_dir}" --family "${family}")"
  belebele_job="$(sbatch --parsable --dependency="afterok:${hybrid_job}" --job-name="flores-dose-belebele-${family}-b${beta_tag}" \
    --export="ALL,CHECKPOINT_PATH=${hybrid_dir},FAMILY=${family},TAG=${tag}" \
    "${REPO}/slurm/slurm_family_belebele_eval.sh")"
  printf 'safeguards %s beta=%s: hybrid=%s ppl=%s belebele=%s\n' \
    "${family}" "${beta}" "${hybrid_job}" "${ppl_job}" "${belebele_job}"
}

for pair in "indic:${INDIC_EXPERT_MODEL}" "austronesian:${AUSTRO_EXPERT_MODEL}"; do
  family="${pair%%:*}"
  expert="${pair#*:}"
  IFS=',' read -r -a beta_values <<< "${BETAS}"
  for beta in "${beta_values[@]}"; do
    printf 'flores %s beta=%s: %s\n' "${family}" "${beta}" "$(submit_flores "${family}" "${expert}" "${beta}")"
  done
  IFS=',' read -r -a intermediate_values <<< "${INTERMEDIATE_BETAS}"
  for beta in "${intermediate_values[@]}"; do
    submit_intermediate_safeguards "${family}" "${expert}" "${beta}"
  done
done
