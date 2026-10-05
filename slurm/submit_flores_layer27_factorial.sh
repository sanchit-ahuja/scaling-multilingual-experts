#!/bin/bash
# Submit the three missing 20-task Dense-to-Base layer-27 factorial cells.
# This script only submits evaluation jobs; it performs no CPT and does not
# inspect held-out languages.  Set BASELINE_EVAL_DIR to the archived Dense
# screen samples so each result is scored against the registered baseline.
set -euo pipefail

REPO="/u/sahuja1/scaling-multilingual-experts"
cd "${REPO}"

: "${BASELINE_EVAL_DIR:?Set BASELINE_EVAL_DIR to the existing 20-task Dense FLORES samples}"

declare -a variants=(
    "28 5 dense-revert-28-33"
    "27 7 dense-revert-27-34"
    "27 1 dense-revert-27-28"
)

for variant in "${variants[@]}"; do
    read -r start window tag <<<"${variant}"
    job_id=$(sbatch --parsable \
        --job-name="${tag}" \
        --export="ALL,PHASE=screen,STARTS=${start},WINDOW=${window},RUN_TAG=${tag},BASELINE_EVAL_DIR=${BASELINE_EVAL_DIR}" \
        slurm/slurm_flores_layer_localization.sh)
    echo "submitted ${tag} -> ${job_id}"
done
