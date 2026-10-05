#!/bin/bash
# Submit the FLORES seed-sweep jobs (7 core checkpoints). Usage:
#   bash slurm/submit_flores_seeds.sh                 # all 7, seeds 1,2
#   SEEDS=1,2,3 bash slurm/submit_flores_seeds.sh
set -euo pipefail
cd /u/sahuja1/scaling-multilingual-experts
source /u/sahuja1/x-elm-v2/.venv/bin/activate
SEEDS="${SEEDS:-1,2}"
python scripts/flores_jobs.py | cut -d'|' -f1 | while read -r tag; do
    jid=$(sbatch --parsable slurm/slurm_flores_seed_sweep.sh "${tag}" "${SEEDS}")
    echo "submitted ${tag} -> ${jid}"
done
