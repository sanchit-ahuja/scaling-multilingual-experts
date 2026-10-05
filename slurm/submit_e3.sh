#!/bin/bash
# Submit all (or a filtered subset of) E3 seed-sweep jobs, belebele first (headline),
# then global_piqa. Usage:
#   bash slurm/submit_e3.sh                 # all 48 jobs
#   bash slurm/submit_e3.sh belebele        # only belebele jobs
#   bash slurm/submit_e3.sh global_piqa     # only piqa jobs
#   SEEDS=1,2,3 bash slurm/submit_e3.sh     # override seeds
set -euo pipefail
cd "${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${VENV:-${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/.venv}/bin/activate"
FILTER="${1:-}"
SEEDS="${SEEDS:-1,2,3,4,5}"

submit_bench () {
    local bench="$1"
    python scripts/e3_jobs.py | grep "|${bench}|" | cut -d'|' -f1 | while read -r tag; do
        jid=$(sbatch --parsable slurm/slurm_e3_seed_sweep.sh "${tag}" "${SEEDS}")
        echo "submitted ${tag} -> ${jid}"
    done
}

if [ -z "${FILTER}" ] || [ "${FILTER}" = "belebele" ]; then submit_bench belebele; fi
if [ -z "${FILTER}" ] || [ "${FILTER}" = "global_piqa" ]; then submit_bench global_piqa; fi
