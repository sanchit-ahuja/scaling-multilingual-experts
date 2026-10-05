#!/bin/bash
#SBATCH --job-name=flores-localize
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
# Full 60-direction confirmations can exceed the original 10-hour screen
# limit; a 24-hour allocation avoids an unnecessary cache-resume cycle.
#SBATCH --time=24:00:00
#SBATCH --requeue

# Usage (one job per phase/window):
# sbatch --export=ALL,PHASE=screen,STARTS=0,5,... slurm/slurm_flores_layer_localization.sh
# Set WINDOW to evaluate a non-default-width screen range, for example
# STARTS=28,WINDOW=5 for [28,33).
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${VENV:-${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/.venv}/bin/activate"
export DATA_ROOT="${DATA_ROOT:-${DATA_ROOT:-data}}"
export CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${DATA_ROOT}/checkpoints}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"
export RESULTS_ROOT="${RESULTS_ROOT:-${RESULTS_ROOT:-results}/${USER}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "${REPO}/logs"
cd "${REPO}"

PHASE="${PHASE:?Set PHASE=screen,confirm,or transfer}"
DENSE_MODEL="${DENSE_MODEL:-${CHECKPOINTS_ROOT}/gemma_4b_dense_25b_v2/final}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-4b-pt}"
# Keep durable samples and summaries on HDD; hybrid checkpoints live on NVMe
# only briefly, then the Python driver deletes them after scoring.
# Include the window width in the default tag.  Boundary diagnostics can use
# the same start layer at different widths (e.g. [27,28) and [27,34)); without
# it, those jobs would incorrectly share outputs and response caches.
RUN_TAG="${RUN_TAG:-${PHASE}_start_${STARTS:-range_${START:-unset}_${END:-unset}}_window_${WINDOW:-6}}"
# A stable per-window directory means a requeue or resubmission can reuse a
# completed result rather than re-running it. Partial attempts are never treated
# as complete because only result_<tag>.json is the completion marker.
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/flores_layer_localization/${RUN_TAG}}"
TEMP_ROOT="${TEMP_ROOT:-${DATA_ROOT}/flores_layer_localization_tmp/${SLURM_JOB_ID}}"
ARGS=(--phase "${PHASE}" --dense_model "${DENSE_MODEL}" --base_model "${BASE_MODEL}" --output_dir "${OUTPUT_DIR}" --checkpoint_dir "${TEMP_ROOT}" --batch_size "${BATCH_SIZE:-32}" --device cuda)
[ -n "${START:-}" ] && ARGS+=(--start "${START}" --end "${END:?END is required with START}")
[ -n "${STARTS:-}" ] && ARGS+=(--starts "${STARTS}")
[ -n "${WINDOW:-}" ] && ARGS+=(--window "${WINDOW}")
[ -n "${FAMILIES:-}" ] && ARGS+=(--families "${FAMILIES}")
[ -n "${BETA:-}" ] && ARGS+=(--beta "${BETA}")
[ "${MATCHED_DENSE_CONTROL:-0}" = "1" ] && ARGS+=(--matched-dense-control)
[ -n "${EXPERT_MODEL:-}" ] && ARGS+=(--expert_model "${EXPERT_MODEL}")
[ -n "${DONOR_MODEL:-}" ] && ARGS+=(--donor_model "${DONOR_MODEL}")
[ -n "${BASELINE_EVAL_DIR:-}" ] && ARGS+=(--baseline_eval_dir "${BASELINE_EVAL_DIR}")
[ -n "${FLORES_DIR:-}" ] && ARGS+=(--include_path "${FLORES_DIR}")
python scripts/flores_layer_localization.py "${ARGS[@]}"
printf '{"experiment_name":"flores-layer-localization","job_id":"%s","job_type":"eval","status":"completed","finished_at":"%s"}\n' "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" > "logs/flores-layer-localization_${SLURM_JOB_ID}.done.json"
