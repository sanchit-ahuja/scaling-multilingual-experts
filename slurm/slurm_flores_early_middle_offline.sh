#!/bin/bash
#SBATCH --job-name=flores-early-middle-offline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
# DeltaAI rejects zero-GPU jobs on its available partitions.  The analysis is
# CPU-parallel, but one GH200 must be reserved for scheduler admission.
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
# Four processes keep the paired-sample state within the 96 GB allocation.
# Each process receives a full serialized bootstrap shard; 32 processes caused
# an OOM kill on job 3072772.
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO}/.venv/bin/activate"
cd "${REPO}"
mkdir -p logs
ARGS=(--output-dir "${OUTPUT_DIR:-${RESULTS_ROOT:-results}/${USER}/flores_early_middle_followup/offline}"
  --n-bootstrap "${N_BOOTSTRAP:-10000}" --seed "${SEED:-20260902}"
  --workers "${SLURM_CPUS_PER_TASK:-1}" --sample-size "${SAMPLE_SIZE:-0}")
if [ -n "${PHASE_B_SUMMARY:-}" ]; then
  ARGS+=(--phase-b-summary "${PHASE_B_SUMMARY}")
else
  ARGS+=(--paper-task-csv "${PAPER_TASK_CSV:-${PAPER_DATA_ROOT:-data}/flores-truncation-figure/flores_alpha_sweep_fixed_task.csv}"
    --sliding-root "${SLIDING_ROOT:-${RESULTS_ROOT:-results}/flores_layer_localization}")
fi
python scripts/analyze_flores_early_middle.py "${ARGS[@]}"
