#!/bin/bash
#SBATCH --job-name=flores-early-middle-offline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ghx4
#SBATCH --account=bfzp-dtai-gh
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
REPO="/u/sahuja1/scaling-multilingual-experts"
source "${REPO}/.venv/bin/activate"
cd "${REPO}"
mkdir -p logs
ARGS=(--output-dir "${OUTPUT_DIR:-/work/hdd/bfzp/${USER}/flores_early_middle_followup/offline}"
  --n-bootstrap "${N_BOOTSTRAP:-10000}" --seed "${SEED:-20260902}"
  --workers "${SLURM_CPUS_PER_TASK:-1}" --sample-size "${SAMPLE_SIZE:-0}")
if [ -n "${PHASE_B_SUMMARY:-}" ]; then
  ARGS+=(--phase-b-summary "${PHASE_B_SUMMARY}")
else
  ARGS+=(--paper-task-csv "${PAPER_TASK_CSV:-/work/nvme/bfzp/sahuja1/flores-truncation-figure/flores_alpha_sweep_fixed_task.csv}"
    --sliding-root "${SLIDING_ROOT:-/work/hdd/bfzp/sahuja1/flores_layer_localization}")
fi
python scripts/analyze_flores_early_middle.py "${ARGS[@]}"
