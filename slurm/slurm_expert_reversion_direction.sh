#!/bin/bash
#SBATCH --job-name=expert-rev-direction
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ghx4
#SBATCH --account=bfzp-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# CPU-bound rescore of saved generations using the paper's first-line protocol.
# DeltaAI accepts batch jobs only with a GH200 allocation.
set -euo pipefail
REPO="/u/sahuja1/scaling-multilingual-experts"
source /u/sahuja1/x-elm-v2/.venv/bin/activate
cd "${REPO}"
mkdir -p logs
export RESULTS_BASE="${RESULTS_BASE:-/work/nvme/bfzp}"
python scripts/expert_reversion_direction.py
