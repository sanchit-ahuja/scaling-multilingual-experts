#!/bin/bash
#SBATCH --job-name=expert-rev-direction
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# CPU-bound rescore of saved generations using the paper's first-line protocol.
# DeltaAI accepts batch jobs only with a GH200 allocation.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${VENV:-${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/.venv}/bin/activate"
cd "${REPO}"
mkdir -p logs
export RESULTS_BASE="${RESULTS_BASE:-${DATA_ROOT:-data}}"
python scripts/expert_reversion_direction.py
