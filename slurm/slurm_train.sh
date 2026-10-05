#!/bin/bash
#SBATCH --job-name=xelm-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --time=120:00:00

# =============================================================================
# NOTE: SBATCH directives above are templated for the Vienna (dgx-h100) cluster.
#       Edit partition, account, nodelist, gres, time, etc. for your own cluster.
# =============================================================================

source "$(dirname "$0")/common.sh"

### Run your job
srun --label python "${PROJECT_ROOT}/train.py" "${@}"
