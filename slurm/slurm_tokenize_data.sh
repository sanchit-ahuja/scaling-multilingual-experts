#!/bin/bash
#SBATCH --job-name=tokenize_data
#SBATCH --output=logs/tokenize_data_%j.out
#SBATCH --error=logs/tokenize_data_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# =============================================================================
# NOTE: SBATCH directives above are templated. Edit partition, account, time
#       etc. for your cluster.
# =============================================================================

# Source common setup (venv, HF login, env vars)
source "$(dirname "$0")/common.sh"
hf_login

python "${PROJECT_ROOT}/tokenize_data.py" --sample-percentage 5
