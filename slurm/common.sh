#!/bin/bash
# =============================================================================
# Common setup for x-elm-v2 slurm scripts
# Source this file at the top of your slurm scripts:
#   source slurm/common.sh
#
# Configuration (set these in your environment or in a .env file at PROJECT_ROOT):
#   PROJECT_ROOT      - path to this repository  (default: $HOME/x-elm-v2)
#   DATA_ROOT         - root for tokenized data / checkpoints / hf cache
#                       (default: $PROJECT_ROOT/data)
#   TOKENIZED_DATA    - tokenized dataset root   (default: $DATA_ROOT/tokenized)
#   CHECKPOINTS_ROOT  - checkpoint root          (default: $DATA_ROOT/checkpoints)
#   HF_HOME           - HuggingFace cache        (default: $DATA_ROOT/hf_cache)
#   HF_TOKEN          - HuggingFace access token (required for gated models)
#
# See scripts/setup_env.sh.example for a template.
# =============================================================================

# Load required modules (adjust for your cluster)
module load cuda/12.6.1 2>/dev/null || true

# Activate virtual environment
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/x-elm-v2}"
source "${PROJECT_ROOT}/.venv/bin/activate"

# =============================================================================
# Paths
# =============================================================================

export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
export TOKENIZED_DATA="${TOKENIZED_DATA:-${DATA_ROOT}/tokenized}"
export CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${DATA_ROOT}/checkpoints}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"

# =============================================================================
# Runtime Environment Variables
# =============================================================================

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# NCCL settings for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0

# =============================================================================
# Helper Functions
# =============================================================================

# Load environment variables from .env file
load_env() {
    local env_file="${1:-${PROJECT_ROOT}/.env}"
    if [ -f "$env_file" ]; then
        export $(grep -v '^#' "$env_file" | xargs)
        echo "[common.sh] Loaded environment from $env_file"
    fi
}

# Login to HuggingFace (call after load_env)
hf_login() {
    if [ -n "${HF_TOKEN:-}" ]; then
        huggingface-cli login --token ${HF_TOKEN}
        echo "[common.sh] Logged in to HuggingFace"
    else
        echo "[common.sh] Warning: HF_TOKEN not set, skipping HuggingFace login"
    fi
}

# Print GPU info
print_gpu_info() {
    echo "[common.sh] GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo ""
}

# =============================================================================
# Default Load (call load_env automatically)
# =============================================================================

load_env
