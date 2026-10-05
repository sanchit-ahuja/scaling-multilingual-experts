#!/bin/bash

# ============================================
# Language Family Configuration
# Usage:
#   sbatch --job-name=Germanic_layer_range \
#          --export=FAMILY=Germanic,CONFIG_TYPE=layer_range,RESUME=false \
#          slurm/slurm_train_single_experts.sh
#
# Options:
#   FAMILY: Germanic, Slavic, Romance, Indic, Austronesian (default: Germanic)
#   CONFIG_TYPE: single_expert, layer_range, freeze, dense (default: layer_range)
#   RESUME: true (auto-resume from latest) or /path/to/checkpoint (default: false)
#
# Log files: logs/<job-name>_<jobid>.out/err
#   Set --job-name=FAMILY_CONFIG_TYPE for descriptive log filenames
# ============================================

#SBATCH --job-name=austronesian_gemma_4b_expert
#SBATCH --output=logs/austronesian_gemma_4b_expert_%j.out
#SBATCH --error=logs/austronesian_gemma_4b_expert_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G

# Set defaults if not provided via --export
: ${FAMILY:="austronesian"}
: ${CONFIG_TYPE:="single_expert"}
: ${RESUME:="false"}

# Source common setup (venv, HF login, env vars, CHECKPOINTS_ROOT/TOKENIZED_DATA/HF_HOME)
source "$(dirname "$0")/common.sh"
hf_login
print_gpu_info

# Set config path and output directory based on CONFIG_TYPE
case ${CONFIG_TYPE} in
    "single_expert")
        CONFIG_PATH="configs/yaml/train_gemma_single_expert.yaml"
        SERIALIZATION_DIR="${CHECKPOINTS_ROOT}/${FAMILY,,}_gemma_4b_expert"
        RUN_NAME="${FAMILY,,}-expert-gemma4b"
        ;;
    "layer_range")
        CONFIG_PATH="configs/yaml/train_gemma_layer_range.yaml"
        SERIALIZATION_DIR="${CHECKPOINTS_ROOT}/gemma_4b_${FAMILY,,}_layer_reg"
        RUN_NAME="${FAMILY,,}-layer-range-l2sp-gemma4b"
        ;;
    "freeze")
        CONFIG_PATH="configs/yaml/train_gemma_freeze.yaml"
        SERIALIZATION_DIR="${CHECKPOINTS_ROOT}/gemma_4b_${FAMILY,,}_freeze"
        RUN_NAME="${FAMILY,,}-freeze-gemma4b"
        ;;
    "dense")
        CONFIG_PATH="configs/yaml/train_gemma_dense.yaml"
        SERIALIZATION_DIR="${CHECKPOINTS_ROOT}/gemma_4b_dense_25b"
        RUN_NAME="${FAMILY,,}-dense-gemma4b"
        ;;
    *)
        echo "ERROR: Unknown CONFIG_TYPE: ${CONFIG_TYPE}"
        echo "Valid options: single_expert, layer_range, freeze, dense"
        exit 1
        ;;
esac

echo "============================================"
echo "Training Configuration:"
echo "  Family: ${FAMILY}"
echo "  Config Type: ${CONFIG_TYPE}"
echo "  Config File: ${CONFIG_PATH}"
echo "  Output Directory: ${SERIALIZATION_DIR}"
echo "  Run Name: ${RUN_NAME}"
echo "  Resume: ${RESUME}"
echo "============================================"

# Set up resume flags if requested
RESUME_FLAGS=""
if [ "${RESUME}" != "false" ]; then
    RESUME_FLAGS="--checkpoint.resume_from_checkpoint true"
    if [ "${RESUME}" != "true" ]; then
        # User provided a checkpoint path
        RESUME_FLAGS="${RESUME_FLAGS} --checkpoint.checkpoint_path ${RESUME}"
        echo "Resuming from checkpoint: ${RESUME}"
    else
        echo "Auto-resuming from latest checkpoint in ${SERIALIZATION_DIR}"
    fi
fi

# Run training with YAML config + CLI overrides
# Dense runs use families: null from the YAML (train on all families)
FAMILIES_FLAG=""
if [ "${CONFIG_TYPE}" != "dense" ]; then
    FAMILIES_FLAG="--data.families [${FAMILY}]"
fi

accelerate launch train.py \
    --config_path ${CONFIG_PATH} \
    --checkpoint.serialization_dir ${SERIALIZATION_DIR} \
    --checkpoint.run_name "${RUN_NAME}" \
    ${FAMILIES_FLAG} \
    ${RESUME_FLAGS}

echo "Training completed for ${FAMILY}"
