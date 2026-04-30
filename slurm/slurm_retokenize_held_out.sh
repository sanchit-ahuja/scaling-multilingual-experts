#!/bin/bash
#SBATCH --job-name=retokenize_held_out
#SBATCH --output=logs/retokenize_held_out_%j.out
#SBATCH --error=logs/retokenize_held_out_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:h100:1
#SBATCH --account=bfzp-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# =============================================================================
# NOTE: SBATCH directives above are templated for NCSA Delta. Edit for your
#       own cluster (partition, account, gres, time, etc.).
# =============================================================================

# Source common setup (venv, HF login, env vars)
source "$(dirname "$0")/common.sh"
hf_login

OUTPUT_DIR="${TOKENIZED_DATA_HELD_OUT:-${DATA_ROOT}/madlad-gemma-held-out-tokenized-dataset}"

# Remove partial/broken directories for the 4 failed families so save_to_disk
# can write cleanly. Romance/ca is intact and is left untouched.
echo "Cleaning up broken partial directories..."
for family in Slavic Germanic Indic Austronesian; do
    lower=$(echo "$family" | tr '[:upper:]' '[:lower:]')
    if [ -d "${OUTPUT_DIR}/${lower}" ]; then
        echo "  Removing ${OUTPUT_DIR}/${lower}/train and ${OUTPUT_DIR}/${lower}/valid"
        rm -rf "${OUTPUT_DIR}/${lower}/train"
        rm -rf "${OUTPUT_DIR}/${lower}/valid"
    fi
done
echo "Cleanup done."

# Re-run tokenization for the 4 failed families only.
# Romance is already complete so we skip it.
python "${PROJECT_ROOT}/tokenize_data.py" \
    --families Slavic Germanic Indic Austronesian \
    --samples-per-family 10000 \
    --output-path "${OUTPUT_DIR}"
