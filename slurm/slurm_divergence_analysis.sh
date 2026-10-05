#!/bin/bash
# =============================================================================
# Weight-space divergence analysis across training strategies
#
# Usage:
#   # Run all strategies (submits one job per strategy, then a comparison job)
#   bash slurm/slurm_divergence_analysis.sh
#
#   # Run a single strategy
#   sbatch --export=STRATEGY=dense slurm/slurm_divergence_analysis.sh
#
#   # Compare only (after all analyze jobs are done)
#   bash slurm/slurm_divergence_analysis.sh --compare-only
#
# Strategies: dense, layer_range, freeze, single_expert
# =============================================================================

#SBATCH --job-name=divergence_analysis
#SBATCH --output=logs/divergence_analysis_%j.out
#SBATCH --error=logs/divergence_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

# =============================================================================
# Configuration
# =============================================================================

CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${DATA_ROOT}/checkpoints}"
# BASE_MODEL_PATH can be a local checkpoint or a HuggingFace model ID.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-google/gemma-3-4b-pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-divergence_results}"

# Family names and their expert checkpoint name casing
# Expert checkpoints: {family}_gemma_4b_expert (Indic is capitalized)
# Layer-range/freeze: gemma_4b_{family}_layer_reg / gemma_4b_{family}_freeze
FAMILIES=("slavic" "germanic" "indic" "austronesian" "romance")

# Expert checkpoint names (case-sensitive as on disk)
declare -A EXPERT_NAMES
EXPERT_NAMES[slavic]="slavic_gemma_4b_expert"
EXPERT_NAMES[germanic]="germanic_gemma_4b_expert"
EXPERT_NAMES[indic]="Indic_gemma_4b_expert"
EXPERT_NAMES[austronesian]="austronesian_gemma_4b_expert"
EXPERT_NAMES[romance]="romance_gemma_4b_expert"

declare -A STRATEGY_LABEL
STRATEGY_LABEL[dense]="Dense (no reg)"
STRATEGY_LABEL[layer_range]="Layer-Range L2-SP"
STRATEGY_LABEL[freeze]="Freeze Middle"
STRATEGY_LABEL[single_expert]="Single Expert"

# =============================================================================
# Helper: resolve checkpoint path for a strategy + family
# =============================================================================

resolve_checkpoint() {
    local strategy=$1
    local family=$2
    local ckpt_dir=""

    case ${strategy} in
        "single_expert")
            ckpt_dir="${CHECKPOINTS_ROOT}/${EXPERT_NAMES[${family}]}"
            ;;
        "layer_range")
            ckpt_dir="${CHECKPOINTS_ROOT}/gemma_4b_${family}_layer_reg"
            ;;
        "freeze")
            ckpt_dir="${CHECKPOINTS_ROOT}/gemma_4b_${family}_freeze"
            ;;
        "dense")
            ckpt_dir="${CHECKPOINTS_ROOT}/gemma_4b_dense_25b"
            ;;
    esac

    # Prefer "final" subdir
    if [ -d "${ckpt_dir}/final" ]; then
        echo "${ckpt_dir}/final"
    else
        echo "${ckpt_dir}"
    fi
}

# =============================================================================
# Batch mode: submit one job per strategy + a comparison job
# =============================================================================

if [ -z "${SLURM_JOB_ID:-}" ]; then
    # We're NOT inside a SLURM job — act as a submission script

    if [ "${1:-}" == "--compare-only" ]; then
        echo "Running comparison only..."
        source "$(dirname "$0")/common.sh"
        RESULT_DIRS=""
        COMPARE_LABELS=""
        for STRAT in dense layer_range freeze single_expert; do
            RDIR="${OUTPUT_ROOT}/${STRAT}"
            if [ -f "${RDIR}/divergence_results.json" ]; then
                RESULT_DIRS="${RESULT_DIRS} ${RDIR}"
                COMPARE_LABELS="${COMPARE_LABELS} \"${STRATEGY_LABEL[${STRAT}]}\""
            else
                echo "WARNING: Missing results for ${STRAT} at ${RDIR}/divergence_results.json, skipping"
            fi
        done

        if [ -z "${RESULT_DIRS}" ]; then
            echo "ERROR: No results found to compare. Run analyze first."
            exit 1
        fi

        eval python analyze_divergence.py compare \
            --result_dirs ${RESULT_DIRS} \
            --compare_labels ${COMPARE_LABELS} \
            --output_dir "${OUTPUT_ROOT}/comparison"
        exit 0
    fi

    mkdir -p logs

    echo "Submitting divergence analysis jobs..."
    JOB_IDS=""
    for STRAT in dense layer_range freeze single_expert; do
        JID=$(sbatch --export=STRATEGY=${STRAT} \
                     --job-name="div_${STRAT}" \
                     --output="logs/div_${STRAT}_%j.out" \
                     --error="logs/div_${STRAT}_%j.err" \
                     slurm/slurm_divergence_analysis.sh | awk '{print $4}')
        echo "  Submitted ${STRAT}: job ${JID}"
        JOB_IDS="${JOB_IDS}:${JID}"
    done

    # Submit comparison job that depends on all analyze jobs
    COMPARE_JID=$(sbatch --dependency=afterok${JOB_IDS} \
                         --export=STRATEGY=compare \
                         --job-name="div_compare" \
                         --output="logs/div_compare_%j.out" \
                         --error="logs/div_compare_%j.err" \
                         --time=00:30:00 \
                         --gres=gpu:0 \
                         --mem=16G \
                         slurm/slurm_divergence_analysis.sh | awk '{print $4}')
    echo "  Submitted comparison: job ${COMPARE_JID} (depends on${JOB_IDS})"
    echo ""
    echo "Monitor with: squeue -u \$USER -n div_dense,div_layer_range,div_freeze,div_single_expert,div_compare"
    exit 0
fi

# =============================================================================
# Inside SLURM job — run analysis for one strategy (or comparison)
# =============================================================================

source "$(dirname "$0")/common.sh"
cd "${PROJECT_ROOT}"

: ${STRATEGY:="dense"}

echo "============================================"
echo "Divergence Analysis"
echo "  Strategy: ${STRATEGY}"
echo "  Base model: ${BASE_MODEL_PATH}"
echo "  Output: ${OUTPUT_ROOT}/${STRATEGY}"
echo "============================================"

# --- Comparison mode ---
if [ "${STRATEGY}" == "compare" ]; then
    echo "Running cross-strategy comparison..."
    RESULT_DIRS=""
    COMPARE_LABELS=""
    for STRAT in dense layer_range freeze single_expert; do
        RDIR="${OUTPUT_ROOT}/${STRAT}"
        if [ -f "${RDIR}/divergence_results.json" ]; then
            RESULT_DIRS="${RESULT_DIRS} ${RDIR}"
            COMPARE_LABELS="${COMPARE_LABELS} \"${STRATEGY_LABEL[${STRAT}]}\""
        else
            echo "WARNING: Missing results for ${STRAT}, skipping"
        fi
    done

    if [ -z "${RESULT_DIRS}" ]; then
        echo "ERROR: No results found to compare"
        exit 1
    fi

    eval python analyze_divergence.py compare \
        --result_dirs ${RESULT_DIRS} \
        --compare_labels ${COMPARE_LABELS} \
        --output_dir "${OUTPUT_ROOT}/comparison"
    exit 0
fi

# --- Analyze mode: build expert paths for this strategy ---
EXPERT_PATHS=""
EXPERT_LABELS=""
MISSING=0

for FAMILY in "${FAMILIES[@]}"; do
    CKPT_DIR=$(resolve_checkpoint "${STRATEGY}" "${FAMILY}")

    # Dense has one checkpoint for all families — use it once
    if [ "${STRATEGY}" == "dense" ]; then
        if [ -z "${EXPERT_PATHS}" ]; then
            if [ -d "${CKPT_DIR}" ]; then
                EXPERT_PATHS="${CKPT_DIR}"
                EXPERT_LABELS="dense_all"
            else
                echo "WARNING: Dense checkpoint not found: ${CKPT_DIR}"
                MISSING=1
            fi
        fi
        continue
    fi

    if [ -d "${CKPT_DIR}" ]; then
        EXPERT_PATHS="${EXPERT_PATHS} ${CKPT_DIR}"
        EXPERT_LABELS="${EXPERT_LABELS} ${FAMILY}"
    else
        echo "WARNING: Checkpoint not found for ${FAMILY}: ${CKPT_DIR}"
        MISSING=1
    fi
done

if [ -z "${EXPERT_PATHS}" ]; then
    echo "ERROR: No expert checkpoints found for strategy ${STRATEGY}"
    exit 1
fi

if [ ${MISSING} -eq 1 ]; then
    echo "WARNING: Some checkpoints missing, proceeding with available experts"
fi

echo "Expert paths:${EXPERT_PATHS}"
echo "Expert labels:${EXPERT_LABELS}"

python analyze_divergence.py analyze \
    --base_path "${BASE_MODEL_PATH}" \
    --expert_paths ${EXPERT_PATHS} \
    --expert_labels ${EXPERT_LABELS} \
    --output_dir "${OUTPUT_ROOT}/${STRATEGY}" \
    --strategy_label "${STRATEGY_LABEL[${STRATEGY}]}" \
    --plot

echo "============================================"
echo "Done: ${STRATEGY}"
echo "Results: ${OUTPUT_ROOT}/${STRATEGY}/divergence_results.json"
echo "============================================"
