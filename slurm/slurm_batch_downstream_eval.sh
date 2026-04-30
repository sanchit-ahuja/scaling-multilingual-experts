#!/bin/bash
# =============================================================================
# Batch Downstream Eval Submission Script
# 
# Submits one SLURM job per checkpoint for a given benchmark.
# Skips checkpoints that already have results.
#
# Usage:
#   bash slurm/slurm_batch_downstream_eval.sh <benchmark> [--dry-run]
#
# Examples:
#   bash slurm/slurm_batch_downstream_eval.sh belebele
#   bash slurm/slurm_batch_downstream_eval.sh global_piqa_completions
#   bash slurm/slurm_batch_downstream_eval.sh flores
#   bash slurm/slurm_batch_downstream_eval.sh perplexity
#   bash slurm/slurm_batch_downstream_eval.sh belebele --dry-run
#   bash slurm/slurm_batch_downstream_eval.sh perplexity --force
# =============================================================================

set -euo pipefail

BENCHMARK="${1:-}"
DRY_RUN=false
FORCE=false

# Parse flags
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --force)
            FORCE=true
            ;;
    esac
done

if [ -z "$BENCHMARK" ]; then
    echo "Error: benchmark argument required"
    echo "Usage: bash slurm/slurm_batch_downstream_eval.sh <benchmark> [--dry-run]"
    echo "  Benchmarks: belebele, global_piqa_completions, flores, perplexity"
    exit 1
fi

# =============================================================================
# Configuration
# =============================================================================

# Resolve paths from environment (see scripts/setup_env.sh.example).
# Defaults assume a flat layout under $DATA_ROOT; override individually if needed.
: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${DATA_ROOT:=${PROJECT_ROOT}/data}"
CHECKPOINT_BASE="${CHECKPOINTS_ROOT:-${DATA_ROOT}/checkpoints}"
RESULTS_BASE="${RESULTS_BASE:-${DATA_ROOT}}"
WORKER_SCRIPT="${PROJECT_ROOT}/slurm/slurm_baseline_downstream_eval.sh"
PERPLEXITY_WORKER_SCRIPT="${PROJECT_ROOT}/slurm/slurm_perplexity_eval.sh"

# All checkpoint paths to evaluate (relative to CHECKPOINT_BASE unless absolute/HF)
# Format: "CHECKPOINT_PATH|FAMILY|STRATEGY"
CHECKPOINTS=(
    # Expert models - final + reverted
    # "${CHECKPOINT_BASE}/austronesian_gemma_4b_expert/final|Austronesian|expert"
    # "${CHECKPOINT_BASE}/austronesian_gemma_4b_expert/reverted-9000|Austronesian|expert-reverted"
    # "${CHECKPOINT_BASE}/germanic_gemma_4b_expert/final|Germanic|expert"
    # "${CHECKPOINT_BASE}/germanic_gemma_4b_expert/reverted|Germanic|expert-reverted"
    # "${CHECKPOINT_BASE}/Indic_gemma_4b_expert/checkpoint-7000|Indic|expert"
    # "${CHECKPOINT_BASE}/Indic_gemma_4b_expert/reverted-7000|Indic|expert-reverted"
    # "${CHECKPOINT_BASE}/romance_gemma_4b_expert/final|Romance|expert"
    # "${CHECKPOINT_BASE}/romance_gemma_4b_expert/reverted|Romance|expert-reverted"
    # "${CHECKPOINT_BASE}/slavic_gemma_4b_expert/final|Slavic|expert"
    # Slavic expert-reverted — newly created checkpoint (2026-03-12)
    # "${CHECKPOINT_BASE}/slavic_gemma_4b_expert/reverted|Slavic|expert-reverted"

    # Freeze models - final only (no reverted available, Slavic skipped)
    # "${CHECKPOINT_BASE}/gemma_4b_austronesian_freeze/final|Austronesian|freeze"
    # "${CHECKPOINT_BASE}/gemma_4b_slavic_freeze/final|Slavic|freeze"
    # "${CHECKPOINT_BASE}/gemma_4b_germanic_freeze/final|Germanic|freeze"
    # "${CHECKPOINT_BASE}/gemma_4b_indic_freeze/final|Indic|freeze"
    # "${CHECKPOINT_BASE}/gemma_4b_romance_freeze/final|Romance|freeze"

    # Layer-reg models - final only (no reverted available)
    # "${CHECKPOINT_BASE}/gemma_4b_austronesian_layer_reg/final|Austronesian|layer-reg"
    # "${CHECKPOINT_BASE}/gemma_4b_germanic_layer_reg/final|Germanic|layer-reg"
    # "${CHECKPOINT_BASE}/gemma_4b_indic_layer_reg/final|Indic|layer-reg"
    # "${CHECKPOINT_BASE}/gemma_4b_romance_layer_reg/final|Romance|layer-reg"
    # "${CHECKPOINT_BASE}/gemma_4b_slavic_layer_reg/final|Slavic|layer-reg"

    # Dense model - final + reverted
    "${CHECKPOINT_BASE}/gemma_4b_dense_25b/final|Dense|dense"
    "${CHECKPOINT_BASE}/gemma_4b_dense_25b/reverted|Dense|dense-reverted"

    # Base model (HuggingFace)
    "google/gemma-3-4b-pt|Base|base"

    # Expert soup (uniform average)
    # "${CHECKPOINT_BASE}/gemma_4b_expert_soup|All|expert-soup"
)

# =============================================================================
# Already-completed evaluations (results exist as CSVs in gemma_results/)
# Format: "checkpoint_path|benchmark" entries that should be skipped
# =============================================================================

ALREADY_DONE_CSV=(
    # Belebele — Indic expert + reverted (result dirs exist)
    "${CHECKPOINT_BASE}/Indic_gemma_4b_expert/checkpoint-7000|belebele"
    "${CHECKPOINT_BASE}/Indic_gemma_4b_expert/reverted-7000|belebele"
    # Belebele — Austronesian expert + reverted (result dirs exist)
    "${CHECKPOINT_BASE}/austronesian_gemma_4b_expert/final|belebele"
    "${CHECKPOINT_BASE}/austronesian_gemma_4b_expert/reverted-8000|belebele"
    # Belebele — Slavic expert (result dir exists; expert-reverted is NEW)
    "${CHECKPOINT_BASE}/slavic_gemma_4b_expert/final|belebele"

    # PiQA — Indic expert + reverted (result dirs exist)
    "${CHECKPOINT_BASE}/Indic_gemma_4b_expert/checkpoint-7000|global_piqa_completions"
    "${CHECKPOINT_BASE}/Indic_gemma_4b_expert/reverted-7000|global_piqa_completions"
    # PiQA — Austronesian expert + reverted (result dirs exist)
    "${CHECKPOINT_BASE}/austronesian_gemma_4b_expert/final|global_piqa_completions"
    "${CHECKPOINT_BASE}/austronesian_gemma_4b_expert/reverted-8000|global_piqa_completions"
    # PiQA — Slavic expert (result dir exists; expert-reverted is NEW)
    "${CHECKPOINT_BASE}/slavic_gemma_4b_expert/final|global_piqa_completions"

    # Flores — Indic expert + reverted (result dirs exist)
    "${CHECKPOINT_BASE}/Indic_gemma_4b_expert/checkpoint-7000|flores"
    "${CHECKPOINT_BASE}/Indic_gemma_4b_expert/reverted-7000|flores"
    # Flores — Austronesian expert + reverted (result dirs exist)
    "${CHECKPOINT_BASE}/austronesian_gemma_4b_expert/final|flores"
    "${CHECKPOINT_BASE}/austronesian_gemma_4b_expert/reverted-8000|flores"
    # Flores — Slavic expert (result dir exists; expert-reverted is NEW)
    "${CHECKPOINT_BASE}/slavic_gemma_4b_expert/final|flores"

    # Perplexity — Slavic expert (result dir exists; expert-reverted is NEW)
    "${CHECKPOINT_BASE}/slavic_gemma_4b_expert/final|perplexity"
)

# =============================================================================
# Helper functions
# =============================================================================

# Generate the expected result directory name for a checkpoint+benchmark combo
# Uses the same naming logic as slurm_baseline_downstream_eval.sh
get_result_name() {
    local checkpoint_path="$1"
    local benchmark="$2"
    local family="$3"
    
    local checkpoint_name
    checkpoint_name="$(basename "$(dirname "$checkpoint_path")")_$(basename "$checkpoint_path")"
    
    if [ "$benchmark" = "perplexity" ]; then
        echo "perplexity_eval_${checkpoint_name}"
    else
        # Worker script prepends FAMILY_ when --family is passed
        if [ -n "$family" ]; then
            echo "baseline_lm_eval_${family}_${checkpoint_name}_${benchmark}"
        else
            echo "baseline_lm_eval_${checkpoint_name}_${benchmark}"
        fi
    fi
}

# Check if results already exist (in result dirs OR in CSV skip list)
results_exist() {
    local result_name="$1"
    local checkpoint_path="$2"
    local benchmark="$3"
    
    # Check 1: result directory exists with expected output file
    if [ -d "${RESULTS_BASE}/${result_name}" ]; then
        if [ "$benchmark" = "perplexity" ]; then
            # Perplexity results are stored as per_language_eval_results.json
            if [ -f "${RESULTS_BASE}/${result_name}/per_language_eval_results.json" ]; then
                return 0
            fi
        else
            # lm_eval nests results*.json inside a model-path subdirectory
            if find "${RESULTS_BASE}/${result_name}" -name 'results*.json' -print -quit 2>/dev/null | grep -q .; then
                return 0
            fi
        fi
    fi
    
    # Check 2: In the CSV-based skip list
    local skip_key="${checkpoint_path}|${benchmark}"
    for done_entry in "${ALREADY_DONE_CSV[@]}"; do
        if [ "$done_entry" = "$skip_key" ]; then
            return 0
        fi
    done
    
    return 1
}

# =============================================================================
# Main loop
# =============================================================================

# Benchmark-specific SLURM time limits
# Flores takes ~10hrs per checkpoint — set 14hrs for safety margin
# Belebele and PiQA are much faster — 10hrs is plenty
case "$BENCHMARK" in
    flores)
        TIME_LIMIT="14:00:00"
        ;;
    *)
        TIME_LIMIT="10:00:00"
        ;;
esac

echo "========================================================"
echo "BATCH DOWNSTREAM EVAL SUBMISSION"
echo "========================================================"
echo "Benchmark:  ${BENCHMARK}"
echo "Time limit: ${TIME_LIMIT}"
echo "Dry-run:    ${DRY_RUN}"
echo "Force:      ${FORCE}"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "========================================================"
echo ""

SUBMITTED=0
SKIPPED=0
MISSING=0
ERRORS=""

for entry in "${CHECKPOINTS[@]}"; do
    # Parse entry
    IFS='|' read -r CHECKPOINT_PATH FAMILY STRATEGY <<< "$entry"
    
    # Validate checkpoint exists (skip for HuggingFace model IDs)
    if [[ "$CHECKPOINT_PATH" == /* ]] && [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "  ⚠ MISSING: $CHECKPOINT_PATH"
        MISSING=$((MISSING + 1))
        ERRORS="${ERRORS}\n  Missing: ${CHECKPOINT_PATH}"
        continue
    fi
    
    # Check if results already exist
    RESULT_NAME=$(get_result_name "$CHECKPOINT_PATH" "$BENCHMARK" "$FAMILY")
    
    if results_exist "$RESULT_NAME" "$CHECKPOINT_PATH" "$BENCHMARK"; then
        if [ "$FORCE" = true ]; then
            echo "  ✗ FORCE DELETE: ${RESULTS_BASE}/${RESULT_NAME}"
            if [ "$DRY_RUN" = false ]; then
                rm -rf "${RESULTS_BASE}/${RESULT_NAME}"
            fi
        else
            echo "  ✓ SKIP (exists): ${FAMILY}/${STRATEGY} → ${RESULT_NAME}"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi
    
    # Select worker script
    if [ "$BENCHMARK" = "perplexity" ]; then
        ACTIVE_WORKER="$PERPLEXITY_WORKER_SCRIPT"
    else
        ACTIVE_WORKER="$WORKER_SCRIPT"
    fi
    
    # Set per-benchmark batch size
    EXTRA_ARGS=""
    if [ "$BENCHMARK" = "flores" ]; then
        EXTRA_ARGS="--batch_size 256"
    fi
    
    # Submit job
    echo "  → SUBMIT: ${FAMILY}/${STRATEGY}"
    echo "    Checkpoint: ${CHECKPOINT_PATH}"
    echo "    Expected output: ${RESULT_NAME}"
    
    if [ "$DRY_RUN" = false ]; then
        JOB_ID=$(sbatch \
            --job-name="eval_${FAMILY}_${STRATEGY}_${BENCHMARK}" \
            --output="logs/eval_${FAMILY}_${STRATEGY}_${BENCHMARK}_%j.out" \
            --error="logs/eval_${FAMILY}_${STRATEGY}_${BENCHMARK}_%j.err" \
            --time="${TIME_LIMIT}" \
            "$ACTIVE_WORKER" \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --dataset "$BENCHMARK" \
            --family "$FAMILY" \
            $EXTRA_ARGS \
            2>&1)
        echo "    Job: ${JOB_ID}"
    else
        echo "    [DRY-RUN] Would submit: sbatch --time=${TIME_LIMIT} $ACTIVE_WORKER --checkpoint_path $CHECKPOINT_PATH --dataset $BENCHMARK --family $FAMILY $EXTRA_ARGS"
    fi
    
    SUBMITTED=$((SUBMITTED + 1))
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "========================================================"
echo "SUMMARY"
echo "========================================================"
echo "  Submitted: ${SUBMITTED}"
echo "  Skipped (already done): ${SKIPPED}"
echo "  Missing checkpoints: ${MISSING}"
echo "  Total: $((SUBMITTED + SKIPPED + MISSING)) / ${#CHECKPOINTS[@]}"

if [ -n "$ERRORS" ]; then
    echo ""
    echo "Errors:"
    echo -e "$ERRORS"
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "*** DRY-RUN MODE — No jobs were actually submitted ***"
fi

echo "========================================================"
