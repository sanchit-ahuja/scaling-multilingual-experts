#!/bin/bash
#SBATCH --job-name=e3_seed
#SBATCH --output=logs/e3_seed_%j.out
#SBATCH --error=logs/e3_seed_%j.err
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --account=bfzp-dtai-gh
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G

# E3 few-shot seed-robustness for one (checkpoint x benchmark) job.
# Positional args avoid the --export comma-splitting bug (task lists/seeds have commas):
#   sbatch slurm_e3_seed_sweep.sh <TAG> [SEEDS] [LIMIT]
# TAG must match a line from scripts/e3_jobs.py (tag|pretrained|benchmark|include|tasks).

set -euo pipefail

REPO="/u/sahuja1/scaling-multilingual-experts"
module load cuda/12.6.1 2>/dev/null || true
source /u/sahuja1/x-elm-v2/.venv/bin/activate
export DATA_ROOT="${DATA_ROOT:-/work/nvme/bfzp}"
export CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-/work/nvme/bfzp/checkpoints}"
export HF_HOME="${HF_HOME:-/work/nvme/bfzp/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd "${REPO}"

TAG="${1:?usage: sbatch slurm_e3_seed_sweep.sh <TAG> [SEEDS] [LIMIT]}"
SEEDS="${2:-1,2,3,4,5}"
LIMIT="${3:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/e3_seed_sweep}"

LINE="$(python scripts/e3_jobs.py | grep -m1 "^${TAG}|")"
if [ -z "${LINE}" ]; then echo "TAG '${TAG}' not found in e3_jobs.py"; exit 1; fi
PRETRAINED="$(echo "${LINE}" | cut -d'|' -f2)"
BENCH="$(echo "${LINE}" | cut -d'|' -f3)"
INCLUDE="$(echo "${LINE}" | cut -d'|' -f4)"
TASKS="$(echo "${LINE}" | cut -d'|' -f5)"

echo "=== E3 seed sweep: ${TAG} ==="
echo "pretrained=${PRETRAINED}"
echo "benchmark=${BENCH}  seeds=${SEEDS}  limit='${LIMIT}'  bs=${BATCH_SIZE}"
echo "tasks=${TASKS}"

LIMIT_ARG=()
[ -n "${LIMIT}" ] && LIMIT_ARG=(--limit "${LIMIT}")

python scripts/e3_seed_sweep.py \
    --pretrained "${PRETRAINED}" \
    --benchmark "${BENCH}" \
    --tasks "${TASKS}" \
    --include_path "${INCLUDE}" \
    --seeds "${SEEDS}" \
    --batch_size "${BATCH_SIZE}" \
    --output "${OUT_ROOT}/${TAG}.jsonl" \
    "${LIMIT_ARG[@]}"
echo "Done: ${OUT_ROOT}/${TAG}.jsonl"
