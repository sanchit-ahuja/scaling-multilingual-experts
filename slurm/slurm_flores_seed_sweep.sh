#!/bin/bash
#SBATCH --job-name=flores_seed
#SBATCH --output=logs/flores_seed_%j.out
#SBATCH --error=logs/flores_seed_%j.err
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --account=bfzp-dtai-gh
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G

# FLORES seed-robustness for one checkpoint. Positional args (avoid --export comma bug):
#   sbatch slurm_flores_seed_sweep.sh <TAG> [SEEDS] [LIMIT]
# TAG must match scripts/flores_jobs.py (tag|pretrained|tasks). SEEDS are the NEW fewshot
# seeds (baseline 1234 is read offline by the aggregator).

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

TAG="${1:?usage: sbatch slurm_flores_seed_sweep.sh <TAG> [SEEDS] [LIMIT]}"
SEEDS="${2:-1,2}"
LIMIT="${3:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/flores_seed_sweep}"

LINE="$(python scripts/flores_jobs.py | grep -m1 "^${TAG}|")"
if [ -z "${LINE}" ]; then echo "TAG '${TAG}' not found in flores_jobs.py"; exit 1; fi
PRETRAINED="$(echo "${LINE}" | cut -d'|' -f2)"
TASKS="$(echo "${LINE}" | cut -d'|' -f3)"

echo "=== FLORES seed sweep: ${TAG} ==="
echo "pretrained=${PRETRAINED}  seeds=${SEEDS}  limit='${LIMIT}'  bs=${BATCH_SIZE}"
echo "n_tasks=$(echo "${TASKS}" | tr ',' '\n' | wc -l)"

LIMIT_ARG=()
[ -n "${LIMIT}" ] && LIMIT_ARG=(--limit "${LIMIT}")

python scripts/flores_seed_sweep.py \
    --pretrained "${PRETRAINED}" \
    --tasks "${TASKS}" \
    --seeds "${SEEDS}" \
    --batch_size "${BATCH_SIZE}" \
    --output "${OUT_ROOT}/${TAG}.jsonl" \
    "${LIMIT_ARG[@]}"
echo "Done: ${OUT_ROOT}/${TAG}.jsonl"
