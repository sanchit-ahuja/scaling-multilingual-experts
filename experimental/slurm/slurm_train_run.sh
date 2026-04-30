#!/bin/bash
#SBATCH --job-name=dense_25b_train
#SBATCH --output=dense_25b_train_%j.out
#SBATCH --error=dense_25b_train_%j.err
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --account=bfzp-dtai-gh
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G

module load cuda/12.6.1
source /u/sahuja1/x-elm-v2/.venv/bin/activate

export HF_HOME="/work/nvme/bfzp/hf_cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# NCCL settings for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL

if [ -f /u/sahuja1/x-elm-v2/.env ]; then
    export $(grep -v '^#' /u/sahuja1/x-elm-v2/.env | xargs)
fi

SERIALIZATION_DIR=/work/nvme/bfzp/qwen_base_checkpoints

hf auth login --token ${HF_TOKEN}

# Multi-GPU training with accelerate config
accelerate launch \
    --config_file /u/sahuja1/x-elm-v2/accelerate_config.yaml \
    train_trl.py \
    --data_prefix /work/nvme/bfzp/madlad-qwen-tokenized-5B \
    --serialization_dir ${SERIALIZATION_DIR} \
    --model_name Qwen/Qwen3-1.7B-Base \
    --train_bsz 16 \
    --grad_accum 3 \
    --max_steps 50000 \
    --lr 2e-5 \
    --warmup_steps 1000 \
    --logging_steps 100 \
    --eval_steps 5000 \
    --max_eval_samples 10000 \
    --save_steps 2500 \
    --data_fraction 1.0 \
    --resume_from_checkpoint true \
    --num_proc 4 \
    --use_gradient_checkpointing 