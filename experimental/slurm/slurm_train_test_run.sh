#!/bin/bash
#SBATCH --job-name=gemma3_pretrain_h200
#SBATCH --output=gemma3_pretrain_h200_%j.out
#SBATCH --error=gemma3_pretrain_h200_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=500G

module load cuda/12.8.0
module load anaconda3/2024.06
source /home/ahuja.sanc/x-elm-v2/.venv/bin/activate

export HF_HOME="/scratch/ahuja.sanc/hf_cache"
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# Run with accelerate for multi-GPU
accelerate launch --config_file accelerate_config.yaml \
    train_gemma.py \
    --data_prefix /scratch/ahuja.sanc/madlad-tokenized-dataset \
    --serialization_dir /scratch/ahuja.sanc/gemma3_full_checkpoints \
    --model_name google/gemma-3-1b-pt \
    --train_bsz 16 \
    --valid_bsz 16 \
    --grad_accum 8 \
    --max_steps 50000 \
    --lr 5e-5 \
    --warmup_steps 1000 \
    --eval_steps 2500 \
    --save_steps 2500 \
    --num_proc 90 \
    --data_fraction 1.0
