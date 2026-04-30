#!/bin/bash
#SBATCH --job-name=dataset_buckets
#SBATCH --output=dataset_buckets_%j.out
#SBATCH --error=dataset_buckets_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32


module load anaconda3/2024.06
# Activate your conda environment if needed
source activate /home/ahuja.sanc/.conda/envs/xelm-env
pip install datasets==3.6.0
export HF_HOME="/scratch/ahuja.sanc/hf_cache"
hf auth login --token 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
export HF_AUTH_TOKEN="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"
#export HF_XET_DISABLE=1  # Disable XET entirely
#export HF_HUB_DISABLE_PROGRESS_BARS=1
# Run the script

python /home/ahuja.sanc/x-elm-v2/create_dataset_buckets.py
