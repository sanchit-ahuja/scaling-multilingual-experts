#!/bin/bash
#SBATCH --job-name=copy_en_ds
#SBATCH --output=copy_en_ds_%j.out
#SBATCH --error=copy_en_ds_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4


module load anaconda3/2024.06
# Activate your conda environment if needed
source activate /home/ahuja.sanc/.conda/envs/xelm-env
export HF_HOME="/scratch/ahuja.sanc/hf_cache"
hf auth login --token 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
export HF_AUTH_TOKEN="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"
#export HF_XET_DISABLE=1  # Disable XET entirely
#export HF_HUB_DISABLE_PROGRESS_BARS=1
# Run the script
cp -r /scratch/ahuja.sanc/madlad-dataset-updated_3/Germanic/en /projects/lilac_lab/sanchit/madlad-dataset/Germanic
