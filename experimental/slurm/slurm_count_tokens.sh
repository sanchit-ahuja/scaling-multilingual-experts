#!/bin/bash
#SBATCH --job-name=count_tokens_madlad
#SBATCH --output=count_tokens_madlad_%j.out
#SBATCH --error=count_tokens_madlad_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Load any required modules
# module load python/3.x  # Uncomment and adjust version as needed

#rm -rf /scratch/ahuja.sanc/hf_cache/.locks
#rm -rf /scratch/ahuja.sanc/hf_cache/.xet*
#find /scratch/ahuja.sanc/hf_cache -name "*.lock" -delete 2>/dev/null || true

# Run the scriptk

module load anaconda3/2024.06
# Activate your conda environment if needed
source /home/ahuja.sanc/x-elm-v2/.venv/bin/activate
export HF_HOME="/scratch/ahuja.sanc/hf_cache"
hf auth login --token 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
export HF_AUTH_TOKEN="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"

python /home/ahuja.sanc/x-elm-v2/analyze_processed_tokens.py
