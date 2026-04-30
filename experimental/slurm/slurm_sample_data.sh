#!/bin/bash
#SBATCH --job-name=tokenize_data
#SBATCH --output=tokenize_data_%j.out
#SBATCH --error=tokenize_data_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32


module load anaconda3/2024.06
# Activate your conda environment if needed
source /home/ahuja.sanc/x-elm-v2/.venv/bin/activate
python sample_tokenized_data.py --input_dir /scratch/ahuja.sanc/madlad-qwen-tokenized-dataset --output_dir /scratch/ahuja.sanc/madlad-qwen-tokenized-5B --num_proc 30
