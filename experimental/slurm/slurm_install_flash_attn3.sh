#!/bin/bash
#SBATCH --job-name=install_flash_attn
#SBATCH --output=install_flash_attn_%j.out
#SBATCH --error=install_flash_attn_%j.err
#SBATCH --partition=multigpu  # or gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1     # Just 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=4                   # Number of tasks
#SBATCH --cpus-per-task=2            # Number of CPUs per task
#SBATCH --mem=64G   


module load cuda/12.8.0
module load anaconda3/2024.06
# Activate your conda environment if needed
source /home/ahuja.sanc/x-elm-v2/.venv/bin/activate
export MAX_JOBS=190

export FLASH_ATTENTION_DISABLE_BACKWARD=FALSE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
export FLASH_ATTENTION_DISABLE_FP16=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_CLUSTER=FALSE
export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=FALSE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE

cd /projects/lilac_lab/sanchit/flash-attention/hopper
python setup.py install
