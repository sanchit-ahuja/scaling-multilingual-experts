#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --time=24:00:00
#SBATCH --account=csunivie
#SBATCH --partition=p_csunivie
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue

### Run your job
python /srv/home/users/blevinst24cs/x-elm-v2/clustering/random_clustering.py "${@}"