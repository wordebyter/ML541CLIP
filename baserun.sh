#!/bin/bash
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem=30g
#SBATCH -J "ML541-final"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100

module load cuda
source activate pytorch
python3 finetune.py