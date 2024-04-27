#!/bin/bash
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem=30g
#SBATCH -J "ML541-final"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "EPYC-7543&(A100|V100)"

module load python/3.12.3/mftt2ua
module load cuda11.7/toolkit/11.7.1

python -m venv env

source env/bin/activate

pip install -r requirements.txt

python3 finetune.py
