#!/bin/bash
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem=30g
#SBATCH -J "ML541-final"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100

module load python/3.12.3/mftt2ua
module load cuda11.7/toolkit/11.7.1

# Create a virtual environment in a directory named 'env'
python -m venv env

# Activate the virtual environment
source env/bin/activate

# Install required Python packages from requirements.txt
pip install -r requirements.txt

# Execute the Python script
python3 finetune.py
