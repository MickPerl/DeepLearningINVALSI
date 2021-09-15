#!/bin/bash
#SBATCH --job-name=dense_batch_activation_reg
#SBATCH --mail-type=END
#SBATCH --mail-user=marco.ferrati@studio.unibo.it
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=dense_batch_activation_reg
#SBATCH --gres=gpu:1

cd ../../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

export JOB_NAME=dense_batch_activation
export PROBLEM_TYPE=regression
export BATCH_NORMALIZATION=dense_batch_activation

python3 src/invalsi.py
