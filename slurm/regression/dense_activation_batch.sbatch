#!/bin/bash
#SBATCH --job-name=dense_activation_batch_cla
#SBATCH --mail-type=END
#SBATCH --mail-user=marco.ferrati@studio.unibo.it
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=dense_activation_batch_cla
#SBATCH --gres=gpu:1

cd ../../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

export JOB_NAME=dense_activation_batch
export PROBLEM_TYPE=regression
export BATCH_NORMALIZATION=dense_activation_batch

python3 src/invalsi.py
