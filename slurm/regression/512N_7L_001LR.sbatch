#!/bin/bash
#SBATCH --job-name=512N_7L_001LR_cla
#SBATCH --mail-type=END
#SBATCH --mail-user=marco.ferrati@studio.unibo.it
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=512N_7L_001LR_cla
#SBATCH --gres=gpu:1

cd ../../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

export JOB_NAME=512N_7L_001LR
export PROBLEM_TYPE=regression
export NEURONS=512
export NUMBER_OF_LAYERS=7
export LEARNING_RATE=0.01

python3 src/invalsi.py
