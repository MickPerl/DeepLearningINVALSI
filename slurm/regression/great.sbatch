#!/bin/bash
#SBATCH --job-name=great_reg
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.azzalin@studio.unibo.it
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=great_reg
#SBATCH --gres=gpu:1

cd ../../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

export JOB_NAME=great
export PROBLEM_TYPE=regression
export NUMBER_OF_LAYERS=20
export NEURONS=512
export LEARNING_RATE=0.005

python3 src/invalsi.py
