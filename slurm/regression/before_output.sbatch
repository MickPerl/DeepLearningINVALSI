#!/bin/bash
#SBATCH --job-name=before_output_reg
#SBATCH --mail-type=END
#SBATCH --mail-user=marco.ferrati@studio.unibo.it
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=before_output_reg
#SBATCH --gres=gpu:1

cd ../../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

export JOB_NAME=before_output
export PROBLEM_TYPE=regression
export BATCH_NORMALIZATION=before_output

python3 src/invalsi.py
