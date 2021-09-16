#!/bin/bash
#SBATCH --job-name=smotenc_reg
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.azzalin@studio.unibo.it
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=smotenc_reg
#SBATCH --gres=gpu:1

cd ../../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

export JOB_NAME=smotenc
export PROBLEM_TYPE=regression
export SAMPLING_TO_PERFORM=SMOTENC
export TEST_SET_PERCENT=0.25
export VALIDATION_SET_PERCENT=0.25

python3 src/invalsi.py
