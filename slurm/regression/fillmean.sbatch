#!/bin/bash
#SBATCH --job-name=fillmean_reg
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.azzalin@studio.unibo.it
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=fillmean_reg
#SBATCH --gres=gpu:1

cd ../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

export JOB_NAME=fillmean
export PROBLEM_TYPE=regression
export FILL_NAN=mean

python3 src/invalsi.py