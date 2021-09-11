#!/bin/bash
#SBATCH --job-name=noearlyepoch100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.azzalin@studio.unibo.it
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=noearlyepoch100
#SBATCH --gres=gpu:1

cd ../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

export JOB_NAME=noearlyepoch100
export EPOCH=100
export EARLY_STOPPING=False

python3 src/invalsi.py