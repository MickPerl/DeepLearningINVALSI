#!/bin/bash
#SBATCH --job-name=batch128
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.azzalin@studio.unibo.it
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=batch128
#SBATCH --gres=gpu:1

cd ../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

export JOB_NAME=batch128
export BATCH_SIZE=128

python3 src/invalsi.py