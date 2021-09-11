#!/bin/bash
#SBATCH --job-name=dense15_cla
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.azzalin@studio.unibo.it
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=dense15_cla
#SBATCH --gres=gpu:1

cd ../../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

export JOB_NAME=dense15
export NUMBER_OF_LAYERS=15

python3 src/invalsi.py