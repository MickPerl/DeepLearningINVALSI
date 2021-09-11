#!/bin/bash
#SBATCH --job-name=neurons256
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.azzalin@studio.unibo.it
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=neurons256
#SBATCH --gres=gpu:1

cd ../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

export JOB_NAME=neurons256
export NEURONS=256

python3 src/invalsi.py