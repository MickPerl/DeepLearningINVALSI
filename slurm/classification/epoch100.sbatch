#!/bin/bash
#SBATCH --job-name=epoch100_cla
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.azzalin@studio.unibo.it
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=epoch100_cla
#SBATCH --gres=gpu:1

cd ../../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

export JOB_NAME=epoch100
export EPOCH=100

python3 src/invalsi.py