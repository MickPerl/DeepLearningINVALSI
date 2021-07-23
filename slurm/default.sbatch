#!/bin/bash
#SBATCH --job-name=default
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.azzalin@studio.unibo.it
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=default
#SBATCH --gres=gpu:1

. bin/activate # per attivare il virtual environment python

pip3 install -r requirements.txt

python3 src/base.py