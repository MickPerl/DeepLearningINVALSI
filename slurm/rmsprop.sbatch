#!/bin/bash
#SBATCH --job-name=rmsprop
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.perlino@studio.unibo.it
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=rmsprop
#SBATCH --gres=gpu:1

cd  ../

. venv/bin/activate # per attivare il virtual environment python

pip3 install -r requirements.txt

python3 src/base.py --optimizer rms
