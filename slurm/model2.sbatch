#!/bin/bash
#SBATCH --job-name=model2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.perlino@studio.unibo.it
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=model2
#SBATCH --gres=gpu:1

cd  ../

. venv/bin/activate # per attivare il virtual environment python

pip3 install -r requirements.txt

python3 src/base.py --model 2
