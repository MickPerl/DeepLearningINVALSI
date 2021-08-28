#!/bin/bash
#SBATCH --job-name=sgd001
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.azzalin@studio.unibo.it
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=sgd001
#SBATCH --gres=gpu:1

cd  ../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

python3 src/base.py --optimizer sgd --learningrate 0.01 
