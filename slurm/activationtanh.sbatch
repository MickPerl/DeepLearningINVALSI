#!/bin/bash
#SBATCH --job-name=activationtanh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.azzalin@studio.unibo.it
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=activationtanh
#SBATCH --gres=gpu:1

cd ../

. venv/bin/activate # per attivare il virtual environment python

pip3 install --no-cache-dir -r requirements.txt

export LEAKY_RELU=False
export DENSE_LAYER_ACTIVATION=tanh

python3 src/invalsi.py