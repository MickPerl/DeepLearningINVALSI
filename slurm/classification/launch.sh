#!/bin/bash
for f in *.sbatch
do
    sbatch "$f"
done