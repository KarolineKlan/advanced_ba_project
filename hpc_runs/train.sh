#!/bin/bash
#BSUB -J train_unet

## output files
#BSUB -o ../hpc_runs/Run_%J.out.txt
#BSUB -e ../hpc_runs/Run_%J.err.txt

## GPU
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -n 4

## runtime
#BSUB -W 24:00

## mail when done
#BSUB -N

source ../aba_env/bin/activate

## Run training with Hydra (full dataset)
python src/advanced_ba_project/train.py
