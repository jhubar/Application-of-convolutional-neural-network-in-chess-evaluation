#!/bin/bash
#SBATCH --job-name=JUJU_PIERRE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=PostP
#SBATCH --gres=gpu:1

srun python3 deepEvaluator.py
