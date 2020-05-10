#!/bin/bash
#
#SBATCH --job-name=Chess-deepEvaluator-Pierre
#SBATCH --output=output-Pierre-MLP.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-72:00:00
#SBATCH --mem-per-cpu=32g
#SBATCH --partition=all
#SBATCH --gres=gpu:1

python3 deepEvaluator-Pierre-MLP.py