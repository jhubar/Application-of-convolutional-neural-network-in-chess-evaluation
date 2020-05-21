#!/bin/bash
#
#SBATCH --job-name=PO_B_256
#SBATCH --output=output-Pierre-Pooling-Batch256.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-12:00:00
#SBATCH --mem-per-cpu=32g
#SBATCH --partition=all
#SBATCH --gres=gpu:1

python3 deepEvaluator-Pierre-Pooling.py
