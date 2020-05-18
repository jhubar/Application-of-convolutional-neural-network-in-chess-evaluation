#!/bin/bash
#
#SBATCH --job-name=Chess-deepEvaluator-Julien
#SBATCH --output=output-Julien-mediumDS.txt
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=julien1941@live.be
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-72:00:00
#SBATCH --mem-per-cpu=32g
#SBATCH --partition=all
#sbatch --mem=16go
#SBATCH --gres=gpu:1

python3 deepEvaluator-Julien.py