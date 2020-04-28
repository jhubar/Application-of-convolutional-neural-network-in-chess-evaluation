#!/bin/bash
#
#SBATCH --job-name=Chess-generateDataset
#SBATCH --output=res-generateDataset2014.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-05:00:00
#SBATCH --mem-per-cpu=32g
#SBATCH --partition=all
#SBATCH --gres=gpu:1

python3 generateDataset2014.py
