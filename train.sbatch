#!/bin/bash
#SBATCH -c 2
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=16G
#SBATCH --job-name=trioformers
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL


/home/tomer.erez/miniconda3/envs/nnn/bin/python -u main.py --batch_size 16