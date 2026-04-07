#!/bin/bash
#SBATCH --partition=long
#SBATCH --output=logs/subnets-%j.out
#SBATCH --error=logs/subnets-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=subnets

export HOME="/home/mila/k/kondrupe"
module load python/3.10
source ~/CrediGraphSub/.venv/bin/activate

uv run python get_topk.py $SCRATCH 50 > topk.csv
