#!/bin/bash
#SBATCH --job-name=BBVI
#SBATCH --account=def-rngubc
#SBATCH --mail-user=ali.seyfi.12@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=4000
#SBATCH --time=50:00:00

module load python/3.7
source $HOME/pytorch/bin/activate

wandb login 47c7b14ac80e20c67500d9e9ff90f61f5bc447e2

python -u blackbox_VI.py --L 5 --lr 0.05 > bbvi.out