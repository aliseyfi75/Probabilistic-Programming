#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=def-rngubc
#SBATCH --mail-user=ali.seyfi.12@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256
#SBATCH --time=0:03:00

module load python/3.7
source $HOME/pytorch/bin/activate

python test.py