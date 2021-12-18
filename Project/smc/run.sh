#!/bin/bash
#SBATCH --job-name=SMC
#SBATCH --account=def-rngubc
#SBATCH --mail-user=ali.seyfi.12@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=50:00:00

module load python/3.7
source $HOME/pytorch/bin/activate

python -u smc.py > SMC.out