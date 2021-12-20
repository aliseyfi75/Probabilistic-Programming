#!/bin/bash
#SBATCH --array=100,250,500,750,1000,2500,5000,7500
#SBATCH --job-name=smc_without
#SBATCH --account=def-rngubc
#SBATCH --mail-user=ali.seyfi.12@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=10:00:00

module load python/3.7
source $HOME/pytorch/bin/activate

python -u smc_IS.py --N $SLURM_ARRAY_TASK_ID > smc_$SLURM_ARRAY_TASK_ID.out