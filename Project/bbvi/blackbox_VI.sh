#!/bin/bash
#SBATCH --array=10,15,20,25,30
#SBATCH --job-name=BBVI
#SBATCH --account=def-rngubc
#SBATCH --mail-user=ali.seyfi.12@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4000
#SBATCH --time=3:00:00

module load python/3.7
source $HOME/pytorch/bin/activate

python blackbox_VI.py --L $SLURM_ARRAY_TASK_ID --lr 0.05