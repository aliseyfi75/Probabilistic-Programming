#!/bin/bash
#SBATCH --job-name=BBVI
#SBATCH --account=def-rngubc
#SBATCH --mail-user=ali.seyfi.12@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=2000
#SBATCH --time=10:00:00

module load python/3.7
source /home/jlovrod/projects/def-condon/jlovrod/dp_gen_env/bin/activate

wandb login 33f5ffa304d256a59512bb634dcf8da21304837f

python -u blackbox_VI.py --L $SLURM_ARRAY_TASK_ID --lr 0.005 > bbvi.out