#!/bin/bash
#SBATCH --job-name=BBVI
#SBATCH --account=def-condon
#SBATCH --mail-user=jlovrod@cs.ubc.ca 
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4000
#SBATCH --time=3:00:00

module load python/3.7
source /home/jlovrod/projects/def-condon/jlovrod/dp_gen_env/bin/activate

wandb login 33f5ffa304d256a59512bb634dcf8da21304837f

python blackbox_VI.py