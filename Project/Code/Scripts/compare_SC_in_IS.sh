#!/bin/bash
#SBATCH --array=1,2,3,4,5
#SBATCH --job-name=IS_ks_squared_hairpins
#SBATCH --account=def-condon
#SBATCH --mail-user=jlovrod@cs.ubc.ca 
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=4000
#SBATCH --time=50:00:00

module load python/3.7
source /home/jlovrod/projects/def-condon/jlovrod/dp_gen_env/bin/activate

python compare_SC_in_IS.py --alpha $SLURM_ARRAY_TASK_ID