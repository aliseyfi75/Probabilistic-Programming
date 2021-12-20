#!/bin/bash
#SBATCH --array=1,2,3,4,5
#SBATCH --job-name=IS_SC_kssquare_hairpins_only
#SBATCH --account=def-condon
#SBATCH --mail-user=jlovrod@cs.ubc.ca 
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4000
#SBATCH --time=5:00:00

module load python/3.7
source /home/jlovrod/projects/def-condon/jlovrod/dp_gen_env/bin/activate

wandb login 33f5ffa304d256a59512bb634dcf8da21304837f

python IS_with_stochastic_conditionning.py --alpha $SLURM_ARRAY_TASK_ID