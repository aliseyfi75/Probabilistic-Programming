#!/bin/bash
#SBATCH --array=10,50,100,200,300,400,500,750,1000,1250,1500
#SBATCH --job-name=alpha0001_wasserstein_hairpins
#SBATCH --account=def-condon
#SBATCH --mail-user=jlovrod@cs.ubc.ca 
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=8000
#SBATCH --time=50:00:00

module load python/3.7
source /home/jlovrod/projects/def-condon/jlovrod/dp_gen_env/bin/activate

python compare_SC_in_IS.py --alpha=0.0001 --n_samples $SLURM_ARRAY_TASK_ID --fulldata=False --squaredLik=False --BBVI_prior=False --distance=wasserstein