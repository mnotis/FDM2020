#!/bin/bash
#SBATCH --job-name=pics    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=mnotis@princeton.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -o OUTPUTpics.lsf
#SBATCH -e ERRORpics.lsf

module purge
module load anaconda3

srun python pics.py

#EOF
