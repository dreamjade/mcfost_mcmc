#!/bin/bash

#SBATCH --job-name=RZPSC
#SBATCH --time=160:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=94
#SBATCH --account=douglase
#SBATCH --partition=high_priority

module add python/3.9/3.9.10
module add mcfost
export MCFOST_UTILS=~/mcfost/utils
export OMP_NUM_THREADS=94
mcfost -setup
source ~/YU_Python3/bin/activate
python3 RZ_Psc_mcmc_py_c.py
