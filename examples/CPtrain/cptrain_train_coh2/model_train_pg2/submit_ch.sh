#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="train_ch"
#SBATCH --get-user-env
#SBATCH -o JobOutput_%x_%j.out # Name of stdout output file (%j expands to jobId)
# #SBATCH --output=_scheduler-%J_stdout.txt
# #SBATCH --error=_scheduler-%J_stderr.txt
#SBATCH --partition=i8cpu
#SBATCH -N 2
#SBATCH -n 256
#SBATCH --mail-type=all        #available type:BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=tragic44cg@icloud.com

. ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate test_dieltools

export OMP_NUM_THREADS=128
module purge
# module load openmpi
module load gcc
module load oneapi_mkl
module load oneapi_mpi
module list

CPtrain.py train --i input.yaml

