#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="PG6_test"
#SBATCH --get-user-env
#SBATCH -o JobOutput_%x_%j.out # Name of stdout output file (%j expands to jobId)
# #SBATCH --output=_scheduler-%J_stdout.txt
# #SBATCH --error=_scheduler-%J_stderr.txt
#SBATCH --partition=i8cpu
#SBATCH -N 2
#SBATCH -n 128
#SBATCH --mail-type=all        #available type:BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=tragic44cg@icloud.com

export OMP_NUM_THREADS=128
module purge
module load gcc
module load oneapi_mkl
module load oneapi_mpi
module list

for i in {1..1};
do
    # ../build/test /work/k0151/k015124/14_pg/20230518_1ns_gromacs/analyze_data/60mol_1/gromacs_trajectory_cell.xyz ../bondlist.txt
    #  /work/k0151/k015124/14_pg/20230518_1ns_gromacs/60mol_${i}/gromacs_test/gromacs_trajectory_cell.xyz
    # xyzを一旦linkして，計算終了後にlinkを解除する
    # ln -s /home/k0151/k015124/14_pg/ppg725_1ns_gromacs/30mol400K_${i}/gromacs_test/gromacs_trajectory_cell.xyz ./
    ~/works/dieltools/notebook/c++/src/build/dieltools descriptor_cc.inp
    # ~/works/dieltools/notebook/c++/src/build/dieltools descriptor_cc.inp
    # ~/c++/build_normal/test  ../PPG725_bondlist.txt descripter.inp
    # CPml.py descripter.inp
    # unlink gromacs_trajectory_cell.xyz
    # mv total_dipole.txt total_dipole_30mol${i}_20000.txt
done

# 
