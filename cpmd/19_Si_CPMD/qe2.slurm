#!/bin/bash
#SBATCH -J cp.x               # Job name
#SBATCH -o JobOutput_%j.out   # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 24                 # Total number of mpi tasks requested
#SBATCH -t 72:00:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p sky6126            # partition name (node name) sky6126

# tanryo_test2でborn effective chargeの計算を行う．


# Assuming module load prun (as well as any mpi module e.g. impi)
module load oneapi/compiler
module load oneapi/mkl
module load oneapi/mpi


# unlimit the maximum number of the processors
ulimit -s unlimited



# output calculation settings
echo
echo    START DATE       : `date`
echo   SLURM_JOBID       : ${SLURM_JOBID}
echo SLURM_SUBMIT_DIR    : ${SLURM_SUBMIT_DIR}
echo SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK}
echo SLURM_JOB_NUM_NODES : ${SLURM_JOB_NUM_NODES} # num of nodes allocated to the job
echo   SLURM_NTASKS      : ${SLURM_NTASKS}

#echo   PBS_NODENUM    : ${PBS_NODENUM}
#echo    PBS_O_HOST    : ${PBS_O_HOST}
#echo    PBS_SERVER    : ${PBS_SERVER}
#echo   PBS_O_QUEUE    : ${PBS_O_QUEUE}
#echo   PBS_ARRAYID    : ${PBS_ARRAYID}
#echo     PBS_QUEUE    : ${PBS_QUEUE}
#echo      used_nodes  : ${used_nodes}
#echo       num_nodes  : ${num_nodes}
#echo   proc_per_node  : ${proc_per_node}
#echo        num_proc  : ${num_proc}
echo     hostname     : `hostname`
echo


#
#######################  End of Initialization   ##############################
#

# move to working dir
cd $SLURM_SUBMIT_DIR

# dependent files dir (slack_send.sh, environment variables....)
DEPENDENT_DIR=/home/amano/sample

# set the needed environment variables
. $DEPENDENT_DIR/environment_variables
echo
echo "environment setting done"
echo 


# check if pw and pseudo directories exist
. $DEPENDENT_DIR/check_directory.sh
echo
echo "check directory done"
echo 

# 保存用にdirを作成
mkdir ${SLURM_JOBID}


####################  Input ######################################
# prefixをファイルから読み込む
#scf=h2o_bec
if [ ! -e prefix ]; then
    echo
    echo " !!!error:::prefixファイルが存在しません!!!"
    echo " abort"
    exit 1;
fi

# prefix読み込み
. ./prefix

# 結果の確認
echo prefix name is ::${prefix}



####################  Input Files (modify here)######################################
#

# phin= ${とするとダメだった．spaceがあるとまずい？
scfin=${prefix}.pw.in
phin=${prefix}.ph.in
nscfin=${prefix}.nscf.in


# echo
# echo "scf file name "${scfin}
# #echo "nscf file name "${nscfin}
# echo

# #check if input file exists
# if [ -f ${scfin} ]; then
#     echo
#     echo "入力ファイルが存在しています．"
#     echo
# else
#     echo
#     echo "!!!error:::入力ファイルが存在しません!!!"
#     echo
#     exit 1;
# fi

####################  Output Files ######################################
#
scfout=${prefix}.pw.out
phout=${prefix}.ph.out
nscfout=${prefix}.nscf.out

# echo "scf file name "${scfout}
# echo "nscf file name "${nscfout}

#-> STEP 1: relax electrons (optionally ions) to the ground state
mpirun -np 24 cp.x < si.1_erelax.in > si.1_erelax.out

# (optionally: check the stress of the cell)

# -> STEP 2: add some random displacement
mpirun -np 24 cp.x < si.2_randDisplace.in > si.2_randDisplace.out

remove si.* in the output directory

# -> STEP 3: start Car-Parrinello Molecular Dynamics (CPMD) calculation (NVE)
mpirun -np 24 cp.x < si.3_cp_NVE.in > si.3_cp_NVE.out

plot temperature
    gnuplot
    plot "outdir/si.evp" using 2:5 with linespoints


#################################
## post commands


# slack notification
# echo JOBID::${SLURM_JOBID} prefix::${prefix} finished | $DEPENDENT_DIR/slack_send.sh


# finish
echo
echo "PROGRAM END"
echo    END DATE       : `date`
